"""Per-request prompt synthesis for the load senders.

Why this exists
---------------
Both senders used to reuse one constant prompt (``" ".join(["token"] * n)``) for every
request of a run. Against an engine with prefix caching enabled that makes prefill free
after the first request: the measured capacity then *rises* with prompt length instead
of falling, which silently invalidates any capacity/theta calibration built on it.

A prompt built here is therefore **unique per request**: two different seed keys differ
inside the first few tokens by construction, so no shared prefix of any useful length
exists between two requests.

Seed policy
-----------
Content is a pure function of the caller's ``seed_key`` string - nothing here reads the
clock, the process id, or global RNG state - so a replay of the same run/cell/sequence
reproduces byte-identical prompts. The senders build the key as:

* replayer (:mod:`tre_replayer.engine.http_sender`): ``"<model>|<request_id>"``. Trace
  request ids are already deterministic per trace.
* grid driver (``deploy/scripts/r3_grid.py``): ``"<run_key>|<cell_id>|<sequence>"``,
  where ``sequence`` is a monotone per-cell counter. Which worker thread picks up which
  sequence number depends on timing, so the *set* of prompts a cell sends is
  reproducible while the send order is not.

The key is hashed with BLAKE2b (not :func:`hash`, which is salted per process) into a
64-bit seed.

Modes
-----
:data:`MODE_NATURAL` (the default) sends English prose - see
:mod:`tre_replayer.engine.corpus` - fitted to the exact target token count with the
model's own tokenizer, loaded from local disk (see
:mod:`tre_replayer.engine.model_tokenizer`). It keeps every property the calibration
depends on (exact realised length, determinism, per-request uniqueness) and adds a
realistic token distribution and attention pattern. It costs one tokenizer load per
model plus a few milliseconds of encoding per request, and it needs the model name; a
caller that can supply neither has to fall back to :data:`MODE_TOKEN_IDS`.

:data:`MODE_TOKEN_IDS` sends the OpenAI-compatible ``prompt`` field as an explicit list
of token ids, which vLLM consumes verbatim: the realised ``usage.prompt_tokens`` is then
exactly the requested length, with no tokenizer in the sender and no whitespace-splitting
approximation. Ids are drawn from ``[TOKEN_ID_MIN, TOKEN_ID_MAX)``, a window inside the
vocabulary of every model in the fleet (smallest vocab 128256) and below every
special/added token id, so decoding is well defined for all of them. What it sends is
*not* language: uniformly random ids are semantically meaningless, so neither the
attention pattern they produce nor the text they decode to resembles real traffic.

:data:`MODE_TEXT` is the fallback for a gateway that will not forward a token-id list
where no local tokenizer is available either. Its realised length is only nominal: it
relies on every word in :data:`_TEXT_WORDS` costing exactly one token. Measured against
the three fleet tokenizers (DeepSeek-R1-Distill Qwen-7B / Llama-8B / Qwen-14B) at
128 / 512 / 1024 tokens the error is 0.00 %, but that is a property of those
vocabularies, not a guarantee.

All three modes hold the same two invariants: content is a pure function of the seed
key, and two different seed keys differ within the first few tokens by construction.
"""
from __future__ import annotations

import hashlib
import random

MODE_TOKEN_IDS = "token_ids"
MODE_TEXT = "text"
MODE_NATURAL = "natural"
MODES = (MODE_TOKEN_IDS, MODE_TEXT, MODE_NATURAL)

#: The mode a sender uses unless told otherwise. Natural language, because the
#: calibration it feeds is meant to predict behaviour on real traffic and uniformly
#: random token ids are not real traffic - and because it holds the exactness,
#: determinism and uniqueness guarantees that made the random ids necessary in the first
#: place. Callers that cannot reach a tokenizer must pass :data:`MODE_TOKEN_IDS`
#: explicitly; the failure is loud, never a silent downgrade.
DEFAULT_MODE = MODE_NATURAL

#: Inclusive lower / exclusive upper bound of the token-id window used for synthesis.
#: Below 1024 sit byte-fallback and control ids; 100000 is under the smallest fleet
#: vocabulary (Llama-8B: 128256) and under the lowest special-token id of any of them.
TOKEN_ID_MIN = 1024
TOKEN_ID_MAX = 100000

#: Leading tokens that encode the seed itself. 4 tokens of base 98976 address 9.6e19
#: values > 2**64, so distinct seeds are guaranteed to differ within the preamble.
PREAMBLE_TOKENS = 4

#: Same guarantee for MODE_TEXT: 11 words of base 64 address 7.4e19 values > 2**64.
TEXT_PREAMBLE_WORDS = 11

#: Words for MODE_TEXT. Each is a single token (with its leading space) under the BPE
#: vocabularies of the fleet; keeping the list short and lowercase keeps it that way.
_TEXT_WORDS = (
    "time", "year", "people", "way", "day", "man", "thing", "woman", "life", "child",
    "world", "school", "state", "family", "student", "group", "country", "problem",
    "hand", "part", "place", "case", "week", "company", "system", "program", "question",
    "work", "number", "night", "point", "home", "water", "room", "mother", "area",
    "money", "story", "fact", "month", "book", "eye", "job", "word", "business", "issue",
    "side", "kind", "head", "house", "service", "friend", "father", "power", "hour",
    "game", "line", "end", "member", "law", "car", "city", "name", "team", "minute",
)

#: English prose costs roughly 1.35 tokens per whitespace word under these BPE
#: vocabularies (punctuation included). Used only to size the first draft; the fit loop
#: below is what makes the length exact, so an inaccurate ratio costs an extra round,
#: never correctness.
WORDS_PER_TOKEN = 0.78

#: Rounds the fit loop may take before giving up. Each round either truncates in token
#: space (which moves the count by exactly the overshoot, modulo a decode/re-encode
#: boundary effect of at most a token or two) or closes a small deficit with
#: single-token fillers, so convergence normally takes two or three.
MAX_FIT_ROUNDS = 16

#: A deficit at or below this is closed with filler words rather than another sentence,
#: so the loop cannot oscillate between overshooting and undershooting.
SMALL_DEFICIT_TOKENS = 12


class PromptFitError(RuntimeError):
    """The natural-language fitter could not hit the exact target token count."""


def prompt_seed(seed_key: str) -> int:
    """Stable 64-bit seed for ``seed_key`` (BLAKE2b; identical across processes/runs)."""
    digest = hashlib.blake2b(seed_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def build_token_id_prompt(
    token_count: int,
    seed_key: str,
    *,
    token_id_min: int = TOKEN_ID_MIN,
    token_id_max: int = TOKEN_ID_MAX,
) -> list[int]:
    """Exactly ``token_count`` token ids, unique per ``seed_key``.

    The first :data:`PREAMBLE_TOKENS` ids encode the seed positionally (so two keys
    diverge immediately); the remainder is drawn from a seeded RNG.
    """
    count = max(1, int(token_count))
    span = token_id_max - token_id_min
    if span < 2:
        raise ValueError("token id window must hold at least two ids")
    seed = prompt_seed(seed_key)
    ids: list[int] = []
    value = seed
    for _ in range(min(PREAMBLE_TOKENS, count)):
        ids.append(token_id_min + value % span)
        value //= span
    rng = random.Random(seed)
    while len(ids) < count:
        ids.append(token_id_min + rng.randrange(span))
    return ids


def build_text_prompt(token_count: int, seed_key: str) -> str:
    """Whitespace-joined words of approximately ``token_count`` tokens, unique per key.

    The leading :data:`TEXT_PREAMBLE_WORDS` words spell the seed out positionally in
    base ``len(_TEXT_WORDS)`` (so distinct seeds diverge inside the preamble, exactly as
    in the token-id mode); the rest is drawn from a seeded RNG. Realised length is
    approximate - see the module docstring - so :func:`build_natural_prompt` or
    :func:`build_token_id_prompt` is preferred wherever either is available.
    """
    count = max(1, int(token_count))
    seed = prompt_seed(seed_key)
    base = len(_TEXT_WORDS)
    words: list[str] = []
    value = seed
    for _ in range(min(TEXT_PREAMBLE_WORDS, count)):
        words.append(_TEXT_WORDS[value % base])
        value //= base
    rng = random.Random(seed)
    while len(words) < count:
        words.append(_TEXT_WORDS[rng.randrange(base)])
    return " ".join(words[:count])


def build_natural_prompt(
    token_count: int,
    seed_key: str,
    *,
    model: str | None = None,
    tokenizer=None,
    tokenizer_path: str | None = None,
) -> str:
    """English prose of *exactly* ``token_count`` tokens, unique per ``seed_key``.

    ``token_count`` is the count vLLM will report as ``usage.prompt_tokens`` - special
    tokens included - not the plain token count, because that is the number the caller
    asked the engine for and the number the calibration grid is indexed by.

    ``tokenizer`` (a :class:`~tre_replayer.engine.model_tokenizer.ModelTokenizer`) is the
    seam the tests inject; otherwise ``model`` is resolved to a tokenizer on local disk.

    The fit is a loop rather than a formula because a BPE tokenizer is not additive
    across a truncation boundary: cutting the id list at N and decoding can re-encode to
    N-1 or N+1 tokens. Each round either truncates by the exact overshoot or closes a
    small deficit with a filler word that is known to cost one token under this
    tokenizer, so the count converges monotonically. The head of the text - the sentence
    carrying the seed - is never touched, which is what preserves uniqueness.
    """
    from tre_replayer.engine import corpus

    target = max(1, int(token_count))
    tok = tokenizer if tokenizer is not None else _load_tokenizer(model, tokenizer_path)
    # The tokenizer's own special tokens are part of what vLLM counts, so a prompt of one
    # token below them is unrepresentable - and an empty prompt is not a request.
    minimum = tok.overhead + 1
    if target < minimum:
        raise ValueError(
            f"a natural prompt for {model or tok.path!r} cannot be shorter than {minimum} "
            f"tokens ({tok.overhead} special token(s) plus at least one of its own); "
            f"asked for {target}"
        )

    builder = corpus.TextBuilder(prompt_seed(seed_key))
    text = builder.ensure_words(int(target * WORDS_PER_TOKEN) + 24)
    for _ in range(MAX_FIT_ROUNDS):
        realised = tok.count(text)
        if realised == target:
            return text
        if realised > target:
            ids = tok.encode_plain(text)
            keep = max(1, len(ids) - (realised - target))
            text = tok.decode_plain(ids[:keep])
            continue
        deficit = target - realised
        if deficit <= SMALL_DEFICIT_TOKENS:
            text = text + tok.filler * deficit
            continue
        text = builder.ensure_words(builder.words + int(deficit * WORDS_PER_TOKEN) + 8)
    raise PromptFitError(
        f"could not fit a natural prompt to {target} tokens for model {model!r} in "
        f"{MAX_FIT_ROUNDS} rounds (last realised {tok.count(text)})"
    )


def _load_tokenizer(model: str | None, tokenizer_path: str | None):
    from tre_replayer.engine.model_tokenizer import TokenizerUnavailable, load_tokenizer

    if not model and not tokenizer_path:
        raise TokenizerUnavailable(
            f"prompt mode {MODE_NATURAL!r} needs a model name (or an explicit tokenizer "
            "path) to load the tokenizer that makes the length exact"
        )
    return load_tokenizer(model or "", tokenizer_path=tokenizer_path)


def build_prompt(
    token_count: int,
    seed_key: str,
    *,
    mode: str = DEFAULT_MODE,
    model: str | None = None,
    tokenizer=None,
    tokenizer_path: str | None = None,
) -> list[int] | str:
    """Dispatch to the per-mode builder. ``model`` is required by :data:`MODE_NATURAL`."""
    if mode == MODE_TOKEN_IDS:
        return build_token_id_prompt(token_count, seed_key)
    if mode == MODE_TEXT:
        return build_text_prompt(token_count, seed_key)
    if mode == MODE_NATURAL:
        return build_natural_prompt(
            token_count, seed_key, model=model, tokenizer=tokenizer, tokenizer_path=tokenizer_path
        )
    raise ValueError(f"unknown prompt mode: {mode!r} (expected one of {MODES})")
