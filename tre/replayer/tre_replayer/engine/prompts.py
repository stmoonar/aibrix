"""Per-request prompt synthesis for the load senders.

Why this exists
---------------
Both senders used to reuse one constant prompt (``" ".join(["token"] * n)``) for every
request of a run. Against an engine with prefix caching enabled that makes prefill free
after the first request: the measured capacity then *rises* with prompt length instead
of falling, which silently invalidates any capacity/theta calibration built on it.

A prompt built here is therefore **unique per request**: two different seed keys differ
inside the first :data:`PREAMBLE_TOKENS` tokens by construction, so no shared prefix of
any useful length exists between two requests.

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

Token ids vs text
-----------------
:data:`MODE_TOKEN_IDS` (the default) sends the OpenAI-compatible ``prompt`` field as an
explicit list of token ids, which vLLM consumes verbatim: the realised
``usage.prompt_tokens`` is then exactly the requested length, with no tokenizer in the
sender and no whitespace-splitting approximation. Ids are drawn from
``[TOKEN_ID_MIN, TOKEN_ID_MAX)``, a window inside the vocabulary of every model in the
fleet (smallest vocab 128256) and below every special/added token id, so decoding is
well defined for all of them.

:data:`MODE_TEXT` is the fallback for a gateway that will not forward a token-id list.
Its realised length is only nominal: it relies on every word in :data:`_TEXT_WORDS`
costing exactly one token. Measured against the three fleet tokenizers
(DeepSeek-R1-Distill Qwen-7B / Llama-8B / Qwen-14B) at 128 / 512 / 1024 tokens the error
is 0.00 %, but that is a property of those vocabularies, not a guarantee - which is why
token ids are the default.
"""
from __future__ import annotations

import hashlib
import random

MODE_TOKEN_IDS = "token_ids"
MODE_TEXT = "text"
MODES = (MODE_TOKEN_IDS, MODE_TEXT)

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
    approximate - see the module docstring - so :func:`build_token_id_prompt` is
    preferred wherever the endpoint accepts token ids.
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


def build_prompt(
    token_count: int,
    seed_key: str,
    *,
    mode: str = MODE_TOKEN_IDS,
) -> list[int] | str:
    """Dispatch to :func:`build_token_id_prompt` / :func:`build_text_prompt`."""
    if mode == MODE_TOKEN_IDS:
        return build_token_id_prompt(token_count, seed_key)
    if mode == MODE_TEXT:
        return build_text_prompt(token_count, seed_key)
    raise ValueError(f"unknown prompt mode: {mode!r} (expected one of {MODES})")
