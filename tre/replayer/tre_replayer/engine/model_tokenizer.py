"""Offline resolution and loading of the fleet's real tokenizers.

The natural-language prompt mode has to hit an *exact* ``usage.prompt_tokens``, and the
only way to do that is to count with the same tokenizer the engine uses. This module
finds that tokenizer on local disk and never touches the network: the cluster is
network-restricted, so a download attempt does not degrade gracefully, it hangs.

Resolution order for a model name (first hit wins):

1. an explicit path passed by the caller;
2. ``TRE_TOKENIZER_PATHS`` - a JSON object ``{"<model>": "<dir>"}``;
3. the TRE registry (``TRE_REGISTRY_PATH`` or ``/etc/tre/registry.yaml``), whose
   ``models[].weights_path`` is the directory the vLLM pods mount and load from, so it
   is by definition the right tokenizer;
4. :data:`FLEET_TOKENIZER_PATHS`, the committed fallback for the three fleet models.

Counting contract
-----------------
``/v1/completions``: :class:`ModelTokenizer` counts ``len(plain ids) + overhead``, where
``overhead`` is what the ``transformers`` wrapper adds around a prompt (one BOS for the
three fleet tokenizers), measured at load time rather than assumed. (Measured
2026-09-30: the fleet's vLLM 0.30 engines add no BOS to a completions prompt, so a
completions prompt built here realises one token less than asked. Kept as it was: the
trace replays send completions and must keep sending the same bytes.)

``/v1/chat/completions``: :func:`for_api` returns a :class:`ChatTemplateTokenizer`
view that counts the prompt the engine actually tokenises - the chat template rendered
around the content (``apply_chat_template(add_generation_prompt=True)``), tokenised
without added special tokens, as vLLM does. The template is rendered once at load time
into a prefix and a suffix, and both the render and the additivity of the count are
checked on probe contents; a template that transforms its content (trim, escaping) or
whose count is not additive is refused rather than approximated.

Thread safety
-------------
The hot path uses ``transformers``' *backend* tokenizer - the Rust
:class:`tokenizers.Tokenizer` - whose ``encode``/``decode`` do not mutate it. The
``transformers`` wrapper's own ``encode`` sets truncation and padding state on that same
backend object, which is not safe from the sender's worker threads; it is therefore used
only once, at load time, to establish ``overhead``.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

#: Committed fallback: where the three fleet models' weights (and tokenizers) sit on the
#: shared NFS mount that the vLLM pods load from. Kept in sync with deploy/registry.yaml.
FLEET_TOKENIZER_PATHS = {
    "dsqwen-7b": "/data/nfs_shared_data/DeepSeek-R1-Distill-Qwen-7B",
    "dsllama-8b": "/data/nfs_shared_data/DeepSeek-R1-Distill-Llama-8B",
    "dsqwen-14b": "/data/nfs_shared_data/Models/DeepSeek-R1-Distill-Qwen-14B",
}

#: Env var holding a JSON ``{model: dir}`` override.
PATHS_ENV = "TRE_TOKENIZER_PATHS"
#: Env var pointing at the registry file to read ``weights_path`` from.
REGISTRY_ENV = "TRE_REGISTRY_PATH"
DEFAULT_REGISTRY_PATH = "/etc/tre/registry.yaml"

#: Candidate filler words for the exact-length fit. The first one that costs exactly one
#: token under a given tokenizer is used; " the" holds for every BPE vocabulary in the
#: fleet, the rest are there so a future model cannot silently break the fit.
FILLER_CANDIDATES = (" the", " and", " of", " a")


class TokenizerUnavailable(RuntimeError):
    """No local tokenizer could be resolved or loaded for a model."""


#: Stand-in content the chat template is rendered around to find its prefix and suffix.
CHAT_SENTINEL = "\u2063TRE-CHAT-CONTENT\u2063"

#: Contents the template render and the count's additivity are checked on: English,
#: Chinese, the mix, and the edges a template could trim or merge across.
CHAT_PROBES = (
    "Hello world",
    "你好，世界。今天的天气很好。",
    "Reference 12345. 这是一个测试。 The end",
    " leading space",
    "trailing space ",
    "line one\nline two",
)


def _from_env_paths(model: str) -> str | None:
    raw = os.environ.get(PATHS_ENV)
    if not raw:
        return None
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TokenizerUnavailable(f"{PATHS_ENV} is not valid JSON: {exc}") from exc
    value = mapping.get(model)
    return str(value) if value else None


def _from_registry(model: str) -> str | None:
    path = os.environ.get(REGISTRY_ENV, DEFAULT_REGISTRY_PATH)
    if not os.path.exists(path):
        return None
    try:
        import yaml  # local import: the replayer must import without PyYAML
    except ImportError:  # pragma: no cover - PyYAML is present wherever the registry is
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return None
    for entry in doc.get("models", []) or []:
        if isinstance(entry, dict) and entry.get("name") == model:
            weights = entry.get("weights_path")
            return str(weights) if weights else None
    return None


def resolve_tokenizer_path(model: str, *, tokenizer_path: str | None = None) -> str:
    """Local directory holding ``model``'s tokenizer. Raises if nothing resolves."""
    for candidate in (
        tokenizer_path,
        _from_env_paths(model),
        _from_registry(model),
        FLEET_TOKENIZER_PATHS.get(model),
    ):
        if candidate:
            return candidate
    raise TokenizerUnavailable(
        f"no local tokenizer for model {model!r}: pass one explicitly, set {PATHS_ENV} "
        f"to a JSON {{model: dir}} map, point {REGISTRY_ENV} at a registry that names it, "
        f"or add it to FLEET_TOKENIZER_PATHS (known: {sorted(FLEET_TOKENIZER_PATHS)})"
    )


class ModelTokenizer:
    """The counting/truncating operations the prompt fitter needs, and nothing else.

    ``backend`` is a :class:`tokenizers.Tokenizer`; injecting one directly is what the
    unit tests do, so they never need a real model on disk.
    """

    __slots__ = ("model", "path", "overhead", "filler", "_backend",
                 "chat_prefix", "chat_suffix", "chat_error", "_chat_view")

    #: The endpoint whose count :meth:`count` reproduces.
    api = "completions"

    def __init__(
        self,
        model: str,
        path: str,
        backend: Any,
        overhead: int,
        *,
        chat_prefix: str | None = None,
        chat_suffix: str | None = None,
        chat_error: str | None = None,
    ) -> None:
        self.model = model
        self.path = path
        self.overhead = int(overhead)
        self._backend = backend
        self.filler = self._pick_filler()
        # The chat template rendered around CHAT_SENTINEL (see _chat_template_parts);
        # None, with chat_error saying why, when the model has no usable template.
        self.chat_prefix = chat_prefix
        self.chat_suffix = chat_suffix
        self.chat_error = chat_error
        self._chat_view = None

    def _pick_filler(self) -> str:
        for candidate in FILLER_CANDIDATES:
            if len(self.encode_plain("word" + candidate)) - len(self.encode_plain("word")) == 1:
                return candidate
        raise TokenizerUnavailable(
            f"tokenizer for {self.model!r} has no single-token filler among "
            f"{FILLER_CANDIDATES}; the exact-length fit cannot close a small deficit"
        )

    def encode_plain(self, text: str) -> list[int]:
        return list(self._backend.encode(text, add_special_tokens=False).ids)

    def decode_plain(self, ids) -> str:
        return self._backend.decode(list(ids), skip_special_tokens=True)

    def count(self, text: str) -> int:
        """Token count as vLLM will report it in ``usage.prompt_tokens``."""
        return len(self.encode_plain(text)) + self.overhead

    def cjk_token_count(self, text: str) -> int:
        """Plain tokens of ``text`` that carry Chinese: the token decodes to text holding
        a CJK character, or to U+FFFD (a byte-level piece of a multi-byte character).

        This is how the Chinese share of a corpus prompt is *measured*, independently of
        how :func:`tre_replayer.engine.corpus.budgeted_text` *budgeted* it. The U+FFFD
        rule is exact for the corpus, whose only multi-byte characters are CJK. Token
        offsets are not used: the fleet's Llama-3 tokenizer reports empty spans for some
        merged CJK tokens.
        """
        from tre_replayer.engine.corpus import is_cjk

        count = 0
        for token_id in self.encode_plain(text):
            piece = self._backend.decode([token_id], skip_special_tokens=True)
            if any(ch == "\ufffd" or is_cjk(ch) for ch in piece):
                count += 1
        return count


class ChatTemplateTokenizer:
    """``base`` counted the way ``/v1/chat/completions`` counts: the content rendered
    inside the chat template (``prefix + content + suffix``) and tokenised without added
    special tokens - the template carries its own BOS and role markers, and the backend
    matches them as the special tokens they are.

    Everything else (plain encode / decode, the filler, the Chinese-token count) is the
    base tokenizer's, i.e. about the *content*: the corpus mix ratio is a property of what
    the user wrote, not of the template. ``overhead`` is the template's token count, and
    the constructor refuses a template whose count is not ``plain + overhead`` on
    :data:`CHAT_PROBES` - the fitter's budget arithmetic relies on it (its exactness does
    not: every round re-counts the whole templated prompt).
    """

    api = "chat"

    def __init__(self, base: Any, prefix: str, suffix: str) -> None:
        self.base = base
        self.model = getattr(base, "model", None)
        self.path = getattr(base, "path", None)
        self.filler = base.filler
        self.chat_prefix = prefix
        self.chat_suffix = suffix
        overheads = {self.count(p) - len(base.encode_plain(p)) for p in CHAT_PROBES}
        if len(overheads) != 1:
            raise TokenizerUnavailable(
                f"the chat template of {self.model or self.path!r} does not add a fixed number "
                f"of tokens to its content (saw {sorted(overheads)} on the probes): the exact "
                "templated length cannot be budgeted"
            )
        self.overhead = overheads.pop()

    def render(self, text: str) -> str:
        """The prompt string the engine tokenises for this content."""
        return self.chat_prefix + text + self.chat_suffix

    def encode_plain(self, text: str) -> list[int]:
        return self.base.encode_plain(text)

    def decode_plain(self, ids) -> str:
        return self.base.decode_plain(ids)

    def count(self, text: str) -> int:
        """Token count as vLLM reports it in ``usage.prompt_tokens`` for a chat request
        whose one user message is ``text``."""
        return len(self.base.encode_plain(self.render(text)))

    def cjk_token_count(self, text: str) -> int:
        return self.base.cjk_token_count(text)


def for_api(tok: Any, api: str) -> Any:
    """``tok`` counting for ``api``: itself for completions, its chat view for chat.

    Raises :class:`TokenizerUnavailable` when the model has no chat template this module
    can reproduce exactly - loudly, never a silent fallback to the completions count.
    """
    if api == "completions":
        if getattr(tok, "api", "completions") != "completions":
            raise ValueError("a chat-template view cannot count a completions prompt")
        return tok
    if api != "chat":
        raise ValueError(f"unknown API: {api!r}")
    if getattr(tok, "api", None) == "chat":
        return tok
    cached = getattr(tok, "_chat_view", None)
    if cached is not None:
        return cached
    prefix = getattr(tok, "chat_prefix", None)
    suffix = getattr(tok, "chat_suffix", None)
    if prefix is None or suffix is None:
        reason = getattr(tok, "chat_error", None) or "the tokenizer carries no chat template"
        name = getattr(tok, "model", None) or getattr(tok, "path", "?")
        raise TokenizerUnavailable(f"no exact chat-template count for {name!r}: {reason}")
    view = ChatTemplateTokenizer(tok, prefix, suffix)
    try:
        tok._chat_view = view
    except AttributeError:  # a stub without the slot: rebuilt per call, which is fine
        pass
    return view


def _chat_template_parts(wrapper: Any) -> tuple[str | None, str | None, str | None]:
    """``(prefix, suffix, error)`` of the wrapper's chat template around one user turn with
    the generation prompt, as vLLM renders a ``/v1/chat/completions`` request.

    Refused (prefix/suffix None, error set) when the template is missing, does not place
    the content exactly once and verbatim, or renders a probe content differently from
    ``prefix + content + suffix`` - e.g. a template that trims its content, which the
    prefix/suffix count would then misstate.
    """
    def render(content: str) -> str:
        return wrapper.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )

    try:
        rendered = render(CHAT_SENTINEL)
    except Exception as exc:  # noqa: BLE001 - no template, a jinja error, ...
        return None, None, f"apply_chat_template failed: {exc}"
    if not isinstance(rendered, str) or rendered.count(CHAT_SENTINEL) != 1:
        return None, None, "the chat template does not place the user content exactly once"
    prefix, suffix = rendered.split(CHAT_SENTINEL)
    for probe in CHAT_PROBES:
        try:
            got = render(probe)
        except Exception as exc:  # noqa: BLE001
            return None, None, f"apply_chat_template failed on a probe: {exc}"
        if got != prefix + probe + suffix:
            return None, None, (
                "the chat template transforms its content (a probe did not render as "
                "prefix + content + suffix)"
            )
    return prefix, suffix, None


_CACHE: dict[str, ModelTokenizer] = {}
_CACHE_LOCK = threading.Lock()


def load_tokenizer(model: str, *, tokenizer_path: str | None = None) -> ModelTokenizer:
    """Load (and cache) ``model``'s tokenizer from local disk.

    Cached per resolved path, because the senders build one prompt per request from many
    threads and loading costs ~100 ms and several MB.
    """
    path = resolve_tokenizer_path(model, tokenizer_path=tokenizer_path)
    cached = _CACHE.get(path)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _CACHE.get(path)
        if cached is not None:
            return cached
        loaded = _load(model, path)
        _CACHE[path] = loaded
        return loaded


def _load(model: str, path: str) -> ModelTokenizer:
    # Belt and braces: even with local_files_only a stray HF call would try the network,
    # and on this cluster that blocks rather than fails.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise TokenizerUnavailable(
            "the natural prompt mode needs `transformers` to count tokens exactly; "
            "install it or select a different --prompt-mode"
        ) from exc
    try:
        wrapper = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - transformers raises a zoo of types
        raise TokenizerUnavailable(f"could not load tokenizer for {model!r} from {path}: {exc}") from exc
    backend = getattr(wrapper, "backend_tokenizer", None)
    if backend is None:
        raise TokenizerUnavailable(
            f"tokenizer for {model!r} at {path} is not a fast tokenizer; the sender needs "
            "the thread-safe Rust backend"
        )
    # Measured, not assumed: how many special tokens the wrapper adds around a prompt.
    overhead = len(wrapper.encode("")) - len(wrapper.encode("", add_special_tokens=False))
    prefix, suffix, error = _chat_template_parts(wrapper)
    return ModelTokenizer(model, path, backend, overhead,
                          chat_prefix=prefix, chat_suffix=suffix, chat_error=error)


def clear_cache() -> None:
    """Drop cached tokenizers (tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()
