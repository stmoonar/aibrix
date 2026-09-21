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
vLLM tokenises a ``/v1/completions`` prompt the way ``transformers`` does, special
tokens included: all three fleet tokenizers prepend exactly one BOS, so
``usage.prompt_tokens`` is one more than the plain token count. :class:`ModelTokenizer`
reproduces that as ``len(plain ids) + overhead``, where ``overhead`` is measured from
the tokenizer itself at load time rather than assumed.

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

    __slots__ = ("model", "path", "overhead", "filler", "_backend")

    def __init__(self, model: str, path: str, backend: Any, overhead: int) -> None:
        self.model = model
        self.path = path
        self.overhead = int(overhead)
        self._backend = backend
        self.filler = self._pick_filler()

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
    return ModelTokenizer(model, path, backend, overhead)


def clear_cache() -> None:
    """Drop cached tokenizers (tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()
