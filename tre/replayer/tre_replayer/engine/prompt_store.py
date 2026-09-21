"""Ahead-of-time prompt materialisation, so no prompt is built on the send path.

Why this exists
---------------
The senders used to build every prompt inside the send call itself - see
:meth:`tre_replayer.engine.http_sender.StreamingHttpSender._send_one` - *after* the
dispatcher had already decided the request was due. A natural-language prompt costs one
tokenizer fit (measured on an idle 64-core host against the dsqwen-7b tokenizer: 1.31 ms
at 256 tokens, 7.34 ms at 1600), and the Rust tokenizer's pyo3 binding does not release
the GIL, so sender threads do not overlap that work at all: 8 threads fitting 32 prompts
measured *slower* than 1 thread (55/s vs 107/s). All of it sat between the request's
scheduled instant and the socket call - i.e. inside the open loop's own definition of
"offered at time t" - and nothing measured it.

This module moves that cost off the send path. Before a cell starts, every request of
its schedule has its prompt built once, in a **process** pool (processes, not threads:
the GIL is precisely why the thread pool did not scale), and written to a JSONL file.
The sender then does a dict lookup.

Where the file goes, and why not into the schedule
--------------------------------------------------
The prompts are deliberately *not* written back into the committed schedules under
``replayer/traces_v2/``: a calibration schedule is a few kB of segments, while one
cell's prompt text is tens of MB, and it is run-specific. The materialised file belongs
to one run and is written under that run's output directory (``--prompt-dir``, which
:mod:`scripts.calibration_campaign` points at ``<out-dir>/prompts``), outside the
repository tree. It is also evidence: it is byte-for-byte what the run put on the wire.

What is preserved
-----------------
A materialised prompt is identical to what the sender would have built inline:

* the same builder (:func:`tre_replayer.engine.prompts.build_prompt`) and the same seed
  key (:func:`sender_seed_key`), so content is a pure function of ``(model, request_id)``
  and a re-run with the same schedule seed produces a byte-identical file;
* **one distinct prompt per request** - never a pool of prompts cycled round, which
  would hand a prefix-caching engine a repeat to serve for free;
* the **per-request** token count, including the length a sampled segment
  (:class:`tre_replayer.engine.schedule.TokenRange`) drew for that one request, because
  the specs are taken from the built schedule rather than from the segment.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from tre_replayer.engine.prompts import DEFAULT_MODE, MODE_NATURAL, build_prompt

#: File name suffix for one cell's materialised prompts.
PROMPT_FILE_SUFFIX = ".prompts.jsonl"

#: Requests handed to one pool task. Large enough that the per-task pickling overhead is
#: negligible against a few milliseconds of tokenizer fitting per request, small enough
#: that the last worker does not hold up the whole batch.
DEFAULT_CHUNK_SIZE = 32

#: Ceiling on pool size. Past this the tokenizer loads (one per worker, each reading a
#: ~7 MB vocabulary off the shared NFS mount) cost more than the fitting they save.
MAX_PROCESSES = 32

#: Prompts a worker must have to be worth starting. Forking a process and loading a
#: tokenizer into it costs a few hundred milliseconds; a handful of prompts does not.
MIN_PROMPTS_PER_PROCESS = 64


def sender_seed_key(model: str, request_id: str) -> str:
    """The seed key :class:`~tre_replayer.engine.http_sender.StreamingHttpSender` uses.

    Defined here rather than inline in the sender so the materialiser and the sender's
    own fallback cannot drift apart: if they did, a materialised run and an inline run of
    the same schedule would send different bytes and their capacity numbers would not be
    comparable.
    """
    return f"{model}|{request_id}"


def prompt_file_path(prompt_dir: str | Path, cell_id: str) -> Path:
    """Where one cell's materialised prompts live under ``prompt_dir``."""
    return Path(prompt_dir) / f"{cell_id}{PROMPT_FILE_SUFFIX}"


@dataclass(frozen=True)
class PromptSpec:
    """Everything needed to build one request's prompt, and nothing about timing."""

    request_id: str
    model: str
    token_count: int
    seed_key: str


def prompt_specs(
    requests: Iterable[Any], *, default_prompt_tokens: int = 64
) -> list[PromptSpec]:
    """One spec per request that needs a synthesised prompt, in schedule order.

    A request that already carries its own ``prompt`` (a trace with recorded text) is
    skipped: there is nothing to synthesise and nothing to look up later.
    """
    specs: list[PromptSpec] = []
    for request in requests:
        if getattr(request, "prompt", ""):
            continue
        token_count = int(getattr(request, "prompt_tokens", None) or default_prompt_tokens)
        model = str(request.model)
        request_id = str(request.request_id)
        specs.append(
            PromptSpec(
                request_id=request_id,
                model=model,
                token_count=token_count,
                seed_key=sender_seed_key(model, request_id),
            )
        )
    return specs


class PromptStore:
    """Materialised prompts, keyed by request id, plus a count of what it did not hold.

    ``misses`` is the point of the class as much as the lookups are. A miss means some
    request fell outside the materialisation - a schedule built with a different seed, a
    cell driven without ``--prompt-dir``, a hold schedule generated after the
    materialisation ran - and the sender then pays the inline tokenizer fit on the send
    path, which is the exact regression this module exists to remove. The count is
    reported per cell so the regression cannot be silent.
    """

    __slots__ = ("_prompts", "path", "_misses", "_lock")

    def __init__(
        self, prompts: dict[str, Any] | None = None, *, path: str | Path | None = None
    ) -> None:
        self._prompts: dict[str, Any] = dict(prompts or {})
        self.path = None if path is None else Path(path)
        self._misses = 0
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: str | Path) -> "PromptStore":
        prompts: dict[str, Any] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompts[str(row["request_id"])] = row["prompt"]
        return cls(prompts, path=path)

    def get(self, request_id: str) -> Any | None:
        """The prompt for ``request_id``, or None - counting the miss."""
        found = self._prompts.get(request_id)
        if found is None:
            with self._lock:
                self._misses += 1
        return found

    @property
    def misses(self) -> int:
        return self._misses

    def __len__(self) -> int:
        return len(self._prompts)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._prompts


# ------------------------------------------------------------------- pool worker state

#: Per-worker-process state. A module global because a ``multiprocessing`` initializer
#: has nowhere else to put what it loaded, and the pool is single-purpose.
_WORKER: dict[str, Any] = {}


def _init_worker(
    mode: str, model: str, tokenizer_path: str | None, tokenizer: Any | None = None
) -> None:
    from tre_replayer.engine import model_tokenizer

    _WORKER["mode"] = mode
    _WORKER["model"] = model
    _WORKER["tokenizer"] = tokenizer
    if mode == MODE_NATURAL and tokenizer is None:
        # Load this worker's own tokenizer rather than inheriting the parent's across the
        # fork: the Rust backend is shared memory after a fork, and a tokenizer that has
        # already been used in the parent brings its thread pool with it.
        model_tokenizer.clear_cache()
        _WORKER["tokenizer"] = model_tokenizer.load_tokenizer(
            model, tokenizer_path=tokenizer_path
        )


def _build_chunk(chunk: Sequence[tuple[str, int, str]]) -> list[tuple[str, Any]]:
    mode = _WORKER["mode"]
    model = _WORKER["model"]
    tokenizer = _WORKER["tokenizer"]
    return [
        (
            request_id,
            build_prompt(token_count, seed_key, mode=mode, model=model, tokenizer=tokenizer),
        )
        for request_id, token_count, seed_key in chunk
    ]


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


def default_processes(work_items: int) -> int:
    """Pool size for ``work_items`` prompts.

    Bounded by the cores available, by :data:`MAX_PROCESSES`, and by the work itself: a
    short schedule builds in this process rather than paying for a pool it cannot fill.
    """
    cpus = os.cpu_count() or 1
    return max(1, min(MAX_PROCESSES, cpus, work_items // MIN_PROMPTS_PER_PROCESS))


# ------------------------------------------------------------------------ materialiser


def build_prompts(
    specs: Sequence[PromptSpec],
    *,
    mode: str = DEFAULT_MODE,
    processes: int | None = None,
    tokenizer: Any | None = None,
    tokenizer_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """``{request_id: prompt}`` for ``specs``, built off the send path.

    Specs are grouped by model because a natural-language fit needs that model's own
    tokenizer; each group gets its own pool, whose workers load it once.

    A pool is not used when ``tokenizer`` is injected (the tests' seam: a stub tokenizer
    cannot be pickled to a worker and does not need to be) or when ``processes`` is 1.
    Ordering of the result is irrelevant to reproducibility: every prompt is a pure
    function of its own spec, so which worker built it cannot change what it is.
    """
    if not specs:
        return {}
    by_model: dict[str, list[PromptSpec]] = {}
    for spec in specs:
        by_model.setdefault(spec.model, []).append(spec)

    prompts: dict[str, Any] = {}
    for model, model_specs in by_model.items():
        workers = default_processes(len(model_specs)) if processes is None else int(processes)
        items = [(s.request_id, s.token_count, s.seed_key) for s in model_specs]
        if tokenizer is not None or workers <= 1:
            _init_worker(mode, model, tokenizer_path, tokenizer)
            prompts.update(dict(_build_chunk(items)))
            continue
        import multiprocessing

        try:
            context = multiprocessing.get_context("fork")
        except ValueError:  # pragma: no cover - platform without fork
            context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(mode, model, tokenizer_path),
        ) as pool:
            for built in pool.imap_unordered(_build_chunk, _chunks(items, chunk_size)):
                prompts.update(dict(built))
    return prompts


def write_prompt_file(path: str | Path, specs: Sequence[PromptSpec], prompts: dict[str, Any]) -> int:
    """Write ``specs`` (in schedule order) with their prompts as JSONL; returns rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for spec in specs:
            if spec.request_id not in prompts:
                continue
            handle.write(
                json.dumps(
                    {
                        "request_id": spec.request_id,
                        "model": spec.model,
                        "prompt_tokens": spec.token_count,
                        "prompt": prompts[spec.request_id],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1
    return written


def materialize_prompts(
    requests: Sequence[Any],
    *,
    path: str | Path,
    mode: str = DEFAULT_MODE,
    default_prompt_tokens: int = 64,
    processes: int | None = None,
    tokenizer: Any | None = None,
    tokenizer_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> PromptStore:
    """Build every prompt of ``requests``, write them to ``path``, return the store.

    This is the one entry point a driver calls. It runs before the dispatch loop starts,
    so the tokenizer cost lands in setup where it is visible and parallel, instead of
    inside the interval the open loop calls "on time".
    """
    specs = prompt_specs(requests, default_prompt_tokens=default_prompt_tokens)
    prompts = build_prompts(
        specs,
        mode=mode,
        processes=processes,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        chunk_size=chunk_size,
    )
    write_prompt_file(path, specs, prompts)
    return PromptStore(prompts, path=path)
