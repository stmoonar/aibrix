"""Materialising prompts ahead of the run must not change a single byte of what is sent.

The point of :mod:`tre_replayer.engine.prompt_store` is to move the tokenizer fit off the
send path, and the only way that is a free win is if the three properties the calibration
depends on survive it: one distinct prompt per request, an exact realised token count
under the model's *own* tokenizer, and byte-identical content for a given seed.
"""
from __future__ import annotations

import json

import pytest

from tre_replayer.engine import model_tokenizer, prompt_store, prompts
from tre_replayer.engine.schedule import RpsSegment, ScheduledRequest, build_poisson_schedule

FLEET_MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")


class StubTokenizer:
    """One token per whitespace word, one leading special, exact decode roundtrip."""

    overhead = 1
    filler = " the"
    path = "<stub>"

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_word: dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self._to_id:
            index = len(self._to_id) + 1
            self._to_id[word] = index
            self._to_word[index] = word
        return self._to_id[word]

    def encode_plain(self, text: str) -> list[int]:
        return [self._id(word) for word in text.split()]

    def decode_plain(self, ids) -> str:
        return " ".join(self._to_word[i] for i in ids)

    def count(self, text: str) -> int:
        return len(self.encode_plain(text)) + self.overhead


def _requests(count: int, *, model: str = "m", tokens: int = 48) -> list[ScheduledRequest]:
    return [
        ScheduledRequest(
            request_id=f"{model}-{i:06d}",
            model=model,
            scheduled_offset_s=i * 0.1,
            prompt_tokens=tokens,
            max_output_tokens=16,
        )
        for i in range(count)
    ]


def _rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------- content is unchanged


def test_a_materialised_prompt_is_what_the_sender_would_have_built_inline() -> None:
    """If the two builders could drift, a materialised run and an inline run of the same
    schedule would send different bytes and their capacity numbers would not compare."""
    request = _requests(1)[0]
    inline = prompts.build_natural_prompt(
        48, prompt_store.sender_seed_key("m", "m-000000"), tokenizer=StubTokenizer()
    )
    built = prompt_store.build_prompts(
        prompt_store.prompt_specs([request]), tokenizer=StubTokenizer()
    )
    assert built["m-000000"] == inline


def test_every_request_gets_its_own_prompt(tmp_path) -> None:
    """Not a pool of prompts cycled round: a repeat is a free prefill on any engine with
    prefix caching, which is exactly what invalidated the earlier capacity scans."""
    path = tmp_path / "cell.prompts.jsonl"
    prompt_store.materialize_prompts(
        _requests(200), path=path, tokenizer=StubTokenizer()
    )
    rows = _rows(path)
    assert len(rows) == 200
    texts = [row["prompt"] for row in rows]
    assert len(set(texts)) == 200


def test_the_same_seed_materialises_the_same_file(tmp_path) -> None:
    segments = [RpsSegment("m", 0.0, 3.0, 20.0, input_tokens=48, max_output_tokens=16)]
    first = tmp_path / "a.prompts.jsonl"
    second = tmp_path / "b.prompts.jsonl"
    for path in (first, second):
        prompt_store.materialize_prompts(
            build_poisson_schedule(segments, seed=7), path=path, tokenizer=StubTokenizer()
        )
    assert first.read_bytes() == second.read_bytes()


def test_a_request_that_carries_its_own_prompt_is_left_alone(tmp_path) -> None:
    """A trace with recorded text has nothing to synthesise and nothing to look up."""
    path = tmp_path / "cell.prompts.jsonl"
    requests = [
        ScheduledRequest(request_id="m-0", model="m", scheduled_offset_s=0.0, prompt="verbatim"),
        ScheduledRequest(request_id="m-1", model="m", scheduled_offset_s=0.1, prompt_tokens=48),
    ]
    store = prompt_store.materialize_prompts(requests, path=path, tokenizer=StubTokenizer())
    assert [row["request_id"] for row in _rows(path)] == ["m-1"]
    assert "m-0" not in store


# ------------------------------------------------- per-request sampled token lengths


def test_each_request_is_built_at_the_length_that_request_actually_drew(tmp_path) -> None:
    """T9-shaped segments draw a per-request length from a log-uniform range. Building
    against the segment's nominal length instead would send a different prompt from the
    one the schedule (and the raw log, and the fitted grid) says was sent."""
    from tre_replayer.engine.schedule import TokenRange

    segments = [
        RpsSegment(
            "m", 0.0, 4.0, 25.0,
            input_tokens_range=TokenRange(low=32, high=512),
            max_output_tokens=16,
        )
    ]
    schedule = build_poisson_schedule(segments, seed=11)
    drawn = {event.request_id: event.prompt_tokens for event in schedule}
    assert len(set(drawn.values())) > 5  # the range really is being sampled

    path = tmp_path / "cell.prompts.jsonl"
    prompt_store.materialize_prompts(schedule, path=path, tokenizer=StubTokenizer())

    tok = StubTokenizer()
    for row in _rows(path):
        assert row["prompt_tokens"] == drawn[row["request_id"]]
        assert tok.count(row["prompt"]) == drawn[row["request_id"]]


# --------------------------------------------------------------- the store's lookups


def test_the_store_counts_what_it_did_not_hold() -> None:
    """A miss means the sender paid an inline tokenizer fit inside its own lateness. It
    is counted so a cell that silently fell back cannot look like one that did not."""
    store = prompt_store.PromptStore({"m-0": "hello"})
    assert store.get("m-0") == "hello" and store.misses == 0
    assert store.get("m-404") is None and store.misses == 1


def test_a_store_round_trips_through_its_file(tmp_path) -> None:
    path = tmp_path / "cell.prompts.jsonl"
    written = prompt_store.materialize_prompts(
        _requests(5), path=path, tokenizer=StubTokenizer()
    )
    reloaded = prompt_store.PromptStore.load(path)
    assert len(reloaded) == len(written) == 5
    assert reloaded.get("m-000003") == written.get("m-000003")


def test_prompt_file_path_is_named_after_the_cell(tmp_path) -> None:
    path = prompt_store.prompt_file_path(tmp_path, "i256_o128_c60")
    assert path.name == "i256_o128_c60" + prompt_store.PROMPT_FILE_SUFFIX


# ------------------------------------------------------- the real multiprocessing path


def test_the_process_pool_builds_the_same_prompts_as_a_single_process(tmp_path) -> None:
    """Processes, not threads - the tokenizer's pyo3 binding holds the GIL, so a thread
    pool made this *slower*. Which worker built a prompt must not change what it is."""
    serial = tmp_path / "serial.prompts.jsonl"
    parallel = tmp_path / "parallel.prompts.jsonl"
    requests = _requests(64, tokens=24)
    prompt_store.materialize_prompts(
        requests, path=serial, mode=prompts.MODE_TOKEN_IDS, processes=1
    )
    prompt_store.materialize_prompts(
        requests, path=parallel, mode=prompts.MODE_TOKEN_IDS, processes=4, chunk_size=7
    )
    assert serial.read_bytes() == parallel.read_bytes()


def test_a_mixed_model_schedule_is_grouped_by_model(tmp_path) -> None:
    """Each model needs its own tokenizer, so the pool is per model; the file still
    carries every request, in schedule order."""
    path = tmp_path / "mixed.prompts.jsonl"
    requests = _requests(4, model="a") + _requests(4, model="b")
    prompt_store.materialize_prompts(
        requests, path=path, mode=prompts.MODE_TOKEN_IDS, processes=1
    )
    rows = _rows(path)
    assert [row["model"] for row in rows] == ["a"] * 4 + ["b"] * 4


# --------------------------------------------------- exactness against real tokenizers


def _fleet_tokenizer_or_skip(model: str):
    model_tokenizer.clear_cache()
    try:
        return model_tokenizer.load_tokenizer(model)
    except model_tokenizer.TokenizerUnavailable as exc:  # pragma: no cover - env dependent
        pytest.skip(f"no local tokenizer for {model}: {exc}")


@pytest.mark.parametrize("model", FLEET_MODELS)
def test_materialised_prompts_hit_the_exact_token_count_for_every_fleet_model(
    model: str, tmp_path
) -> None:
    """``usage.prompt_tokens`` is the coordinate the calibration grid is indexed by. A
    prompt that is one token off is filed under a cell it did not measure - and the fit
    is done with the *model's own* tokenizer, so this has to be checked per model rather
    than once against a stub.

    This also exercises the real process pool: the workers load the tokenizer themselves.
    """
    tok = _fleet_tokenizer_or_skip(model)
    targets = [64, 128, 257, 1024]
    requests = [
        ScheduledRequest(
            request_id=f"{model}-{i:06d}",
            model=model,
            scheduled_offset_s=i * 0.1,
            prompt_tokens=targets[i % len(targets)],
            max_output_tokens=16,
        )
        for i in range(8)
    ]
    path = tmp_path / f"{model}.prompts.jsonl"
    prompt_store.materialize_prompts(requests, path=path, processes=2, chunk_size=2)

    rows = _rows(path)
    assert len(rows) == len(requests)
    for row in rows:
        assert tok.count(row["prompt"]) == row["prompt_tokens"], row["request_id"]
    assert len({row["prompt"] for row in rows}) == len(rows)
