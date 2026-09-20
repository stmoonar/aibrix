"""Natural-language prompt mode: exactness, determinism, uniqueness, offline resolution.

These run with a stub tokenizer so they never need a model on disk; the exactness of the
fit against the *real* fleet tokenizers is checked live (see the task report), because a
unit test cannot both stay hermetic and prove a property of a 150k-entry vocabulary.
"""
from __future__ import annotations

import json

import pytest

from tre_replayer.engine import corpus, model_tokenizer, prompts


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


@pytest.mark.parametrize("target", [2, 8, 32, 128, 512, 1024, 1600])
def test_natural_prompt_hits_the_exact_token_count(target: int) -> None:
    """Approximate is not good enough: the capacity grid is indexed by input length, so a
    cell that silently sends 1590 tokens instead of 1600 fits the wrong curve."""
    tok = StubTokenizer()
    text = prompts.build_natural_prompt(target, f"unit|{target}", tokenizer=tok)
    assert tok.count(text) == target


def test_natural_prompt_below_the_special_token_floor_is_refused() -> None:
    """Rounding down to a shorter prompt would put a cell's real input length out of step
    with the grid coordinate it is filed under."""
    with pytest.raises(ValueError, match="cannot be shorter"):
        prompts.build_natural_prompt(1, "unit", tokenizer=StubTokenizer())


def test_natural_prompt_is_byte_identical_for_the_same_seed_key() -> None:
    first = prompts.build_natural_prompt(256, "run|cell|7", tokenizer=StubTokenizer())
    second = prompts.build_natural_prompt(256, "run|cell|7", tokenizer=StubTokenizer())
    assert first == second


def test_natural_prompts_diverge_inside_the_head_for_every_seed_key() -> None:
    """The whole point of the module: identical prompts plus prefix caching made prefill
    free and the measured capacity rose with prompt length."""
    tok = StubTokenizer()
    texts = [prompts.build_natural_prompt(128, f"run|cell|{i}", tokenizer=tok) for i in range(200)]
    assert len(set(texts)) == 200
    assert len({tuple(t.split()[:3]) for t in texts}) == 200


def test_natural_prompt_reads_as_english_not_as_a_token_soup() -> None:
    text = prompts.build_natural_prompt(128, "run|cell|1", tokenizer=StubTokenizer())
    assert text.count(" ") > 60  # words, not one long blob
    assert "." in text  # sentences
    assert all(ord(ch) < 128 for ch in text)


def test_reference_id_is_a_fixed_width_injection_of_the_seed() -> None:
    """Fixed width so no id is a prefix of another and the opening costs a fixed number
    of tokens; injective so two seeds cannot share an opening."""
    ids = {corpus.reference_id(seed) for seed in range(5000)}
    assert len(ids) == 5000
    assert {len(i) for i in ids} == {13}
    assert corpus.reference_id(0) == "0" * 13


def test_text_builder_only_ever_appends() -> None:
    builder = corpus.TextBuilder(1234)
    short = builder.ensure_words(20)
    longer = builder.ensure_words(200)
    assert longer.startswith(short)


def test_build_prompt_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown prompt mode"):
        prompts.build_prompt(16, "k", mode="nonsense")


def test_natural_mode_without_a_model_fails_loudly_rather_than_downgrading() -> None:
    """A silent fall back to random ids would put two incomparable runs in one campaign
    with nothing in the artifacts to say which was which."""
    with pytest.raises(model_tokenizer.TokenizerUnavailable):
        prompts.build_prompt(16, "k", mode=prompts.MODE_NATURAL)


def test_tokenizer_paths_resolve_from_the_environment_before_the_builtin_map(monkeypatch) -> None:
    monkeypatch.setenv(model_tokenizer.PATHS_ENV, json.dumps({"dsqwen-7b": "/somewhere/else"}))
    assert model_tokenizer.resolve_tokenizer_path("dsqwen-7b") == "/somewhere/else"


def test_tokenizer_paths_resolve_from_the_registry_weights_path(tmp_path, monkeypatch) -> None:
    """The registry's weights_path is the directory the vLLM pods actually load, so it is
    by definition the tokenizer that decides usage.prompt_tokens."""
    registry = tmp_path / "registry.yaml"
    registry.write_text("models:\n- name: newmodel\n  weights_path: /mnt/models/newmodel\n")
    monkeypatch.delenv(model_tokenizer.PATHS_ENV, raising=False)
    monkeypatch.setenv(model_tokenizer.REGISTRY_ENV, str(registry))
    assert model_tokenizer.resolve_tokenizer_path("newmodel") == "/mnt/models/newmodel"


def test_tokenizer_path_falls_back_to_the_committed_fleet_map(monkeypatch) -> None:
    monkeypatch.delenv(model_tokenizer.PATHS_ENV, raising=False)
    monkeypatch.setenv(model_tokenizer.REGISTRY_ENV, "/nonexistent/registry.yaml")
    assert model_tokenizer.resolve_tokenizer_path("dsqwen-14b") == (
        model_tokenizer.FLEET_TOKENIZER_PATHS["dsqwen-14b"]
    )


def test_unknown_model_names_the_ways_out(monkeypatch) -> None:
    monkeypatch.delenv(model_tokenizer.PATHS_ENV, raising=False)
    monkeypatch.setenv(model_tokenizer.REGISTRY_ENV, "/nonexistent/registry.yaml")
    with pytest.raises(model_tokenizer.TokenizerUnavailable, match="TRE_TOKENIZER_PATHS"):
        model_tokenizer.resolve_tokenizer_path("no-such-model")


def test_fleet_tokenizer_map_matches_the_committed_registry() -> None:
    """The fallback map is a copy of registry.yaml's weights_path. A copy that drifts
    would tokenise with the wrong vocabulary and quietly miss the target length."""
    import pathlib

    import yaml

    registry = pathlib.Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
    if not registry.exists():  # pragma: no cover - repo layout guard
        pytest.skip("registry.yaml not in this tree")
    doc = yaml.safe_load(registry.read_text())
    on_disk = {m["name"]: m["weights_path"] for m in doc["models"]}
    for name, path in model_tokenizer.FLEET_TOKENIZER_PATHS.items():
        assert on_disk.get(name) == path, f"{name} drifted from registry.yaml"
