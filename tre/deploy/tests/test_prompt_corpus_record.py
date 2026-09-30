"""The prompt corpus is recorded, compared and enforced where calibration artefacts meet."""
from __future__ import annotations

import json

import pytest

from scripts import dline_refit
from scripts import prompt_corpus


def test_the_mirror_matches_the_replayer() -> None:
    from tre_replayer.engine import corpus

    assert prompt_corpus.CORPUS_LANGS == corpus.CORPUS_LANGS
    assert prompt_corpus.DEFAULT_CORPUS_LANG == corpus.DEFAULT_CORPUS_LANG
    assert prompt_corpus.DEFAULT_ZH_RATIO == corpus.DEFAULT_ZH_RATIO


def test_a_missing_record_is_english_and_ratios_are_effective() -> None:
    assert prompt_corpus.normalize(None) == {"corpus_lang": "en", "zh_ratio": 0.0}
    assert prompt_corpus.normalize({}) == prompt_corpus.LEGACY
    assert prompt_corpus.normalize({"corpus_lang": "en", "zh_ratio": 0.5}) == prompt_corpus.LEGACY
    assert prompt_corpus.normalize({"corpus_lang": "zh", "zh_ratio": 0.5})["zh_ratio"] == 1.0
    assert prompt_corpus.same({"corpus_lang": "mix", "zh_ratio": 0.5},
                              {"corpus_lang": "mix", "zh_ratio": 0.5, "prompt_mode": "natural"})
    assert not prompt_corpus.same(None, {"corpus_lang": "mix", "zh_ratio": 0.5})


def test_m_and_t14_refuse_a_freeze_trained_on_another_corpus(capsys) -> None:
    legacy = {"models": {"m": {}}}
    mixed = {"models": {"m": {"prompt_corpus": {"corpus_lang": "mix", "zh_ratio": 0.5}}}}
    mix = {"corpus_lang": "mix", "zh_ratio": 0.5}
    with pytest.raises(ValueError, match="trained on en prompts"):
        prompt_corpus.check_matches_freeze(legacy, "m", mix, what="freeze")
    assert prompt_corpus.check_matches_freeze(mixed, "m", mix, what="freeze") == \
        {"corpus_lang": "mix", "zh_ratio": 0.5}
    assert prompt_corpus.check_matches_freeze(legacy, "m", {"corpus_lang": "en"}, what="freeze") == \
        prompt_corpus.LEGACY
    prompt_corpus.check_matches_freeze(legacy, "m", mix, what="freeze", allow_mismatch=True)
    assert "WARNING" in capsys.readouterr().out


def test_the_freeze_reads_the_training_corpus_from_the_datasets(tmp_path) -> None:
    def dataset(name: str, manifest: dict) -> dict:
        d = tmp_path / name
        d.mkdir()
        (d / "windows.csv").write_text("x\n", encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"run": name, "windows_csv": str(d / "windows.csv"), "sealed_to_h2": False}

    old = dataset("old", {"campaigns": [{"campaign": "c"}]})
    new = dataset("new", {"prompt_corpus": {"corpus_lang": "mix", "zh_ratio": 0.5}})
    new2 = dataset("new2", {"campaigns": [{"prompt": {"corpus_lang": "mix", "zh_ratio": 0.5}}]})
    assert dline_refit.training_prompt_corpora({"sources": [old]}) == {"en": prompt_corpus.LEGACY}
    assert list(dline_refit.training_prompt_corpora({"sources": [new, new2]})) == ["mix(zh_ratio=0.5)"]
    assert len(dline_refit.training_prompt_corpora({"sources": [old, new]})) == 2
