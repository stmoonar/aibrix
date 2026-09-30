"""The load path (prompt corpus, routing) is recorded, compared and enforced."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import dline_refit
from scripts import prompt_corpus

MIX = {"corpus_lang": "mix", "zh_ratio": 0.5}


def test_the_mirror_matches_the_replayer() -> None:
    from tre_replayer.engine import corpus

    assert prompt_corpus.CORPUS_LANGS == corpus.CORPUS_LANGS
    assert prompt_corpus.DEFAULT_CORPUS_LANG == corpus.DEFAULT_CORPUS_LANG
    assert prompt_corpus.DEFAULT_ZH_RATIO == corpus.DEFAULT_ZH_RATIO


def test_a_missing_record_is_the_legacy_load_path_and_ratios_are_effective() -> None:
    assert prompt_corpus.normalize(None) == {"corpus_lang": "en", "zh_ratio": 0.0}
    assert prompt_corpus.normalize({"corpus_lang": "en", "zh_ratio": 0.5}) == prompt_corpus.LEGACY
    assert prompt_corpus.normalize({"corpus_lang": "zh", "zh_ratio": 0.5})["zh_ratio"] == 1.0
    assert prompt_corpus.load_path(None) == prompt_corpus.LEGACY_LOAD_PATH
    assert prompt_corpus.load_path({"prompt": MIX, "routing_strategy": "none"}) == \
        {"prompt": MIX, "routing_strategy": None}
    assert prompt_corpus.dataset_load_path({"campaigns": [{"prompt": MIX, "routing_strategy": "x"}]}) == \
        {"prompt": MIX, "routing_strategy": "x"}


def test_a_freeze_of_another_load_path_is_refused_unless_allowed(capsys) -> None:
    legacy = {"models": {"m": {}}}
    trained = {"models": {"m": {"prompt_corpus": MIX, "routing_strategy": "least-gpu-cache"}}}
    current = {"prompt": MIX, "routing_strategy": "least-gpu-cache"}
    with pytest.raises(ValueError, match="made with en prompts"):
        prompt_corpus.check_matches_freeze(legacy, "m", current, what="freeze")
    with pytest.raises(ValueError, match="routed with no routing header"):
        prompt_corpus.check_matches_freeze(legacy, "m", {**current, "prompt": prompt_corpus.LEGACY},
                                           what="freeze")
    assert prompt_corpus.check_matches_freeze(trained, "m", current, what="freeze")["recorded"] == current
    # routing not compared (T14)
    prompt_corpus.check_matches_freeze(legacy, "m", {**current, "prompt": prompt_corpus.LEGACY},
                                       what="freeze", check_routing=False)
    report = prompt_corpus.check_matches_freeze(legacy, "m", current, what="freeze",
                                                allow_corpus_mismatch=True, allow_routing_mismatch=True)
    assert report["corpus_mismatch_allowed"] and report["routing_mismatch_allowed"]
    assert "WARNING" in capsys.readouterr().out


def _dataset(root: Path, name: str, manifest: dict) -> dict:
    """A dataset directory and the trainset stage's source record of it (the real shape:
    dline_refit's trainset writer records ``directory`` and ``manifest_sha256``)."""
    d = root / name
    d.mkdir()
    (d / "windows.csv").write_text("x\n", encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source = dline_refit.DatasetSource(name=name, directory=d, sealed_to_h2=False)
    return {"run": source.name, "directory": str(source.directory), "sealed_split": "m",
            "windows_csv_sha256": hashlib.sha256(b"x\n").hexdigest(),
            "manifest_sha256": dline_refit.sha256_file(d / "manifest.json"), "format_revision": 2}


def test_the_freeze_reads_the_training_load_path_from_the_trainset_sources(tmp_path) -> None:
    old = _dataset(tmp_path, "old", {"campaigns": [{"campaign": "c"}]})
    new = _dataset(tmp_path, "new", {"load_path": {"prompt": MIX, "routing_strategy": "least-gpu-cache"}})
    new2 = _dataset(tmp_path, "new2", {"campaigns": [{"prompt": MIX, "routing_strategy": "least-gpu-cache"}]})
    found, problems = dline_refit.training_load_paths({"sources": [old]})
    assert problems == [] and list(found.values()) == [prompt_corpus.LEGACY_LOAD_PATH]
    found, problems = dline_refit.training_load_paths({"sources": [new, new2]})
    assert problems == [] and len(found) == 1
    assert next(iter(found.values())) == {"prompt": MIX, "routing_strategy": "least-gpu-cache"}
    found, problems = dline_refit.training_load_paths({"sources": [old, new]})
    assert len(found) == 2


def test_an_unreadable_or_changed_training_manifest_is_a_problem_not_a_default(tmp_path) -> None:
    gone = _dataset(tmp_path, "gone", {})
    (Path(gone["directory"]) / "manifest.json").unlink()
    changed = _dataset(tmp_path, "changed", {})
    (Path(changed["directory"]) / "manifest.json").write_text(json.dumps({"load_path": {}}))
    _found, problems = dline_refit.training_load_paths({"sources": [gone, changed]})
    assert len(problems) == 2
    assert "no dataset manifest" in problems[0] and "changed since" in problems[1]
    assert dline_refit.training_load_paths({"sources": []})[1]


def test_the_trainset_writer_records_what_the_freeze_reads(tmp_path) -> None:
    """Guard against the two drifting apart again (a freeze once read a key the trainset
    manifest never had, and silently labelled every training set English)."""
    import inspect

    writer = inspect.getsource(dline_refit)
    assert '"directory": str(s.directory)' in writer and '"manifest_sha256": sha256_file(man)' in writer


def test_a_training_supplement_must_extend_its_base_run_on_the_same_load_path(tmp_path) -> None:
    import argparse

    from scripts import calibration_training_supplement as training

    base = tmp_path / "base"
    for model, prov in (("m", {}), ("other", {"prompt": MIX, "routing_strategy": "least-gpu-cache"})):
        (base / model).mkdir(parents=True)
        (base / model / "plan.json").write_text(json.dumps({"models": [model], "provenance": prov}))
    legacy = argparse.Namespace(base_run=base, corpus_lang="en", zh_ratio=0.0, routing_strategy=None)
    assert len(training.check_base_load_path(legacy, "m")) == 1
    mixed = argparse.Namespace(base_run=base, corpus_lang="mix", zh_ratio=0.5,
                               routing_strategy="least-gpu-cache")
    with pytest.raises(ValueError, match="base run"):
        training.check_base_load_path(mixed, "m")
    assert training.check_base_load_path(mixed, "other")[0]["recorded"]["routing_strategy"] == "least-gpu-cache"
