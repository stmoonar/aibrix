"""The load path (prompt corpus, routing, API) is recorded, compared and enforced."""
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
    from tre_replayer.engine import api

    assert prompt_corpus.APIS == api.APIS
    assert prompt_corpus.API_PATHS == api.API_PATHS
    assert (prompt_corpus.API_CHAT, prompt_corpus.API_COMPLETIONS) == (api.API_CHAT, api.API_COMPLETIONS)
    # The replayer keeps completions (trace replays unchanged); calibration sends chat.
    assert api.DEFAULT_API == api.API_COMPLETIONS and prompt_corpus.CALIBRATION_API == api.API_CHAT


def test_a_missing_record_is_the_legacy_load_path_and_ratios_are_effective() -> None:
    assert prompt_corpus.normalize(None) == {"corpus_lang": "en", "zh_ratio": 0.0}
    assert prompt_corpus.normalize({"corpus_lang": "en", "zh_ratio": 0.5}) == prompt_corpus.LEGACY
    assert prompt_corpus.normalize({"corpus_lang": "zh", "zh_ratio": 0.5})["zh_ratio"] == 1.0
    assert prompt_corpus.load_path(None) == prompt_corpus.LEGACY_LOAD_PATH
    assert prompt_corpus.load_path({"prompt": MIX, "routing_strategy": "none"}) == \
        {"prompt": MIX, "routing_strategy": None, "api": "completions"}
    assert prompt_corpus.dataset_load_path({"campaigns": [{"prompt": MIX, "routing_strategy": "x"}]}) == \
        {"prompt": MIX, "routing_strategy": "x", "api": "completions"}
    # run_provenance records {"endpoint": ...}; a dataset's load_path the bare name
    assert prompt_corpus.load_path({"api": {"endpoint": "chat", "path": "/v1/chat/completions"}})["api"] == "chat"
    assert prompt_corpus.dataset_load_path({"load_path": {"prompt": MIX, "api": "chat"}})["api"] == "chat"
    assert prompt_corpus.dataset_load_path({"load_path": {"prompt": MIX}})["api"] == "completions"
    with pytest.raises(ValueError, match="unknown API"):
        prompt_corpus.normalize_api("responses")


def test_a_freeze_of_another_load_path_is_refused_unless_allowed(capsys) -> None:
    legacy = {"models": {"m": {}}}
    trained = {"models": {"m": {"prompt_corpus": MIX, "routing_strategy": "least-gpu-cache"}}}
    current = {"prompt": MIX, "routing_strategy": "least-gpu-cache"}
    with pytest.raises(ValueError, match="made with en prompts"):
        prompt_corpus.check_matches_freeze(legacy, "m", current, what="freeze")
    with pytest.raises(ValueError, match="routed with no routing header"):
        prompt_corpus.check_matches_freeze(legacy, "m", {**current, "prompt": prompt_corpus.LEGACY},
                                           what="freeze")
    assert prompt_corpus.check_matches_freeze(trained, "m", current, what="freeze")["recorded"] == \
        {**current, "api": "completions"}
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
    assert next(iter(found.values())) == {"prompt": MIX, "routing_strategy": "least-gpu-cache",
                                          "api": "completions"}
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


def test_a_training_supplement_must_extend_its_base_run_on_the_same_load_path(tmp_path) -> None:
    import argparse

    from scripts import calibration_training_supplement as training

    base = tmp_path / "base"
    for model, prov in (("m", {}), ("other", {"prompt": MIX, "routing_strategy": "least-gpu-cache"})):
        (base / model).mkdir(parents=True)
        (base / model / "plan.json").write_text(json.dumps({"models": [model], "provenance": prov}))
    legacy = argparse.Namespace(base_run=base, boundary_supplement_run=base, corpus_lang="en",
                                zh_ratio=0.0, routing_strategy=None, api="completions")
    assert len(training.check_source_load_paths(legacy, "m")) == 2  # base and boundary run
    mixed = argparse.Namespace(base_run=base, boundary_supplement_run=base, corpus_lang="mix",
                               zh_ratio=0.5, routing_strategy="least-gpu-cache", api="completions")
    with pytest.raises(ValueError, match="--base-run"):
        training.check_source_load_paths(mixed, "m")
    reports = training.check_source_load_paths(mixed, "other")
    assert reports[0]["recorded"]["routing_strategy"] == "least-gpu-cache"
    # a dry run reports instead of refusing
    dry = argparse.Namespace(**{**vars(mixed), "dry_run": True})
    assert training.check_source_load_paths(dry, "m")[0]["corpus_mismatch_allowed"]
    # a source with no campaign of the model has an unknown load path
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no campaign"):
        training.check_source_load_paths(
            argparse.Namespace(**{**vars(legacy), "boundary_supplement_run": empty}), "m")


# ------------------------------------------------------------------------------ the API


CHAT = {"prompt": MIX, "routing_strategy": "least-gpu-cache", "api": "chat"}


def test_an_api_mismatch_is_refused_whatever_the_allow_flags_say(capsys) -> None:
    completions = {**CHAT, "api": "completions"}
    for recorded, current in ((completions, CHAT), (CHAT, completions), ({"prompt": MIX}, CHAT)):
        with pytest.raises(ValueError, match="API"):
            prompt_corpus.check_load_path(recorded, current, what="x", allow_corpus_mismatch=True,
                                          allow_routing_mismatch=True)
    # a freeze without an API record is a completions freeze
    with pytest.raises(ValueError, match="made through the completions API"):
        prompt_corpus.check_matches_freeze(
            {"models": {"m": {"prompt_corpus": MIX, "routing_strategy": "least-gpu-cache"}}}, "m", CHAT,
            what="freeze")
    ok = prompt_corpus.check_matches_freeze(
        {"models": {"m": {"prompt_corpus": MIX, "routing_strategy": "least-gpu-cache", "api": "chat"}}},
        "m", CHAT, what="freeze")
    assert ok["recorded"] == CHAT and not ok["api_mismatch_reported"]
    # a dry run (every flag on) reports instead
    report = prompt_corpus.check_load_path(completions, CHAT, what="x", allow_api_mismatch=True)
    assert report["api_mismatch_reported"] and "WARNING" in capsys.readouterr().out


def test_the_description_names_the_api_so_a_dataset_refuses_two() -> None:
    a = prompt_corpus.describe_load_path(CHAT)
    b = prompt_corpus.describe_load_path({**CHAT, "api": None})
    assert a != b and a.endswith("chat API") and b.endswith("completions API")


def test_campaign_args_carry_the_api_into_the_load_path_and_the_provenance() -> None:
    import argparse

    from scripts import calibration_campaign as campaign

    args = argparse.Namespace(corpus_lang="mix", zh_ratio=0.5, routing_strategy="least-gpu-cache")
    assert campaign.load_path(args)["api"] == "chat"  # the calibration default
    assert campaign.api_record(args)["path"] == "/v1/chat/completions"
    args.api, args.request_seed = "completions", 7
    assert campaign.load_path(args)["api"] == "completions"
    assert campaign.api_record(args)["request_seed"] == 7
    assert campaign.mismatch_flags(args)["allow_api_mismatch"] is False


@pytest.mark.parametrize("url,api,ok", [
    ("http://h:1/v1/chat/completions", "chat", True),
    ("http://h:1/v1/chat/completions/", "chat", True),
    ("http://h:1/v1/completions", "chat", False),
    ("http://h:1", "chat", False),
    ("http://h:1/v1/completions", "completions", True),
    ("http://gw", "completions", True),
    ("http://h:1/v1/chat/completions", "completions", False),
])
def test_the_gateway_url_must_name_the_apis_path(url: str, api: str, ok: bool) -> None:
    from tre_replayer.engine import api as replayer_api

    for check in (prompt_corpus.check_gateway_url, replayer_api.check_api_url):
        if ok:
            check(url, api)
        else:
            with pytest.raises(ValueError):
                check(url, api)
