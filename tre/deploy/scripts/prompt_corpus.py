"""How a calibration run's load reached the engine - recorded, compared, enforced.

Two properties of the load path decide what a measured window means, and neither can be
recovered from the windows themselves:

* the **prompt corpus** (:mod:`tre_replayer.engine.corpus`): ``en``, ``zh`` or ``mix``
  (default: English and Chinese sentences, ``zh_ratio`` of each prompt's tokens
  Chinese). Prefill cost per token and KV footprint per character differ between them;
* the **routing strategy** header (``least-gpu-cache`` by default; None = no header, the
  per-model HTTPRoute). The plugin path adds the ext_proc hop and its own admission
  limits, both of which reach the TTFT label.

A theta fitted on one load path is not evidence about another. So every calibration
artefact records its load path, and every place where two artefacts meet refuses to
combine different ones:

* ``calibration_campaign`` pins both on every ``r3_grid`` cell and records them in
  ``run_provenance`` (``prompt``, ``routing_strategy``);
* ``calibration_dataset`` refuses to build one dataset from campaigns of different load
  paths, and records the dataset's in its manifest (``load_path``);
* ``dline_refit freeze`` reads the training datasets' manifests and records the training
  load path in each model's freeze entry (``prompt_corpus``, ``routing_strategy``),
  refusing a training set that mixes load paths or whose manifests it cannot read;
* M (``calibration_acceptance``) refuses a freeze - and retained cells - of another load
  path; T14 refuses another corpus (its routing is fixed to least-gpu-cache by its
  preregistration, so routing is recorded there, not compared); the training
  supplement refuses a base run of another load path.

Records made before these options existed read as :data:`LEGACY_LOAD_PATH`: English
prompts, no routing header - what every campaign sent until 2026-09-30.
``--allow-prompt-corpus-mismatch`` / ``--allow-routing-mismatch`` turn a refusal into a
recorded warning for a deliberate cross-path evaluation; a pre-registered collection
(T14 with ``--preregistration-json``) refuses them.

Dependency-free on purpose (the replayer package is not importable everywhere these
scripts run); its constants mirror the replayer's and a test pins the two together.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

#: Mirror of ``tre_replayer.engine.corpus`` (guarded by a test).
LANG_EN = "en"
LANG_ZH = "zh"
LANG_MIX = "mix"
CORPUS_LANGS = (LANG_EN, LANG_ZH, LANG_MIX)
DEFAULT_CORPUS_LANG = LANG_MIX
DEFAULT_ZH_RATIO = 0.5

#: What a corpus record from before the corpus option means.
LEGACY = {"corpus_lang": LANG_EN, "zh_ratio": 0.0}
#: What a load-path record from before these options means.
LEGACY_LOAD_PATH = {"prompt": dict(LEGACY), "routing_strategy": None}
#: Routing values that mean "no header".
NO_ROUTING = ("", "none")


def effective_zh_ratio(corpus_lang: str, zh_ratio: Optional[float]) -> float:
    """The Chinese share a corpus actually targets: ``en`` 0, ``zh`` 1, ``mix`` its ratio.

    Recording the nominal ratio for ``en`` / ``zh`` would misstate what was sent.
    """
    if corpus_lang == LANG_EN:
        return 0.0
    if corpus_lang == LANG_ZH:
        return 1.0
    return float(DEFAULT_ZH_RATIO if zh_ratio is None else zh_ratio)


def normalize(doc: Optional[Mapping[str, Any]]) -> dict:
    """``{"corpus_lang", "zh_ratio"}`` of a recorded corpus; a missing record is LEGACY."""
    if not doc or not doc.get("corpus_lang"):
        return dict(LEGACY)
    lang = str(doc["corpus_lang"])
    return {"corpus_lang": lang, "zh_ratio": round(effective_zh_ratio(lang, doc.get("zh_ratio")), 6)}


def normalize_routing(value: Any) -> Optional[str]:
    """None for "no header" (missing, '', 'none'), else the strategy."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in NO_ROUTING else text


def same(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]]) -> bool:
    return normalize(a) == normalize(b)


def describe(doc: Optional[Mapping[str, Any]]) -> str:
    n = normalize(doc)
    return n["corpus_lang"] if n["corpus_lang"] != LANG_MIX else f"mix(zh_ratio={n['zh_ratio']})"


def load_path(provenance: Optional[Mapping[str, Any]]) -> dict:
    """The load path of a ``run_provenance`` (or a dataset's campaign entry)."""
    prov = provenance or {}
    return {"prompt": normalize(prov.get("prompt")),
            "routing_strategy": normalize_routing(prov.get("routing_strategy"))}


def dataset_load_path(manifest: Mapping[str, Any]) -> dict:
    """The load path a standard dataset records (``load_path``), or, for a dataset built
    before that, its first campaign's provenance (all campaigns of a dataset share one)."""
    recorded = manifest.get("load_path")
    if recorded:
        return {"prompt": normalize(recorded.get("prompt")),
                "routing_strategy": normalize_routing(recorded.get("routing_strategy"))}
    campaigns = manifest.get("campaigns") or [{}]
    return load_path(campaigns[0] or {})


def describe_load_path(path: Mapping[str, Any]) -> str:
    routing = normalize_routing(path.get("routing_strategy"))
    return f"{describe(path.get('prompt'))} prompts, routing {routing or 'none (per-model HTTPRoute)'}"


def check_load_path(recorded: Mapping[str, Any], current: Mapping[str, Any], *, what: str,
                    allow_corpus_mismatch: bool = False, allow_routing_mismatch: bool = False,
                    check_routing: bool = True) -> dict:
    """Refuse (ValueError) to combine ``current`` with an artefact of load path
    ``recorded``; with an allow flag the mismatch is printed and reported instead."""
    rec = {"prompt": normalize(recorded.get("prompt")),
           "routing_strategy": normalize_routing(recorded.get("routing_strategy"))}
    cur = {"prompt": normalize(current.get("prompt")),
           "routing_strategy": normalize_routing(current.get("routing_strategy"))}
    report = {"recorded": rec, "current": cur, "corpus_mismatch_allowed": False,
              "routing_mismatch_allowed": False, "routing_compared": bool(check_routing)}
    if rec["prompt"] != cur["prompt"]:
        message = (f"{what}: made with {describe(rec['prompt'])} prompts, but this run sends "
                   f"{describe(cur['prompt'])} prompts; a theta says nothing about a corpus it "
                   "was not fitted on (pass --corpus-lang/--zh-ratio to match, or "
                   "--allow-prompt-corpus-mismatch for a deliberate cross-corpus evaluation)")
        if not allow_corpus_mismatch:
            raise ValueError(message)
        print(f"WARNING: {message}")
        report["corpus_mismatch_allowed"] = True
    if check_routing and rec["routing_strategy"] != cur["routing_strategy"]:
        message = (f"{what}: routed with {rec['routing_strategy'] or 'no routing header'}, but "
                   f"this run routes with {cur['routing_strategy'] or 'no routing header'}; the "
                   "ext_proc path changes the TTFT the labels are made of (pass "
                   "--routing-strategy to match, or --allow-routing-mismatch)")
        if not allow_routing_mismatch:
            raise ValueError(message)
        print(f"WARNING: {message}")
        report["routing_mismatch_allowed"] = True
    return report


def freeze_load_path(freeze_doc: Mapping[str, Any], model: str) -> dict:
    """The training load path a freeze recorded for ``model`` (absent = LEGACY)."""
    entry = (freeze_doc.get("models") or {}).get(model) or {}
    return {"prompt": normalize(entry.get("prompt_corpus")),
            "routing_strategy": normalize_routing(entry.get("routing_strategy"))}


def check_matches_freeze(freeze_doc: Mapping[str, Any], model: str,
                         current: Mapping[str, Any], *, what: str, **flags) -> dict:
    """:func:`check_load_path` against the freeze's training load path of ``model``."""
    return check_load_path(freeze_load_path(freeze_doc, model), current,
                           what=f"{what} ({model}'s training data)", **flags)
