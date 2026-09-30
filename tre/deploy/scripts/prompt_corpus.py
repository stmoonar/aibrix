"""What the synthesised prompts of a run were written in - recorded, compared, enforced.

The replayer writes natural prompts in one of three corpora
(:mod:`tre_replayer.engine.corpus`): ``en``, ``zh`` or ``mix`` (default: English and
Chinese sentences, ``zh_ratio`` of each prompt's tokens Chinese). A theta fitted on one
corpus is not evidence about another - prefill cost per token and the KV footprint per
character both differ - so every calibration artefact records the corpus it was made
with, and the places where two artefacts meet refuse to combine different ones:

* ``calibration_campaign`` pins the corpus on every ``r3_grid`` cell and records it in
  ``run_provenance["prompt"]``;
* ``calibration_dataset`` refuses to build one dataset from campaigns of different
  corpora and records the dataset's corpus in its manifest;
* ``dline_refit freeze`` records the training corpus in each model's freeze entry;
* the M (``calibration_acceptance``) and T14 (``calibration_t14``) collections refuse to
  run under a freeze whose training corpus differs from theirs, unless told to.

Everything recorded before the corpus option existed was English: a missing record reads
as :data:`LEGACY`. This module is dependency-free on purpose (the replayer package is
not importable everywhere these scripts run); its constants mirror the replayer's and a
test pins the two together.
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

#: What an artefact from before the corpus option was made with.
LEGACY = {"corpus_lang": LANG_EN, "zh_ratio": 0.0}


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


def same(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]]) -> bool:
    return normalize(a) == normalize(b)


def describe(doc: Optional[Mapping[str, Any]]) -> str:
    n = normalize(doc)
    return n["corpus_lang"] if n["corpus_lang"] != LANG_MIX else f"mix(zh_ratio={n['zh_ratio']})"


def check_matches_freeze(freeze_doc: Mapping[str, Any], model: str,
                         corpus: Optional[Mapping[str, Any]], *, what: str,
                         allow_mismatch: bool = False) -> dict:
    """Refuse (ValueError) to collect a held-out set with a corpus other than the one the
    frozen parameters of ``model`` were trained on; returns the frozen corpus.

    A freeze entry without ``prompt_corpus`` predates the option: its training data was
    English. ``allow_mismatch`` turns the refusal into a printed warning - for a
    deliberate cross-corpus evaluation, never for a pre-registered one.
    """
    entry = (freeze_doc.get("models") or {}).get(model) or {}
    trained = normalize(entry.get("prompt_corpus"))
    if not same(trained, corpus):
        message = (f"{what}: {model} was trained on {describe(trained)} prompts, but this "
                   f"collection would send {describe(corpus)} prompts; a theta says nothing "
                   "about a corpus it was not fitted on (pass --corpus-lang/--zh-ratio to "
                   "match, or --allow-prompt-corpus-mismatch for a deliberate cross-corpus "
                   "evaluation)")
        if not allow_mismatch:
            raise ValueError(message)
        print(f"WARNING: {message}")
    return trained
