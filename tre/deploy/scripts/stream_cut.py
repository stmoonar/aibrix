"""The stream-cut rule of the held-out sets (T14 since 2026-10-04; M2 by user decision
2026-10-05): one definition, shared by the T14 scorer and ``dline_refit accept``.

* runtime: a cell whose ``model_errors / sent`` exceeds :data:`RUNTIME_MODEL_ERROR_LIMIT`
  (0.10, the collection's ``--max-model-error-rate``) is void and re-driven by the driver;
* audit: a request with outcome ``model_error`` and ``e2e_ms >=`` :data:`ROUTE_TIMEOUT_CUT_MS`
  (the route's 150 s timeout cut it mid-stream) is a CENSORED CUT, not an engine error; any
  other ``model_error`` (one without ``e2e_ms`` included) is a non-cut error. A cell whose
  non-cut errors / sent > :data:`NON_CUT_ERROR_LIMIT` (0.05) is void at audit and excluded
  from evaluation (disclosed). Scope: the evaluated attempt, every request of it (warm-up
  included, the whole-cell denominator of ``openloop.check_cell``).
* the label is unchanged: a cut request is an unserved request (``model_errors`` of its
  send window, ``openloop.mark_unserved_request_windows``), so its window is a violation
  under every label attribution - the audit only changes the void decision.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Optional

#: The route timeout (``audit_rule.cut``: "e2e_ms >= 150000").
ROUTE_TIMEOUT_CUT_MS = 150_000.0
#: The limit the non-cut errors of a cell are judged against (``audit_rule.rule``).
NON_CUT_ERROR_LIMIT = 0.05
#: The runtime void limit (``runtime_limit.max_model_error_rate``).
RUNTIME_MODEL_ERROR_LIMIT = 0.10
OUTCOME_MODEL_ERROR = "model_error"
RULE = {"cut": f"outcome {OUTCOME_MODEL_ERROR} and e2e_ms >= {ROUTE_TIMEOUT_CUT_MS:.0f}",
        "non_cut": "any other model_error, e2e_ms missing included (D4)",
        "non_cut_limit": NON_CUT_ERROR_LIMIT, "runtime_limit": RUNTIME_MODEL_ERROR_LIMIT,
        "scope": "the evaluated (valid) attempt only; voided earlier attempts excluded",
        "denominator": "every request of that attempt in requests.csv, warm-up included"}


def classify(row: Mapping[str, Any]) -> Optional[str]:
    """``cut`` / ``non_cut`` for a ``model_error`` request row (requests.csv), else None."""
    if (row.get("outcome") or "").strip() != OUTCOME_MODEL_ERROR:
        return None
    try:
        e2e = float(row.get("e2e_ms") or "nan")
    except ValueError:
        e2e = math.nan
    return "cut" if math.isfinite(e2e) and e2e >= ROUTE_TIMEOUT_CUT_MS else "non_cut"


def audit(requests_csv: Path, cells: Mapping[tuple, Mapping[str, Any]]) -> dict:
    """The audit per cell ``(cell_id, attempt)`` of ``cells`` (their extra fields are copied
    into each row) over one dataset's ``requests.csv``."""
    per: dict[tuple, dict] = {k: {"sent": 0, "sent_in_warmup": 0, "model_error": 0, "cut": 0, "non_cut": 0,
                                  "non_cut_no_e2e": 0} for k in cells}
    with open(requests_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                key = (str(r["cell_id"]), int(float(r["attempt"] or 1)))
            except (KeyError, ValueError):
                continue
            c = per.get(key)
            if c is None:
                continue
            c["sent"] += 1
            c["sent_in_warmup"] += str(r.get("in_warmup") or "").strip().lower() in ("1", "true", "yes")
            kind = classify(r)
            if kind is None:
                continue
            c["model_error"] += 1
            c[kind] += 1
            if kind == "non_cut":
                try:
                    c["non_cut_no_e2e"] += not math.isfinite(float(r.get("e2e_ms") or "nan"))
                except ValueError:
                    c["non_cut_no_e2e"] += 1
    out = []
    for (cid, att), c in sorted(per.items()):
        sent = c["sent"]
        rate = (c["non_cut"] / sent) if sent else None
        out.append({"cell_id": cid, "attempt": att, **cells[(cid, att)], **c,
                    "cut_share_of_sent": (c["cut"] / sent) if sent else None,
                    "non_cut_rate": rate,
                    "model_error_rate": (c["model_error"] / sent) if sent else None,
                    "void_at_audit": bool(sent == 0 or (rate is not None and rate > NON_CUT_ERROR_LIMIT)),
                    "runtime_limit_exceeded": bool(sent and c["model_error"] / sent > RUNTIME_MODEL_ERROR_LIMIT)})
    return {"rule": dict(RULE), "requests_csv": str(requests_csv), "cells": out,
            "void_at_audit": [c["cell_id"] for c in out if c["void_at_audit"]],
            "totals": {k: sum(c[k] for c in out) for k in ("sent", "model_error", "cut", "non_cut")}}


def record() -> dict:
    """What a collection binds in its plan / manifest (M2: ``stream_cut``)."""
    return {"max_model_error_rate": RUNTIME_MODEL_ERROR_LIMIT, "cut_e2e_ms": ROUTE_TIMEOUT_CUT_MS,
            "non_cut_error_limit": NON_CUT_ERROR_LIMIT, "rule": dict(RULE),
            "implementation": "scripts.stream_cut (shared by the T14 scorer and dline_refit accept)",
            "label": ("unchanged: a cut request is an unserved request (model_errors of its send window) - a "
                      "violation under every attribution")}
