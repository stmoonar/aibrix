"""The unified per-request SLO and the SLO-clean window judge (user decisions 2026-11-04).

One SLO for every request of every tool that calibrates on it (PreServe's mu first):

* TTFT_slo(L) = ``max(floor, k * (c_m + b_m * L))``, the one threshold function
  ``tre_common.slo_labels.request_ttft_slo_ms`` (c / b = registry ``models[].slo.ttft_idle_c_ms`` /
  ``ttft_idle_b_ms_per_token``, k = ``ttft_slowdown_k`` (5), floor = ``ttft_floor_ms`` (500 ms),
  L = the request's prompt tokens);
* TPOT_slo = the registry ``slo.tpot_p95_ms`` (75 ms).

The rule is the slowdown formula whatever the registry's ``ttft_slo_mode`` says (that key picks the
online / scoring mode; it is recorded, not obeyed).

Input is a list of :class:`Request` (unix seconds), independent of where it was logged; one thin
adapter per source turns that source's rows into it (:func:`from_ladder_record` for the
``velocity_ladder`` raw logs, :func:`from_client_row` for the replayer's per-request client log
``performance_metrics.json`` of the campaign runs). A request violates the SLO when it

1. failed (``not ok``) or ran to the route timeout (``done - send >= route_timeout_s``) -> ``failed``;
2. else TTFT = ``first - send`` > TTFT_slo(L) -> ``ttft``;
3. else TPOT = ``(done - first) / (completion_tokens - 1)`` > TPOT_slo -> ``tpot`` (one output
   token: TPOT is undefined, TTFT alone is judged).

A **window** [lo, hi) holds the requests *sent* in it; it is SLO-clean when it has at least one
request and none violates. A served request without a first-token time (not streamed) cannot be
judged: the window is refused (:class:`UnjudgeableWindow`), never judged on the rest.
:func:`judge_windows` cuts a span into tumbling windows (10 s by default, user 2026-11-04).

**NL rule** (PreServe's own SLO, arXiv 2504.03702v2: normalized latency NL = e2e / output tokens):
:func:`judge_request_nl` / :func:`judge_window_nl` judge the same :class:`Request` records with one
NL threshold per model (ms per output token; user 2026-11-04: the NL equivalent of the unified SLO at
the model's mix shape, an input of the caller, never a constant here). Step 1 (failed / route
timeout) is the same; else NL = ``(done - send) / completion_tokens`` > NL_slo -> ``nl``. A served
request without an output-token count cannot be judged (:class:`UnjudgeableWindow`).
:func:`from_dataset_row` adapts a ``calibration_dataset`` ``requests.csv`` row (the TSS / theta
training set).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from tre_common.slo_labels import TTFT_SLO_MODE_SLOWDOWN, LabelDefinition, label_def_for_model, request_ttft_slo_ms

DEFAULT_ROUTE_TIMEOUT_S = 150.0
DEFAULT_WINDOW_S = 10.0
CAUSES = ("failed", "ttft", "tpot")

#: one SLO, or model -> SLO for records of several models
Slo = Union[LabelDefinition, Mapping[str, LabelDefinition]]


class UnjudgeableWindow(ValueError):
    """A request in the window cannot be judged (served without a first-token time, or no
    prompt length / no SLO for its model)."""


@dataclass(frozen=True)
class Request:
    """One client request, unix seconds. ``first`` None = no first-token time (not streamed);
    ``done`` = the last token (None when it failed)."""

    model: str
    send: float
    first: Optional[float]
    done: Optional[float]
    ok: bool
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    id: Any = None


# ------------------------------------------------------------------ adapters

def from_ladder_record(rec: Mapping[str, Any], model: str, default_prompt_tokens: Optional[int] = None) -> Request:
    """A ``velocity_ladder`` raw record (``send`` / ``first`` / ``done`` / ``ok`` / token counts);
    ``default_prompt_tokens`` (the step's input length) stands in for a missing ``prompt_tokens``."""
    L = rec.get("prompt_tokens")
    return Request(model=model, send=float(rec["send"]), first=rec.get("first"), done=rec.get("done"),
                   ok=bool(rec.get("ok")), prompt_tokens=default_prompt_tokens if L is None else int(L),
                   completion_tokens=rec.get("completion_tokens"), id=rec.get("k"))


def from_client_row(row: Mapping[str, Any]) -> Request:
    """A replayer client row (``performance_metrics.json``), read as ``tre_eval.metrics`` scores it:
    send = ``start_time``, first = send + ``ttft``, done = send + ``e2e_latency``, ok =
    ``success_strict`` (rows before 2026-10-29 lack it: refused, completion unknown)."""
    strict = row.get("success_strict")
    if strict is None:
        raise UnjudgeableWindow(f"client row {row.get('request_id')}: no success_strict (completion unknown)")
    send = float(row["start_time"])
    ttft, e2e = row.get("ttft"), row.get("e2e_latency")
    return Request(model=str(row.get("model_name")), send=send,
                   first=None if ttft is None else send + float(ttft),
                   done=None if e2e is None else send + float(e2e), ok=strict is True,
                   prompt_tokens=row.get("input_tokens"), completion_tokens=row.get("output_tokens"),
                   id=row.get("request_id"))


# ------------------------------------------------------------------ SLO

def unified_slo(model: str, registry: Any) -> LabelDefinition:
    """The model's unified SLO from a registry (path or loaded): slowdown TTFT with the registry's
    c / b / k / floor, TPOT = the registry's ``tpot_p95_ms``. Refuses (SystemExit) without c / b."""
    if registry is None or isinstance(registry, str):
        from tre_common.registry import load_registry

        registry = load_registry(registry)
    spec = registry.model(model).slo
    return label_def_for_model(model, mode=TTFT_SLO_MODE_SLOWDOWN, tpot_p95_ms=spec.tpot_p95_ms,
                               ttft_p95_ms=spec.ttft_p95_ms, registry=registry)


def slo_record(slo: LabelDefinition, registry_mode: Optional[str] = None) -> Dict[str, Any]:
    """What a result file records of the SLO it judged with."""
    return {"rule": "TTFT <= max(floor_ms, k * (c_ms + b_ms_per_token * L)), TPOT <= tpot_ms",
            "c_ms": slo.ttft_idle_c_ms, "b_ms_per_token": slo.ttft_idle_b_ms_per_token,
            "k": slo.ttft_slowdown_k, "floor_ms": slo.ttft_floor_ms, "tpot_ms": slo.tpot_p95_ms,
            "registry_ttft_slo_mode": registry_mode}


def _slo_of(slo: Slo, model: str) -> LabelDefinition:
    if isinstance(slo, LabelDefinition):
        return slo
    if model not in slo:
        raise UnjudgeableWindow(f"no SLO for model {model!r}")
    return slo[model]


# ------------------------------------------------------------------ judges

def judge_request(req: Request, slo: Slo, *, route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S) -> Dict[str, Any]:
    """One request: ``{cause: failed|ttft|tpot|None, ttft_ms, tpot_ms, ttft_slo_ms}``."""
    if not req.ok or req.done is None or req.done - req.send >= route_timeout_s:
        return {"cause": "failed", "ttft_ms": None, "tpot_ms": None, "ttft_slo_ms": None}
    if req.first is None:
        raise UnjudgeableWindow(f"request {req.id} ({req.model}, sent {req.send:.3f}) was served without a "
                                "first-token time (not streamed): TTFT / TPOT unknown")
    if req.prompt_tokens is None:
        raise UnjudgeableWindow(f"request {req.id} ({req.model}): no prompt_tokens, TTFT SLO unknown")
    s = _slo_of(slo, req.model)
    thr = float(s.ttft_slo_ms(req.prompt_tokens))
    ttft = (req.first - req.send) * 1000.0
    ct = req.completion_tokens
    tpot = (req.done - req.first) * 1000.0 / (int(ct) - 1) if ct is not None and int(ct) > 1 else None
    cause = "ttft" if ttft > thr else ("tpot" if tpot is not None and tpot > float(s.tpot_p95_ms) else None)
    return {"cause": cause, "ttft_ms": ttft, "tpot_ms": tpot, "ttft_slo_ms": thr}


def _q(vals: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank quantile."""
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))] if v else None


def _r(x: Optional[float], nd: int = 1) -> Optional[float]:
    return None if x is None else round(x, nd)


def judge_window(requests: Sequence[Request], lo: float, hi: float, slo: Slo, *,
                 route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S) -> Dict[str, Any]:
    """The requests sent in [lo, hi): count, violations by cause, TTFT / TPOT p95 and max, the
    TTFT thresholds used; ``meets_slo`` = at least one request and no violation. Raises
    :class:`UnjudgeableWindow` when a request in it cannot be judged."""
    win = [r for r in requests if lo <= r.send < hi]
    judged = [judge_request(r, slo, route_timeout_s=route_timeout_s) for r in win]
    by = {c: sum(1 for j in judged if j["cause"] == c) for c in CAUSES}
    ttft = [j["ttft_ms"] for j in judged if j["ttft_ms"] is not None]
    tpot = [j["tpot_ms"] for j in judged if j["tpot_ms"] is not None]
    thr = [j["ttft_slo_ms"] for j in judged if j["ttft_slo_ms"] is not None]
    n_viol = sum(by.values())
    return {"lo": lo, "hi": hi, "n": len(win), "violations": n_viol, **{f"v_{c}": by[c] for c in CAUSES},
            "ttft_p95_ms": _r(_q(ttft, 0.95)), "ttft_max_ms": _r(max(ttft, default=None)),
            "tpot_p95_ms": _r(_q(tpot, 0.95), 2), "tpot_max_ms": _r(max(tpot, default=None), 2),
            "ttft_slo_ms_min": _r(min(thr, default=None)), "ttft_slo_ms_max": _r(max(thr, default=None)),
            "meets_slo": bool(win) and n_viol == 0}


def judge_windows(requests: Sequence[Request], slo: Slo, *, window_s: float = DEFAULT_WINDOW_S,
                  start: Optional[float] = None, end: Optional[float] = None,
                  route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S) -> List[Dict[str, Any]]:
    """Tumbling ``window_s`` windows over [start, end) (default: first to last send), each judged by
    :func:`judge_window`; a window that would end after ``end`` is left out (not complete)."""
    if not window_s > 0:
        raise ValueError("window_s must be > 0")
    sends = [r.send for r in requests]
    if not sends:
        return []
    t0 = min(sends) if start is None else float(start)
    t1 = (max(sends) + 1e-9) if end is None else float(end)
    out, k = [], 0
    while t0 + (k + 1) * window_s <= t1 + 1e-9:
        lo = t0 + k * window_s
        out.append(judge_window(requests, lo, lo + window_s, slo, route_timeout_s=route_timeout_s))
        k += 1
    return out


# ------------------------------------------------------------------ NL rule (PreServe)

#: one NL SLO (ms per output token), or model -> NL SLO
NlSlo = Union[float, Mapping[str, float]]
NL_CAUSES = ("failed", "nl")


def _num(v: Any) -> Optional[float]:
    """A CSV / JSON number; '' / 'nan' / None -> None."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if not v or v.lower() == "nan":
            return None
    x = float(v)
    return None if math.isnan(x) else x


def from_dataset_row(row: Mapping[str, Any]) -> Request:
    """A ``calibration_dataset`` ``requests.csv`` row (ms, csv strings or numbers): send / first / done
    = ``send_ts_ms`` / ``first_token_ts_ms`` / ``done_ts_ms``, ok = ``outcome == "ok"``, prompt =
    ``input_tokens`` (a failed request has none: ``expected_prompt_tokens``), completion =
    ``output_tokens`` (None for a failed request)."""
    def sec(key: str) -> Optional[float]:
        x = _num(row.get(key))
        return None if x is None else x / 1000.0

    L = _num(row.get("input_tokens"))
    if L is None:
        L = _num(row.get("expected_prompt_tokens"))
    ct = _num(row.get("output_tokens"))
    return Request(model=str(row.get("model")), send=float(sec("send_ts_ms")), first=sec("first_token_ts_ms"),
                   done=sec("done_ts_ms"), ok=str(row.get("outcome")) == "ok",
                   prompt_tokens=None if L is None else int(L), completion_tokens=None if ct is None else int(ct),
                   id=row.get("request_id"))


def _nl_of(nl_slo: NlSlo, model: str) -> float:
    if isinstance(nl_slo, Mapping):
        if model not in nl_slo:
            raise UnjudgeableWindow(f"no NL SLO for model {model!r}")
        nl_slo = nl_slo[model]
    v = float(nl_slo)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"NL SLO for {model!r} must be > 0 ms, got {v}")
    return v


def judge_request_nl(req: Request, nl_slo: NlSlo, *,
                     route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S) -> Dict[str, Any]:
    """One request under PreServe's NL rule: ``{cause: failed|nl|None, nl_ms, nl_slo_ms}``."""
    if not req.ok or req.done is None or req.done - req.send >= route_timeout_s:
        return {"cause": "failed", "nl_ms": None, "nl_slo_ms": None}
    ct = req.completion_tokens
    if ct is None or int(ct) < 1:
        raise UnjudgeableWindow(f"request {req.id} ({req.model}, sent {req.send:.3f}) was served without an "
                                "output-token count: NL unknown")
    thr = _nl_of(nl_slo, req.model)
    nl = (req.done - req.send) * 1000.0 / int(ct)
    return {"cause": "nl" if nl > thr else None, "nl_ms": nl, "nl_slo_ms": thr}


def judge_window_nl(requests: Sequence[Request], lo: float, hi: float, nl_slo: NlSlo, *,
                    route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S) -> Dict[str, Any]:
    """The requests sent in [lo, hi) under the NL rule: count, violations by cause, NL p95 / max;
    ``meets_slo`` = at least one request and no violation. Raises :class:`UnjudgeableWindow`."""
    win = [r for r in requests if lo <= r.send < hi]
    judged = [judge_request_nl(r, nl_slo, route_timeout_s=route_timeout_s) for r in win]
    by = {c: sum(1 for j in judged if j["cause"] == c) for c in NL_CAUSES}
    nl = [j["nl_ms"] for j in judged if j["nl_ms"] is not None]
    n_viol = sum(by.values())
    return {"lo": lo, "hi": hi, "n": len(win), "violations": n_viol, **{f"v_{c}": by[c] for c in NL_CAUSES},
            "nl_p95_ms": _r(_q(nl, 0.95), 2), "nl_max_ms": _r(max(nl, default=None), 2),
            "meets_slo": bool(win) and n_viol == 0}


# ------------------------------------------------------------------ TTFT + TPOT limits from idle multiples

TT_CAUSES = ("failed", "ttft", "tpot")
TT_KEYS = ("c_ms", "b_ms_per_token", "k_ttft", "k_tpot", "ttft_floor_ms", "tpot_idle_ms")


def judge_request_ttft_tpot(req: Request, params: Mapping[str, Mapping[str, float]], *,
                            route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S) -> Dict[str, Any]:
    """One request under the idle-multiple TTFT + TPOT rule (the limits of ``preserve_nl_slo``):
    ``ttft <= max(floor, k_ttft * (c + b L))`` and ``tpot = (done - first) / (out - 1) <= k_tpot * tpot_idle``
    (one output token: TTFT only). ``params[model]`` holds ``TT_KEYS``. Cause: failed | ttft | tpot | None."""
    if not req.ok or req.done is None or req.done - req.send >= route_timeout_s:
        return {"cause": "failed", "ttft_ms": None, "tpot_ms": None}
    if req.first is None or req.prompt_tokens is None or req.completion_tokens is None:
        raise UnjudgeableWindow(f"request {req.id} ({req.model}, sent {req.send:.3f}) was served without a "
                                "first-token time or token counts: TTFT / TPOT unknown")
    if req.model not in params:
        raise UnjudgeableWindow(f"no TTFT / TPOT limits for model {req.model!r}")
    q = params[req.model]
    thr = request_ttft_slo_ms(req.prompt_tokens, c_ms=q["c_ms"], b_ms_per_token=q["b_ms_per_token"],
                              k=q["k_ttft"], floor_ms=q["ttft_floor_ms"])
    ttft = (req.first - req.send) * 1000.0
    ct = int(req.completion_tokens)
    tpot = (req.done - req.first) * 1000.0 / (ct - 1) if ct > 1 else None
    cause = "ttft" if ttft > thr else ("tpot" if tpot is not None and tpot > q["k_tpot"] * q["tpot_idle_ms"] else None)
    return {"cause": cause, "ttft_ms": ttft, "tpot_ms": tpot}


def judge_window_ttft_tpot(requests: Sequence[Request], lo: float, hi: float, params: Mapping[str, Mapping[str, float]],
                           *, route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S) -> Dict[str, Any]:
    """The requests sent in [lo, hi): violations by cause; ``meets_slo`` = at least one request and none violates."""
    win = [r for r in requests if lo <= r.send < hi]
    judged = [judge_request_ttft_tpot(r, params, route_timeout_s=route_timeout_s) for r in win]
    by = {c: sum(1 for j in judged if j["cause"] == c) for c in TT_CAUSES}
    ttft = [j["ttft_ms"] for j in judged if j["ttft_ms"] is not None]
    tpot = [j["tpot_ms"] for j in judged if j["tpot_ms"] is not None]
    n_viol = sum(by.values())
    return {"lo": lo, "hi": hi, "n": len(win), "violations": n_viol, **{f"v_{c}": by[c] for c in TT_CAUSES},
            "ttft_p95_ms": _r(_q(ttft, 0.95), 2), "tpot_p95_ms": _r(_q(tpot, 0.95), 2),
            "meets_slo": bool(win) and n_viol == 0}
