"""Offline audit of a generated plan against the design rules (docs/trace-design-v2.md).

Load model (as in trace-audit-20261007): a request costs ``capacity.ModelCap.cost`` replica-seconds
at rho = 1 (the knee, or V_slo with a calibrated capacity file); ``rho_m`` = cost per 10 s bin / 10;
``G = sum rho_m * gpus_m``; replicas needed
at the SLO margin = ``max(floor, ceil(rho / u))``, ``u`` = 0.85, capped at ``max_awake``.

* **R1** contention, judged on the *design* load of a synthetic plan (rho(t) on a 1 s grid:
  the offered rates; Poisson and length noise in realised 10 s / 30 s bins is reported
  beside it, not judged) and on realised 30 s windows of a real slice:
  ``max G <= u * pool`` (outside the spec's
  ``audit.overload_windows_s``; ``audit.r1a_stat: p99`` judges the p99 bin instead, for
  real slices) **and** ``static = sum_m replicas(peak rho_m) * gpus_m > pool``.
* **R2** timescale: every complete hot run lasts ``>= audit.r2_min_s`` (150 s). A hot run is a
  maximal interval where the *design* rho (1 s grid) is at or above the model's threshold
  (``audit.r2_slack_s`` = 3 s, one ramp, of slack: where a 3 s ramp crosses the threshold
  moves the measured length by up to one ramp):
  ``audit.hot_rho[model]``, else the midpoint of its design range when the model moves
  (max >= 1.25 min and max - min >= 0.3). Runs cut by the trace start/end are not judged.
* **R3** arrivals: 1 s counts detrended by a centred 31 s moving mean; CV of the residual
  vs the Azure reference process at the same mean rate (:func:`azure.reference_cv`, dataset
  ``audit.r3_reference``, default conv2023); met when the ratio is within [0.8, 1.25].
* **R5** depth: the median design rho over a model's hot runs (= the plateau) is ``>= 2.0``.
* Exemptions: ``audit.exempt`` {"R1a"|"R1b"|"R2"|"R3"|"R5": reason} or
  {"R5": {"<model>": reason}} - reported as "exempt", never as "met".
* **R7** contention, on the design rho (1 s grid): seconds where the replicas needed at V_slo
  (``ceil(rho)``, floor / max_awake applied) summed over models (x gpus) exceed the pool, and the
  same with TokenScale's knee view (``ceil(rho_factor * rho)``); seconds a model's design rho
  sits in TokenScale's knee band ``(k, k / rho_factor]`` (k = 1..3), on plateaus (rho constant)
  and in all. ``audit.r7.contention`` true: met when both need sums exceed the pool somewhere;
  false: met when no run of seconds outside ``overload_windows_s`` longer than one ramp
  (``r2_slack_s``, 3 s: models switching shape at the same edge) needs more than the pool, in
  the additive and in the shape-aware unit (design rho x the 30 s bin's shape factor, below). ``audit.r7.blind_band_free``: no plateau second
  in the band.
* **R8** pre-registered decision points (``audit.r8.events``), checked with a fluid queue: per
  model, work arrives at the design rho (replica-seconds at V_slo per second) and is served at
  ``n / rho_factor`` (the knee); the backlog diverges above it and drains below it. An option
  is an allocation that holds from ``start + lag_s`` to ``end + lag_s`` (base outside it). An
  arrival violates when the backlog wait ``B / (n / rho_factor)`` is over 0.5 s, else with
  probability ``(u - 1) / (1 / rho_factor - 1)`` clipped to [0, 1] for per-replica load
  ``u = rho / n`` (0 at V_slo, 1 at the knee). Scored over ``[start, until_s)``: violated
  requests (V_req) and the worst model's violated fraction (max-min); met when the expected
  option is best on both, at ``lag_s`` (``2 lag_s`` reported beside it). Also the static
  max-min per-replica rho and when every backlog is drained.
* **Shape-aware capacity** (:mod:`.shape`, report; judged only through R7 = false): 30 s bins of
  the plan with the KV knee, ``G_shape`` and the summed need.
* **In cap**: no prompt above ``audit.in_max`` (8192) tokens (``in_cap_ok``).
* **Out cap**: no request's ``max_output_tokens`` above the model's route-timeout cap
  (``capacity.route_cap``; ``out_cap_ok``).
* **In flight** (Little's law at the knee): a request occupies its replica for
  ``W = ttft_knee + out * tpot_knee``. ``L(t)`` = requests in flight at knee speed. Per
  replica: static allocation (peak) = ``L / r``; lagged oracle (replicas follow the design
  rho delayed by ``audit.lag_s`` = 30 s, each model alone, pool ignored) adds a fluid backlog
  ``B += work - r`` and gives ``(L * min(1, r / rho) + B / cost) / r``; compared with the
  ~420 open-file limit the reissue sidecar had on node10 before its fd fix
  (``audit.inflight_limit``; reported, no longer a design limit). Also the
  backlog wait ``B / r`` and the share of requests whose ``W`` alone exceeds the 150 s route
  timeout (``audit.route_timeout_s``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import azure
from .capacity import Capacity, load_capacity, load_fits
from .generate import DESIGN_FILE, MANIFEST_FILE, model_plans
from .lengths import DEFAULT_TRUNC_Q  # noqa: F401  (documented default)
from . import rates
from . import shape as shape_mod

BIN_S = 10
R2_MIN_S = 150.0
R5_MIN_RHO = 2.0
R3_BAND = (0.8, 1.25)
LAG_S = 30.0
INFLIGHT_LIMIT = 420
IN_MAX = 8192
FLUID_WAIT_S = 0.5
ROUTE_TIMEOUT_S = 150.0


def _cap_from_manifest(manifest: dict) -> Capacity:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(manifest["capacity"], fh)
    try:
        return load_capacity(fh.name)
    finally:
        Path(fh.name).unlink()


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i)); start = None
    return out


def _detrended_cv(counts: list[int], window: int = azure.TREND_WINDOW_S) -> tuple[float, float]:
    n = len(counts)
    half = window // 2
    pre = [0]
    for c in counts:
        pre.append(pre[-1] + c)
    resid = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        resid.append(counts[i] - (pre[hi] - pre[lo]) / (hi - lo))
    mean = sum(counts) / n
    mr = sum(resid) / n
    var = sum((r - mr) ** 2 for r in resid) / n
    return (math.sqrt(var) / mean if mean else float("nan")), mean


def audit_run(run_dir: str | Path, *, fits: dict | None = None) -> dict:
    run = Path(run_dir)
    manifest = json.loads((run / MANIFEST_FILE).read_text())
    rows = json.loads((run / DESIGN_FILE).read_bytes())
    spec = manifest["spec"]
    a = spec.get("audit", {})
    exempt = a.get("exempt", {})
    cap = _cap_from_manifest(manifest)
    fits = fits if fits is not None else load_fits()
    dur = float(spec["duration_s"])
    nb = int(math.ceil(dur / BIN_S))
    ns = int(math.ceil(dur))
    models = sorted({r["model_name"] for r in rows} | set(spec.get("models", {})) | set(spec.get("real", {}).get("rho", {})))
    seed = int(manifest["seed"])
    plans = model_plans(spec, seed, cap, fits) if "models" in spec else {}

    per = {}
    for m in models:
        mc = cap.models[m]
        work = [0.0] * nb
        counts = [0] * ns
        work_s = [0.0] * ns
        diff = [0] * (ns + 2)
        n = 0
        wsum = 0.0
        over_route = 0
        wmax = 0.0
        omax = 0
        imax = 0
        for r in rows:
            if r["model_name"] != m:
                continue
            t = r["timestamp"]
            c = mc.cost(r["prompt_length"], r["max_output_tokens"])
            work[min(nb - 1, int(t // BIN_S))] += c
            k = min(ns - 1, int(t))
            counts[k] += 1
            work_s[k] += c
            omax = max(omax, r["max_output_tokens"])
            imax = max(imax, r["prompt_length"])
            w = mc.service_s(r["max_output_tokens"])
            wmax = max(wmax, w)
            over_route += w > a.get("route_timeout_s", ROUTE_TIMEOUT_S)
            diff[k] += 1
            diff[min(ns + 1, k + int(math.ceil(w)))] -= 1
            n += 1
            wsum += c
        rho_bins = [x / BIN_S for x in work]
        L, acc = [], 0
        for k in range(ns):
            acc += diff[k]
            L.append(acc)
        mean_cost = wsum / n if n else 1.0
        per[m] = {"rho_bins": rho_bins, "counts": counts, "work_s": work_s, "L": L, "n": n,
                  "mean_cost": mean_cost, "w_max_s": wmax, "out_max_seen": omax, "in_max_seen": imax, "over_route": over_route}

    # realised rho on 10 s and 30 s bins; design rho on a 1 s grid (synthetic plans) - real
    # slices have no design, their "design" is the realised 30 s series
    def rebin(m, width):
        n = int(math.ceil(dur / width))
        w = [0.0] * n
        for k, x in enumerate(per[m]["work_s"]):
            w[min(n - 1, k // width)] += x
        return [x / width for x in w]
    rho30 = {m: rebin(m, 30) for m in models}
    design_rho = {}
    for m in models:
        if m in plans:
            design_rho[m] = rates.sample(plans[m].rho, dur, 1.0)
        else:
            design_rho[m] = [rho30[m][min(len(rho30[m]) - 1, i // 30)] for i in range(ns)]
    synthetic = bool(plans)

    overload = a.get("overload_windows_s", [])

    def overlaps(lo, hi):
        return any(lo < w[1] and hi > w[0] for w in overload)

    def gseries(series, width):
        n = len(next(iter(series.values())))
        g = [sum(series[m][i] * cap.models[m].gpus for m in models) for i in range(n)]
        keep = [x for i, x in enumerate(g) if not overlaps(i * width, (i + 1) * width)]
        return g, keep

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))]
    Gd, Gd_in = gseries(design_rho, 1)
    G10, G10_in = gseries({m: per[m]["rho_bins"] for m in models}, BIN_S)
    G30, G30_in = gseries(rho30, 30)

    def static_of(series, width, excl):
        tot = 0
        for m in models:
            vals = [x for i, x in enumerate(series[m]) if not (excl and overlaps(i * width, (i + 1) * width))]
            tot += cap.replicas_needed(m, max(vals)) * cap.models[m].gpus
        return tot
    if synthetic:   # R1 on the design load (the offered rates); realised bins reported beside it
        basis, static, static_in = "design", static_of(design_rho, 1, False), static_of(design_rho, 1, True)
        g_all, g_in = Gd, Gd_in
    else:           # real slices: realised 30 s windows
        basis, static, static_in = "realised_30s", static_of(rho30, 30, False), static_of(rho30, 30, True)
        g_all, g_in = G30, G30_in
    peak = {m: max(design_rho[m]) for m in models}
    limit_g = cap.u_target * cap.pool_gpus
    stat = a.get("r1a_stat", "max")
    g_judged = pct(g_in, 0.99) if stat == "p99" else max(g_in)
    out = {"trace": manifest["trace"], "seed": seed, "duration_s": dur, "requests": len(rows),
           "R1_basis": basis, "G_max": max(g_all), "G_max_excl_overload": max(g_in), "G_mean": sum(g_all) / len(g_all),
           "G_p99": pct(g_all, 0.99), "G_judged": g_judged, "G_judged_stat": stat,
           "G_realised_10s_max": max(G10), "G_realised_30s_max": max(G30),
           "G_realised_30s_max_excl_overload": max(G30_in), "G_realised_30s_mean": sum(G30) / len(G30),
           "static": static, "static_excl_overload": static_in,
           "static_realised_30s": static_of(rho30, 30, False),
           "overload_windows_s": overload, "models": {}}
    r1a = g_judged <= limit_g + 1e-9
    r1b = static > cap.pool_gpus
    out["R1a"] = "exempt" if "R1a" in exempt else ("met" if r1a else "FAIL")
    out["R1b"] = "exempt" if "R1b" in exempt else ("met" if r1b else "FAIL")

    r2_min = float(a.get("r2_min_s", R2_MIN_S))
    hot_rho = a.get("hot_rho", {})
    ref_name = a.get("r3_reference", "conv2023")
    lag = float(a.get("lag_s", LAG_S))
    ramp_slack = float(a.get("r2_slack_s", 3.0))
    r2_all, r3_all, r5_all = [], [], []
    for m in models:
        mc = cap.models[m]
        d = design_rho[m]
        p = per[m]
        lo, hi = min(d), max(d)
        moving = bool(a.get("hot", True)) and hi >= 1.25 * lo and hi - lo >= 0.3
        thr = hot_rho.get(m, (lo + hi) / 2 if moving else None)
        runs = _runs([x >= thr for x in d]) if thr is not None and synthetic else []
        complete = [(s, e) for s, e in runs if s > 0 and e < ns]
        hot_s = [e - s for s, e in complete]
        hot_vals = sorted(x for s, e in runs for x in d[s:e])
        hot_level = hot_vals[len(hot_vals) // 2] if hot_vals else None
        hot_bins = [p["rho_bins"][b] for b in range(nb)
                    if any(s <= b * BIN_S and (b + 1) * BIN_S <= e for s, e in runs)]
        cv, mean_rate = _detrended_cv(p["counts"])
        ref_cv = azure.reference_cv(fits[ref_name], mean_rate) if ref_name in fits and mean_rate > 0 else None
        ratio = cv / ref_cv if ref_cv else None
        # in flight: static allocation and the lagged oracle with a fluid backlog
        r_static = cap.replicas_needed(m, peak[m])
        backlog, worst_lag, worst_wait = 0.0, 0.0, 0.0
        lag_k = int(lag)
        for k in range(ns):
            rho_t = d[k]
            r = cap.replicas_needed(m, d[k - lag_k] if k >= lag_k else d[0])
            backlog = max(0.0, backlog + p["work_s"][k] - r)
            inflight = (p["L"][k] * min(1.0, r / rho_t if rho_t > 0 else 1.0) + backlog / p["mean_cost"]) / r
            worst_lag = max(worst_lag, inflight)
            worst_wait = max(worst_wait, backlog / r)
        mres = {
            "requests": p["n"], "mean_rps": p["n"] / dur, "mean_rho": sum(p["rho_bins"]) / nb, "peak_rho": peak[m],
            "design_rho_range": [lo, hi], "hot_threshold": thr, "hot_runs_s": [[s, e] for s, e in runs],
            "hot_run_min_s": min(hot_s) if hot_s else None, "hot_level_design": hot_level,
            "hot_level_realised": sum(hot_bins) / len(hot_bins) if hot_bins else None,
            "cv_1s": cv, "cv_ref": ref_cv, "cv_ratio": ratio,
            "inflight_total_max": max(p["L"]) if p["L"] else 0,
            "peak_rho_realised_10s": max(p["rho_bins"]), "peak_rho_realised_30s": max(rho30[m]),
            "inflight_per_replica_static_max": max(p["L"]) / r_static if p["L"] else 0,
            "replicas_static": r_static,
            "inflight_per_replica_lag_max": worst_lag, "backlog_wait_max_s": worst_wait,
            "out_max_seen": p["out_max_seen"], "out_cap": mc.out_max, "in_max_seen": p["in_max_seen"],
            "w_max_s": p["w_max_s"], "over_route_timeout_frac": p["over_route"] / p["n"] if p["n"] else 0.0,
        }
        r2 = None if not complete else all(x >= r2_min - ramp_slack for x in hot_s)
        r5 = None if hot_level is None else hot_level >= R5_MIN_RHO - 1e-6
        r3 = None if ratio is None else R3_BAND[0] <= ratio <= R3_BAND[1]
        r5x = exempt.get("R5")
        if isinstance(r5x, dict) and m in r5x:
            r5 = "exempt"
        mres.update({"R2": r2, "R3": r3, "R5": r5})
        out["models"][m] = mres
        r2_all.append(r2); r3_all.append(r3); r5_all.append(r5)

    def agg(rule, vals):
        if rule in exempt and not isinstance(exempt[rule], dict):
            return "exempt"
        judged = [v for v in vals if v is not None and v != "exempt"]
        if not judged:
            return "n/a" if "exempt" not in vals else "exempt"
        return "met" if all(judged) else "FAIL"
    out["R2"] = agg("R2", r2_all)
    out["R3"] = agg("R3", r3_all)
    out["R5"] = agg("R5", r5_all)
    out["exempt"] = exempt
    limit = int(a.get("inflight_limit", INFLIGHT_LIMIT))
    out["inflight_limit"] = limit
    out["inflight_lag_max"] = max(v["inflight_per_replica_lag_max"] for v in out["models"].values())
    out["inflight_ok"] = out["inflight_lag_max"] <= limit
    out["out_cap_ok"] = all(v["out_cap"] is None or v["out_max_seen"] <= v["out_cap"] for v in out["models"].values())
    in_max = int(a.get("in_max", IN_MAX))
    out["in_max"] = in_max
    out["in_cap_ok"] = all(v["in_max_seen"] <= in_max for v in out["models"].values())

    # shape-aware capacity on realised 30 s bins
    sc = shape_mod.load_shape()
    srho = {m: shape_mod.rho_series(rows, m, cap.models[m], sc[m], dur) for m in models if m in sc}
    nbin = int(math.ceil(dur / 30))
    g_sh, need_sh, need_add = [], [], []
    for b in range(nbin):
        g_sh.append(sum(srho[m][b]["rho_shape"] * cap.models[m].gpus for m in srho))
        need_sh.append(sum(_need(cap, m, srho[m][b]["rho_shape"]) * cap.models[m].gpus for m in srho))
        need_add.append(sum(_need(cap, m, srho[m][b]["rho_add"]) * cap.models[m].gpus for m in srho))
    keep = [b for b in range(nbin) if not overlaps(b * 30, (b + 1) * 30)]
    # design rho in the shape-aware unit: design rho x the bin's shape factor (rho_shape / rho_add
    # of its requests), so Poisson noise does not move the judged need
    kfac = {m: [(x["rho_shape"] / x["rho_add"]) if x["rho_add"] > 0 else 1.0 for x in srho[m]] for m in srho}
    need_sh_design = [sum(_need(cap, m, design_rho[m][k] * kfac[m][min(nbin - 1, k // 30)]) * cap.models[m].gpus
                          for m in srho) for k in range(ns)]
    out["shape"] = {
        "need_shape_design_max": max(need_sh_design),
        "need_shape_design_over_pool_s_excl_overload": sum(1 for k in range(ns) if need_sh_design[k] > cap.pool_gpus
                                                           and not overlaps(k, k + 1)),
        "need_shape_design_over_pool_runs": [[s0, e0] for s0, e0 in _runs([need_sh_design[k] > cap.pool_gpus
                                                                           and not overlaps(k, k + 1) for k in range(ns)])],
        "rho_shape_design_hot": {m: max(design_rho[m][k] * kfac[m][min(nbin - 1, k // 30)] for k in range(ns)) for m in srho},
        "G_shape_30s_max": max(g_sh), "G_add_30s_max": max(G30),
        "need_shape_30s_max": max(need_sh), "need_add_30s_max": max(need_add),
        "need_shape_over_pool_bins": [b * 30 for b in keep if need_sh[b] > cap.pool_gpus],
        "need_add_over_pool_bins": [b * 30 for b in keep if need_add[b] > cap.pool_gpus],
        "kv_bound_bins": {m: [x["t0"] for x in srho[m] if x["bind"] == "kv"] for m in srho},
        "rho_shape_max": {m: max(x["rho_shape"] for x in srho[m]) for m in srho},
        "rho_add_max": {m: max(x["rho_add"] for x in srho[m]) for m in srho}}

    # R7 on the design grid
    r7cfg = a.get("r7")
    pool = cap.pool_gpus
    need_slo = [sum(_need(cap, m, design_rho[m][k]) * cap.models[m].gpus for m in models) for k in range(ns)]
    need_ts = [sum(_need(cap, m, cap.models[m].rho_factor * design_rho[m][k]) * cap.models[m].gpus for m in models)
               for k in range(ns)]
    in_ov = [overlaps(k, k + 1) for k in range(ns)]
    blind = {}
    for m in models:
        f = cap.models[m].rho_factor
        d = design_rho[m]
        inb = [any(kk < d[k] <= kk / f + 1e-12 for kk in (1, 2, 3)) for k in range(ns)]
        flat = [k > 0 and abs(d[k] - d[k - 1]) < 1e-9 for k in range(ns)]
        blind[m] = {"all_s": sum(inb), "plateau_s": sum(1 for k in range(ns) if inb[k] and flat[k])}
    r7 = {"need_slo_over_pool_s": sum(1 for k in range(ns) if need_slo[k] > pool),
          "need_ts_over_pool_s": sum(1 for k in range(ns) if need_ts[k] > pool),
          "need_slo_over_pool_s_excl_overload": sum(1 for k in range(ns) if need_slo[k] > pool and not in_ov[k]),
          "need_slo_max": max(need_slo), "need_ts_max": max(need_ts), "knee_band_s": blind, "config": r7cfg,
          "need_slo_over_pool_runs_excl_overload": [[s0, e0] for s0, e0 in _runs([need_slo[k] > pool and not in_ov[k]
                                                                                  for k in range(ns)])]}
    slack = float(a.get("r2_slack_s", 3.0))   # one 3 s ramp: shapes switching at the same edge overlap for <= one ramp
    if r7cfg is None:
        r7["verdict"] = "n/a"
    else:
        ok = True
        if r7cfg.get("contention"):
            ok = r7["need_slo_over_pool_s"] > 0 and r7["need_ts_over_pool_s"] > 0
        else:
            ok = (all(e0 - s0 <= slack for s0, e0 in r7["need_slo_over_pool_runs_excl_overload"])
                  and all(e0 - s0 <= slack for s0, e0 in out["shape"]["need_shape_design_over_pool_runs"]))
        if r7cfg.get("blind_band_free"):
            ok = ok and all(v["plateau_s"] == 0 for v in blind.values())
        r7["verdict"] = "met" if ok else "FAIL"
    out["R7"] = r7["verdict"]
    out["r7"] = r7

    # R8 fluid decision points
    r8cfg = a.get("r8")
    if r8cfg and plans:
        out["r8"] = fluid_r8(r8cfg, plans, cap, design_rho, dur)
        out["R8"] = "met" if all(e["met"] for e in out["r8"]) else "FAIL"
    else:
        out["R8"] = "n/a"
    return out


def _need(cap: Capacity, m: str, rho: float) -> int:
    """Replicas needed to keep a model at or under rho = 1 per replica (floor / max_awake)."""
    mc = cap.models[m]
    return min(mc.max_awake, max(mc.floor, int(math.ceil(rho - 1e-9))))


def _fluid(alloc_at, plans, cap, design_rho, t0: int, t1: int) -> dict:
    """Fluid queues over [t0, t1) for allocation ``alloc_at(k) -> {model: n}`` (see R8)."""
    res = {m: {"arr": 0.0, "viol": 0.0, "u_sum": 0.0, "u_n": 0, "drained_at": None} for m in plans}
    back = {m: 0.0 for m in plans}
    for k in range(t0, t1):
        al = alloc_at(k)
        for m, plan in plans.items():
            f = cap.models[m].rho_factor
            n = al[m]
            c = n / f
            a = design_rho[m][k]
            back[m] = max(0.0, back[m] + a - c)
            u = a / n
            w = back[m] / c
            v = 1.0 if w > FLUID_WAIT_S else min(1.0, max(0.0, (u - 1.0) / (1.0 / f - 1.0)))
            lam = plan.lam(k + 0.5)
            r = res[m]
            r["arr"] += lam; r["viol"] += lam * v; r["u_sum"] += u; r["u_n"] += 1
            if back[m] > 1e-9:
                r["drained_at"] = None
            elif r["drained_at"] is None:
                r["drained_at"] = k
    return {"V_req": sum(r["viol"] for r in res.values()),
            "arrivals": sum(r["arr"] for r in res.values()),
            "worst_model_frac": max(r["viol"] / r["arr"] for r in res.values() if r["arr"] > 0),
            "per_model_frac": {m: (r["viol"] / r["arr"] if r["arr"] else 0.0) for m, r in res.items()},
            "backlog_end": dict(back),
            "drained_at_s": {m: r["drained_at"] for m, r in res.items()}}


def fluid_r8(cfg: dict, plans: dict, cap: Capacity, design_rho: dict, dur: float) -> list[dict]:
    lag = int(cfg.get("lag_s", 30))
    base = cfg["base"]
    out = []
    for ev in cfg["events"]:
        s, e = (int(x) for x in ev["window_s"])
        until = int(ev.get("until_s", min(dur, e + 2 * lag)))
        rows = {}
        for lg in (lag, 2 * lag):
            for name, alloc in ev["options"].items():
                def alloc_at(k, alloc=alloc, lg=lg):
                    return alloc if s + lg <= k < e + lg else base
                fl = _fluid(alloc_at, plans, cap, design_rho, s, until)
                u_static = max(sum(design_rho[m][k] / alloc[m] for k in range(s + lg, e)) / max(1, e - s - lg)
                               for m in plans)
                fl["maxmin_rho_static"] = u_static
                rows.setdefault(name, {})[f"lag{lg}"] = fl
        def rank(key, lg):
            return sorted(rows, key=lambda n: rows[n][f"lag{lg}"][key])
        best_v, best_mm = rank("V_req", lag)[0], rank("worst_model_frac", lag)[0]
        out.append({"name": ev["name"], "surge_model": ev.get("surge_model"), "window_s": [s, e], "until_s": until,
                    "lag_s": lag, "expect": ev["expect"], "best_V_req": best_v, "best_maxmin": best_mm,
                    "best_V_req_2lag": rank("V_req", 2 * lag)[0], "best_maxmin_2lag": rank("worst_model_frac", 2 * lag)[0],
                    "met": best_v == ev["expect"] and best_mm == ev["expect"], "options": rows})
    return out


SHORT = {"dsllama-8b": "8b", "dsqwen-7b": "7b", "dsqwen-14b": "14b"}
ORDER = ("dsllama-8b", "dsqwen-7b", "dsqwen-14b")


def _f(x, d=2):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def markdown(results: list[dict]) -> str:
    lines = ["| trace | seed | dur s | req | mean ρ 8b/7b/14b | peak ρ 8b/7b/14b (design) | G judged (basis) | G realised 30 s max | static Σ | R1a / R1b | R2 (min hot s) | R3 (CV ratio) | R5 (hot ρ) | in-flight/replica lag 30 s (static) | max out / cap | W>150 s | max in ≤ cap | R7 (need SLO/TS s>8; max) | shape: need design max / realised 30 s max (add) / G 30 s max (add) | R8 |",
             "|---|---|---:|---:|---|---|---|---:|---|---|---|---|---|---|---|---:|---|---|---|---|"]
    for r in results:
        ms = [m for m in ORDER if m in r["models"]]
        g = lambda k, d=2: "/".join(_f(r["models"][m][k], d) for m in ms)
        hot = "/".join(_f(r["models"][m]["hot_run_min_s"], 0) for m in ms)
        cvr = "/".join(_f(r["models"][m]["cv_ratio"]) for m in ms)
        lvl = "/".join(_f(r["models"][m]["hot_level_design"]) for m in ms)
        inf = "/".join(f"{r['models'][m]['inflight_per_replica_lag_max']:.0f}" for m in ms)
        inf_s = "/".join(f"{r['models'][m]['inflight_per_replica_static_max']:.0f}" for m in ms)
        wt = max(r["models"][m]["over_route_timeout_frac"] for m in ms)
        basis = "design" if r["R1_basis"] == "design" else "30 s"
        gx = f"{r['G_judged']:.2f} ({basis} {r['G_judged_stat']})"
        if r["overload_windows_s"]:
            gx += f"; overload {r['G_max']:.2f}"
        st = f"{r['static_excl_overload']}" + (f" → {r['static']}" if r["static"] != r["static_excl_overload"] else "")
        flag = "" if r["inflight_ok"] else f" (>{r['inflight_limit']})"
        oc = "/".join(f"{r['models'][m]['out_max_seen']}≤{r['models'][m]['out_cap']}" for m in ms) + ("" if r["out_cap_ok"] else " **FAIL**")
        lines.append(f"| {r['trace']} | {r['seed']} | {r['duration_s']:.0f} | {r['requests']} | {g('mean_rho')} | {g('peak_rho')} | "
                     f"{gx} | {r['G_realised_30s_max']:.2f} | {st} | {r['R1a']} / {r['R1b']} | {r['R2']} ({hot}) | {r['R3']} ({cvr}) | "
                     f"{r['R5']} ({lvl}) | {inf} ({inf_s}){flag} | {oc} | {wt * 100:.2f}% | "
                     f"{'/'.join(str(r['models'][m]['in_max_seen']) for m in ms)} ≤ {r['in_max']}{'' if r['in_cap_ok'] else ' **FAIL**'} | "
                     f"{r['R7']} ({r['r7']['need_slo_over_pool_s']}/{r['r7']['need_ts_over_pool_s']}; {r['r7']['need_slo_max']}) | "
                     f"{r['shape']['need_shape_design_max']} / {r['shape']['need_shape_30s_max']} ({r['shape']['need_add_30s_max']}) / {r['shape']['G_shape_30s_max']:.2f} ({r['shape']['G_add_30s_max']:.2f}) | "
                     f"{r['R8']} |")
    r8 = [(r["trace"], r["seed"], e) for r in results for e in r.get("r8") or []]
    if r8:
        lines += ["", "R8 (fluid queue, lag {}): violated requests / worst model's violated fraction per option".format(r8[0][2]["lag_s"]), "",
                  "| trace | seed | event | expect | options: V_req / worst frac / static max-min rho | best (V_req, max-min) | best at 2x lag | met |",
                  "|---|---|---|---|---|---|---|---|"]
        for t, sd, e in r8:
            key = f"lag{e['lag_s']}"
            opts = "; ".join(f"{n} {v[key]['V_req']:.0f} / {v[key]['worst_model_frac']:.2f} / {v[key]['maxmin_rho_static']:.2f}"
                             for n, v in e["options"].items())
            lines.append(f"| {t} | {sd} | {e['name']} | {e['expect']} | {opts} | {e['best_V_req']}, {e['best_maxmin']} | "
                         f"{e['best_V_req_2lag']}, {e['best_maxmin_2lag']} | {'met' if e['met'] else '**FAIL**'} |")
    return "\n".join(lines) + "\n"
