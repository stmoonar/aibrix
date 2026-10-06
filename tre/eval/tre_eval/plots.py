"""Figures (matplotlib, Agg). Every function writes ``<base>.png`` and ``<base>.pdf`` and returns
the PNG path, or None when its inputs are absent. One quantity per axis (no dual axes)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import style as S  # noqa: E402

S.apply_rc()


def _save(fig, base: str) -> str:
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".png", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return base + ".png"


def _legend(ax, **kw):
    if kw.get("handles") or ax.get_legend_handles_labels()[0]:
        ax.legend(**kw)


def _ecdf(v):
    v = np.sort(np.asarray([x for x in v if x is not None and x == x], dtype=float))
    if not len(v):
        return None, None
    return v, np.arange(1, len(v) + 1) / len(v)


METRICS = (("ttft_ms", "TTFT (ms)", True, 1.0), ("tpot_ms", "TPOT (ms)", False, 1.0), ("e2e_ms", "E2E (s)", False, 1e-3))


def _onset_lines(ax, onsets, models, alpha=0.5):
    for o in onsets or []:
        ax.axvline(o["t_on"], color=S.model_color(models, o["model"]), lw=0.8, ls=(0, (2, 2)), alpha=alpha, zorder=0)


# ---------------------------------------------------------------- per arm
def arm_cdfs(res: dict, base: str) -> str | None:
    recs = [r for r in res["records"] if r.get("ok") and "viol" in r]
    if not recs:
        return None
    models = res["models"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, (k, lab, logx, sc) in zip(axs, METRICS):
        for m in models + ["ALL"]:
            x, y = _ecdf([r[k] * sc for r in recs if r[k] is not None and (m == "ALL" or r["model"] == m)])
            if x is None:
                continue
            ax.plot(x, y, color=S.model_color(models, m), ls="--" if m == "ALL" else "-", lw=1.2 if m == "ALL" else 1.5,
                    label=m)
        if logx:
            ax.set_xscale("log")
        if k == "tpot_ms":
            ax.axvline(75, color=S.INK2, lw=0.8, ls=":")
        if k == "e2e_ms":
            ax.axvline(149, color=S.INK2, lw=0.8, ls=":")
        ax.set_xlabel(lab)
        ax.set_ylim(0, 1.01)
    axs[0].set_ylabel("CDF (successful, scored requests)")
    _legend(axs[0], loc="lower right")
    fig.suptitle(f"{res['label_full']} - latency CDFs", x=0.01, ha="left", fontsize=9.5)
    return _save(fig, base)


def arm_lengths(res: dict, base: str) -> str | None:
    recs = [r for r in res["records"] if "viol" in r]
    if not recs:
        return None
    models = res["models"]
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.6))
    for ax, k, lab in ((axs[0], "in", "input tokens"), (axs[1], "out", "output tokens")):
        for m in models:
            x, y = _ecdf([r[k] for r in recs if r["model"] == m])
            if x is not None:
                ax.step(x, y, where="post", color=S.model_color(models, m), label=m)
        ax.set_xlabel(lab)
    axs[0].set_ylabel("CDF")
    _legend(axs[0], loc="lower right")
    return _save(fig, base)


def arm_timeseries(res: dict, base: str) -> str | None:
    ts = res["ts"]
    models = res["models"]
    t = ts["_t"]["t"]
    rows = ["supply", "rps", "z", "kv", "queue", "viol"]
    titles = {"supply": "replicas · ρ", "rps": "req/s (5 s)", "z": "Z_m = TSS/θ_m", "kv": "KV usage",
              "queue": "requests", "viol": "V_req % (10 s)"}
    notes = {"supply": "ρ = offered load in replica-equivalents (shaded); awake / routable / target replicas; ▲▼ decisions",
             "rps": "offered (grey) vs completed", "z": "gateway TSS signal; horizontal lines τ_crit / τ_low / τ_high",
             "kv": "vLLM KV cache usage of routable pods (mean, max)", "queue": "vLLM running / waiting, summed over routable pods",
             "viol": "share of requests (by send time) that violate the SLO"}
    fig, axs = plt.subplots(len(rows), len(models), figsize=(4.1 * len(models), 1.75 * len(rows)),
                            sharex=True, squeeze=False)
    dec = res.get("decisions") or []
    for c, m in enumerate(models):
        col = S.model_color(models, m)
        f = ts[m]
        for r, key in enumerate(rows):
            ax = axs[r][c]
            _onset_lines(ax, [o for o in res.get("onsets", []) if o["model"] == m], models)
            if key == "supply":
                if np.isfinite(f["rho"]).any():
                    rho = _smooth(f["rho"], 10)
                    ax.fill_between(t, 0, rho, color=col, alpha=0.18, lw=0, label="ρ offered (10 s)")
                ax.step(t, f["awake"], where="post", color=col, lw=1.6, label="awake")
                ax.step(t, f["routable"], where="post", color=S.INK, lw=1.0, ls="--", label="routable")
                if np.isfinite(f["target"]).any():
                    ax.step(t, f["target"], where="post", color=S.INK2, lw=1.0, ls=":", label="target")
                top = np.nanmax(np.concatenate([f["awake"], f["target"], _smooth(f["rho"], 10), [1]]))
                ups = [d["ts_rel"] for d in dec if d["model"] == m and d["dir"] > 0]
                dns = [d["ts_rel"] for d in dec if d["model"] == m and d["dir"] < 0]
                if ups:
                    ax.plot(ups, [top + 0.4] * len(ups), ls="none", marker="^", ms=5, color=col, label="decision up")
                if dns:
                    ax.plot(dns, [top + 0.4] * len(dns), ls="none", marker="v", ms=5, mfc="white", color=col,
                            label="decision down")
                ax.set_ylim(0, top + 1)

            elif key == "rps":
                ax.plot(t, _smooth(f["offered_rps"], 5), color=S.INK2, lw=0.9, label="offered")
                ax.plot(t, _smooth(f["completed_rps"], 5), color=col, lw=1.4, label="completed")
                if c == 0:
                    _legend(ax, loc="upper left", fontsize=6.5)
            elif key == "z":
                z = f["z"] if np.isfinite(f["z"]).any() else f["ctrl_z"]
                if np.isfinite(z).any():
                    ok = np.isfinite(z)
                    ax.plot(t[ok], z[ok], color=col, marker="o", ms=2.5, lw=1.0)
                    trs = res["trs"].get(m) or {}
                    for k, ls in (("tau_crit", "-"), ("tau_low", "--"), ("tau_high", ":")):
                        if trs.get(k) is not None:
                            ax.axhline(trs[k], color=S.INK2, lw=0.8, ls=ls)
                            if c == len(models) - 1:
                                ax.text(1.0, trs[k], " " + k, transform=ax.get_yaxis_transform(), va="center",
                                        fontsize=6.5, color=S.INK2)
                    hi = np.nanpercentile(z[ok], 99)
                    ax.set_ylim(0, max(hi * 1.1, (trs.get("tau_high") or 1) * 1.15))
                else:
                    ax.text(0.5, 0.5, "no TSS signal logged for this arm", transform=ax.transAxes, ha="center",
                            va="center", color=S.INK2, fontsize=7.5)
            elif key == "kv":
                kv = f["kv_mean"]
                if np.isfinite(kv).any():
                    ok = np.isfinite(kv)
                    ax.plot(t[ok], kv[ok], color=col, lw=1.3, label="mean")
                    ax.plot(t[ok], f["kv_max"][ok], color=col, lw=0.7, alpha=0.6, label="max")
                elif np.isfinite(f["ctrl_kv"]).any():
                    ok = np.isfinite(f["ctrl_kv"])
                    ax.plot(t[ok], f["ctrl_kv"][ok], color=col, lw=1.3, label="controller sample")
                ax.set_ylim(0, 1.05)
                if c == 0:
                    _legend(ax, loc="upper left", fontsize=6.5)
            elif key == "queue":
                okq = np.isfinite(f["running"])
                if okq.any():
                    ax.plot(t[okq], f["running"][okq], color=col, lw=1.3, label="running")
                    ax.plot(t[okq], f["waiting"][okq], color=S.INK, lw=1.0, ls="--", label="waiting")
                if c == 0:
                    _legend(ax, loc="upper left", fontsize=6.5)
            elif key == "viol":
                v = f["viol_frac_10s"] * 100
                ax.fill_between(t, 0, np.nan_to_num(v), step="post", color=col, alpha=0.5, lw=0)
                ax.set_ylim(0, 100)
            if c == 0:
                ax.set_ylabel(titles[key], fontsize=8)
                ax.text(0.0, 1.02, notes[key], transform=ax.transAxes, fontsize=6.5, color=S.INK2, va="bottom")
            if r == 0:
                ax.set_title(m, color=S.INK, pad=14)
            if r == len(rows) - 1:
                ax.set_xlabel("t since load start (s)")
    seen: dict = {}
    for ax in axs[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    if seen:
        fig.legend(list(seen.values()), list(seen), loc="upper right", ncol=len(seen), fontsize=7, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.suptitle(f"{res['label_full']} - time series (dotted verticals = hot-phase onsets)", x=0.01, y=0.995, ha="left",
                 fontsize=9.5)
    return _save(fig, base)


def _smooth(x, w):
    x = np.asarray(x, dtype=float)
    if not len(x) or w <= 1:
        return x
    ok = np.isfinite(x).astype(float)
    k = np.ones(w)
    num = np.convolve(np.nan_to_num(x), k, mode="same")
    den = np.convolve(ok, k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def arm_gpu_map(res: dict, base: str) -> str | None:
    iv = res.get("gpu_intervals") or {}
    if not iv:
        return None
    models = res["models"]
    gpus = list(iv)
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(gpus) + 1.2))
    for i, g in enumerate(gpus):
        for x in iv[g]:
            col = S.model_color(models, x["model"])
            ax.broken_barh([(x["t0"], max(x["t1"] - x["t0"], 0.5))], (i - 0.38, 0.76), facecolors=col, edgecolor="white",
                           lw=1.0)
            for h0, h1 in x["hidden"]:
                ax.broken_barh([(h0, max(h1 - h0, 0.5))], (i - 0.38, 0.76), facecolors="none", edgecolor=S.INK,
                               hatch="////", lw=0)
    ymap = {g: i for i, g in enumerate(gpus)}
    for e in res.get("events") or []:
        if e["source"] != "sm" or e["t"] is None or not e["target"]:
            continue
        from .load import parse_binding
        _, node, gl = parse_binding(e["target"])
        for gg in gl:
            y = ymap.get(f"{node}/{gg}")
            if y is None:
                continue
            if e["kind"] == "wake_done":
                ax.plot(e["t"], y, marker="|", ms=9, color=S.INK, mew=1.2)
            elif e["kind"] == "sleep":
                ax.plot(e["t"], y, marker="x", ms=5, color=S.INK, mew=1.0)
    _onset_lines(ax, res.get("onsets"), models, alpha=0.7)
    ax.set_yticks(range(len(gpus)))
    ax.set_yticklabels(gpus, fontsize=7)
    ax.set_ylim(-0.6, len(gpus) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("t since load start (s)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color=S.model_color(models, m), label=m) for m in models]
    handles += [Patch(facecolor="white", edgecolor=S.INK, hatch="////", label="hidden (not routable)"),
                Line2D([], [], marker="|", ls="none", color=S.INK, label="wake done (SM)"),
                Line2D([], [], marker="x", ls="none", color=S.INK, label="sleep (SM)")]
    _legend(ax, handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    ax.set_title(f"{res['label_full']} - per-GPU occupancy (awake model per GPU, SM layout 1 s)", loc="left")
    return _save(fig, base)


def arm_switch(res: dict, base: str) -> str | None:
    sw = res.get("switch") or {}
    wu = [w["wake_up_s"] for w in sw.get("wakes", []) if w.get("wake_up_s") is not None]
    wt = [w["total_s"] for w in sw.get("wakes", []) if w.get("total_s") is not None]
    sl = [x for x in (res.get("interruption") or {}).get("sidecar_sleep_durations_s", []) if x is not None]
    if not (wu or sl):
        return None
    fig, ax = plt.subplots(figsize=(5, 2.4))
    data, labels = [], []
    for v, lab in ((wu, "wake_up (engine)"), (wt, "wake total (SM)"), (sl, "sleep (sidecar)")):
        if v:
            data.append(v)
            labels.append(f"{lab}\nn={len(v)}")
    ax.boxplot(data, vert=False, labels=labels, widths=0.5, medianprops={"color": S.INK})
    for i, v in enumerate(data, 1):
        ax.plot(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), ls="none", marker="o",
                ms=3, color="#2a78d6", alpha=0.6)
    ax.set_xlabel("seconds")
    ax.set_title(f"{res['label_full']} - hot-switch durations", loc="left")
    return _save(fig, base)


# ---------------------------------------------------------------- cross arm
def cross_cdfs(results: list[dict], styles, base: str) -> str | None:
    models = results[0]["models"]
    cols = models + ["ALL"]
    fig, axs = plt.subplots(3, len(cols), figsize=(3.2 * len(cols), 7.2), squeeze=False)
    for r, (k, lab, logx, sc) in enumerate(METRICS):
        for c, m in enumerate(cols):
            ax = axs[r][c]
            for res in results:
                x, y = _ecdf([q[k] * sc for q in res["records"] if q.get("ok") and "viol" in q and q[k] is not None
                              and (m == "ALL" or q["model"] == m)])
                if x is None:
                    continue
                st = styles[res["name"]]
                ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.3, label=res["label_full"])
            if logx:
                ax.set_xscale("log")
            if r == 0:
                ax.set_title(m)
            if c == 0:
                ax.set_ylabel(f"CDF - {lab}")
            ax.set_xlabel(lab)
            ax.set_ylim(0, 1.01)
    _legend(axs[0][0], loc="lower right", fontsize=6.5)
    fig.tight_layout()
    return _save(fig, base)


def cross_bars(results: list[dict], styles, base: str) -> str | None:
    models = results[0]["models"] + ["ALL"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.2))
    w = 0.8 / len(results)
    for ax, key, ci_key, lab in ((axs[0], "V_req_pct", "ci_vreq", "V_req (%), 95 % block-bootstrap CI"),
                                 (axs[1], "ttft_p95_ms", "ci_ttft_p95", "P95 TTFT (ms), 95 % CI")):
        for i, res in enumerate(results):
            st = styles[res["name"]]
            xs = np.arange(len(models)) + (i - (len(results) - 1) / 2) * w
            ys = [(res["summary"].get(m) or {}).get(key) for m in models]
            ys = [np.nan if y is None else y for y in ys]
            lo = [(res["ci"].get(m) or {}).get(ci_key, (None, None))[0] for m in models]
            hi = [(res["ci"].get(m) or {}).get(ci_key, (None, None))[1] for m in models]
            err = np.array([[max(0, y - (l if l is not None else y)) for y, l in zip(ys, lo)],
                            [max(0, (h if h is not None else y) - y) for y, h in zip(ys, hi)]])
            ax.bar(xs, ys, width=w * 0.9, color=st["color"], alpha=0.85, edgecolor="white", lw=1,
                   hatch=st["hatch"], label=res["label_full"])
            ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=S.INK, lw=0.8, capsize=2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_ylabel(lab)
        if key == "ttft_p95_ms":
            ax.set_yscale("log")
    _legend(axs[0], fontsize=6.5, ncol=2)
    fig.tight_layout()
    return _save(fig, base)


def cross_scatter(results: list[dict], styles, base: str) -> str | None:
    pts = [(res, (res["gpu"].get("ALL") or {}).get("gpu_s"), res["summary"]["ALL"]["V_req_pct"])
           for res in results if (res.get("gpu") or {}).get("ALL")]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for res, x, y in pts:
        st = styles[res["name"]]
        ax.plot(x, y, marker=st["marker"], ms=8, color=st["color"], ls="none", mec="white", mew=1.5)
        ax.annotate(res["label_full"], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color=S.INK)
    ax.set_xlabel("GPU-seconds (awake GPUs over the load window)")
    ax.set_ylabel("V_req (%) - ALL")
    ax.set_title("cost vs SLO violation (lower-left is better)", loc="left")
    return _save(fig, base)


def cross_onsets(results: list[dict], styles, base: str) -> str | None:
    stages = ("decision", "awake", "routable", "donor")
    have = any(r.get("onset_rows") for r in results)
    if not have:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.4))
    w = 0.8 / len(results)
    for i, res in enumerate(results):
        st = styles[res["name"]]
        for j, s in enumerate(stages):
            v = [o[f"lat_{s}_s"] for o in res.get("onset_rows", []) if o.get(f"lat_{s}_s") is not None
                 and not o.get("at_start")]
            x = j + (i - (len(results) - 1) / 2) * w
            if v:
                ax.boxplot([v], positions=[x], widths=w * 0.8, patch_artist=True, showfliers=False,
                           boxprops={"facecolor": st["color"], "alpha": 0.35, "edgecolor": st["color"]},
                           medianprops={"color": st["color"], "lw": 1.6}, whiskerprops={"color": st["color"]},
                           capprops={"color": st["color"]})
                ax.plot(np.full(len(v), x), v, ls="none", marker=st["marker"], ms=3.5, color=st["color"])
            n_on = sum(1 for o in res.get("onset_rows", []) if not o.get("at_start"))
            if n_on:
                ax.text(x, 1.0, f"{len(v)}/{n_on}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=5.5, color=S.INK2, rotation=90)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([f"onset → {s}" for s in stages])
    ax.set_ylabel("seconds after onset")
    from matplotlib.lines import Line2D
    _legend(ax, handles=[Line2D([], [], color=styles[r["name"]]["color"], marker=styles[r["name"]]["marker"],
                              ls=styles[r["name"]]["ls"], label=r["label_full"]) for r in results], fontsize=6.5,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("hot-onset latency breakdown (k/n above = onsets where the stage happened inside the hot phase)",
                 loc="left", pad=22)
    return _save(fig, base)


def cross_paired(paired: list[dict], styles, ref_label: str, base: str) -> str | None:
    rows = [p for p in paired if p.get("n_pairs")]
    if not rows:
        return None
    models = sorted({p["model"] for p in rows}, key=lambda m: (m == "ALL", m))
    arms = list(dict.fromkeys(p["arm"] for p in rows))
    fig, axs = plt.subplots(1, len(models), figsize=(3.0 * len(models), 0.35 * len(arms) + 1.4), sharey=True,
                            squeeze=False)
    for c, m in enumerate(models):
        ax = axs[0][c]
        for i, a in enumerate(arms):
            p = next((q for q in rows if q["arm"] == a and q["model"] == m), None)
            if not p:
                continue
            st = styles[a]
            lo, hi = p["d_vreq_ci"]
            ax.plot([lo, hi], [i, i], color=st["color"], lw=2)
            ax.plot(p["d_vreq_pp"], i, marker=st["marker"], color=st["color"], ms=6, mec="white")
        ax.axvline(0, color=S.INK2, lw=0.8)
        ax.set_title(m)
        ax.set_xlabel(f"ΔV_req (pp) vs {ref_label}")
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([next(p["arm_label"] for p in rows if p["arm"] == a) for a in arms])
    fig.tight_layout()
    return _save(fig, base)


def cross_replicas(results: list[dict], styles, base: str) -> str | None:
    models = results[0]["models"]
    fig, axs = plt.subplots(len(models), 1, figsize=(11, 2.0 * len(models)), sharex=True, squeeze=False)
    for r, m in enumerate(models):
        ax = axs[r][0]
        rho_done = False
        for res in results:
            f = res["ts"].get(m)
            if f is None:
                continue
            t = res["ts"]["_t"]["t"]
            if not rho_done and np.isfinite(f["rho"]).any():
                ax.fill_between(t, 0, _smooth(f["rho"], 10), color=S.model_color(models, m), alpha=0.18, lw=0,
                                label="ρ offered (10 s)")
                rho_done = True
            st = styles[res["name"]]
            ax.step(t, f["awake"], where="post", color=st["color"], ls=st["ls"], lw=1.3, label=res["label_full"])
        ax.set_ylabel(f"{m}\nawake replicas", fontsize=7.5)
    _legend(axs[0][0], loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    axs[-1][0].set_xlabel("t since load start (s)")
    fig.tight_layout()
    return _save(fig, base)
