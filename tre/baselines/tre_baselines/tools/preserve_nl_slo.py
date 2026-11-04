"""PreServe's NL SLO per model from raw data (the input ``nl_slo`` of ``preserve_mu alg1``; user decisions 2026-11-04).

    python3 -m tre_baselines.tools.preserve_nl_slo --config tre_baselines/tools/preserve_nl_slo.yaml \\
        --data-root R --registry <tre>/deploy/registry.yaml --out DIR

The SLO is the registry's ``idle_multiple`` definition (``tre_common.slo``, whatever ``slo_mode`` is in force:
this tool calibrates for that definition), every parameter from the model's ``models[].slo`` block:

* TTFT_slo(L) = max(idle_multiple.ttft_floor_ms, idle_multiple.ttft_k * (c + b L)), c / b = ``ttft_idle_c_ms`` /
  ``ttft_idle_b_ms_per_token`` (the raw idle TTFT fit, ``ttft_idle_fit.json``, is reported next to them, not used);
* TPOT_slo = idle_multiple.tpot_k * idle_multiple.tpot_idle_ms. The tool recomputes TPOT_idle from raw data, the
  median ``tpot_ms`` of the ``outcome == "ok"`` records of the idle capture (``idle_raw``: one request in flight,
  16 output tokens), and refuses a registry whose ``tpot_idle_ms`` differs (the registry value is this median);
* PreServe judges one normalized latency NL = e2e / output tokens per request. A request on both limits at the
  model's mix shape (L_mix x out_mix, the ``mix`` shape of the isolated run's ``isolated-meta.json``) has
  NL_SLO = (TTFT_slo(L_mix) + (out_mix - 1) TPOT_slo) / out_mix; ``nl_slo_ms`` is that value rounded to
  ``round_decimals`` (the threshold ``preserve_mu alg1`` judges with).

Writes ``DIR/nl_slo.yaml``: ``nl_slo_ms`` per model, every intermediate value per model (also the isolated NL of
the mix shape and the ratio), and the provenance (tool sha, config / registry / every input with sha256; data
paths relative to ``--data-root``). No time stamp, host or output path: the same inputs and commit give the same
bytes. Refuses an existing ``nl_slo.yaml``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from tre_common import slo as slo_def

CONFIG_KEYS = ("round_decimals", "inputs", "models")
INPUT_KEYS = ("idle_raw", "idle_fit", "isolated_dir", "mix_shape")
#: Relative tolerance of the registry ``tpot_idle_ms`` against the median recomputed from raw.
TPOT_IDLE_RTOL = 1e-9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    miss = [k for k in CONFIG_KEYS if k not in cfg] + [f"inputs.{k}" for k in INPUT_KEYS if k not in (cfg.get("inputs") or {})]
    if miss:
        raise SystemExit(f"preserve_nl_slo: {path}: missing {miss}")
    if not isinstance(cfg["models"], list) or not all(isinstance(m, str) for m in cfg["models"]):
        raise SystemExit(f"preserve_nl_slo: {path}: models must be a list of model names (the SLO parameters are "
                         "the registry's models[].slo.idle_multiple)")
    return cfg


def registry_thresholds(registry_path: str, model: str) -> "slo_def.SloThresholds":
    """The ``idle_multiple`` thresholds of ``models[name == model].slo`` (``tre_common.slo``; plain YAML read,
    so a registry snapshot without the cluster sections works too)."""
    with open(registry_path, "r", encoding="utf-8") as fh:
        reg = yaml.safe_load(fh) or {}
    for m in reg.get("models") or []:
        if m.get("name") == model:
            try:
                return slo_def.slo_thresholds(m.get("slo") or {}, slo_def.SLO_MODE_IDLE_MULTIPLE)
            except slo_def.SloConfigError as exc:
                raise SystemExit(f"preserve_nl_slo: {registry_path}: {model}: {exc}") from None
    raise SystemExit(f"preserve_nl_slo: {registry_path}: no model {model}")


def tpot_idle(records: Sequence[Mapping[str, Any]]) -> tuple:
    """(median tpot_ms of the ok records, n used, n left out)."""
    vals = [float(r["tpot_ms"]) for r in records if r.get("outcome") == "ok" and r.get("tpot_ms") is not None]
    if not vals:
        raise SystemExit("preserve_nl_slo: no ok idle record with tpot_ms")
    return statistics.median(vals), len(vals), len(records) - len(vals)


def isolated_nl_ms(records: Sequence[Mapping[str, Any]], shape: str) -> Optional[float]:
    """Mean isolated NL (ms) of the clean measured runs of ``shape`` (``velocity_ladder.isolated_nl``)."""
    from tre_baselines.tools.velocity_ladder import isolated_nl

    by, _bad = isolated_nl(list(records))
    return by[shape]["mean"] * 1000.0 if shape in by else None


def nl_slo_ms(L: float, out: float, thresholds: "slo_def.SloThresholds") -> dict:
    """The NL equivalent of the per-request SLO at one shape (L prompt, out output tokens)."""
    ttft = thresholds.ttft_ms(L)
    tpot = thresholds.tpot_ms
    return {"ttft_slo_ms": ttft, "tpot_slo_ms": tpot, "nl_slo_ms": (ttft + (out - 1) * tpot) / out}


def _read_jsonl(path: Path) -> list:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compute(cfg: Mapping, data_root: Path, registry: str) -> dict:
    inp = cfg["inputs"]
    nd = int(cfg["round_decimals"])
    files: list = []

    def use(rel: str) -> Path:
        p = data_root / rel
        if not p.is_file():
            raise SystemExit(f"preserve_nl_slo: missing {p}")
        if rel not in [f["path"] for f in files]:
            files.append({"path": rel, "sha256": sha256(p)})
        return p

    fit = json.loads(use(inp["idle_fit"]).read_text(encoding="utf-8"))
    meta_rel = f"{inp['isolated_dir']}/isolated-meta.json"
    meta = json.loads(use(meta_rel).read_text(encoding="utf-8"))
    models: dict = {}
    for model in cfg["models"]:
        th = registry_thresholds(registry, model)
        c, b = th.c_ms, th.b_ms_per_token
        t_idle, n_ok, n_out = tpot_idle(_read_jsonl(use(inp["idle_raw"].format(model=model))))
        if abs(t_idle - float(th.tpot_idle_ms)) > TPOT_IDLE_RTOL * t_idle:
            raise SystemExit(f"preserve_nl_slo: {registry}: {model}: slo.idle_multiple.tpot_idle_ms {th.tpot_idle_ms} "
                             f"!= the median of the idle capture {t_idle!r}; publish that median")
        shapes = {s[0]: (int(s[1]), int(s[2])) for s in (meta.get("shapes") or {}).get(model, [])}
        if inp["mix_shape"] not in shapes:
            raise SystemExit(f"preserve_nl_slo: {meta_rel}: no shape {inp['mix_shape']!r} for {model}")
        L, out = shapes[inp["mix_shape"]]
        v = nl_slo_ms(L, out, th)
        iso = isolated_nl_ms(_read_jsonl(use(f"{inp['isolated_dir']}/{model}.jsonl")), inp["mix_shape"])
        fm = (fit.get("models") or {}).get(model) or {}
        models[model] = {
            "k_ttft": th.ttft_k, "k_tpot": th.tpot_k, "ttft_floor_ms": th.ttft_floor_ms,
            "ttft_idle_c_ms": c, "ttft_idle_b_ms_per_token": b,
            "ttft_idle_fit": {"c_ms": fm.get("c_ms"), "b_ms_per_token": fm.get("b_ms_per_token"), "n": fm.get("n")},
            "tpot_idle_ms": t_idle, "tpot_idle_n": n_ok, "tpot_idle_left_out": n_out,
            "mix": {"shape": inp["mix_shape"], "prompt_tokens": L, "output_tokens": out},
            "ttft_slo_mix_ms": v["ttft_slo_ms"], "tpot_slo_ms": v["tpot_slo_ms"],
            "nl_slo_ms_exact": v["nl_slo_ms"], "nl_slo_ms": round(v["nl_slo_ms"], nd),
            "isolated_nl_mix_ms": iso, "nl_slo_over_isolated_nl": None if not iso else v["nl_slo_ms"] / iso}
        if fm.get("c_ms") is not None and (abs(c - fm["c_ms"]) > 0.02 * fm["c_ms"]
                                           or abs(b - fm["b_ms_per_token"]) > 0.02 * fm["b_ms_per_token"]):
            print(f"# WARN {model}: registry c/b {c}/{b} differ > 2 % from the fit "
                  f"{fm['c_ms']:.3f}/{fm['b_ms_per_token']:.5f}", file=sys.stderr)
    return {"nl_slo_ms": {m: v["nl_slo_ms"] for m, v in models.items()}, "models": models, "files": files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="preserve_nl_slo.yaml")
    ap.add_argument("--data-root", required=True, help="root the config's input paths are relative to")
    ap.add_argument("--registry", required=True, help="registry with models[].slo ttft_idle_c_ms / _b_ms_per_token and "
                                                      "idle_multiple (ttft_k, tpot_k, ttft_floor_ms, tpot_idle_ms)")
    ap.add_argument("--out", required=True, help="output dir (must not hold an nl_slo.yaml)")
    a = ap.parse_args(argv)
    from tre_baselines.tools.preserve_mu import _git_sha

    cfg = load_config(a.config)
    out = Path(a.out) / "nl_slo.yaml"
    if out.exists():
        raise SystemExit(f"preserve_nl_slo: {out} exists; use a new --out")
    res = compute(cfg, Path(a.data_root), a.registry)
    doc = {"nl_slo_ms": res["nl_slo_ms"],
           "rule": {"ttft_slo_ms": "max(ttft_floor_ms, k_ttft * (c + b * L))", "tpot_slo_ms": "k_tpot * tpot_idle_ms",
                    "tpot_idle_ms": "median tpot_ms of the outcome == ok records of the idle capture",
                    "nl_slo_ms": f"(ttft_slo(L_mix) + (out_mix - 1) * tpot_slo) / out_mix, rounded to "
                                 f"{int(cfg['round_decimals'])} decimals"},
           "models": res["models"],
           "provenance": {"tool": "tre_baselines.tools.preserve_nl_slo", "tool_sha": _git_sha(),
                          "config": {"path": a.config, "sha256": sha256(Path(a.config))},
                          "registry": {"path": a.registry, "sha256": sha256(Path(a.registry))},
                          "inputs": res["files"]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    out.write_text("# preserve_nl_slo: PreServe NL SLO (ms per output token) per model; input of preserve_mu alg1 "
                   "--nl-slo\n" + text, encoding="utf-8")
    sys.stdout.write(yaml.safe_dump({"nl_slo_ms": res["nl_slo_ms"]}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
