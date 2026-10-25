"""Write the measured TokenScale / PreServe values into the policy ConfigMaps (the freeze step).

The shipped ConfigMaps (``deploy/baselines/tre/policy-configmaps.yaml``) carry
``calibration: pending`` until the GPU calibration of 2026-10-26 is done, and the policies refuse
to start on them. This tool renders each policy's whole parameter block from the calibration
outputs, so nothing is copied by hand:

* ``tokenscale --report <buckets-report.yaml> --profile <ladder profile.yaml>``: ``bucket_centers``
  from the report (``tools/tokenscale_buckets``), ``velocity`` = the ladder's peak V per cell
  (``tools/velocity_ladder ladder`` / ``summarize``); a cell without V refuses;
* ``preserve --mu <mu.yaml>``: ``mu`` = variant B of ``tools/velocity_ladder mu``.

``--pending`` renders the block with ``calibration: pending`` and null values (the state before
the calibration). Every input's sha256 goes into the block's header comment. Without
``--write`` the block is printed; with it the ConfigMap file is rewritten in place (only that
policy's ``<policy>.yaml: |`` block changes).

    python3 -m tre_baselines.tools.freeze_params tokenscale --report R --profile P --write
    python3 -m tre_baselines.tools.freeze_params preserve --mu M --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

TRE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CM = TRE_ROOT / "deploy" / "baselines" / "tre" / "policy-configmaps.yaml"
MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
#: PreServe mLSTM weights shipped in the image (relative to tre/baselines), v1 paper_past (2026-10-24)
WEIGHTS = "weights/preserve-mlstm-20261024-paper"
WEIGHT_SETS = ("code", "conv", "pooled")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flow(x: Any) -> str:
    return json.dumps(x, separators=(", ", ": "))


def render_tokenscale(centers: Optional[Mapping[str, Any]], velocity: Optional[Mapping[str, Any]],
                      sources: Mapping[str, str], models=MODELS) -> str:
    frozen = centers is not None and velocity is not None
    L = [f"# {'FROZEN' if frozen else 'PENDING'} (tools/freeze_params tokenscale). Sources: "
         + (", ".join(f"{k} sha256 {v}" for k, v in sources.items()) or "none"),
         f"calibration: {'frozen' if frozen else 'pending'}",
         "bucket_edges:            # paper (TokenScale Table II / DynamoLLM Table IV): in < 256 / < 1024 / else",
         '  "*": {in: [255, 1023], out: [99, 349]}   # (> 8192 is long too), out < 100 / < 350 / >= 350',
         "bucket_centers:          # [in, out] per cell = mean of the campaign's requests in it (v1 D49), "
         "tools/tokenscale_buckets"]
    for m in models:
        L.append(f"  {m}: {_flow(centers[m]) if frozen else 'null'}")
    L.append("velocity:                # (in+out) tok/s per replica at each centre: the PEAK of effective full steps")
    L.append("                         # (waiting slope > 2 % x offered rps, v1 D53; tools/velocity_ladder), no V_P (D51)")
    for m in models:
        if frozen:
            L.append(f"  {m}:")
            L.append(f"    buckets: {_flow(velocity[m])}")
        else:
            L.append(f"  {m}: null")
    L += ["out_len_source: max_tokens  # the true output picks the bucket before the simulated error",
          "default_out_tokens: 256  # not in paper",
          "misbucket_rate: 0.15     # paper: 85 %-accurate length predictor; draw = hash(seed, model, client id)",
          "seed: 0",
          "skip_reissue: true",
          "window_s: 30             # = TRE TSS window (30 s); paper gives no value",
          "max_estimate_frac: 0.05  # not in paper"]
    return "\n".join(L) + "\n"


def render_preserve(mu: Optional[Mapping[str, Any]], sources: Mapping[str, str], weights: Mapping[str, str],
                    models=MODELS) -> str:
    frozen = mu is not None
    L = [f"# {'FROZEN' if frozen else 'PENDING'} (tools/freeze_params preserve). Sources: "
         + (", ".join(f"{k} sha256 {v}" for k, v in sources.items()) or "none"),
         f"calibration: {'frozen' if frozen else 'pending'}",
         "# The trace: replaced per run by the campaign runner (its oracle copy in the trace volume); the",
         "# history before t0 is next to it: pre_t0.json (synthetic, the generator's), else preserve_history.json",
         "# (Real_*, tools/preserve_history real); see tre_baselines.policies.preserve_history.",
         "trace_path: /etc/tre-baselines-traces/case/seed1/traces_tre.effective.json",
         "trace_seed: 0",
         "trace_schedule: poisson",
         "trace_match_parts: 2",
         "window_s: 30             # v1 D18 (paper 600 s; the traces are 12 min)",
         "lead_s: 0",
         "tier1: mlstm             # paper forecaster; v1 paper_past weights (train before the Real_* slices)",
         "seq_len: 25",
         "predictors:              # relative paths: under tre/baselines (in the image /app/tre/baselines)"]
    for name in WEIGHT_SETS:
        L.append(f"  {name}: {{path: {WEIGHTS}/{name}-paper.npz, sha256: {weights[name]}}}")
    L += ["predictor_rules: [[Real_code, code], [Real_conv, conv]]   # any other trace: pooled",
          "mu:                      # tok/s per replica, PreServe Alg.1 on the V ladders: NL SLO = 3 x isolated NL,",
          "                         # every request of a step, variant B (P share in the traces' p5-p95), tools/velocity_ladder mu"]
    for m in models:
        L.append(f"  {m}: {_flow({k: mu[m][k] for k in ('p', 'd', 't')}) if frozen else 'null'}")
    L += ["load_error_mode: paper_tab2   # paper RQ2 'PreServe w/ load errors' (Tab. 2 error model)",
          "load_error_seed: 0       # the same draws in every arm and replay seed (as TokenScale seed: 0)",
          "max_model_len: 131072",
          "map_factor: 1.2",
          "lookahead_iters: 100     # paper l",
          "kv_high: 0.95            # paper",
          "overload_frac: 0.10      # paper",
          "t_f: 0.30                # paper T_f",
          "out_len_is_upper_bound: false   # load errors: D_hat is a prediction (paper refinement)",
          "ext_frac: 0.2            # paper virtual extension",
          "t2_reset_at_window: true # v1 D54 #2",
          "kv_agree_tol: 0.15       # logged only (kv_disagree)",
          "req_ttl_s: 1800"]
    return "\n".join(L) + "\n"


def replace_block(text: str, policy: str, block: str) -> str:
    """``text`` (the ConfigMaps file) with the ``<policy>.yaml: |`` literal block replaced."""
    head = f"  {policy}.yaml: |\n"
    i = text.index(head) + len(head)
    j = text.find("\n---", i)
    j = len(text) if j < 0 else j + 1
    body = "".join("    " + ln + "\n" if ln else "\n" for ln in block.rstrip("\n").split("\n"))
    return text[:i] + body + text[j:]


def weights_sha256() -> dict[str, str]:
    root = TRE_ROOT / "baselines" / WEIGHTS
    return {n: sha256_file(root / f"{n}-paper.npz") for n in WEIGHT_SETS}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="policy", required=True)
    t = sub.add_parser("tokenscale")
    t.add_argument("--report")
    t.add_argument("--profile")
    p = sub.add_parser("preserve")
    p.add_argument("--mu")
    for s in (t, p):
        s.add_argument("--pending", action="store_true")
        s.add_argument("--configmaps", default=str(DEFAULT_CM))
        s.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.policy == "tokenscale":
            if a.pending:
                block = render_tokenscale(None, None, {})
            else:
                if not a.report or not a.profile:
                    raise ValueError("--report and --profile (or --pending)")
                rep = yaml.safe_load(Path(a.report).read_text(encoding="utf-8"))
                prof = yaml.safe_load(Path(a.profile).read_text(encoding="utf-8"))
                centers = {m: rep["bucket_centers"][m] for m in MODELS}
                vel = {m: prof["velocity"][m]["buckets"] for m in MODELS}
                bad = [f"{m}:b{i}{j}" for m in MODELS for i, row in enumerate(vel[m]) for j, v in enumerate(row)
                       if not isinstance(v, (int, float)) or v <= 0]
                if bad:
                    raise ValueError(f"cells without V (not saturated / no full step): {bad}")
                if rep.get("bucket_edges") != {"*": {"in": [255, 1023], "out": [99, 349]}}:
                    raise ValueError(f"report edges {rep.get('bucket_edges')} are not the paper's")
                block = render_tokenscale(centers, vel, {"report": sha256_file(a.report),
                                                         "profile": sha256_file(a.profile)})
        else:
            w = weights_sha256()
            if a.pending:
                block = render_preserve(None, {}, w)
            else:
                if not a.mu:
                    raise ValueError("--mu (or --pending)")
                doc = yaml.safe_load(Path(a.mu).read_text(encoding="utf-8"))
                mu = doc.get("mu_B") or {}
                missing = [m for m in MODELS if not all((mu.get(m) or {}).get(k) for k in ("p", "d", "t"))]
                if missing:
                    raise ValueError(f"mu_B lacks p / d / t for {missing}")
                block = render_preserve(mu, {"mu": sha256_file(a.mu)}, w)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    yaml.safe_load(block)  # the block itself must parse
    if not a.write:
        print(block, end="")
        return 0
    path = Path(a.configmaps)
    new = replace_block(path.read_text(encoding="utf-8"), a.policy, block)
    list(yaml.safe_load_all(new))
    path.write_text(new, encoding="utf-8")
    print(f"wrote the {a.policy} block of {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
