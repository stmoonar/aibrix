"""Write the measured TokenScale / PreServe values into the policy ConfigMaps (the freeze step).

The shipped ConfigMaps (``deploy/baselines/tre/policy-configmaps.yaml``) carry
``calibration: pending`` until the GPU calibration of 2026-10-26 is done, and the policies refuse
to start on them. This tool renders each policy's whole parameter block from the calibration
outputs, so nothing is copied by hand:

* ``tokenscale --report <buckets-report.yaml> --profile <ladder profile.yaml>``: ``bucket_centers``
  from the report (``tools/tokenscale_buckets``), ``velocity`` = the ladder's peak V per cell
  (``tools/velocity_ladder ladder`` / ``summarize``); a cell without V refuses;
* ``preserve --mu <mu.yaml>``: ``mu`` = the ``mu`` block of ``tools/preserve_mu alg1`` (2026-11-04, paper
  Alg. 1; the file's first comment line names the tool commit) or, for older files, ``mu_B`` of
  ``tools/velocity_ladder mu``. ``--weights <dir>`` (relative to tre/baselines; default the 2026-11-04 per-trace
  weights) and ``--predictor-variant v2|v1`` pick the Tier-1 predictors: a dir with ``v1/`` / ``v2/`` files
  (``preserve_train/trace_train.py``) renders one set per variant and trace (``v2``: one file per model, every
  trace; ``v1-<trace>``: the trace's own files) with ``predictor_normalization`` and ``history_variant`` from its
  manifest, and ``predictor_rules`` mapping every trace to the chosen variant (a V1 arm overrides the rules with
  :func:`predictor_rules` ``("v1")``); an older dir (``code`` / ``conv`` / ``pooled``) renders those three sets.

``--pending`` renders the block with ``calibration: pending`` and null values (the state before
the calibration). ``--note TEXT`` (repeatable) adds a ``# NOTE:`` disclosure line under the header
(e.g. a cell measured with another setting). Every input's sha256 goes into the block's header comment. Without
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
#: PreServe mLSTM weights rendered by default (relative to tre/baselines): trained on each trace's own past
#: (2026-11-04, ``preserve_train/trace_train.py``; V1 per trace x model, V2 = the 7 traces concatenated, min-max)
WEIGHTS = "weights/preserve-mlstm-20261104-trace-w10"
#: the v1 paper_past protocol at W = 10 s (2026-11-03; code / conv / pooled, per_sequence_mean), for past runs
LEGACY_WEIGHTS = "weights/preserve-mlstm-20261103-paper-w10"
WEIGHTS_SUFFIX = "-paper-w10"
WEIGHT_SETS = ("code", "conv", "pooled")
PREDICTOR_VARIANTS = ("v2", "v1")


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
          "window_s: 10             # = TRE metrics window (TRE_METRICS_WINDOW_MS 10000, tune F); paper gives no value",
          "max_estimate_frac: 0.05  # not in paper"]
    return "\n".join(L) + "\n"


def _manifest(weights: str) -> dict:
    return json.loads((TRE_ROOT / "baselines" / weights / "manifest.json").read_text(encoding="utf-8"))


def _per_trace(weights: str) -> bool:
    """A ``trace_train.py`` dir (files ``v1/<trace>/<model>`` and ``v2/concat/<model>``)."""
    files = _manifest(weights).get("files") or {}
    return any(str(k).startswith("v2/") for k in files) and any(str(k).startswith("v1/") for k in files)


def trace_names(weights: str = WEIGHTS) -> list[str]:
    """The traces of a per-trace weights dir, in training-config order."""
    out: list[str] = []
    for k in _manifest(weights)["files"]:
        bits = str(k).split("/")
        if bits[0] == "v1" and bits[1] not in out:
            out.append(bits[1])
    return out


def predictor_rules(variant: str, weights: str = WEIGHTS) -> list[list[str]]:
    """``predictor_rules`` that give every trace of a per-trace weights dir its ``variant`` set."""
    if variant not in PREDICTOR_VARIANTS:
        raise ValueError(f"predictor variant must be one of {PREDICTOR_VARIANTS}, got {variant!r}")
    return [[t, "v2" if variant == "v2" else f"v1-{t}"] for t in trace_names(weights)]


def _predictor_lines(weights: str, w: Mapping[str, str], variant: str, models) -> list[str]:
    if not _per_trace(weights):
        L = ["predictor_normalization: per_sequence_mean   # the 2026-11-03 weights (D15)",
             "predictors:              # relative paths: under tre/baselines (in the image /app/tre/baselines)"]
        L += [f"  {name}: {{path: {weights}/{name}{WEIGHTS_SUFFIX}.npz, sha256: {w[name]}}}" for name in WEIGHT_SETS]
        L.append("predictor_rules: [[Real_code, code], [Real_conv, conv]]   # any other trace: pooled")
        return L
    man = _manifest(weights)
    files = man["files"]
    norm = str((man.get("normalization") or {}).get("name") or "")
    hv = (man.get("history") or {}).get("variant")
    L = [f"predictor_normalization: {norm}   # paper Alg. 1 line 3 (min / max of each file's training series)",
         f"history_variant: {hv}   # history before t0 = the end of the trace's training series (weights' REPORT.md)",
         f"predictors:              # under tre/baselines: {weights} (manifest.json); one file per model",
         "  v2:                     # V2: trained on the 7 traces concatenated"
         " (any trace)",
         "    models:"]
    L += [f"      {m}: {{path: {weights}/{files[f'v2/concat/{m}']['file']}, sha256: {w[f'v2/concat/{m}']}}}" for m in models]
    for t in trace_names(weights):
        L += [f"  v1-{t}:   # V1: trained on this trace's own past", "    models:"]
        L += [f"      {m}: {{path: {weights}/{files[f'v1/{t}/{m}']['file']}, sha256: {w[f'v1/{t}/{m}']}}}" for m in models]
    rules = predictor_rules(variant, weights)
    L.append(f"predictor_rules: {_flow(rules)}   # {variant.upper()} for every trace (default); "
             f"a {'V1' if variant == 'v2' else 'V2'} arm overrides the rules (freeze_params.predictor_rules)")
    return L


def render_preserve(mu: Optional[Mapping[str, Any]], sources: Mapping[str, str], weights: Mapping[str, str],
                    models=MODELS, *, weights_dir: str = WEIGHTS, variant: str = "v2", mu_kind: str = "alg1") -> str:
    frozen = mu is not None
    L = [f"# {'FROZEN' if frozen else 'PENDING'} (tools/freeze_params preserve). Sources: "
         + (", ".join(f"{k} {v}" for k, v in sources.items()) or "none"),
         f"calibration: {'frozen' if frozen else 'pending'}",
         "# The trace: replaced per run by the campaign runner (its oracle copy in the trace volume); the",
         "# history before t0 is next to it, named by the window (preserve_history.history_names): at 10 s",
         "# preserve_history_w10.json (tools/preserve_history_azure), with history_variant V",
         "# preserve_history_w10_<V>.json; at 30 s pre_t0.json, else preserve_history.json.",
         "trace_path: /etc/tre-baselines-traces/case/seed1/traces_tre.effective.json",
         "trace_seed: 0",
         "trace_schedule: poisson",
         "trace_match_parts: 2",
         "window_s: 10             # = TRE metrics window (2026-11-03; 30 s before, v1 D18; paper 600 s)",
         "lead_s: 0",
         "tier1: mlstm             # paper forecaster (Tier-1 mLSTM)",
         "tier1_horizon: paper_two_step   # paper Alg. 2: two steps at the start of T_i -> N_{i+1} (2026-11-04)",
         "tier1_d_source: done_tokens     # paper: observed D = done out_tokens, known at completion (2026-11-04)",
         "seq_len: 25"]
    L += _predictor_lines(weights_dir, weights, variant, models)
    if mu_kind == "alg1":
        L += ["mu:                      # tok/s per replica, PreServe Alg. 1 (tools/preserve_mu alg1): 10 s windows of the",
              "                         # TSS / theta training set, clean = every request within NL_SLO, max over clean windows"]
    else:
        L += ["mu:                      # tok/s per replica, PreServe Alg.1 on the V ladders: NL SLO = 3 x isolated NL,",
              "                         # every request of a step, variant B (P share in the traces' p5-p95), tools/velocity_ladder mu"]
    for m in models:
        L.append(f"  {m}: {_flow({k: mu[m][k] for k in ('p', 'd', 't')}) if frozen else 'null'}")
    L += ["load_error_mode: paper_tab2_relative   # paper RQ2 'w/ load errors': Tab. 2 as a relative error (2026-11-04)",
          "underload_pending_gate: false   # paper §4.3.2 scale-in rule only (2026-11-04)",
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


def read_mu(path: str) -> tuple[dict, str, str]:
    """(mu, kind, provenance) of a mu file: ``preserve_mu alg1`` (``mu:``; its first comment line names the
    tool commit and config) or ``velocity_ladder mu`` (``mu_B:``). Refuses a file that lacks p / d / t."""
    text = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text) or {}
    if isinstance(doc.get("mu"), Mapping):
        mu, kind = doc["mu"], "alg1"
    elif isinstance(doc.get("mu_B"), Mapping):
        mu, kind = doc["mu_B"], "velocity_ladder"
    else:
        raise ValueError(f"{path}: no mu (preserve_mu alg1) or mu_B (velocity_ladder mu) block")
    missing = [m for m in MODELS if not all((mu.get(m) or {}).get(k) for k in ("p", "d", "t"))]
    if missing:
        raise ValueError(f"{path}: mu lacks p / d / t for {missing}")
    head = next((ln[1:].strip() for ln in text.splitlines() if ln.startswith("#")), "")
    return {m: mu[m] for m in MODELS}, kind, " ".join(head.split())


def replace_block(text: str, policy: str, block: str) -> str:
    """``text`` (the ConfigMaps file) with the ``<policy>.yaml: |`` literal block replaced."""
    head = f"  {policy}.yaml: |\n"
    i = text.index(head) + len(head)
    j = text.find("\n---", i)
    j = len(text) if j < 0 else j + 1
    body = "".join("    " + ln + "\n" if ln else "\n" for ln in block.rstrip("\n").split("\n"))
    return text[:i] + body + text[j:]


def with_notes(block: str, notes) -> str:
    """``block`` with one ``# NOTE: <text>`` line per note under its header comment."""
    head, sep, rest = block.partition("\n")
    lines = "".join(f"# NOTE: {' '.join(str(n).split())}\n" for n in notes or ())
    return head + sep + lines + rest


def weights_sha256(weights: str = WEIGHTS) -> dict[str, str]:
    """sha256 of every file of a weights dir, computed from the files (checked against its manifest)."""
    root = TRE_ROOT / "baselines" / weights
    if not _per_trace(weights):
        return {n: sha256_file(root / f"{n}{WEIGHTS_SUFFIX}.npz") for n in WEIGHT_SETS}
    out = {}
    for k, f in _manifest(weights)["files"].items():
        out[k] = sha256_file(root / f["file"])
        if out[k] != f.get("sha256"):
            raise ValueError(f"{weights}/{f['file']}: sha256 {out[k]} != manifest {f.get('sha256')}")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="policy", required=True)
    t = sub.add_parser("tokenscale")
    t.add_argument("--report")
    t.add_argument("--profile")
    p = sub.add_parser("preserve")
    p.add_argument("--mu")
    p.add_argument("--weights", default=WEIGHTS, help="weights dir relative to tre/baselines")
    p.add_argument("--predictor-variant", default="v2", choices=PREDICTOR_VARIANTS,
                   help="the predictor_rules default of a per-trace weights dir")
    for s in (t, p):
        s.add_argument("--pending", action="store_true")
        s.add_argument("--configmaps", default=str(DEFAULT_CM))
        s.add_argument("--write", action="store_true")
        s.add_argument("--note", action="append", default=[],
                       help="a disclosure line in the block's header comment (repeatable)")
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
            w = weights_sha256(a.weights)
            wsrc = {}
            if (TRE_ROOT / "baselines" / a.weights / "manifest.json").is_file():
                wsrc = {"weights manifest": f"sha256 {sha256_file(TRE_ROOT / 'baselines' / a.weights / 'manifest.json')}"}
            kw = {"weights_dir": a.weights, "variant": a.predictor_variant}
            if a.pending:
                block = render_preserve(None, {}, w, **kw)
            else:
                if not a.mu:
                    raise ValueError("--mu (or --pending)")
                mu, kind, prov = read_mu(a.mu)
                src = {"mu": f"sha256 {sha256_file(a.mu)}" + (f" ({prov})" if prov else ""), **wsrc}
                block = render_preserve(mu, src, w, mu_kind=kind, **kw)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    block = with_notes(block, a.note)
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
