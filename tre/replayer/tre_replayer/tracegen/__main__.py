"""CLI: python3 -m tre_replayer.tracegen {fit,generate,materialize,verify,audit} ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not v:
            raise SystemExit(f"expected NAME=PATH, got {it!r}")
        out[k] = v
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tre_replayer.tracegen")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="lognormal length fits + arrival dispersion of Azure CSVs")
    f.add_argument("--csv", action="append", required=True, help="NAME=PATH (repeatable)")
    f.add_argument("--out", required=True, help="fits JSON to write (merged with an existing one)")

    g = sub.add_parser("generate", help="spec + seed -> design.json + manifest.json")
    g.add_argument("--spec", required=True)
    g.add_argument("--seed", type=int, action="append", help="repeatable; default: the spec's seeds")
    g.add_argument("--out-root", required=True, help="writes <out-root>/<trace>/seed<k>/")
    g.add_argument("--capacity")
    g.add_argument("--fits")
    g.add_argument("--azure-csv", action="append", help="NAME=PATH for real slices")

    m = sub.add_parser("materialize", help="design -> traces_tre.effective.json (prompts) + verify")
    m.add_argument("run_dir", nargs="+")
    m.add_argument("--processes", type=int, default=4)
    m.add_argument("--tokenizer-path", action="append", help="MODEL=DIR")

    v = sub.add_parser("verify", help="effective == design")
    v.add_argument("run_dir", nargs="+")
    v.add_argument("--no-recount", action="store_true")
    v.add_argument("--tokenizer-path", action="append", help="MODEL=DIR")

    a = sub.add_parser("audit", help="R1/R2/R3/R5 + in-flight")
    a.add_argument("run_dir", nargs="+")
    a.add_argument("--fits")
    a.add_argument("--json", help="write all results here")
    a.add_argument("--md", help="write the markdown table here")

    c = sub.add_parser("loadgen-configs", help="<config root>/<trace>/config.yaml + <trace>_s<k> links")
    c.add_argument("--spec", nargs="+", required=True)
    c.add_argument("--config-root", required=True)
    c.add_argument("--capacity")

    n = sub.add_parser("link-names", help="<out root>/by-name/<trace>_s<k> -> ../<trace>/seed<k>")
    n.add_argument("out_root")

    args = p.parse_args(argv)
    if args.cmd == "loadgen-configs":
        from .names import write_configs
        for f in write_configs(args.spec, args.config_root, args.capacity):
            print(f)
        return 0
    if args.cmd == "link-names":
        from .names import link_names
        print(len(link_names(args.out_root)), "links")
        return 0
    if args.cmd == "fit":
        from . import azure
        out = Path(args.out)
        fits = json.loads(out.read_text()) if out.exists() else {}
        for name, path in _kv(args.csv).items():
            fits[name] = azure.fit_csv(path, name)
            print(name, json.dumps({k: fits[name][k] for k in ("in", "out")}), file=sys.stderr)
        out.write_text(json.dumps(fits, indent=1, sort_keys=True) + "\n")
        return 0
    if args.cmd == "generate":
        from .generate import generate
        spec = json.loads(Path(args.spec).read_text())
        for seed in args.seed or spec["seeds"]:
            d = Path(args.out_root) / spec["trace"] / f"seed{seed}"
            man = generate(args.spec, seed, d, capacity_path=args.capacity, fits_path=args.fits,
                           azure_csv=_kv(args.azure_csv))
            print(d, man["design"]["requests"], man["design"]["sha256"][:16])
        return 0
    if args.cmd == "materialize":
        from .materialize import materialize
        for d in args.run_dir:
            man = materialize(d, processes=args.processes, tokenizer_paths=_kv(args.tokenizer_path))
            print(d, man["effective"]["bytes"], man["effective"]["sha256"][:16], "verified")
        return 0
    if args.cmd == "verify":
        from .materialize import verify
        for d in args.run_dir:
            print(d, verify(d, recount=not args.no_recount, tokenizer_paths=_kv(args.tokenizer_path)))
        return 0
    if args.cmd == "audit":
        from .audit import audit_run, markdown
        from .capacity import load_fits
        fits = load_fits(args.fits)
        res = [audit_run(d, fits=fits) for d in args.run_dir]
        md = markdown(res)
        print(md)
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
        if args.md:
            Path(args.md).write_text(md)
        return 0 if all(r["inflight_ok"] for r in res) else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
