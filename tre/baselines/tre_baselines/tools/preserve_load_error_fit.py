"""Fit the relative response-length error of PreServe's predictor to its Tab. 2 (2026-11-04).

The paper (§5.2.2, Tab. 2, PreServe row) reports the predictor's error on ShareGPT in tokens:
MAE 78.25, Acc-25 56.77 %, Acc-50 68.79 %, Acc-100 77.95 % (lengths up to 4096). Our traces have
much shorter outputs, so an absolute error rebuilt from those numbers is a far larger *relative*
error than the paper's predictor makes. This tool assumes the predictor's error is relative,
``err = r * D`` with ``|r|`` independent of the true length ``D``, and ``|r|`` a two-component
lognormal mixture with a shared shape:

    |r| ~ (1 - w) LogNormal(mu1, sigma) + w LogNormal(mu2, sigma)

Its four parameters are solved so that ``|round(r D)|`` over the ShareGPT response lengths
reproduces the four Tab. 2 numbers exactly (least squares on the three shares in percentage points
and the MAE relative; residuals ~0). A single lognormal (2 parameters) misses Acc-100 by ~5 pp.

ShareGPT lengths: per conversation the first ``gpt`` turn that follows a ``human`` turn, tokenized
without special tokens, kept when 1 <= tokens <= 4096 ("up to 4096"). The paper serves LLaMA-2; the
LLaMA-2 sentencepiece vocabulary is the one of CodeLlama (pass its tokenizer directory).

Usage (on the server; the dataset is not in git)::

    python3 -m tre_baselines.tools.preserve_load_error_fit \\
        --sharegpt <datasets>/sharegpt-vicuna-192ab218/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --tokenizer <models>/CodeLlama-13b-hf --out <dir>/fit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from typing import Sequence

TAB2 = {"mae": 78.25, "acc": ((25, 0.5677), (50, 0.6879), (100, 0.7795))}
MAX_LEN = 4096


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_responses(rows: Sequence[dict]) -> list[str]:
    out = []
    for row in rows:
        conv = row.get("conversations") or []
        for a, b in zip(conv, conv[1:]):
            if a.get("from") == "human" and b.get("from") == "gpt":
                out.append(str(b.get("value") or ""))
                break
    return out


def stats(w: float, mu1: float, mu2: float, sigma: float, lengths) -> dict:
    """Tab. 2 statistics of ``|round(r L)|`` over ``lengths`` for the mixture (exact in ``r``; the
    MAE ignores the rounding, < 0.5 token)."""
    import numpy as np
    from scipy.stats import norm

    L = np.asarray(lengths, dtype=float)
    mean_r = (1 - w) * math.exp(mu1 + sigma * sigma / 2.0) + w * math.exp(mu2 + sigma * sigma / 2.0)
    out = {"mae": float(mean_r * L.mean())}
    for b, _ in TAB2["acc"]:
        x = np.log((b + 0.5) / L)  # |round(r L)| <= b  <=>  |r| L < b + 0.5
        out[f"acc{b}"] = float(((1 - w) * norm.cdf((x - mu1) / sigma) + w * norm.cdf((x - mu2) / sigma)).mean())
    return out


def fit(lengths) -> tuple[dict, dict]:
    import numpy as np
    from scipy.optimize import least_squares

    def unpack(x):
        return 1.0 / (1.0 + math.exp(-x[0])), float(x[1]), float(x[2]), math.exp(x[3])

    def resid(x):
        s = stats(*unpack(x), lengths)
        r = [100.0 * (s[f"acc{b}"] - want) for b, want in TAB2["acc"]]
        return np.array(r + [100.0 * (s["mae"] / TAB2["mae"] - 1.0)])

    res = least_squares(resid, x0=[-1.5, -3.0, -0.5, 0.0], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    w, mu1, mu2, sigma = unpack(res.x)
    params = {"w": w, "mu1": mu1, "mu2": mu2, "sigma": sigma}
    return params, stats(w, mu1, mu2, sigma, lengths)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--tokenizer", required=True, help="a LLaMA-2 vocabulary tokenizer directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import numpy as np
    from transformers import AutoTokenizer

    with open(a.sharegpt, encoding="utf-8") as fh:
        rows = json.load(fh)
    texts = first_responses(rows)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    lens = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    kept = np.array([n for n in lens if 1 <= n <= MAX_LEN])
    params, got = fit(kept)
    q = np.percentile(kept, [5, 25, 50, 75, 95]).tolist()
    report = {"sharegpt": a.sharegpt, "sharegpt_sha256": sha256_of(a.sharegpt), "tokenizer": a.tokenizer,
              "conversations": len(rows), "pairs": len(texts), "kept": int(kept.size),
              "length_mean": float(kept.mean()), "length_p5_25_50_75_95": q,
              "mixture": params, "fitted": got, "tab2": TAB2}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
