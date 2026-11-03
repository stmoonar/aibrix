"""PreServe pre-t0 history (``tre-preserve-history-v1``) from real Azure segments, at any Tier-1 window.

Method ``azure-segment-v1-20261031`` (user 2026-10-31): the generator of the ``preserve_history*.json`` files of
the ICSE trace sets. Merged 2026-11-03 from the three copies that wrote the files of the past runs; each copy is
a ``--profile`` (the code path is the same, the profiles differ only in the data listed in PROFILES):

  profile  copy (TRE local workspace, docs/scripts/)              sha256 of the copy   files it wrote
  icse     icse_suite_20261030/preserve_history_azure.py           58335c9c8b2679ab...  traces-icse-ktrace-20261031
           (the set's _src/ copy)
  r3c      trace_suite_r3c_20261101/preserve_history_azure.py      6dac1a1ed7eace6a...  traces-icse-r3c-20261101
  r3e      trace_suite_r3e_20261101/preserve_history_azure.py      ab053d6746e27214...  traces-icse-r3e-20261101

Rule (per trace, seed, model). The history never reads the evaluated trace: inputs are the design files
(``<v1-gen>/config/icse_final/<trace>/``, the v9 slice files), k and the Azure CSVs, never a path inside the trace
set.

1. segment (does not depend on the window): a SEGMENT_S = 750 s segment (25 x 30 s) of the real Azure 2024 week
   that overlaps no zone of the profile's EXCLUDE list (Real_* slices, their pre-slice history, other suites'
   intervals, 1 h guards) and no segment already chosen for the same file (same trace / seed). Family:
   dsllama-8b <- conv week, dsqwen-7b / dsqwen-14b <- code week. Start: sha256("icse-ktrace-history-v1|<trace>|
   seed<s>|<model>|<attempt>") -> offset in the week, rounded down to 30 s; the first free attempt is used.
2. window: the history is the LAST W x n s of the segment (W = --window-s, n = --seq-len; the segment end stays
   t0) in n windows of W. At 30 x 25 that is the whole segment (the files of the past runs).
3. lengths: the v9 tok70 transform (v1 generate_traces_v9.py rescale_trace_file): min-max of the segment's own
   ContextTokens / GeneratedTokens (over the whole segment) to the model's v9 token target of the family used,
   rounded, >= 1, then capped at the trace's planned max (input / output) for that model.
4. level: P (prompt) and D (output) are scaled separately so that their mean per W window equals the trace's
   PLANNED mean per W window over the 720 s. Plan = the generator's per-second count schedule x planned
   lengths, never the effective file:
     synthetic: count(t) = carry-rounded k x round(rps_m(t)) (k x rps_m(t) when the design marker says
                count_rule carry); length(t) = mean of 4000 draws of the v1 generator's own
                _get_request_overrides + _generate_prompt_length for the segment covering t (planning RNG seeded
                by trace / model); output = override max_tokens, else the config's max_tokens (06-07 traces:
                the config's max_tokens unless the design marker says max_tokens_rule generator).
     Real_*:    no marker: count(t) = round(v9 count(t) x v1_factor x k), length(t) = mean v9 lengths of
                second t; marker (r3c / r3e): the design's per-second lists real_traces/<m>.json (k = 1).
   The time shape and noise inside the history stay the real segment's.

Output: ``<suite>/<trace>/seed<s>/<name>``, name = the last of ``policies/preserve_history.history_names(W)``
(30 s: ``preserve_history.json``, the legacy name; another W: ``preserve_history_w<W>.json``), the name
the policy and the campaign runner look for. An existing file is never
overwritten (identical -> skipped, different -> exit 2). ``--verify`` writes nothing: it compares each output
with the existing file. A file records its generator's path and sha256, so a file of an old copy cannot be
byte-identical to this tool's output; ``--verify`` reports the raw comparison and the comparison with those two
fields set to the old file's values.

Run where the v1 generator (``--v1-gen``: CustomTraceGenerator or ``<set>.design``, with ``src/`` and
``config/icse_final/``) and the Azure files are; CPU only, a few minutes per synthetic trace (the plan).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SELECT_WINDOW_S, SEGMENT_WINDOWS = 30.0, 25          # the segment rule (fixed; not the history's window)
SEGMENT_S = SELECT_WINDOW_S * SEGMENT_WINDOWS
LEGACY_WINDOW_S, LEGACY_SEQ_LEN = 30.0, 25
DUR = 720.0
MODELS = ["dsllama-8b", "dsqwen-7b", "dsqwen-14b"]
SYNTH = ["Alternating_hot_model_periodic_A", "Decode_heavy_burst", "Prefill_mixed_corner_decode_mix",
         "Simultaneous_spike_ramp_twice_tps1o2", "Sinusoidal_demand"]
REAL = ["Real_code_2024_slice_a_tok70", "Real_conv_2023_slice_a_tok70"]
RUN_0607 = {"Sinusoidal_demand", "Decode_heavy_burst", "Alternating_hot_model_periodic_A",
            "Prefill_mixed_corner_decode_mix"}
V1_FACTOR = {"Real_code_2024_slice_a_tok70": {"dsllama-8b": 5, "dsqwen-7b": 5, "dsqwen-14b": 2.5},
             "Real_conv_2023_slice_a_tok70": {"dsllama-8b": 4, "dsqwen-7b": 4, "dsqwen-14b": 4}}
UTC = timezone.utc
FILES = {  # Azure LLM inference 2024 week files (under --azure-root)
    "conv": {"rel": "2024/AzureLLMInferenceTrace_conv_1week.csv",
             "sha256": "a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941",
             "start": datetime(2024, 5, 12, tzinfo=UTC), "end": datetime(2024, 5, 19, tzinfo=UTC)},
    "code": {"rel": "2024/AzureLLMInferenceTrace_code_1week.csv",
             "sha256": "71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f",
             "start": datetime(2024, 5, 10, tzinfo=UTC), "end": datetime(2024, 5, 17, tzinfo=UTC)},
}
FAMILY = {"dsllama-8b": "conv", "dsqwen-7b": "code", "dsqwen-14b": "code"}
# v1 generate_traces_v9.py token targets (the tok70 / v9 length transform)
TOKEN_TARGETS = {
    "code": {"dsllama-8b": {"input": (250, 700), "output": (80, 400)},
             "dsqwen-7b": {"input": (260, 710), "output": (78, 395)},
             "dsqwen-14b": {"input": (245, 695), "output": (85, 410)}},
    "conv": {"dsllama-8b": {"input": (250, 700), "output": (80, 500)},
             "dsqwen-7b": {"input": (260, 710), "output": (78, 495)},
             "dsqwen-14b": {"input": (245, 695), "output": (85, 510)}},
}


def _z(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


# Azure time ranges no history may touch (both files; [start, end) UTC). Each Real_* zone covers the v1 ReTrace
# artifact window (all slices a-d, i.e. ICSE slice_a, contend Real_*_c, v2 Real-* seeds) plus its 750 s pre-slice
# history, with a 1 h guard; each other-suite zone is source_start_utc +- 1 h (720 s slice + 750 s history inside).
EXCLUDE_BASE = [
    ("Real_conv_2023_slice_a..d (conv 2024 week [118800, 126000) s) + pre-slice history",
     _z("2024-05-13T08:00:00+00:00"), _z("2024-05-13T12:00:00+00:00")),
    ("Real_code_2024_slice_a..d (code 2024 week [86400, 88200) s) + pre-slice history",
     _z("2024-05-10T23:00:00+00:00"), _z("2024-05-11T01:30:00+00:00")),
] + [(f"tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc {s}",
      _z(s) - timedelta(hours=1), _z(s) + timedelta(hours=1))
     for s in ("2024-05-12T15:24:00+00:00", "2024-05-14T06:50:00+00:00", "2024-05-14T22:53:00+00:00",
               "2024-05-15T04:08:00+00:00", "2024-05-17T13:54:00+00:00", "2024-05-18T02:53:00+00:00")]
# r3c Real_* intervals (real_select.py; filled in after the selection, which itself only sees EXCLUDE_BASE)
R3C_REAL = [   # (name, family, start_utc_iso)
    ('Real_code_2024_slice_a_tok70/dsqwen-7b', 'code', '2024-05-12T20:32:00+00:00'),
    ('Real_code_2024_slice_a_tok70/dsllama-8b', 'code', '2024-05-13T01:18:00+00:00'),
    ('Real_code_2024_slice_a_tok70/dsqwen-14b', 'code', '2024-05-13T02:42:00+00:00'),
    ('Real_conv_2023_slice_a_tok70/dsqwen-7b', 'conv', '2024-05-16T16:13:00+00:00'),
    ('Real_conv_2023_slice_a_tok70/dsllama-8b', 'conv', '2024-05-13T19:03:00+00:00'),
    ('Real_conv_2023_slice_a_tok70/dsqwen-14b', 'conv', '2024-05-13T19:22:00+00:00'),
]
# r3d Real_* intervals (r3d_select.py; the selection itself only saw EXCLUDE_BASE + R3C_REAL)
R3D_REAL = [   # (name, family, start_utc_iso, length_s)
    ('Real_code_2024_slice_a_tok70/dsqwen-7b', 'code', '2024-05-11T06:00:00+00:00', 5760),
    ('Real_code_2024_slice_a_tok70/dsllama-8b', 'code', '2024-05-12T08:40:00+00:00', 5760),
    ('Real_code_2024_slice_a_tok70/dsqwen-14b', 'code', '2024-05-15T12:50:00+00:00', 5760),
    ('Real_conv_2023_slice_a_tok70/dsqwen-7b', 'conv', '2024-05-16T01:30:00+00:00', 5760),
    ('Real_conv_2023_slice_a_tok70/dsllama-8b', 'conv', '2024-05-15T06:00:00+00:00', 5760),
    ('Real_conv_2023_slice_a_tok70/dsqwen-14b', 'conv', '2024-05-16T04:10:00+00:00', 5760),
]
# r3e Real_* intervals (r3e_select.py; the selection itself only saw EXCLUDE_BASE + R3C_REAL + R3D_REAL)
R3E_REAL = [   # (name, family, start_utc_iso, length_s)
    ('Real_code_2024_slice_a_tok70/dsqwen-7b', 'code', '2024-05-12T03:17:00+00:00', 5760),
    ('Real_code_2024_slice_a_tok70/dsqwen-14b', 'code', '2024-05-16T19:04:00+00:00', 5760),
    ('Real_code_2024_slice_a_tok70/dsllama-8b', 'code', '2024-05-14T10:47:00+00:00', 5760),
    ('Real_conv_2023_slice_a_tok70/dsqwen-7b', 'conv', '2024-05-17T15:23:00+00:00', 5760),
    ('Real_conv_2023_slice_a_tok70/dsqwen-14b', 'conv', '2024-05-18T20:58:00+00:00', 5760),
    ('Real_conv_2023_slice_a_tok70/dsllama-8b', 'conv', '2024-05-16T22:13:00+00:00', 5760),
]
_EX_R3C = [(f"r3c {n} interval +- 1 h ({fam})", _z(st) - timedelta(hours=1),
            _z(st) + timedelta(seconds=720) + timedelta(hours=1)) for n, fam, st in R3C_REAL]
_EX_R3D = [(f"r3d {n} interval (time-compressed, {L} s real) +- 1 h ({fam})", _z(st) - timedelta(hours=1),
            _z(st) + timedelta(seconds=L) + timedelta(hours=1)) for n, fam, st, L in R3D_REAL]
_EX_R3E = [(f"r3e {n} interval (time-compressed, {L} s real) +- 1 h ({fam})", _z(st) - timedelta(hours=1),
            _z(st) + timedelta(seconds=L) + timedelta(hours=1)) for n, fam, st, L in R3E_REAL]

#: exclude: Azure zones; marker: design marker file in config/icse_final/<trace>/ (None: no marker, v9 Real_* plan,
#: 06-07 max_tokens rule by trace name only); real_rule: the Real_* plan's rule string with a marker
PROFILES = {
    "icse": {"exclude": EXCLUDE_BASE, "marker": None, "real_rule": None},
    "r3c": {"exclude": EXCLUDE_BASE + _EX_R3C, "marker": "r3c.json",
            "real_rule": "r3c design per-second list (real_select.py), lengths as planned"},
    "r3e": {"exclude": EXCLUDE_BASE + _EX_R3C + _EX_R3D + _EX_R3E, "marker": "r3e.json",
            "real_rule": "r3e design per-second list (r3e_select.py), lengths as planned"},
}
GENERATOR = "tre/baselines/tre_baselines/tools/preserve_history_azure.py"


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cround(x: float) -> int:
    return int(math.floor(x + 0.5))


def out_name(window_s: float) -> str:
    """The file name of this window: the last of ``preserve_history.history_names`` (the policy's and
    the campaign runner's rule)."""
    from tre_baselines.policies import preserve_history as ph
    return ph.history_names(window_s)[-1]


# ------------------------------------------------------------------ plan (design inputs only)
def design_marker(profile: str, icse_dir: Path) -> dict:
    name = PROFILES[profile]["marker"]
    mk = Path(icse_dir) / name if name else None
    return json.loads(mk.read_text()) if mk is not None and mk.is_file() else {}


def flat_max_tokens(profile: str, trace: str, icse_dir: Path) -> bool:
    """06-07 effective rule (config max_tokens for every request) unless the design marker (or, without one,
    an r2.json marker) says max_tokens_rule = generator. Profile icse: by trace name only."""
    if PROFILES[profile]["marker"] is None:
        return trace in RUN_0607
    mk = design_marker(profile, icse_dir)
    if mk:
        return mk.get("max_tokens_rule") != "generator" and trace in RUN_0607
    r2 = Path(icse_dir) / "r2.json"
    if r2.is_file() and json.loads(r2.read_text()).get("max_tokens_rule") == "generator":
        return False
    return trace in RUN_0607


def _v1_generator(icse_dir: Path, v1_gen: Path, tmp: Path):
    """The v1 generator on the ICSE config (cwd must be tmp, holding config.yaml + trace.json; output dir = tmp)."""
    import yaml
    cfg = yaml.safe_load((icse_dir / "config.yaml").read_text())
    cfg["custom_load_test"]["output"]["base_dir"] = str(tmp / "out")
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (tmp / "trace.json").write_bytes((icse_dir / "trace.json").read_bytes())
    if str(v1_gen / "src") not in sys.path:
        sys.path.insert(0, str(v1_gen / "src"))
    from config_manager import ConfigManager
    import trace_generator as tg
    cm = ConfigManager(str(tmp / "config.yaml"))
    cm.load_config()
    g = tg.create_trace_generator(config_manager=cm)
    return g, g.generate_load_timeline(1.0)


def plan_synth(profile: str, trace: str, k: float, icse_dir: Path, v1_gen: Path, window_s: float,
               n_draw: int = 4000) -> dict:
    import yaml
    caps = {m["name"]: int(m["max_tokens"]) for m in
            yaml.safe_load((icse_dir / "config.yaml").read_text())["custom_load_test"]["models"]}
    flat = flat_max_tokens(profile, trace, icse_dir)
    carry = design_marker(profile, icse_dir).get("count_rule") == "carry"
    st = random.getstate()
    cwd = os.getcwd()
    tmp = Path(tempfile.mkdtemp(prefix="icse_hist_plan_"))
    os.chdir(tmp)
    out = {}
    try:
        g, tl = _v1_generator(icse_dir, v1_gen, tmp)
        _, model_segs = g._get_custom_model_segments()
        for m in MODELS:
            acc, prev, cache = 0.0, 0, {}
            P = D = 0.0
            n_tot, pmax, dmax = 0, 0, 0
            for p in tl:
                n0 = max(0.0, p.model_loads.get(m, 0.0)) if carry else max(0, int(round(p.model_loads.get(m, 0.0))))
                acc += k * n0
                tot = cround(acc)
                n, prev = tot - prev, tot
                if n == 0:
                    continue
                t = float(p.timestamp)
                seg = None
                for i, c in enumerate(model_segs.get(m, [])):
                    if c["start_time"] <= t < c["end_time"]:
                        seg = i
                if seg not in cache:
                    random.seed(f"icse-ktrace-plan|{trace}|{m}|seg{seg}")
                    ls, os_ = [], []
                    for _ in range(n_draw):
                        o = g._get_request_overrides(m, t)
                        ls.append(g._generate_prompt_length(o.get("input_token_config")))
                        mo = o.get("max_output_tokens")
                        os_.append(caps[m] if (flat or mo is None) else int(mo))
                    cache[seg] = (statistics.mean(ls), statistics.mean(os_), max(ls), max(os_))
                mi, mo, xi, xo = cache[seg]
                P += n * mi; D += n * mo; n_tot += n
                pmax, dmax = max(pmax, xi), max(dmax, xo)
            out[m] = {"requests": n_tot, "P_per_window": P / DUR * window_s, "D_per_window": D / DUR * window_s,
                      "mean_in": P / n_tot, "mean_out": D / n_tot, "max_in": pmax, "max_out": dmax,
                      "rule": "carry-rounded k x round(rps_m(t)) x mean of the generator's planned lengths "
                              f"({n_draw} draws per segment, planning RNG)"}
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
        random.setstate(st)
    return out


def plan_real(profile: str, trace: str, k: float, v9: Path, icse_dir: Path, window_s: float) -> dict:
    if PROFILES[profile]["marker"] is not None and design_marker(profile, icse_dir):
        return _plan_real_design(profile, k, Path(icse_dir), window_s)
    out = {}
    for m in MODELS:
        P = D = 0.0
        n_tot, pmax, dmax = 0, 0, 0
        for b in json.loads((v9 / trace / f"{m}.json").read_text()):
            rs = b["requests"]
            want = int(round(len(rs) * V1_FACTOR[trace][m] * k))
            if want == 0:
                continue
            mi = statistics.mean(q["input_tokens"] for q in rs)
            mo = statistics.mean(q["output_tokens"] for q in rs)
            P += want * mi; D += want * mo; n_tot += want
            pmax = max(pmax, max(q["input_tokens"] for q in rs))
            dmax = max(dmax, max(q["output_tokens"] for q in rs))
        out[m] = {"requests": n_tot, "P_per_window": P / DUR * window_s, "D_per_window": D / DUR * window_s,
                  "mean_in": P / n_tot, "mean_out": D / n_tot, "max_in": pmax, "max_out": dmax,
                  "v9_sha256": sha(v9 / trace / f"{m}.json"),
                  "rule": "round(v9 count x v1_factor x k) x mean v9 lengths of the second"}
    return out


def _plan_real_design(profile: str, k: float, icse_dir: Path, window_s: float) -> dict:
    """Real_* with a design marker: the design's per-second lists (real_traces/<m>.json, already scaled; k = 1)."""
    assert abs(k - 1.0) < 1e-9, k
    out = {}
    for m in MODELS:
        f = icse_dir / "real_traces" / f"{m}.json"
        rs = [q for b in json.loads(f.read_text()) for q in b["requests"]]
        P = sum(q["input_tokens"] for q in rs)
        D = sum(q["output_tokens"] for q in rs)
        out[m] = {"requests": len(rs), "P_per_window": P / DUR * window_s, "D_per_window": D / DUR * window_s,
                  "mean_in": P / len(rs), "mean_out": D / len(rs), "max_in": max(q["input_tokens"] for q in rs),
                  "max_out": max(q["output_tokens"] for q in rs), "design_list_sha256": sha(f),
                  "rule": PROFILES[profile]["real_rule"]}
    return out


def plan(profile: str, trace: str, k, *, v1_gen: Path, v9: Path, window_s: float) -> dict:
    icse_dir = Path(v1_gen) / "config" / "icse_final" / trace
    if trace in SYNTH:
        return plan_synth(profile, trace, k, icse_dir, Path(v1_gen), window_s)
    return plan_real(profile, trace, k, Path(v9), icse_dir, window_s)


# ------------------------------------------------------------------ Azure segments
def _overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def choose_segments(profile: str, trace: str, seed: int) -> dict:
    chosen: dict = {}
    for m in MODELS:
        fam = FAMILY[m]
        f = FILES[fam]
        span_s = int((f["end"] - f["start"]).total_seconds() - SEGMENT_S)
        for attempt in range(1000):
            h = hashlib.sha256(f"icse-ktrace-history-v1|{trace}|seed{seed}|{m}|{attempt}".encode()).digest()
            off = (int.from_bytes(h[:8], "big") % span_s) // int(SELECT_WINDOW_S) * int(SELECT_WINDOW_S)
            s0 = f["start"] + timedelta(seconds=off)
            s1 = s0 + timedelta(seconds=SEGMENT_S)
            if any(_overlaps(s0, s1, z0, z1) for _n, z0, z1 in PROFILES[profile]["exclude"]):
                continue
            if any(c["family"] == fam and _overlaps(s0, s1, c["start"], c["end"]) for c in chosen.values()):
                continue
            chosen[m] = {"family": fam, "start": s0, "end": s1, "attempt": attempt, "offset_s": off}
            break
        else:
            raise RuntimeError(f"no free segment for {trace}/seed{seed}/{m}")
    return chosen


def read_segment(csv: Path, s0: datetime, s1: datetime) -> list:
    """[(t_rel_s, ContextTokens, GeneratedTokens)] of [s0, s1) (binary search on the sorted CSV)."""
    def stamp(line: bytes) -> datetime:
        return datetime.fromisoformat(line.split(b",", 1)[0].decode())
    rows = []
    with open(csv, "rb") as f:
        f.readline()
        first = f.tell()
        lo, hi = first, csv.stat().st_size
        while hi - lo > 65536:
            mid = (lo + hi) // 2
            f.seek(mid); f.readline()
            line = f.readline()
            if line and stamp(line) < s0:
                lo = f.tell()
            else:
                hi = mid
        f.seek(max(first, lo - 65536))
        if f.tell() != first:
            f.readline()
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{csv} ends before {s1}")
            t = stamp(line)
            if t < s0:
                continue
            if t >= s1:
                break
            _, i, o = line.strip().rsplit(b",", 2)
            rows.append(((t - s0).total_seconds(), int(i), int(o)))
    if not rows or rows[0][0] > SELECT_WINDOW_S:
        raise ValueError(f"segment {s0} has no arrivals in its first window")
    return rows


def _minmax(v, lo, hi, a, b) -> float:
    return (a + b) / 2.0 if hi == lo else (v - lo) / (hi - lo) * (b - a) + a


# ------------------------------------------------------------------ build
def build(profile: str, trace: str, seed: int, k: float, *, plan_doc: dict, azure_root: Path,
          window_s: float = LEGACY_WINDOW_S, seq_len: int = LEGACY_SEQ_LEN) -> dict:
    """The history document. Inputs: the plan (design files) and the Azure CSVs only (no path inside the trace
    set). plan_doc must be planned at window_s (plan(..., window_s=window_s))."""
    window_s, seq_len = float(window_s), int(seq_len)
    span_w = window_s * seq_len
    if span_w > SEGMENT_S + 1e-9:
        raise ValueError(f"{window_s:g} s x {seq_len} is longer than the {SEGMENT_S:g} s segment")
    lead = SEGMENT_S - span_w                    # seconds of the segment before the history
    legacy = lead == 0.0
    segs = choose_segments(profile, trace, seed)
    rows_all, prov = {}, {}
    for m in MODELS:
        sg = segs[m]
        fam = sg["family"]
        raw = read_segment(Path(azure_root) / FILES[fam]["rel"], sg["start"], sg["end"])
        ins, outs = [r[1] for r in raw], [r[2] for r in raw]
        lo_i, hi_i, lo_o, hi_o = min(ins), max(ins), min(outs), max(outs)   # over the whole segment
        tgt = TOKEN_TARGETS[fam][m]
        pl = plan_doc[m]
        cap_in, cap_out = int(pl["max_in"]), int(pl["max_out"])
        win = [[0, 0.0, 0.0] for _ in range(seq_len)]
        used = 0
        for t, i, o in raw:
            if t < lead:
                continue
            j = int((t - lead) // window_s)
            mi = min(cap_in, max(1, int(round(_minmax(i, lo_i, hi_i, *tgt["input"])))))
            mo = min(cap_out, max(1, int(round(_minmax(o, lo_o, hi_o, *tgt["output"])))))
            win[j][0] += 1; win[j][1] += mi; win[j][2] += mo
            used += 1
        mp = statistics.mean(w[1] for w in win)
        md = statistics.mean(w[2] for w in win)
        if mp <= 0 or md <= 0:
            raise ValueError(f"{trace}/seed{seed}/{m}: no requests in the last {span_w:g} s of the segment")
        sp, sd = pl["P_per_window"] / mp, pl["D_per_window"] / md
        rows_all[m] = [[round(w[1] * sp, 2), round(w[2] * sd, 2)] for w in win]
        counts = [w[0] for w in win]
        pm = {"source_file": FILES[fam]["rel"], "source_sha256": FILES[fam]["sha256"], "family": fam,
              "segment_start_utc": sg["start"].isoformat(), "segment_end_utc_exclusive": sg["end"].isoformat(),
              "segment_rule_attempt": sg["attempt"]}
        if not legacy:
            pm["history_start_utc"] = (sg["start"] + timedelta(seconds=lead)).isoformat()
            pm["raw_requests_segment"] = len(raw)
        pm["raw_requests"] = used
        pm["raw_rps_mean"] = round(used / span_w, 3)
        pm[f"raw_count_cv_{window_s:g}s"] = round(statistics.pstdev(counts) / statistics.mean(counts), 4)
        if not legacy:
            pm["empty_windows"] = sum(1 for c in counts if c == 0)
        pm["raw_in_range"] = [lo_i, hi_i]
        pm["raw_out_range"] = [lo_o, hi_o]
        pm["length_transform"] = (f"v9 min-max to {tgt['input']} / {tgt['output']} ({fam} targets), "
                                  f"cap {cap_in} / {cap_out}"
                                  + ("" if legacy else f" (min / max over the whole {SEGMENT_S:g} s segment)"))
        pm["mapped_mean_in"] = round(sum(w[1] for w in win) / used, 2)
        pm["mapped_mean_out"] = round(sum(w[2] for w in win) / used, 2)
        pm["target_P_per_window"] = round(pl["P_per_window"], 2)
        pm["target_D_per_window"] = round(pl["D_per_window"], 2)
        pm["scale_P"], pm["scale_D"] = round(sp, 6), round(sd, 6)
        pm["plan"] = {x: (round(v, 3) if isinstance(v, float) else v) for x, v in pl.items()}
        prov[m] = pm
    where = "" if legacy else (f", its last {span_w:g} s in {seq_len} x {window_s:g} s windows ending at t0 "
                               f"(segment rule unchanged)")
    method = "azure-segment-v1-20261031" + ("" if window_s == LEGACY_WINDOW_S else f"-w{window_s:g}") \
        + ("" if seq_len == LEGACY_SEQ_LEN else f"-n{seq_len}")
    return {"format": "tre-preserve-history-v1", "trace": trace, "seed": seed, "window_s": window_s,
            "models": rows_all,
            "source": {"method": method, "rate_scale_k": k,
                       "history_range_s": [-span_w, 0],
                       "generator": f"{GENERATOR} --profile {profile}",
                       "generator_sha256": sha(__file__),
                       "rule": "real Azure 2024 segment per model (no overlap with any Real_* slice / pre-slice "
                               f"history / other suite interval){where}, v9 tok70 length transform, P and D scaled "
                               f"to the trace's planned mean per {window_s:g} s window (schedule x planned lengths, "
                               "not the effective file)",
                       "per_model": prov}}


def dump(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2) + "\n").encode()


def verify_csvs(azure_root: Path) -> None:
    for fam, f in FILES.items():
        got = sha(Path(azure_root) / f["rel"])
        if got != f["sha256"]:
            raise ValueError(f"{f['rel']}: sha256 {got} != {f['sha256']}")


def _diff_paths(a, b, path: str = "") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                out.append(f"{path}/{k} (only in {'old' if k in a else 'new'})")
            else:
                out += _diff_paths(a[k], b[k], f"{path}/{k}")
        if not out and list(a) != list(b):
            out.append(f"{path} (key order)")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for x, y, i in zip(a, b, range(len(a))) for p in _diff_paths(x, y, f"{path}[{i}]")]
    return [] if a == b and type(a) is type(b) else [path]


def verify_one(doc: dict, existing: Path) -> dict:
    """Raw byte comparison with an existing file, and the comparison with the two generator fields set to
    the existing file's values (the only fields that name the program instead of the data)."""
    old_bytes = existing.read_bytes()
    old = json.loads(old_bytes)
    new_bytes = dump(doc)
    alt = json.loads(new_bytes)
    for f in ("generator", "generator_sha256"):
        if f in old.get("source", {}):
            alt["source"][f] = old["source"][f]
    alt_bytes = dump(alt)
    return {"file": str(existing), "file_sha256": hashlib.sha256(old_bytes).hexdigest(),
            "new_sha256": hashlib.sha256(new_bytes).hexdigest(), "byte_identical": new_bytes == old_bytes,
            "old_generator": old.get("source", {}).get("generator"),
            "identical_with_old_generator_fields": alt_bytes == old_bytes,
            "sha256_with_old_generator_fields": hashlib.sha256(alt_bytes).hexdigest(),
            "models_identical": alt.get("models") == old.get("models"),
            "differences": _diff_paths(old, alt)[:20]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--suite", required=True, help="trace set dir: <suite>/<trace>/seed<s>/")
    ap.add_argument("--k-map", required=True, help="JSON {trace: k} (the set's _checks/k_map.json)")
    ap.add_argument("--v1-gen", required=True, help="v1 CustomTraceGenerator or <set>.design (src/, config/icse_final/)")
    ap.add_argument("--v9", required=True, help="v1 ReTrace traces_v9 dir (Real_* plan without a design marker)")
    ap.add_argument("--azure-root", required=True, help="holds 2024/AzureLLMInferenceTrace_{code,conv}_1week.csv")
    ap.add_argument("--trace", action="append", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--window-s", type=float, default=LEGACY_WINDOW_S)
    ap.add_argument("--seq-len", type=int, default=LEGACY_SEQ_LEN)
    ap.add_argument("--out-name", help="file name (default: the runner's name for the window)")
    ap.add_argument("--verify", action="store_true", help="compare with the existing files, write nothing")
    ap.add_argument("--skip-csv-sha", action="store_true")
    a = ap.parse_args(argv)
    if a.seq_len != LEGACY_SEQ_LEN and not a.out_name:
        ap.error("--seq-len other than 25 needs --out-name (the runner's names assume 25 windows)")
    name = a.out_name or out_name(a.window_s)
    if not a.verify:
        from tre_baselines.policies import preserve_history as ph
    kmap = json.loads(Path(a.k_map).read_text())
    if not a.skip_csv_sha:
        verify_csvs(Path(a.azure_root))
    bad = 0
    for t in a.trace:
        pl = plan(a.profile, t, kmap[t], v1_gen=Path(a.v1_gen), v9=Path(a.v9), window_s=float(a.window_s))
        for s in a.seeds:
            doc = build(a.profile, t, s, float(kmap[t]), plan_doc=pl, azure_root=Path(a.azure_root),
                        window_s=a.window_s, seq_len=a.seq_len)
            dst = Path(a.suite) / t / f"seed{s}" / name
            if a.verify:
                if not dst.is_file():
                    print(json.dumps({"trace": t, "seed": s, "file": str(dst), "missing": True}), flush=True)
                    bad += 1
                    continue
                r = verify_one(doc, dst)
                bad += not r["identical_with_old_generator_fields"]
                print(json.dumps({"trace": t, "seed": s, **r}), flush=True)
                continue
            ph.parse(doc, window_s=float(a.window_s), models=MODELS, seq_len=a.seq_len, trace=t, seed=s)
            data = dump(doc)
            if dst.exists():
                if dst.read_bytes() == data:
                    print(f"{dst}: exists, identical", flush=True)
                    continue
                print(f"{dst} exists and differs: not overwritten", file=sys.stderr)
                return 2
            tmp = dst.with_name(dst.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            print(json.dumps({"trace": t, "seed": s, "file": str(dst), "sha256": hashlib.sha256(data).hexdigest(),
                              "models": {m: {"segment": doc["source"]["per_model"][m]["segment_start_utc"][5:19],
                                             "last": doc["models"][m][-1]} for m in MODELS}}), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
