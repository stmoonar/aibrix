"""Causal PreServe histories for a frozen trace suite (no test-request file is read).

Real traces: sample each model's own preceding Azure interval with its frozen keep
fraction and byte-offset SplitMix64 draw. Synthetic traces: extend the initial
baseline into the past, or extend the declared sinusoid analytically. These are
explicit history conventions, not forecasts of later synthetic peaks.

The frozen suite's prompt_length already denotes chat-template-inclusive tokens.
Apply its input floor, without adding the template a second time. Never scale by
the test interval's realized count or mean length.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

from tre_baselines.policies import preserve_history as ph

MASK64 = (1 << 64) - 1


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_uniform(offset, seed, kind):
    """Frozen suite's splitmix64-source-byte-offset-v1, including its 53-bit draw."""
    x = (offset + seed * 1000003 + {'code': 17, 'conv': 29}[kind] + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return (x >> 11) / (1 << 53)


def source_rows(path, start, end):
    """Read only [start,end) of a timestamp-sorted source, retaining stable byte IDs."""
    def stamp(line):
        return datetime.fromisoformat(line.split(b',', 1)[0].decode())
    with Path(path).open('rb') as f:
        f.readline()
        first = f.tell()
        first_line = f.readline()
        if not first_line or stamp(first_line) > start:
            raise ValueError(f'source does not cover history start: {path} {start}')
        lo, hi = first, Path(path).stat().st_size
        while hi - lo > 65536:
            mid = (lo + hi) // 2
            f.seek(mid)
            f.readline()
            line = f.readline()
            if line and stamp(line) < start:
                lo = f.tell()
            else:
                hi = mid
        f.seek(max(first, lo - 65536))
        if f.tell() != first:
            f.readline()
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                raise ValueError(f'source ends before history boundary: {path} {end}')
            t = stamp(line)
            if t < start:
                continue
            if t >= end:
                return
            _, inp, out = line.strip().rsplit(b',', 2)
            yield off, (t - start).total_seconds(), int(inp), int(out)


def build_history(spec, trace, seed, dataset_root, *, window_s=30.0, seq_len=25):
    if not math.isfinite(window_s) or window_s <= 0 or seq_len < 1:
        raise ValueError('positive window and sequence length required')
    span = window_s * seq_len
    floors = spec['prompt']['min_input_tokens_by_model']
    models, provenance = {}, {}
    for model in spec['model_order']:
        conf = trace['models'][model]
        rows = [[0, 0] for _ in range(seq_len)]
        count = adjusted = 0
        ids = hashlib.sha256()
        if 'source' in trace:
            src = trace['source']
            end = datetime.fromisoformat(conf['source_start_utc'])
            start = end - timedelta(seconds=span)
            prob = float(conf['keep_fraction'])
            if not 0 <= prob <= 1 or not math.isfinite(prob):
                raise ValueError('invalid frozen keep fraction')
            samples = ((off, t, inp, out) for off, t, inp, out in
                       source_rows(Path(dataset_root) / src['file'], start, end)
                       if source_uniform(off, seed, src['kind']) < prob)
            info = {'method': 'preceding-source-interval', 'start_utc': start.isoformat(),
                    'end_utc_exclusive': end.isoformat(), 'keep_fraction': prob,
                    'source_file': src['file'], 'source_sha256': src['sha256']}
        else:
            seg = conf['segments'][0]
            if float(seg['start_s']) != 0:
                raise ValueError('synthetic baseline must start at zero')
            rate = seg['rate']
            rng = random.Random(int.from_bytes(hashlib.sha256(
                f'preserve-history-v1|{trace["name"]}|{seed}|{model}'.encode()).digest(), 'big'))
            def arrivals(rate=rate, rng=rng, seg=seg):
                kind = rate['kind']
                bound = float(rate['rps'] if kind == 'constant' else rate['mean'] + abs(rate['amplitude']))
                if kind not in ('constant', 'sinusoid') or not math.isfinite(bound) or bound < 0:
                    raise ValueError('invalid synthetic history rate')
                if kind == 'sinusoid' and (rate['period_s'] <= 0 or rate['mean'] < abs(rate['amplitude'])):
                    raise ValueError('invalid sinusoid')
                if bound == 0:
                    return
                t, i = -span, 0
                while True:
                    t += rng.expovariate(bound)
                    if t >= 0:
                        return
                    if kind == 'sinusoid':
                        lam = rate['mean'] + rate['amplitude'] * math.sin(
                            2 * math.pi * t / rate['period_s'] + math.radians(rate['phase_deg']))
                        if rng.random() >= lam / bound:
                            continue
                    i += 1
                    yield i, t + span, int(seg['input_tokens']), int(seg['output_tokens'])
            samples = arrivals()
            info = {'method': 'analytic-past-sinusoid' if rate['kind'] == 'sinusoid' else 'initial-baseline-past',
                    'start_s': -span, 'end_s_exclusive': 0, 'rate': rate,
                    'input_tokens': seg['input_tokens'], 'output_tokens': seg['output_tokens']}
        for identity, t, inp, out in samples:
            if inp < 1 or out < 1 or not 0 <= t < span:
                raise ValueError('invalid source request in history')
            i = int(t // window_s)
            adjusted += inp < floors[model]
            rows[i][0] += max(inp, int(floors[model]))
            rows[i][1] += out
            count += 1
            ids.update(f'{identity}\n'.encode())
        models[model] = rows
        provenance[model] = {**info, 'requests': count, 'input_adjusted_requests': adjusted,
                             'input_floor_chat_tokens': floors[model], 'selected_ids_sha256': ids.hexdigest()}
    return {'format': ph.FORMAT, 'trace': trace['name'], 'seed': seed, 'window_s': window_s,
            'models': models, 'source': {'method': 'causal-frozen-suite-v1',
                'input_units': 'chat-template-inclusive tokens; same as online x-tre-bl-in-tokens',
                'history_range_s': [-span, 0], 'per_model': provenance}}


def generate(spec_path, dataset_root, out_root, *, window_s=30.0, seq_len=25):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    if spec['format'] != 'tre-frozen-suite-v1':
        raise ValueError('unsupported frozen suite')
    for src in {t['source']['file']: t['source'] for t in spec['traces'] if 'source' in t}.values():
        if sha_file(Path(dataset_root) / src['file']) != src['sha256']:
            raise ValueError(f'source SHA256 mismatch: {src["file"]}')
    report = []
    for trace in spec['traces']:
        for seed in spec['seeds']:
            doc = build_history(spec, trace, seed, dataset_root, window_s=window_s, seq_len=seq_len)
            doc['source'].update(spec_sha256=sha_file(spec_path), generator_sha256=sha_file(__file__))
            ph.parse(doc, window_s=window_s, models=spec['model_order'], seq_len=seq_len,
                     trace=trace['name'], seed=seed)
            dest = Path(out_root) / trace['name'] / f'seed{seed}' / ph.DEFAULT_NAME
            data = (json.dumps(doc, indent=2, allow_nan=False) + '\n').encode()
            if dest.exists() and dest.read_bytes() != data:
                raise FileExistsError(f'refusing to overwrite history: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            report.append({'trace': trace['name'], 'seed': seed, 'history': str(dest),
                           'sha256': hashlib.sha256(data).hexdigest(),
                           'requests': {m: d['requests'] for m, d in doc['source']['per_model'].items()}})
            print(json.dumps(report[-1]), flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--window-s', type=float, default=30)
    p.add_argument('--seq-len', type=int, default=25)
    a = p.parse_args(argv)
    generate(a.spec, a.dataset_root, a.out_root, window_s=a.window_s, seq_len=a.seq_len)


if __name__ == '__main__':
    main()
