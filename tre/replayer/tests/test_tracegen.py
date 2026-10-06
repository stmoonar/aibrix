"""tracegen: seeded determinism, Poisson arrivals at the design rate, lengths, effective == design,
and the client sending every request's own max_tokens."""
from __future__ import annotations

import json
import math
import random
import re

import pytest

from tre_replayer.engine import corpus
from tre_replayer.tracegen import audit, generate, lengths, materialize

_CJK = "　-〿㐀-䶿一-鿿＀-￯"
_TOKEN = re.compile(rf"<[A-Za-z]+>|\s*(?:[{_CJK}]|[^\s<{_CJK}]+)")


class ChatStub:
    """One token per marker / CJK character / ASCII word; chat template <BOS><U>{c}<A><think>."""

    overhead = 1
    filler = " the"
    path = "<chat-stub>"
    model = "m"
    chat_prefix = "<BOS><U>"
    chat_suffix = "<A><think>"
    chat_error = None

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._pieces: dict[int, str] = {}

    def _id(self, piece: str) -> int:
        if piece not in self._ids:
            self._ids[piece] = len(self._ids) + 1
            self._pieces[self._ids[piece]] = piece
        return self._ids[piece]

    def encode_plain(self, text: str) -> list[int]:
        return [self._id(p) for p in _TOKEN.findall(text)]

    def decode_plain(self, ids) -> str:
        return "".join(self._pieces[i] for i in ids)

    def count(self, text: str) -> int:
        return len(self.encode_plain(text)) + self.overhead

    def cjk_token_count(self, text: str) -> int:
        return sum(1 for p in _TOKEN.findall(text) if any(corpus.is_cjk(c) for c in p))


LEN = {"in": {"dist": "lognormal", "sigma": 0.98, "mean": 120, "min": 32},
       "out": {"dist": "lognormal", "sigma": 0.86, "mean": 60, "min": 1}}


def _spec(**models) -> dict:
    return {"trace": "unit", "duration_s": 400, "defaults": {"lengths": LEN}, "models": models, "audit": {}}


SQUARE_FN = {"kind": "square", "low": 0.5, "high": 2.4, "period_s": 200, "high_s": 100, "offset_s": 50}
SQUARE = {"rate": {"unit": "rps", "fn": SQUARE_FN}}          # small plans for the prompt tests
SQUARE_RHO = {"rate": {"unit": "rho", "fn": SQUARE_FN}}


def _write(tmp_path, spec, name="spec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(spec))
    return p


def test_same_seed_gives_byte_identical_plans_and_another_seed_does_not(tmp_path):
    spec = _write(tmp_path, _spec(**{"dsqwen-7b": SQUARE, "dsllama-8b": {"rate": {"unit": "rps", "fn": 3}}}))
    a = generate.generate(spec, 1, tmp_path / "a")
    b = generate.generate(spec, 1, tmp_path / "b")
    c = generate.generate(spec, 2, tmp_path / "c")
    assert (tmp_path / "a/design.json").read_bytes() == (tmp_path / "b/design.json").read_bytes()
    assert a["design"]["sha256"] == b["design"]["sha256"] != c["design"]["sha256"]
    # streams are independent: dropping a model leaves the other model's requests unchanged
    solo = _write(tmp_path, _spec(**{"dsqwen-7b": SQUARE}), "solo.json")
    generate.generate(solo, 1, tmp_path / "d")
    pick = lambda d: [(r["timestamp"], r["prompt_length"], r["max_output_tokens"])
                      for r in json.loads((tmp_path / d / "design.json").read_text()) if r["model_name"] == "dsqwen-7b"]
    assert pick("a") == pick("d")


def test_arrivals_are_poisson_at_the_design_rate():
    rng = random.Random(7)
    times = generate.nhpp(lambda t: 20.0, 2000.0, rng)
    assert abs(len(times) - 40000) < 4 * math.sqrt(40000)
    counts = [0] * 2000
    for t in times:
        counts[int(t)] += 1
    mean = sum(counts) / len(counts)
    fano = sum((c - mean) ** 2 for c in counts) / len(counts) / mean
    assert 0.9 < fano < 1.1  # Poisson: var = mean (the v1 traces had var = 0)
    # a square wave: each half gets its own integral
    lam = lambda t: 30.0 if (t % 100) < 50 else 5.0
    times = generate.nhpp(lam, 1000.0, random.Random(8))
    hi = sum(1 for t in times if (t % 100) < 50)
    lo = len(times) - hi
    assert abs(hi - 15000) < 4 * math.sqrt(15000) and abs(lo - 2500) < 4 * math.sqrt(2500)


def test_lognormal_lengths_hit_the_mean_inside_the_p995_cut():
    d = lengths.solve(400, 0.86, min_tokens=1)
    assert d.mean == pytest.approx(400, rel=1e-6)
    assert d.hi == pytest.approx(math.exp(d.mu + 0.86 * lengths._N.inv_cdf(0.995)))
    draws = [d.draw(random.Random(i)) for i in range(20000)]
    assert max(draws) <= d.hi and min(draws) >= 1
    assert sum(draws) / len(draws) == pytest.approx(400, rel=0.03)


def _materialized(tmp_path):
    spec = _write(tmp_path, _spec(**{"dsqwen-7b": SQUARE, "dsqwen-14b": {"rate": {"unit": "rps", "fn": 0.5}}}))
    generate.generate(spec, 3, tmp_path / "run")
    stubs = {"dsqwen-7b": ChatStub(), "dsqwen-14b": ChatStub()}
    materialize.materialize(tmp_path / "run", tokenizers=stubs)
    return tmp_path / "run", stubs


def test_effective_equals_design_and_a_changed_or_null_max_tokens_is_caught(tmp_path):
    run, stubs = _materialized(tmp_path)
    eff = json.loads((run / materialize.EFFECTIVE_FILE).read_text())
    design = json.loads((run / generate.DESIGN_FILE).read_text())
    assert len(eff) == len(design) > 100
    assert all(isinstance(r["max_output_tokens"], int) and r["max_output_tokens"] >= 1 for r in eff)
    materialize.verify(run, tokenizers=stubs)  # re-counts every prompt (chat template included)
    for bad in (None, eff[5]["max_output_tokens"] + 1):
        tampered = [dict(r) for r in eff]
        tampered[5]["max_output_tokens"] = bad
        (run / materialize.EFFECTIVE_FILE).write_bytes(generate.dumps_plan(tampered))
        with pytest.raises(AssertionError, match="max_output_tokens"):
            materialize.verify(run, tokenizers=stubs)


def test_the_client_sends_each_requests_own_max_tokens_not_the_configs(tmp_path):
    """The e1_v1 client path (tre_loadgen_v1 --trace-file): trace dict -> RequestTrace ->
    ScheduledRequest -> SDK kwargs, with a config max_tokens that differs from every request."""
    from tre_loadgen_v1.client_dispatcher import scheduled_requests
    from tre_loadgen_v1.trace_types import RequestTrace
    from tre_replayer.engine.profiles import V1ChatOptions

    run, _ = _materialized(tmp_path)
    eff = json.loads((run / materialize.EFFECTIVE_FILE).read_text())
    traces = [RequestTrace(request_id=d["request_id"], timestamp=d["timestamp"], model_name=d["model_name"],
                           prompt=d["prompt"], prompt_length=d["prompt_length"],
                           phase_type=d.get("phase_type", "unknown"), max_output_tokens=d.get("max_output_tokens"))
              for d in eff]  # exactly as cli.stage2_client_dispatch builds them
    opts = V1ChatOptions(model_params={m: {"max_tokens": 99999, "temperature": None} for m in ("dsqwen-7b", "dsqwen-14b")},
                         ignore_eos=True)
    for d, req in zip(eff, scheduled_requests(traces)):
        kw = opts.kwargs_for(req.model, req.prompt, req.max_output_tokens)
        assert kw["max_tokens"] == d["max_output_tokens"]
        assert kw["extra_body"] == {"ignore_eos": True}


def test_audit_measures_hot_runs_depth_and_design_g(tmp_path):
    spec = _spec(**{"dsqwen-7b": SQUARE_RHO, "dsqwen-14b": {"rate": {"unit": "rho", "fn": 1.0}}})
    spec["audit"] = {"r3_reference": "conv2023"}
    generate.generate(_write(tmp_path, spec), 1, tmp_path / "run")
    r = audit.audit_run(tmp_path / "run")
    m = r["models"]["dsqwen-7b"]
    assert m["hot_level_design"] == pytest.approx(2.4)
    assert abs(m["hot_run_min_s"] - 100) <= 3          # one 3 s ramp of slack
    assert r["G_judged"] == pytest.approx(2.4 + 2 * 1.0, abs=1e-6) and r["R1_basis"] == "design"
    assert r["R5"] == "met" and r["R2"] == "FAIL"        # 100 s hot runs < 150 s
    assert r["models"]["dsqwen-14b"]["R2"] is None       # flat: no hot runs to judge


def test_outputs_are_capped_by_the_route_timeout_rule_and_keep_their_mean(tmp_path):
    """out_max = floor((0.8 * 150 s - TTFT_knee_p95) / TPOT_knee_p95): no output may be longer
    (it would be cut by the route timeout even at knee speed), and the target mean still holds."""
    from tre_replayer.tracegen.capacity import load_capacity, route_out_max
    cap = load_capacity()
    m = cap.models["dsqwen-7b"]
    assert m.out_max == route_out_max(150.0, 0.8, m.knee_p95_ttft_s, m.knee_p95_tpot_s)
    assert (m.out_max + 1) * m.knee_p95_tpot_s + m.knee_p95_ttft_s > 120.0 >= m.out_max * m.knee_p95_tpot_s + m.knee_p95_ttft_s
    long_out = dict(LEN, out={"dist": "lognormal", "sigma": 0.86, "mean": 1300, "min": 1})
    spec = {"trace": "unit", "duration_s": 300, "defaults": {"lengths": long_out}, "audit": {},
            "models": {"dsqwen-7b": {"rate": {"unit": "rps", "fn": 20}}}}
    man = generate.generate(_write(tmp_path, spec), 1, tmp_path / "run")
    outs = [r["max_output_tokens"] for r in json.loads((tmp_path / "run/design.json").read_text())]
    assert max(outs) <= m.out_max < max(outs) + 200          # the cap binds
    assert sum(outs) / len(outs) == pytest.approx(1300, rel=0.03)
    assert man["out_cap"]["out_max"]["dsqwen-7b"] == m.out_max
    assert audit.audit_run(tmp_path / "run")["out_cap_ok"]


def test_every_spec_can_be_replayed_by_name(tmp_path):
    """The runner looks up $LOADGEN_CONFIG_ROOT/<trace>_s<k>/config.yaml: the committed configs
    are what the generator writes now, every seed has its name, and the config loads."""
    from pathlib import Path
    from tre_loadgen_v1.config_manager import ConfigManager
    from tre_replayer.tracegen.names import write_configs
    here = Path(generate.__file__).resolve().parent
    committed = here.parents[2] / "loadgen_v1" / "configs" / "traces_v2"
    specs = sorted((here / "specs").glob("*.json"))
    write_configs(specs, tmp_path)
    for sp in specs:
        spec = json.loads(sp.read_text())
        t = spec["trace"]
        assert (committed / t / "config.yaml").read_text() == (tmp_path / t / "config.yaml").read_text()
        for k in spec["seeds"]:
            cfg = ConfigManager(str(committed / f"{t}_s{k}" / "config.yaml")).load_config()
            assert [m.name for m in cfg.models] == ["dsllama-8b", "dsqwen-7b", "dsqwen-14b"]
