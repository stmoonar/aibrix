"""The calibration drivers send /v1/chat/completions, prompts exact after the template.

r3_grid and the campaign default to the chat API; every cell pins it, the run's
preflight checks usage.prompt_tokens against the fitted length for every model and
refuses to run on a difference, each cell records its per-request check, and the API is
part of the load path (tests of the load-path refusals: test_prompt_corpus_record).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from scripts import calibration_campaign as campaign
from scripts import openloop, r3_grid

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = TRE_ROOT / "deploy" / "registry.yaml"
CHAT_URL = "http://gw/v1/chat/completions"


# ------------------------------------------------------------------ per-request record


def test_the_raw_record_keeps_the_expected_length_next_to_the_engines() -> None:
    record = {"request_id": "m-1", "actual_send_ts_ms": 1000, "ttft_ms": 10.0, "e2e_ms": 50.0,
              "input_tokens": 512, "prompt_tokens": 511, "completion_tokens": 8, "http_status": 200}
    raw = openloop._raw_from_sender_record("i512_o8_c1", record)
    assert raw["expected_prompt_tokens"] == 512 and raw["input_tokens"] == 511
    assert set(raw) == set(r3_grid.RAW_COLUMNS) | set(r3_grid.RAW_REQUEST_COLUMNS)


def test_prompt_tokens_check_counts_served_requests_only() -> None:
    records = [
        {"request_id": "a", "http_status": 200, "e2e_ms": 9.0, "input_tokens": 512, "prompt_tokens": 512},
        {"request_id": "b", "http_status": 200, "e2e_ms": 9.0, "input_tokens": 512, "prompt_tokens": 513},
        {"request_id": "c", "http_status": 200, "e2e_ms": 9.0, "input_tokens": 512, "prompt_tokens": None},
        {"request_id": "d", "http_status": 503, "e2e_ms": 9.0, "input_tokens": 512, "prompt_tokens": None},
        # failed inside its 200 stream: not served, so not checked
        {"request_id": "e", "http_status": 200, "e2e_ms": 9.0, "input_tokens": 512, "prompt_tokens": None,
         "stream_error": "boom"},
    ]
    check = openloop.prompt_tokens_check(records)
    assert (check["served"], check["checked"], check["mismatched"], check["missing_usage"]) == (3, 2, 1, 1)
    assert check["max_abs_diff"] == 1 and not check["ok"]
    assert check["examples"] == [{"request_id": "b", "expected": 512, "prompt_tokens": 513}]
    assert openloop.prompt_tokens_check(records[:1])["ok"]
    # the grid path's raw records: expected_prompt_tokens next to input_tokens (= usage)
    raw = [{"http_status": 200, "e2e_ms": 9.0, "expected_prompt_tokens": 512, "input_tokens": n}
           for n in (512, 510)]
    grid = openloop.prompt_tokens_check(raw, expected_key="expected_prompt_tokens", actual_key="input_tokens")
    assert (grid["served"], grid["mismatched"], grid["max_abs_diff"]) == (2, 1, 2)


# ------------------------------------------------------------------------ the preflight


class _FakeTok:
    """Enough of a tokenizer for build_prompt(api=chat): 1 token per word, a 5-token
    template (the fleet's)."""

    overhead = 1
    filler = " the"
    path = "<fake>"
    model = "m"
    chat_prefix = "B U "
    chat_suffix = " A T N"
    chat_error = None

    def encode_plain(self, text):
        return text.split()

    def decode_plain(self, ids):
        return " ".join(ids)

    def count(self, text):
        return len(text.split()) + 1

    def cjk_token_count(self, text):
        return 0


def _stream(prompt_tokens=None, completion_tokens=None, status=200, first=12.0, seen=None):
    from tre_replayer.engine.http_sender import StreamResult

    def call(url, headers, body, timeout):
        payload = json.loads(body)
        if seen is not None:
            seen.append((url, headers, payload))
        return StreamResult(
            status, first if status == 200 else None, 80.0,
            payload_len(payload) if prompt_tokens is None else prompt_tokens,
            payload["max_tokens"] if completion_tokens is None else completion_tokens,
            first_token_field="content" if status == 200 and first is not None else None,
            error=None if status == 200 else f"HTTP {status}",
            error_body=None if status == 200 else "overloaded",
        )
    return call


def payload_len(payload) -> int:
    """What the fake engine reports: the templated length (5 template tokens)."""
    return len(payload["messages"][0]["content"].split()) + 5


def _preflight(**kw):
    kwargs = dict(api="chat", corpus_lang="en", routing_strategy="least-gpu-cache", tokenizer=_FakeTok())
    kwargs.update(kw)
    return openloop.preflight_prompt_tokens(CHAT_URL, "m", **kwargs)


def test_the_preflight_passes_when_the_engine_counts_what_was_fitted() -> None:
    seen: list = []
    verdict = _preflight(stream_call=_stream(seen=seen), request_seed=11)
    assert verdict["ok"] and verdict["reasons"] == []
    from tre_replayer.engine.preflight import PREFLIGHT_INPUT_TOKENS

    assert verdict["expected_prompt_tokens"] == verdict["prompt_tokens"] == PREFLIGHT_INPUT_TOKENS
    assert verdict["template_overhead"] == 5 and verdict["first_token_field"] == "content"
    url, headers, payload = seen[0]
    assert url == CHAT_URL and headers["routing-strategy"] == "least-gpu-cache"
    assert payload["ignore_eos"] is True and payload["seed"] == 11 and "messages" in payload


@pytest.mark.parametrize("stream,reason", [
    (dict(prompt_tokens=511), "usage.prompt_tokens 511 != 512"),
    (dict(completion_tokens=3), "ignore_eos was not honoured"),
    (dict(first=None), "no chunk carried a token"),
    (dict(status=503), "HTTP 503"),
])
def test_the_preflight_fails_closed(stream, reason) -> None:
    verdict = _preflight(stream_call=_stream(**stream))
    assert not verdict["ok"] and any(reason in r for r in verdict["reasons"]), verdict["reasons"]


def test_the_preflight_refuses_a_request_it_cannot_make() -> None:
    verdict = _preflight(stream_call=_stream(), api="chat")
    assert verdict["ok"]
    bad = openloop.preflight_prompt_tokens("http://gw/v1/completions", "m", api="chat", tokenizer=_FakeTok(),
                                           stream_call=_stream())
    assert not bad["ok"] and "chat/completions" in bad["reasons"][0]

    def boom(*a):
        raise OSError("connection refused")

    assert "connection refused" in _preflight(stream_call=boom)["reasons"][0]


def test_the_preflight_checks_the_mix_share_of_the_content() -> None:
    # _FakeTok counts no Chinese token: a mix prompt of it is 0 % Chinese
    verdict = _preflight(stream_call=_stream(), corpus_lang="mix", zh_ratio=0.5)
    assert not verdict["ok"] and verdict["zh_token_ratio"] == 0.0
    assert any("Chinese token share" in r for r in verdict["reasons"])
    # below 128 tokens one token is more than the tolerance: not checked
    assert _preflight(stream_call=_stream(), corpus_lang="mix", input_tokens=64)["ok"]


def test_the_preflight_cli_writes_one_row_per_model_and_target_and_exits_by_exactness(tmp_path) -> None:
    from scripts import calib_preflight

    out = tmp_path / "pf.jsonl"
    argv = ["--models", "m1,m2", "--models", "m3", "--gateway-url", CHAT_URL, "--out", str(out),
            "--targets", "128,512", "--corpus-lang", "en"]
    toks = {m: _FakeTok() for m in ("m1", "m2", "m3")}
    assert calib_preflight.main(argv, stream_call=_stream(), tokenizers=toks) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["model"], r["target"]) for r in rows] == [(m, t) for m in ("m1", "m2", "m3") for t in (128, 512)]
    for row in rows:
        assert {"model", "target", "prompt_tokens", "completion_tokens", "zh_token_ratio"} <= set(row)
        assert row["prompt_tokens"] == row["target"] and row["completion_tokens"] == 8 and row["ok"]
    # an engine whose count is off for one target (here: 512 whatever was sent): exit 1,
    # and each wrong row says why
    assert calib_preflight.main(argv, stream_call=_stream(prompt_tokens=512), tokenizers=toks) == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["ok"] for r in rows] == [False, True] * 3
    assert "usage.prompt_tokens 512 != 128" in rows[0]["reasons"][0]
    with pytest.raises(SystemExit) as exc:
        calib_preflight.parse_args(["--models", "m", "--gateway-url", "http://gw/v1/completions", "--out", "x"])
    assert exc.value.code == 2
    args = calib_preflight.parse_args(["--models", "m", "--gateway-url", CHAT_URL, "--out", "x",
                                       "--routing-strategy", "none"])
    assert args.routing_strategy is None and args.targets == [512] and args.api == "chat"


def _campaign_args(tmp_path, **over) -> argparse.Namespace:
    args = argparse.Namespace(gateway_url=CHAT_URL, models="m1,m2", corpus_lang="mix", zh_ratio=0.5,
                              routing_strategy="least-gpu-cache", api="chat", request_seed=None,
                              prompt_preflight="refuse")
    for key, value in over.items():
        setattr(args, key, value)
    return args


def test_the_campaign_preflight_checks_every_model_and_refuses_on_one_failure(tmp_path, monkeypatch) -> None:
    calls: list = []

    def fake(url, model, **kw):
        calls.append((url, model, kw))
        n = kw["input_tokens"]
        ok = not (model == "m2" and n == 4096)
        return {"model": model, "api": kw["api"], "expected_prompt_tokens": n,
                "prompt_tokens": n if ok else n - 1, "completion_tokens": 8, "first_token_field": "content",
                "template_overhead": 5, "ok": ok, "reasons": [] if ok else [f"usage.prompt_tokens {n - 1} != {n}"]}

    monkeypatch.setattr(openloop, "preflight_prompt_tokens", fake)
    with pytest.raises(SystemExit, match="m2 @ 4096: usage.prompt_tokens 4095 != 4096"):
        campaign.require_prompt_preflight(_campaign_args(tmp_path), out_dir=tmp_path)
    # every model at the shortest cell input of any calibration shape, 512 and the longest
    assert [(c[1], c[2]["input_tokens"]) for c in calls] == \
        [(m, n) for m in ("m1", "m2") for n in (128, 512, 4096)]
    kw = calls[0][2]
    assert (kw["api"], kw["corpus_lang"], kw["zh_ratio"], kw["routing_strategy"]) == \
        ("chat", "mix", 0.5, "least-gpu-cache")
    doc = json.loads((tmp_path / campaign.PROMPT_PREFLIGHT_FILE).read_text())
    assert set(doc["models"]) == {"m1", "m2"}  # written before refusing: the evidence stays
    assert doc["models"]["m1"]["ok"] and not doc["models"]["m2"]["ok"] and doc["input_lengths"] == [128, 512, 4096]

    calls.clear()
    doc = campaign.require_prompt_preflight(_campaign_args(tmp_path, models="m1"), out_dir=tmp_path,
                                            shapes=["S1", "S3"])
    assert doc["models"]["m1"]["ok"] and [c[2]["input_tokens"] for c in calls] == [256, 512, 2048]
    assert campaign.require_prompt_preflight(_campaign_args(tmp_path, prompt_preflight="skip")) is None


# ------------------------------------------------------------------------ the drivers' CLIs


def test_r3_grid_sends_chat_by_default_and_refuses_what_chat_cannot_carry() -> None:
    args = r3_grid.parse_args(["--model", "m", "--gateway-url", CHAT_URL, "--output", "o.csv"])
    assert (args.api, args.request_seed, args.prompt_preflight) == ("chat", None, "refuse")
    for bad in (["--gateway-url", "http://gw/v1/completions"],
                ["--gateway-url", CHAT_URL, "--prompt-mode", "token_ids"],
                ["--gateway-url", CHAT_URL, "--api", "completions"]):
        with pytest.raises(SystemExit):
            r3_grid.parse_args(["--model", "m", "--output", "o.csv", *bad])
    legacy = r3_grid.parse_args(["--model", "m", "--gateway-url", "http://gw/v1/completions", "--output", "o.csv",
                                 "--api", "completions", "--prompt-mode", "token_ids", "--request-seed", "5"])
    assert (legacy.api, legacy.request_seed) == ("completions", 5)


def test_r3_grid_preflight_refuses_skips_and_passes(monkeypatch) -> None:
    args = r3_grid.parse_args(["--model", "m", "--gateway-url", CHAT_URL, "--output", "o.csv"])
    verdict = {"ok": False, "reasons": ["usage.prompt_tokens 511 != 512"], "expected_prompt_tokens": 512,
               "prompt_tokens": 511, "completion_tokens": 8, "first_token_field": "content",
               "template_overhead": 5}
    seen: list = []
    monkeypatch.setattr(openloop, "preflight_prompt_tokens",
                        lambda url, model, **kw: seen.append(kw) or dict(verdict))
    with pytest.raises(SystemExit, match="511 != 512"):
        r3_grid.run_prompt_preflight(args)
    assert seen[0]["api"] == "chat" and seen[0]["corpus_lang"] == "mix"
    verdict.update(ok=True, reasons=[])
    assert r3_grid.run_prompt_preflight(args)["ok"]
    args.prompt_preflight = "skip"
    assert r3_grid.run_prompt_preflight(args) is None


def test_every_campaign_cell_pins_the_api_and_leaves_the_preflight_to_the_run(tmp_path) -> None:
    cell = campaign.Cell(model="dsqwen-7b", shape="S1", primitive="hold", cell_id="i0_o0_c1", schedule="s.json",
                         duration_s=60.0, capacity_rps=1.0)
    args = _campaign_args(
        tmp_path, raw_dir="/r", window_ms=30000, fit_step_ms=10000, instant_sample_ms=1000,
        model_namespace="default", guard_mode="warn", min_slo_windows=3, out_dir="/o",
        max_model_error_rate=0.01, ttft_slo_ms=500.0, tpot_slo_ms=75.0, registry=None, redis_url=None,
        controller_namespace="ctl", no_capture_extras=True, request_seed=9)
    command = campaign.cell_command(cell, args, Path("s.json"), Path("/o/x.csv"))
    assert command[command.index("--api") + 1] == "chat"
    assert command[command.index("--prompt-preflight") + 1] == "skip"
    assert command[command.index("--request-seed") + 1] == "9"
    # and r3_grid accepts exactly that command line
    parsed = r3_grid.parse_args(command[command.index("--model"):])
    assert (parsed.api, parsed.prompt_preflight, parsed.request_seed) == ("chat", "skip", 9)


def test_the_run_provenance_records_the_api(tmp_path) -> None:
    args = _campaign_args(tmp_path, registry=str(REGISTRY_PATH), window_ms=30000, fit_step_ms=10000,
                          instant_sample_ms=1000, fit_window_align="grid", models="",
                          ttft_slo_ms=500.0, tpot_slo_ms=75.0, request_seed=4)
    prov = campaign.run_provenance(args)
    assert prov["api"]["endpoint"] == "chat" and prov["api"]["path"] == "/v1/chat/completions"
    assert prov["api"]["ignore_eos"] is True and prov["api"]["request_seed"] == 4
    from scripts import prompt_corpus

    assert prompt_corpus.load_path(prov)["api"] == "chat"


def test_the_campaign_cli_refuses_a_gateway_url_of_the_other_api(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc:
        campaign.main(["--out-dir", str(tmp_path), "--gateway-url", "http://gw/v1/completions", "--dry-run"])
    assert exc.value.code == 2


# -------------------------------------------------------------- a cell, end to end, via chat


def test_a_chat_cell_sends_chat_and_records_its_prompt_check(tmp_path, monkeypatch) -> None:
    """r3_grid.run_schedule_cell with --api chat: chat bodies on the chat URL, and the
    guard artifact counts the served requests whose usage.prompt_tokens was off."""
    from tre_common.registry import load_registry

    spec = load_registry(str(REGISTRY_PATH)).model("dsqwen-7b")
    schedule = tmp_path / "S1_hold1090.json"
    schedule.write_text(json.dumps({"dsqwen-7b": [
        {"start_time": 0, "end_time": 0.6, "rps": 40.0, "input_tokens": 64, "max_tokens": 8},
    ]}), encoding="utf-8")
    import threading

    bodies: list = []
    lock = threading.Lock()

    def fake_stream(url, headers, body, timeout):
        from tre_replayer.engine.http_sender import StreamResult

        payload = json.loads(body)
        with lock:
            bodies.append((url, payload))
            off = 1 if len(bodies) == 2 else 0  # one request prefilled a token too many
        return StreamResult(200, 11.0, 40.0, 64 + off, payload["max_tokens"], first_token_field="content")

    real_drive = openloop.drive_cell_schedule

    def drive(*args, **kwargs):
        assert kwargs["api"] == "chat" and kwargs["request_seed"] == 3
        kwargs.update(prompt_dir=None, stream_call=fake_stream)
        return real_drive(*args, **kwargs)

    monkeypatch.setattr(openloop, "drive_cell_schedule", drive)
    monkeypatch.setattr("tre_replayer.engine.http_sender.build_prompt", lambda n, key, **kw: f"text {key}")
    monkeypatch.setattr(openloop, "make_pod_metrics_sampler",
                        lambda endpoints: (lambda now: {"waiting": 0.0, "running": 1.0}))
    monkeypatch.setattr(r3_grid, "template_overhead", lambda args: 5)

    class _Store:
        def read_model_window(self, model, start_ms, end_ms):
            from types import SimpleNamespace

            return SimpleNamespace(ttft_p95_ms=None, tpot_p95_ms=None, e2e_p95_ms=None)

    args = r3_grid.parse_args([
        "--model", "dsqwen-7b", "--gateway-url", CHAT_URL, "--schedule", str(schedule),
        "--cell-id", "i64_o8_c1090", "--output", str(tmp_path / "o" / "x_a1.csv"),
        "--raw-dir", str(tmp_path / "raw"), "--window-ms", "400", "--step-ms", "200",
        "--instant-sample-ms", "100", "--min-latency-samples", "1", "--pod-endpoint", "http://pod/metrics",
        "--registry", str(REGISTRY_PATH), "--guard-mode", "warn", "--request-seed", "3",
        "--ttft-slo-ms", "30", "--tpot-slo-ms", "75", "--min-completed-requests", "1",
    ])
    r3_grid.run_schedule_cell(args, _Store(), spec)
    assert bodies and all(url == CHAT_URL and "messages" in p and p["seed"] == 3 for url, p in bodies)
    guard = json.loads(next((tmp_path / "raw").rglob("*.guard.json")).read_text())
    assert guard["api"] == "chat" and guard["chat_template_overhead"] == 5 and guard["request_seed"] == 3
    check = guard["prompt_tokens_check"]
    assert (check["served"], check["mismatched"], check["max_abs_diff"]) == (len(bodies), 1, 1)
    raw = [json.loads(line) for line in next((tmp_path / "raw").rglob("i64_o8_c1090.jsonl")).read_text().splitlines()]
    assert {r["expected_prompt_tokens"] for r in raw} == {64}
    assert sorted(r["input_tokens"] for r in raw).count(65) == 1


def test_the_preflight_lengths_span_the_runs_cell_inputs() -> None:
    assert campaign.preflight_input_lengths() == [128, 512, 4096]   # every calibration shape
    assert campaign.preflight_input_lengths(["S1"]) == [256, 512]
    assert campaign.preflight_input_lengths(["T9"]) == [300, 512, 2200]      # a sampled range's ends
    assert campaign.preflight_input_lengths(["M"]) == [128, 512, 3072]


def test_every_entry_point_runs_the_prompt_preflight_after_the_clock_check() -> None:
    import inspect

    from scripts import (calibration_acceptance, calibration_ladder, calibration_supplement, calibration_t14,
                         calibration_training_supplement)

    for module in (calibration_acceptance, calibration_ladder, calibration_supplement, calibration_t14,
                   calibration_training_supplement):
        src = inspect.getsource(module)
        clock = src.index("campaign.require_capture_clock_domains(args)")
        assert clock < src.index("campaign.require_prompt_preflight(args", clock), module.__name__
    for fn in (campaign.run_campaign, campaign.run_reprobe):
        assert "require_prompt_preflight(args" in inspect.getsource(fn), fn.__name__


def test_the_campaign_needs_a_gateway_url_from_the_cli_or_the_environment(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv(campaign.GATEWAY_URL_ENV, raising=False)
    with pytest.raises(SystemExit) as exc:
        campaign.main(["--out-dir", str(tmp_path / "o"), "--models", "dsqwen-7b"])
    assert exc.value.code == 2 and campaign.GATEWAY_URL_ENV in capsys.readouterr().err
    # the environment supplies it (and is checked against --api like the flag)
    monkeypatch.setenv(campaign.GATEWAY_URL_ENV, "http://gw/v1/completions")
    with pytest.raises(SystemExit) as exc:
        campaign.main(["--out-dir", str(tmp_path / "o"), "--models", "dsqwen-7b"])
    assert exc.value.code == 2 and "chat/completions" in capsys.readouterr().err
