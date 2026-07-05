from scripts import n4b_three_model_precheck as precheck
from scripts.n4b_three_model_precheck import parse_args, parse_models, run_precheck, summarize_latencies


def test_parse_models_trims_and_drops_empty_items() -> None:
    assert parse_models(" dsqwen-7b, ,dsllama-8b ") == ("dsqwen-7b", "dsllama-8b")


def test_parse_args_accepts_runtime_overrides() -> None:
    args = parse_args(
        [
            "--gateway-url",
            "http://gateway.example/v1/completions",
            "--service-manager-url",
            "http://sm.example:8000",
            "--models",
            "a,b",
            "--duration-seconds",
            "12",
            "--phase-seconds",
            "3",
            "--workers",
            "2",
            "--baseline-workers-per-model",
            "1",
            "--max-tokens",
            "16",
            "--sample-seconds",
            "4",
            "--request-timeout",
            "9",
        ]
    )

    assert args.gateway_url == "http://gateway.example/v1/completions"
    assert args.service_manager_url == "http://sm.example:8000"
    assert args.models == ("a", "b")
    assert args.duration_seconds == 12
    assert args.phase_seconds == 3
    assert args.workers == 2
    assert args.baseline_workers_per_model == 1
    assert args.max_tokens == 16
    assert args.sample_seconds == 4
    assert args.request_timeout == 9


def test_summarize_latencies_reports_nearest_rank_p95() -> None:
    assert summarize_latencies([1.0, 2.0, 3.0, 100.0]) == {
        "count": 4,
        "min": 1.0,
        "avg": 26.5,
        "p95": 100.0,
        "max": 100.0,
    }


def test_run_precheck_returns_result_without_live_load(monkeypatch) -> None:
    monkeypatch.setattr(precheck, "pod_restarts", lambda namespace, selector=None: {namespace: 0})
    monkeypatch.setattr(precheck, "http_json", lambda method, url, payload=None, headers=None, timeout=10: {"url": url})
    monkeypatch.setattr(precheck, "rss_kb", lambda namespace, selector: 123)
    monkeypatch.setattr(precheck, "redis_dbsize", lambda: 7)
    args = parse_args(
        [
            "--models",
            "a,b",
            "--duration-seconds",
            "0",
            "--workers",
            "0",
        ]
    )

    result = run_precheck(args)

    assert result["models"] == ["a", "b"]
    assert result["baseline_workers_per_model"] == 0
    assert result["initial_state"] == {"url": "http://10.111.21.116:8000/v2/state"}
    assert result["final_state"] == {"url": "http://10.111.21.116:8000/v2/state"}
    assert result["errors"] == {}
    assert result["latency_ms"]["a"]["count"] == 0
