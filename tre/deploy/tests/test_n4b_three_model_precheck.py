from scripts.n4b_three_model_precheck import parse_args, parse_models, summarize_latencies


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
