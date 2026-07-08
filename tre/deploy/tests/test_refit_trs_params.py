from __future__ import annotations

import csv
import json

from scripts import refit_trs_params


# Latency SLOs used when writing/reading the synthetic CSVs.
_TTFT_SLO = 100.0
_TPOT_SLO = 50.0


def _write_csv(path, rows) -> None:
    fieldnames = [
        "scenario_id",
        "scenario_family",
        "trs",
        "prompt_tokens_total",
        "generation_tokens_total",
        "avg_waiting",
        "avg_running",
        "avg_swapping",
        "p95_ttft",
        "p95_tpot",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _adopt_rows() -> list[dict]:
    """10 ranked windows where only w_p=0.10 recovers the true health order.

    total_tokens = prompt*w_p + generation with prompt=100*r (true signal, rising) and
    generation=1000-9*r (a confounder falling with r). So total = 1000 + r*(100*w_p - 9):
    the slope in r is positive only at w_p=0.10 (100*0.10-9=+1) and negative for every
    smaller grid w_p (incl. inherited 0.08 -> -1). queue_raw is held at 1 (avg_running=1,
    waiting=swapping=0) so lambda_wait/qmin do not move the signal. Latency (ttft) falls
    with r, so higher r is genuinely healthier: at w_p=0.10 trs tracks health
    (spearman=+1, auroc=1); at w_p=0.08 it inverts (spearman=-1, auroc=0).
    """
    rows: list[dict] = []
    for r in range(10):
        ttft = 150.0 - 12.0 * r  # r>=5 -> ratio<=1 (slo_met), monotonically healthier
        rows.append(
            {
                "scenario_id": f"cell-{r}",
                "scenario_family": "sweep",
                "trs": 1.0,  # ignored by the refit; present so the row passes filtering
                "prompt_tokens_total": 100.0 * r,
                "generation_tokens_total": 1000.0 - 9.0 * r,
                "avg_waiting": 0.0,
                "avg_running": 1.0,
                "avg_swapping": 0.0,
                "p95_ttft": ttft,
                "p95_tpot": 10.0,  # ratio 0.2, never gates slo_met/health
            }
        )
    return rows


def _keep_rows() -> list[dict]:
    """10 ranked windows where trs tracks health for every grid w_p.

    generation=0 and prompt=100*(r+1) so total=prompt*w_p rises with r for ALL w_p>0:
    spearman=+1 and auroc=1 everywhere, so best and inherited score identically and the
    recommendation must be keep_inherited.
    """
    rows: list[dict] = []
    for r in range(10):
        ttft = 150.0 - 12.0 * r
        rows.append(
            {
                "scenario_id": f"cell-{r}",
                "scenario_family": "sweep",
                "trs": 1.0,
                "prompt_tokens_total": 100.0 * (r + 1),
                "generation_tokens_total": 0.0,
                "avg_waiting": 0.0,
                "avg_running": 1.0,
                "avg_swapping": 0.0,
                "p95_ttft": ttft,
                "p95_tpot": 10.0,
            }
        )
    return rows


def _load(path):
    return refit_trs_params.load_windows_and_inputs(
        path, latency_slo_ms={"ttft_p95": _TTFT_SLO, "tpot_p95": _TPOT_SLO}
    )


def test_adopt_refit_when_refit_beats_inherited(tmp_path) -> None:
    src = tmp_path / "adopt.csv"
    _write_csv(src, _adopt_rows())
    windows, inputs = _load(src)
    assert len(windows) == len(inputs) == 10
    assert sum(1 for w in windows if w.slo_met) == 5  # r>=5

    report = refit_trs_params.build_report(
        windows,
        inputs,
        model_name="dsqwen-7b",
        inherited_w_p=0.08,
        inherited_lambda_wait=1.875,
        inherited_qmin=1.0,
        generated_at="2026-07-08T00:00:00+00:00",
    )

    # Inherited w_p=0.08 inverts the ranking; the grid best at w_p=0.10 recovers it.
    assert report["inherited"]["auroc"] == 0.0
    assert report["inherited"]["spearman_health"] == -1.0
    assert report["best"]["w_p"] == 0.1
    assert report["best"]["auroc"] == 1.0
    assert report["best"]["spearman_health"] == 1.0
    assert report["comparison"]["auroc_exceeds"] is True
    assert report["comparison"]["spearman_exceeds"] is True
    assert report["recommendation"] == "adopt_refit"


def test_keep_inherited_when_within_threshold(tmp_path) -> None:
    src = tmp_path / "keep.csv"
    _write_csv(src, _keep_rows())
    windows, inputs = _load(src)

    report = refit_trs_params.build_report(
        windows,
        inputs,
        model_name="dsqwen-7b",
        inherited_w_p=0.08,
        inherited_lambda_wait=1.875,
        inherited_qmin=1.0,
        generated_at="2026-07-08T00:00:00+00:00",
    )

    # Every grid point separates perfectly, so best == inherited in score.
    assert report["inherited"]["auroc"] == 1.0
    assert report["inherited"]["spearman_health"] == 1.0
    assert report["best"]["auroc"] == 1.0
    assert report["best"]["spearman_health"] == 1.0
    assert report["comparison"]["auroc_delta"] == 0.0
    assert report["comparison"]["spearman_delta"] == 0.0
    assert report["comparison"]["auroc_exceeds"] is False
    assert report["comparison"]["spearman_exceeds"] is False
    assert report["recommendation"] == "keep_inherited"


def test_report_schema_and_grid(tmp_path) -> None:
    src = tmp_path / "keep.csv"
    _write_csv(src, _keep_rows())
    windows, inputs = _load(src)

    report = refit_trs_params.build_report(
        windows,
        inputs,
        model_name="dsqwen-7b",
        inherited_w_p=0.08,
        inherited_lambda_wait=1.875,
        inherited_qmin=1.0,
        inherited_source="registry",
        generated_at="2026-07-08T00:00:00+00:00",
    )

    assert set(report) == {
        "generated_at",
        "model_name",
        "signal_column",
        "slo",
        "window",
        "grid",
        "inherited",
        "best",
        "top5",
        "comparison",
        "recommendation",
    }
    # Default grid is doc15 s2's grid: 5 * 5 * 1 = 25 candidates.
    assert report["grid"]["w_p_candidates"] == [0.02, 0.04, 0.06, 0.08, 0.10]
    assert report["grid"]["lambda_wait_candidates"] == [1.5, 1.875, 2.25, 2.625, 3.0]
    assert report["grid"]["qmin_candidates"] == [1.0]
    assert report["grid"]["candidate_count"] == 25
    assert len(report["top5"]) == 5
    assert report["inherited"]["source"] == "registry"
    assert report["window"] == {"count": 10, "healthy": 5, "violation": 5}
    for key in ("w_p", "lambda_wait", "qmin", "objective", "auroc", "spearman_health"):
        assert key in report["best"]
        assert key in report["top5"][0]
    assert report["comparison"]["threshold_pct"] == 5.0


def test_relative_pct_undefined_when_inherited_zero() -> None:
    # inherited auroc ~0 -> percent undefined (null) but a real improvement still exceeds.
    assert refit_trs_params._relative_pct(0.5, 0.0) is None
    assert refit_trs_params._metric_exceeds(0.5, None, 5.0) is True
    assert refit_trs_params._metric_exceeds(0.0, None, 5.0) is False
    # defined percent uses the threshold directly.
    assert refit_trs_params._metric_exceeds(0.02, 2.0, 5.0) is False
    assert refit_trs_params._metric_exceeds(0.10, 10.0, 5.0) is True


def test_resolve_inherited_from_cli_short_circuits_registry() -> None:
    triple, source = refit_trs_params.resolve_inherited(
        "does-not-exist", w_p=0.03, lambda_wait=2.0, qmin=1.0
    )
    assert triple == (0.03, 2.0, 1.0)
    assert source == "cli"


def test_resolve_inherited_reads_registry_defaults() -> None:
    # Falls back to the repo registry.yaml for the model's current values.
    triple, source = refit_trs_params.resolve_inherited(
        "dsqwen-7b", w_p=None, lambda_wait=None, qmin=None
    )
    assert triple == (0.02, 3.0, 1.0)
    assert source == "registry"
    triple14, _ = refit_trs_params.resolve_inherited(
        "dsqwen-14b", w_p=None, lambda_wait=None, qmin=None
    )
    assert triple14 == (0.0575, 3.0, 1.0)


def test_cli_writes_report_from_csv(tmp_path) -> None:
    src = tmp_path / "adopt.csv"
    out = tmp_path / "report.json"
    _write_csv(src, _adopt_rows())

    rc = refit_trs_params.main(
        [
            "--input", str(src),
            "--output", str(out),
            "--model-name", "dsqwen-7b",
            "--ttft-p95-ms", str(_TTFT_SLO),
            "--tpot-p95-ms", str(_TPOT_SLO),
            "--inherited-w-p", "0.08",
            "--inherited-lambda-wait", "1.875",
            "--inherited-qmin", "1.0",
            "--generated-at", "2026-07-08T00:00:00+00:00",
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["model_name"] == "dsqwen-7b"
    assert report["inherited"]["source"] == "cli"
    assert report["best"]["w_p"] == 0.1
    assert report["recommendation"] == "adopt_refit"
    assert report["slo"] == {"ttft_p95": _TTFT_SLO, "tpot_p95": _TPOT_SLO}
