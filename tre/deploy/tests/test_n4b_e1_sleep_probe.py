from scripts.n4b_e1_sleep_probe import parse_gpu_memory, percentile95, summarize_used_mib


def test_parse_gpu_memory_groups_rows_by_uuid() -> None:
    text = """GPU-a, 111, 10
GPU-b, 222, 20 MiB
bad row
GPU-a, 333, 30
"""

    parsed = parse_gpu_memory(text)

    assert parsed == {
        "GPU-a": [{"pid": "111", "used_mib": 10}, {"pid": "333", "used_mib": 30}],
        "GPU-b": [{"pid": "222", "used_mib": 20}],
    }


def test_summarize_used_mib_reports_total_for_selected_uuid() -> None:
    rows = {
        "GPU-a": [{"pid": "111", "used_mib": 10}, {"pid": "333", "used_mib": 30}],
        "GPU-b": [{"pid": "222", "used_mib": 20}],
    }

    assert summarize_used_mib(rows, "GPU-a") == {"processes": 2, "used_mib": 40}
    assert summarize_used_mib(rows, "missing") == {"processes": 0, "used_mib": 0}


def test_percentile95_uses_nearest_rank() -> None:
    assert percentile95([1.0, 2.0, 3.0, 100.0]) == 100.0
    assert percentile95([3.0]) == 3.0
    assert percentile95([]) is None
