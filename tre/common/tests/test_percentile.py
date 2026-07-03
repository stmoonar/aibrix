from tre_common.percentile import histogram_percentile


def test_histogram_percentile_supports_bucket_upper_and_interpolated_modes():
    buckets = [(100.0, 1.0), (200.0, 3.0), (300.0, 6.0)]

    assert histogram_percentile(buckets, 0.75, mode="bucket_upper") == 300.0
    assert histogram_percentile(buckets, 0.75, mode="interpolated") == 250.0


def test_histogram_percentile_returns_none_for_empty_or_zero_histograms():
    assert histogram_percentile([], 0.95) is None
    assert histogram_percentile([(100.0, 0.0)], 0.95) is None


def test_histogram_percentile_rejects_invalid_quantile():
    try:
        histogram_percentile([(100.0, 1.0)], 1.5)
    except ValueError as exc:
        assert "quantile" in str(exc)
