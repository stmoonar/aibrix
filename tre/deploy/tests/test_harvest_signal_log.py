import csv

from deploy.scripts.analysis.harvest_signal_log import SIGNAL_LOG_FIELDS, write_signal_csv


def test_harvest_signal_log_writes_fixed_schema_csv(tmp_path):
    fields = {field.encode(): f"value-{field}".encode() for field in SIGNAL_LOG_FIELDS}
    output = tmp_path / "timeline_signals.csv"

    assert write_signal_csv([(b"1000-0", fields)], output) == 1

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == SIGNAL_LOG_FIELDS
    assert rows[0]["model"] == "value-model"
    assert rows[0]["action"] == "value-action"
