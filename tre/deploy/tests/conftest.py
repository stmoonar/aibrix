"""Keep the deploy tests off any cluster: the run provenance reads images with kubectl."""
from __future__ import annotations

import pytest


def _no_kubectl(argv, **_kw):
    raise FileNotFoundError("kubectl is not called from the unit tests")


@pytest.fixture(autouse=True)
def _no_cluster_in_run_provenance(monkeypatch):
    monkeypatch.setattr("scripts.calibration_campaign.IMAGE_PROVENANCE_RUN", _no_kubectl)
