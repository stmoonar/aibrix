"""Export trained torch mLSTMs to the numpy ``.npz`` format read by
``tre_baselines.policies.preserve_predictor`` (v1: ``tre_preserve.predictor``)."""

from __future__ import annotations

import os
from typing import Mapping

import numpy as np

from tre_baselines.policies.preserve_predictor import FORMAT_VERSION, KINDS, PER_SEQUENCE_MEAN

# state_dict key (mLSTMModel) -> exported name
_KEYS = {
    "mLSTM.cells.0.W_ih.weight": "W_ih.weight",
    "mLSTM.cells.0.W_ih.bias": "W_ih.bias",
    "mLSTM.cells.0.W_hh.weight": "W_hh.weight",
    "mLSTM.cells.0.W_hh.bias": "W_hh.bias",
    "mLSTM.cells.0.W_m.weight": "W_m.weight",   # unused by upstream's forward (kept for completeness)
    "mLSTM.cells.0.W_m.bias": "W_m.bias",
    "mLSTM.cells.0.U_m.weight": "U_m.weight",
    "mLSTM.cells.0.U_m.bias": "U_m.bias",
    "fc.weight": "fc.weight",
    "fc.bias": "fc.bias",
}


def export_npz(models: Mapping[str, object], path: str, seq_len: int, hidden: int, eps: float,
               window_s: float) -> None:
    """Write {"prompt": mLSTMModel, "response": mLSTMModel} (per-sequence normalized) to ``path``;
    ``window_s`` = the training window (aibrix 2026-11-03: the policy refuses weights of another window)."""
    if set(models) != set(KINDS):
        raise ValueError(f"need exactly the models {KINDS}, got {sorted(models)}")
    arrays = {
        "format_version": np.array(FORMAT_VERSION),
        "seq_len": np.array(int(seq_len)),
        "hidden": np.array(int(hidden)),
        "eps": np.array(float(eps)),
        "normalization": np.array(PER_SEQUENCE_MEAN),
        "kinds": np.array(list(KINDS)),
        "window_s": np.array(float(window_s)),
    }
    for kind, model in models.items():
        sd = model.state_dict()
        missing = set(_KEYS) - set(sd)
        if missing:
            raise ValueError(f"{kind}: state_dict lacks {sorted(missing)} (not a 1-layer mLSTMModel?)")
        for src, dst in _KEYS.items():
            arrays[f"{kind}.{dst}"] = sd[src].detach().cpu().numpy().astype(np.float32)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
