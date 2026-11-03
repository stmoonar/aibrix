"""Export trained torch mLSTMs to the numpy ``.npz`` format read by
``tre_baselines.policies.preserve_predictor`` (v1: ``tre_preserve.predictor``)."""

from __future__ import annotations

import os
from typing import Mapping, Optional, Tuple

import numpy as np

from tre_baselines.policies.preserve_predictor import FORMAT_VERSION, KINDS, MINMAX, NORMALIZATIONS, PER_SEQUENCE_MEAN

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
               window_s: float, normalization: str = PER_SEQUENCE_MEAN,
               minmax: Optional[Mapping[str, Tuple[float, float]]] = None,
               meta: Optional[Mapping[str, str]] = None) -> None:
    """Write {"prompt": mLSTMModel, "response": mLSTMModel} to ``path``; ``window_s`` = the training window
    (aibrix 2026-11-03: the policy refuses weights of another window). ``normalization`` the models were
    trained with; ``minmax`` = {kind: (min, max)} of the training series for ``minmax`` (2026-11-04, stored as
    ``<kind>.norm_min`` / ``<kind>.norm_max``); ``meta`` = optional scalar strings (``model``, ``trace``,
    ``variant``)."""
    if set(models) != set(KINDS):
        raise ValueError(f"need exactly the models {KINDS}, got {sorted(models)}")
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    if (normalization == MINMAX) != (minmax is not None) or (minmax is not None and set(minmax) != set(KINDS)):
        raise ValueError("minmax constants for every kind go with (and only with) normalization minmax")
    arrays = {
        "format_version": np.array(FORMAT_VERSION),
        "seq_len": np.array(int(seq_len)),
        "hidden": np.array(int(hidden)),
        "eps": np.array(float(eps)),
        "normalization": np.array(normalization),
        "kinds": np.array(list(KINDS)),
        "window_s": np.array(float(window_s)),
    }
    for k, v in (meta or {}).items():
        if k in arrays or "." in k:
            raise ValueError(f"meta key {k!r} clashes with the file layout")
        arrays[k] = np.array(str(v))
    for kind, (mn, mx) in (minmax or {}).items():
        arrays[f"{kind}.norm_min"] = np.array(float(mn))
        arrays[f"{kind}.norm_max"] = np.array(float(mx))
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
