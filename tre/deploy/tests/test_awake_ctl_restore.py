"""awake_ctl.py restore-ids against a fake SM that enforces one awake binding per GPU (409 like
the real SM). Invariants checked after every power call: no GPU holds two awake bindings, no model
that had an awake replica drops to zero."""
import importlib.util
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "release" / "awake_ctl.py"
_spec = importlib.util.spec_from_file_location("awake_ctl_under_test", _P)
awake_ctl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(awake_ctl)

N9, N10 = "nscc-ds-4a100-node9", "nscc-ds-4a100-node10"
CANON = {f"dsqwen-7b/{N9}/0", f"dsllama-8b/{N9}/1", f"dsqwen-14b/{N10}/0,1"}


def binding(model, node, gpu_ids, awake):
    bid = f"{model}/{node}/{','.join(map(str, gpu_ids))}"
    return {"binding_id": bid, "serve_id": bid, "model": model, "node": node, "gpu_ids": list(gpu_ids),
            "awake": awake, "hidden": False}


class FakeSM:
    def __init__(self, bindings):
        self.b = {x["binding_id"]: x for x in bindings}
        self.models = {x["model"] for x in bindings if x["awake"]}

    def state(self):
        return [dict(x) for x in self.b.values()]

    def power(self, serve_id, awake, dry):
        x = self.b[serve_id]
        if awake and not x["awake"]:
            used = {(c["node"], g) for c in self.b.values() if c["awake"] for g in c["gpu_ids"]}
            if any((x["node"], g) in used for g in x["gpu_ids"]):
                raise SystemExit(1)  # 409 slot_occupied
        x["awake"] = awake
        used = [(c["node"], g) for c in self.b.values() if c["awake"] for g in c["gpu_ids"]]
        assert len(used) == len(set(used)), "two awake bindings on one GPU"
        for m in self.models:
            assert any(c["model"] == m and c["awake"] for c in self.b.values()), f"{m} dropped to zero awake"

    def awake(self):
        return {k for k, v in self.b.items() if v["awake"]}


def fleet(awake, single_gpu_slots):
    bs = [binding(m, n, [g], f"{m}/{n}/{g}" in awake) for m in ("dsqwen-7b", "dsllama-8b") for n, g in single_gpu_slots]
    bs += [binding("dsqwen-14b", n, g, f"dsqwen-14b/{n}/{g[0]},{g[1]}" in awake)
           for n in (N9, N10) for g in ((0, 1), (2, 3))]
    return FakeSM(bs)


def run(monkeypatch, sm):
    monkeypatch.setattr(awake_ctl, "state", sm.state)
    monkeypatch.setattr(awake_ctl, "power", sm.power)
    awake_ctl.restore(set(CANON), False)


SWAPPED = {f"dsllama-8b/{N9}/0", f"dsqwen-7b/{N9}/1", f"dsqwen-14b/{N10}/0,1"}


def test_swap_resolves_via_free_gpu(monkeypatch):
    sm = fleet(SWAPPED, [(n, g) for n in (N9, N10) for g in range(4)])
    run(monkeypatch, sm)
    assert sm.awake() == CANON


def test_swap_without_free_slot_fails_clearly(monkeypatch):
    sm = fleet(SWAPPED, [(N9, 0), (N9, 1)])  # 7b / 8b have no binding off the two target GPUs
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, sm)
    assert e.value.code == 3
    assert sm.awake() == SWAPPED
