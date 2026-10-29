"""Drain before sleep (PreServe §4.3.2): invariants of the shell's hide -> drained -> sleep.

* a shrink hides; a pod with work in flight is never slept before the drain bound;
* a drained pod is slept by name (never a count shrink that could pick a busy pod);
* a scale-up brings draining pods back before it wakes one;
* the shell only drains (and on stop unhides) pods it hid itself.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from bl_fakes import FakeRedis, limits, make_config, wait_until
from tre_baselines.drain import DrainTracker
from tre_baselines.loop import BaselineShell, OwnerLock
from tre_baselines.policies.base import Decision
from tre_baselines.sm_client import Call, Dispatcher, SMResult
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot

MODELS = {"m": limits("m", 1, 4)}


@dataclass
class PodCluster:
    """Pods of one model: awake / hidden flags and running + waiting per pod."""

    pods: dict  # name -> {"awake": bool, "hidden": bool, "queued": int}
    log: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_hidden(self, model, *, add=(), remove=()):
        with self.lock:
            self.log.append(("routable", tuple(add), tuple(remove)))
            for name in add:
                if self.pods[name]["awake"]:
                    self.pods[name]["hidden"] = True
            for name in remove:
                self.pods[name]["hidden"] = False
        return SMResult(ok=True, code=200, raw={"model": model})

    def post_plan(self, body):
        out = {"plan_id": body.get("plan_id"), "status": "done", "sleeps": [], "wakes": []}
        with self.lock:
            for item in body.get("sleeps") or ():
                self.log.append(("sleep", dict(item)))
                assert "pods" in item, "a draining shell never sends a count shrink"
                for name in item["pods"]:
                    self.pods[name].update(awake=False, hidden=False)
                out["sleeps"].append({"model": item["model"], "status": "done", "taken": len(item["pods"])})
            for item in body.get("wakes") or ():
                self.log.append(("wake", dict(item)))
                awake = [n for n, p in self.pods.items() if p["awake"]]
                for name in sorted(n for n, p in self.pods.items() if not p["awake"]):
                    if len(awake) >= item["at_least"]:
                        break
                    self.pods[name].update(awake=True, hidden=False, queued=0)
                    awake.append(name)
                out["wakes"].append({"model": item["model"], "status": "done", "unfilled": 0, "refusals": []})
        return SMResult(ok=True, code=200, raw=out)

    def snapshot(self, now_ms, tick, tick_s):
        def pod(name, p):
            return PodSnapshot(pod=name, model="m", node="n", gpu_ids=(0,), running=float(p["queued"]),
                               waiting=0.0, kv_usage=0.1, counters={}, num_gpu_blocks=None, block_size=None,
                               scraped_at_ms=now_ms)
        with self.lock:
            awake = {n: p for n, p in self.pods.items() if p["awake"]}
            lim = MODELS["m"]
            ms = ModelSnapshot(
                model="m", awake=len(awake), min_replicas=lim.min_replicas, max_replicas=lim.max_replicas,
                gpus_per_replica=1, ttft_slo_ms=500.0, tpot_slo_ms=75.0, max_num_seqs=256,
                pods=tuple(pod(n, p) for n, p in sorted(awake.items()) if not p["hidden"]), events=(),
                hidden=tuple(pod(n, p) for n, p in sorted(awake.items()) if p["hidden"]))
        return ClusterSnapshot(now_ms=now_ms, tick_s=tick_s, models={"m": ms}, tick=tick)


class Source:
    def __init__(self, cluster, redis, tick_s):
        self.cluster, self.redis, self.tick_s = cluster, redis, tick_s

    def gather(self, tick=0):
        self.redis.advance(2000)
        return self.cluster.snapshot(self.redis.now_ms, tick, self.tick_s)


class Want:
    """Desired count per tick (sticky), a policy that drains."""

    name = "scripted"
    drain_on_shrink = True
    evidence_backstop = False

    def __init__(self, script):
        self.script, self.want = script, None

    def decide(self, snap):
        self.want = self.script.get(snap.tick, self.want)
        return {} if self.want is None else {"m": Decision(desired=self.want, reason="want", inputs={})}


def _shell(tmp_path, pods, script, **cfg):
    cluster = PodCluster({n: dict(awake=a, hidden=h, queued=q) for n, (a, h, q) in pods.items()})
    config = make_config(tmp_path, MODELS, dry_run=False, **cfg)
    redis = FakeRedis()
    dispatcher = Dispatcher(cluster.post_plan, update_hidden=cluster.update_hidden)
    shell = BaselineShell(config, Source(cluster, redis, config.tick_s), Want(script), dispatcher, redis,
                          lock=OwnerLock(redis, 30.0, token="me"))
    return shell, cluster, dispatcher


def _tick(shell, dispatcher):
    lines = shell.tick_once()
    assert wait_until(lambda: dispatcher.inflight_count() == 0)
    return {line["model"]: line for line in lines}


def test_shrink_hides_least_loaded_and_sleeps_only_once_drained(tmp_path) -> None:
    shell, cluster, dispatcher = _shell(
        tmp_path, {"m-0": (True, False, 5), "m-1": (True, False, 2), "m-2": (True, False, 9)}, {0: 2})
    line = _tick(shell, dispatcher)["m"]
    assert line["action"] == "hide" and line["inputs"]["drain"]["hide"] == ["m-1"]
    assert cluster.pods["m-1"]["hidden"] and cluster.pods["m-1"]["awake"]
    for _ in range(3):  # still busy: never slept
        assert _tick(shell, dispatcher)["m"]["action"] == "none"
    assert cluster.pods["m-1"]["awake"]
    cluster.pods["m-1"]["queued"] = 0
    line = _tick(shell, dispatcher)["m"]
    assert line["action"] == "down" and line["inputs"]["drain"]["drained"] == ["m-1"]
    assert not cluster.pods["m-1"]["awake"]
    assert [e for e in cluster.log if e[0] == "sleep"] == [
        ("sleep", {"model": "m", "pods": ["m-1"], "sleep_path": "urgent"})]
    assert shell.policy_counters()["m"]["drain_drained"] == 1


def test_scale_up_brings_draining_pod_back_before_waking(tmp_path) -> None:
    shell, cluster, dispatcher = _shell(
        tmp_path, {"m-0": (True, False, 5), "m-1": (True, False, 2), "m-2": (False, False, 0)},
        {0: 1, 2: 2, 4: 3})
    assert _tick(shell, dispatcher)["m"]["action"] == "hide"
    _tick(shell, dispatcher)
    line = _tick(shell, dispatcher)["m"]          # 2 wanted, m-1 still draining: unhide, no wake
    assert line["action"] == "unhide" and not cluster.pods["m-1"]["hidden"]
    assert not any(e[0] == "wake" for e in cluster.log)
    _tick(shell, dispatcher)
    assert _tick(shell, dispatcher)["m"]["action"] == "up"   # 3 wanted: nothing draining, wake
    assert cluster.pods["m-2"]["awake"]


def test_up_past_the_draining_pods_unhides_them_before_the_wake(tmp_path) -> None:
    shell, cluster, dispatcher = _shell(
        tmp_path, {"m-0": (True, False, 5), "m-1": (True, False, 2), "m-2": (False, False, 0)}, {0: 1, 2: 3})
    _tick(shell, dispatcher)
    _tick(shell, dispatcher)
    line = _tick(shell, dispatcher)["m"]
    assert line["action"] == "up" and line["target"] == 3
    kinds = [e[0] for e in cluster.log]
    assert kinds.index("wake") > max(i for i, k in enumerate(kinds) if k == "routable")
    assert all(p["awake"] and not p["hidden"] for p in cluster.pods.values())


def test_drain_bound_sleeps_a_busy_pod(tmp_path) -> None:
    shell, cluster, dispatcher = _shell(tmp_path, {"m-0": (True, False, 5), "m-1": (True, False, 2)}, {0: 1},
                                        drain_max_s=7.0)
    _tick(shell, dispatcher)
    actions = [_tick(shell, dispatcher)["m"]["action"] for _ in range(5)]
    assert actions.count("down") == 1 and not cluster.pods["m-1"]["awake"]
    assert shell.policy_counters()["m"]["drain_timeout"] == 1


def test_routable_failure_keeps_the_model_out_of_the_plan() -> None:
    plans = []
    disp = Dispatcher(lambda body: plans.append(body) or SMResult(ok=True, code=200, raw={}),
                      update_hidden=lambda model, add=(), remove=(): SMResult(ok=False, code=409, error="floor"))
    disp.dispatch({"m": Call("up", 3, 1, unhide_first=("m-1",))})
    assert wait_until(lambda: disp.inflight_count() == 0)
    [done] = disp.drain_results()
    assert not done.result.ok and done.result.error == "floor" and plans == []


def test_only_own_or_adopted_pods_are_drained_and_released(tmp_path) -> None:
    # m-9 is hidden before the shell's first owner tick: adopted (a restarted shell resumes).
    shell, cluster, dispatcher = _shell(
        tmp_path, {"m-0": (True, False, 1), "m-1": (True, False, 1), "m-9": (True, True, 3)}, {0: 2, 1: 1})
    line = _tick(shell, dispatcher)["m"]
    assert line["action"] == "none" and "m-9" in line["inputs"]["drain"]["draining"]
    # A pod hidden by someone else after adoption is never drained or released by us.
    cluster.pods["m-1"]["hidden"] = True
    line = _tick(shell, dispatcher)["m"]
    assert "m-1" not in line["inputs"]["drain"]["draining"]
    released = shell.release_drains(cluster.update_hidden)
    assert released["m"]["pods"] == ["m-9"]
    assert not cluster.pods["m-9"]["hidden"] and cluster.pods["m-1"]["hidden"]


def test_tracker_holds_without_a_scraped_pod() -> None:
    ms = ModelSnapshot(model="m", awake=2, min_replicas=1, max_replicas=4, gpus_per_replica=1,
                       ttft_slo_ms=500.0, tpot_slo_ms=75.0, max_num_seqs=None, pods=(), events=(),
                       unscraped=("m-0", "m-1"))
    step = DrainTracker().step(ms, 1, 0)
    assert step.direction == "none" and step.info["hold"] == "no_scraped_pod"
