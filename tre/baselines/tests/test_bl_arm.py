"""Arm tool: command sequences, precondition refusals, print mode has no side effects."""
from __future__ import annotations

import io
import json
import shlex

from tre_baselines.keys import OWNER_KEY, REPLAY_T0_KEY
from tre_baselines.tools import arm
from tre_baselines.tools.arm import CONTROLLER_MODE_KEY, RunResult

APA_NONE = json.dumps({"items": []})
APA_7B = json.dumps({"items": [{"metadata": {"namespace": "default", "name": "dsqwen-7b-apa"},
                                "spec": {"scaleTargetRef": {"name": "dsqwen-7b"}}}]})
DEPLOY = "deploy/tre-v2-baseline-scaler"


class FakeRedis:
    def __init__(self, kv=None, now_ms=1_700_000_000_123):
        self.kv = dict(kv or {})
        self.now_ms = now_ms
        self.calls = []

    def get(self, key):
        self.calls.append(("get", key))
        return self.kv.get(key)

    def set(self, key, value):
        self.calls.append(("set", key, value))
        self.kv[key] = value

    def time_ms(self):
        self.calls.append(("time",))
        return self.now_ms


class FakeRunner:
    """Records argv; ``apa`` is the PodAutoscaler list JSON; healthz answers after N tries."""

    def __init__(self, apa=APA_NONE, healthz_after=0, fail_on=None, redis=None, owner_on_healthy=True,
                 pods=("tre-v2-baseline-scaler-abc12",)):
        self.pods = list(pods)
        self.calls = []
        self.apa = apa
        self.healthz_after = healthz_after
        self.healthz_tries = 0
        self.fail_on = fail_on
        self.redis = redis
        self.owner_on_healthy = owner_on_healthy

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if self.fail_on and self.fail_on in argv:
            return RunResult(1, "", "boom")
        if argv[1:3] == ["get", arm.APA_RESOURCE]:
            return RunResult(0, self.apa)
        if argv[3:5] == ["get", "pods"]:
            return RunResult(0, " ".join(self.pods))
        if "python3" in argv and "-c" in argv:
            self.healthz_tries += 1
            ok = self.healthz_tries > self.healthz_after
            if ok and self.redis is not None and self.owner_on_healthy:
                self.redis.kv[OWNER_KEY] = "tok"
            return RunResult(0 if ok else 1)
        if "--replicas=0" in argv and self.redis is not None:
            self.redis.kv.pop(OWNER_KEY, None)
        return RunResult(0)


def run(argv, runner, redis, **kw):
    out = io.StringIO()
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    code = arm.main(argv, runner=runner, redis=redis, sleep=sleep, clock=lambda: clock["t"], out=out, **kw)
    return code, out.getvalue()


def test_print_mode_has_no_side_effects_and_never_touches_redis() -> None:
    for argv in (["enable", "--policy", "chiron"], ["enable", "--policy", "preserve", "--dry-run-shell"],
                 ["disable"], ["mark-replay", "--trace", "t/a.json", "--seed", "3"]):
        runner, redis = FakeRunner(), FakeRedis()
        code, out = run(argv, runner, redis)
        assert code == 0 and runner.calls == [] and redis.calls == [] and redis.kv == {}
        assert "kubectl" in out or "SET" in out


def test_enable_print_lists_the_commands() -> None:
    _, out = run(["enable", "--policy", "tokenscale"], FakeRunner(), FakeRedis())
    lines = [ln for ln in out.splitlines() if not ln.startswith("#")]
    assert "TRE_BL_POLICY=tokenscale" in lines[0] and "TRE_BL_DRY_RUN=false" in lines[0]
    assert shlex.split(lines[1])[-2:] == [DEPLOY, "--replicas=1"]
    assert "owner" in out  # the wait on the lock is described
    _, out = run(["enable", "--policy", "chiron", "--dry-run-shell", "--namespace", "ns1"], FakeRunner(), FakeRedis())
    assert "TRE_BL_DRY_RUN=true" in out and "-n ns1" in out and "wait until redis GET" not in out


def test_enable_execute_sequence_and_waits() -> None:
    redis = FakeRedis({CONTROLLER_MODE_KEY: "observe"})
    runner = FakeRunner(healthz_after=2, redis=redis)
    code, out = run(["enable", "--policy", "preserve", "--execute"], runner, redis)
    assert code == 0, out
    assert runner.calls[0][1] == "get"  # read-only APA check first
    assert runner.calls[1][3:6] == ["set", "env", DEPLOY]
    assert "TRE_BL_POLICY=preserve" in runner.calls[1] and "TRE_BL_DRY_RUN=false" in runner.calls[1]
    assert runner.calls[2][3:] == ["scale", DEPLOY, "--replicas=1"]
    assert runner.calls[3][3] == "rollout"
    assert runner.healthz_tries == 3  # polled until 200
    assert ("get", OWNER_KEY) in redis.calls and "enabled" in out


def test_enable_missing_controller_mode_counts_as_observe() -> None:
    redis = FakeRedis()
    code, _ = run(["enable", "--policy", "chiron", "--execute"], FakeRunner(redis=redis), redis)
    assert code == 0


def test_enable_refuses_when_controller_is_active_and_does_nothing() -> None:
    redis = FakeRedis({CONTROLLER_MODE_KEY: "active"})
    runner = FakeRunner()
    code, _ = run(["enable", "--policy", "chiron", "--execute"], runner, redis)
    assert code == arm.EXIT_REFUSED and runner.calls == []
    assert not any(c[0] == "set" for c in redis.calls)


def test_enable_refuses_with_apa_cr_for_a_managed_model() -> None:
    redis = FakeRedis({CONTROLLER_MODE_KEY: "observe"})
    runner = FakeRunner(apa=APA_7B)
    code, _ = run(["enable", "--policy", "chiron", "--execute"], runner, redis)
    assert code == arm.EXIT_REFUSED
    assert all(c[1] == "get" for c in runner.calls)  # only the read-only listing ran
    # an APA CR of another model is fine when --models restricts the check
    runner = FakeRunner(apa=APA_7B, redis=redis)
    code, _ = run(["enable", "--policy", "chiron", "--execute", "--models", "dsllama-8b"], runner, redis)
    assert code == 0
    code, _ = run(["enable", "--policy", "chiron", "--execute", "--models", "dsqwen-7b"], FakeRunner(apa=APA_7B), redis)
    assert code == arm.EXIT_REFUSED


def test_enable_dry_run_shell_skips_the_owner_wait() -> None:
    redis = FakeRedis({CONTROLLER_MODE_KEY: "observe"})
    runner = FakeRunner(redis=redis, owner_on_healthy=False)
    code, _ = run(["enable", "--policy", "chiron", "--dry-run-shell", "--execute"], runner, redis)
    assert code == 0 and "TRE_BL_DRY_RUN=true" in runner.calls[1]
    assert ("get", OWNER_KEY) not in redis.calls


def test_enable_times_out_and_reports_a_failing_step() -> None:
    redis = FakeRedis()
    code, _ = run(["enable", "--policy", "chiron", "--execute", "--timeout-s", "5"],
                  FakeRunner(healthz_after=10**6, redis=redis), redis)
    assert code == arm.EXIT_FAIL
    code, _ = run(["enable", "--policy", "chiron", "--execute", "--timeout-s", "5"],
                  FakeRunner(redis=redis, owner_on_healthy=False), redis)  # never gets the lock
    assert code == arm.EXIT_FAIL
    runner = FakeRunner(fail_on="scale", redis=redis)
    code, _ = run(["enable", "--policy", "chiron", "--execute"], runner, redis)
    assert code == arm.EXIT_FAIL and not any("rollout" in c for c in runner.calls)


def test_disable_scales_to_zero_and_waits_for_the_lock_to_go(tmp_path) -> None:
    redis = FakeRedis({OWNER_KEY: "tok"})
    runner = FakeRunner(redis=redis, pods=())
    code, out = run(["disable", "--execute", "--collect-dir", str(tmp_path / "c")], runner, redis)
    assert code == 0 and runner.calls[-1][3:] == ["scale", DEPLOY, "--replicas=0"]
    assert OWNER_KEY not in redis.kv and "disabled" in out and "no decision logs" in out
    redis = FakeRedis({OWNER_KEY: "tok"})  # a lock that never goes away
    code, _ = run(["disable", "--execute", "--skip-collect", "--timeout-s", "4"], FakeRunner(), redis)
    assert code == arm.EXIT_FAIL


def test_disable_copies_the_decision_logs_out_before_scaling(tmp_path) -> None:
    redis = FakeRedis({OWNER_KEY: "tok"})
    runner = FakeRunner(redis=redis, pods=("scaler-a", "scaler-b"))
    dest = tmp_path / "evidence"
    code, out = run(["disable", "--execute", "--collect-dir", str(dest), "--namespace", "ns1"], runner, redis)
    assert code == 0, out
    get_pods, cp_a, cp_b, scale = runner.calls
    assert get_pods[:6] == ["kubectl", "-n", "ns1", "get", "pods", "-l"]
    assert get_pods[6] == "app.kubernetes.io/name=tre-v2-baseline-scaler"
    assert cp_a == ["kubectl", "cp", "ns1/scaler-a:/var/log/tre-baselines", str(dest / "scaler-a")]
    assert cp_b[2:] == ["ns1/scaler-b:/var/log/tre-baselines", str(dest / "scaler-b")]
    assert scale[3:] == ["scale", DEPLOY, "--replicas=0"] and dest.is_dir()
    # a failed copy aborts before the scale-down: the pod (and its logs) stays
    runner = FakeRunner(redis=FakeRedis({OWNER_KEY: "tok"}), fail_on="cp")
    code, _ = run(["disable", "--execute", "--collect-dir", str(dest)], runner, runner.redis)
    assert code == arm.EXIT_FAIL and not any("--replicas=0" in c for c in runner.calls)


def test_disable_execute_requires_a_collect_dir() -> None:
    runner, redis = FakeRunner(), FakeRedis({OWNER_KEY: "tok"})
    code, _ = run(["disable", "--execute"], runner, redis)
    assert code == arm.EXIT_REFUSED and runner.calls == []
    # print mode lists the copy, with a placeholder when no directory is given
    code, out = run(["disable", "--log-dir", "/logs", "--selector", "app=x"], FakeRunner(), FakeRedis())
    assert code == 0 and "tre-v2/<pod>:/logs" in out and "-l app=x" in out


def test_mark_replay_writes_the_marker_from_the_redis_clock() -> None:
    redis = FakeRedis(now_ms=1_700_000_000_123)
    code, _ = run(["mark-replay", "--trace", "traces/e1/trace.json", "--seed", "11", "--execute"],
                  FakeRunner(), redis)
    assert code == 0
    doc = json.loads(redis.kv[REPLAY_T0_KEY])
    assert doc == {"t0_ms": 1_700_000_000_123, "trace_path": "traces/e1/trace.json", "seed": 11}
    _, out = run(["mark-replay", "--trace", "x.json", "--seed", "1"], FakeRunner(), FakeRedis())
    assert REPLAY_T0_KEY in out and "<redis TIME ms>" in out


def test_kubectl_redis_backend_uses_redis_cli_exec() -> None:
    calls = []

    def runner(argv):
        calls.append(list(argv))
        cmd = argv[argv.index("--raw") + 1]
        return RunResult(0, {"GET": "observe\n", "TIME": "1700000000\n123456\n", "SET": "OK\n"}[cmd])

    r = arm.KubectlRedis(runner, "kubectl", "ns9", "redis-x")
    assert r.get("k") == "observe" and r.time_ms() == 1_700_000_000_123
    r.set("k", '{"a": 1}')
    assert calls[0][:8] == ["kubectl", "-n", "ns9", "exec", "deploy/redis-x", "--", "redis-cli", "--raw"]
    assert calls[2][-3:] == ["SET", "k", '{"a": 1}']
