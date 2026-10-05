"""Arm tool: switch the baseline-scaler arm on / off and mark the replay start.

    python3 -m tre_baselines.tools.arm enable --policy chiron [--dry-run-shell] [--execute]
    python3 -m tre_baselines.tools.arm disable [--collect-dir DIR | --skip-collect] [--execute]
    python3 -m tre_baselines.tools.arm mark-replay --trace PATH --seed N [--execute]

Without ``--execute`` nothing is contacted: the tool prints the commands it would run
(and, for ``enable``, the preconditions it would check). With ``--execute``:

* ``enable`` first checks the preconditions and refuses (exit 2) when they do not hold:
  the TRE controller must be in ``observe`` mode (Redis ``tre:v2:controller:mode``; a
  missing key is observe) and no ``PodAutoscaler`` (the AIBrix APA CR) may target a
  managed model. Then it sets ``TRE_BL_POLICY`` (and ``TRE_BL_DRY_RUN=false`` unless
  ``--dry-run-shell``) on the scaler deployment, scales it to 1 and waits until
  ``/healthz`` answers 200 and, when the shell actuates, the owner lock is held.
* ``disable`` first copies the decision-log directory (``--log-dir``, the pod's
  ``TRE_BL_LOG_DIR``) out of every scaler pod (``kubectl cp <ns>/<pod>:<log-dir>
  <collect-dir>/<pod>``; the logs live on an ``emptyDir`` and die with the pod), then
  scales the deployment to 0 and waits until the owner lock is gone. ``--collect-dir`` is
  required with ``--execute`` unless ``--skip-collect`` says the files may be lost (the
  lines are also in the Redis stream ``tre:v2:bl:decisions``, which additionally holds
  the few ticks between the copy and the shutdown). A failed copy aborts before scaling.
* ``mark-replay`` writes ``tre:v2:bl:replay_t0`` = ``{"t0_ms": <Redis TIME>, "trace_path",
  "seed", "gw_bl_dropped0"}``; the last is the gateway plugins' summed
  ``tre_gateway_bl_req_events_dropped_total`` at that moment (null when unreadable).
* Per-run validity: with a collect dir, ``disable`` also writes ``run_validity.json`` (and
  the raw ``<pod>.metrics.txt`` of the scaler) before scaling down: the gateway's dropped
  request events since the replay marker (``gw_bl_dropped_delta``; any value > 0, a
  negative one (plugin restart) or null makes the run's event evidence suspect), the
  shell's arrivals / non-streaming arrivals per model, SM calls dropped / refused, and the
  policy's per-model counters (``ft_without_arr``, ``unknown_req``, ``unknown_pod``,
  ``tier2_below_t1``, ``empty_window_busy``, ...). Metrics are read through the API server
  proxy (``kubectl get --raw .../pods/<pod>:<port>/proxy/metrics``), read only.

Everything that touches the cluster goes through an injectable ``runner`` (``kubectl``)
and a ``RedisOps`` object, so tests use fakes. Redis is reached directly through
``--redis-url`` / ``TRE_REDIS_URL`` when given, else through ``kubectl exec ... redis-cli``
in the Redis deployment (``--redis-deploy``), like ``deploy/scripts/set_run_mode.sh``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

from tre_baselines.keys import CONTROLLER_MODE_KEY, OWNER_KEY, REPLAY_T0_KEY

POLICIES = ("chiron", "tokenscale", "preserve")
APA_RESOURCE = "podautoscalers.autoscaling.aibrix.ai"

EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2


class ArmError(Exception):
    """A step failed or a precondition does not hold (the message says which)."""

    def __init__(self, message: str, code: int = EXIT_FAIL) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str]], RunResult]


def subprocess_runner(argv: Sequence[str]) -> RunResult:
    proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=300)
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


class RedisOps(Protocol):
    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str) -> None: ...
    def time_ms(self) -> int: ...


class DirectRedis:
    """redis-py over a URL."""

    def __init__(self, url: str) -> None:
        import redis as redis_lib

        self._r = redis_lib.Redis.from_url(url, decode_responses=True, socket_timeout=5.0)

    def get(self, key: str) -> Optional[str]:
        return self._r.get(key)

    def set(self, key: str, value: str) -> None:
        self._r.set(key, value)

    def time_ms(self) -> int:
        seconds, micros = self._r.time()
        return int(seconds) * 1000 + int(micros) // 1000


class KubectlRedis:
    """``kubectl exec deploy/<redis> -- redis-cli --raw ...`` through the runner."""

    def __init__(self, runner: Runner, kubectl: str, namespace: str, deploy: str) -> None:
        self._run = runner
        self._prefix = [kubectl, "-n", namespace, "exec", f"deploy/{deploy}", "--", "redis-cli", "--raw"]

    def _cli(self, *args: str) -> str:
        res = self._run([*self._prefix, *args])
        if res.returncode != 0:
            raise ArmError(f"redis-cli {args[0]} failed: {res.stderr.strip() or res.stdout.strip()}")
        return res.stdout

    def get(self, key: str) -> Optional[str]:
        out = self._cli("GET", key).rstrip("\n")
        return out or None

    def set(self, key: str, value: str) -> None:
        out = self._cli("SET", key, value).strip()
        if out != "OK":
            raise ArmError(f"redis SET {key} answered {out!r}")

    def time_ms(self) -> int:
        lines = self._cli("TIME").split()
        return int(lines[0]) * 1000 + int(lines[1]) // 1000


@dataclass
class Env:
    """Everything the commands touch; tests replace the fakes."""

    runner: Runner
    redis: RedisOps
    kubectl: str = "kubectl"
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    out: Any = None

    def say(self, text: str) -> None:
        print(text, file=self.out or sys.stdout)


def _cmd(argv: Sequence[str]) -> str:
    return shlex.join([str(a) for a in argv])


# ------------------------------------------------------------------ helpers


def _healthz_argv(kubectl: str, ns: str, deploy: str, port: int) -> list[str]:
    code = (
        "import sys,urllib.request;"
        f"sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:{port}/healthz',timeout=3).status==200 else 1)"
    )
    return [kubectl, "-n", ns, "exec", f"deploy/{deploy}", "--", "python3", "-c", code]


def _poll(env: Env, what: str, ok: Callable[[], bool], timeout_s: float, interval_s: float) -> None:
    deadline = env.clock() + timeout_s
    while True:
        try:
            if ok():
                return
        except ArmError:
            pass  # a transient failure while the pod restarts: keep polling
        if env.clock() >= deadline:
            raise ArmError(f"timed out after {timeout_s:g}s waiting for {what}")
        env.sleep(interval_s)


def check_controller_observe(env: Env) -> None:
    mode = env.redis.get(CONTROLLER_MODE_KEY)
    if mode not in (None, "", "observe"):
        raise ArmError(
            f"refusing: the TRE controller mode is {mode!r} ({CONTROLLER_MODE_KEY}); it must be "
            "'observe' while a baseline shell actuates (deploy/scripts/set_run_mode.sh observe active)",
            EXIT_REFUSED,
        )


def check_no_apa(env: Env, models: Optional[Sequence[str]]) -> None:
    res = env.runner([env.kubectl, "get", APA_RESOURCE, "-A", "-o", "json"])
    if res.returncode != 0:
        raise ArmError(f"cannot list PodAutoscalers: {res.stderr.strip() or res.stdout.strip()}")
    try:
        items = json.loads(res.stdout or "{}").get("items") or []
    except ValueError as exc:
        raise ArmError(f"cannot parse the PodAutoscaler list: {exc}") from exc
    wanted = set(models or ())
    hits = []
    for item in items:
        meta = item.get("metadata") or {}
        target = ((item.get("spec") or {}).get("scaleTargetRef") or {}).get("name")
        if not wanted or target in wanted:
            hits.append(f"{meta.get('namespace')}/{meta.get('name')} -> {target}")
    if hits:
        raise ArmError(
            "refusing: PodAutoscaler CRs target the managed models (APA and a baseline shell would "
            "both scale them): " + ", ".join(hits) + " (remove them, e.g. deploy/scripts/toggle_tre_apa.sh)",
            EXIT_REFUSED,
        )


def _step(env: Env, argv: Sequence[str]) -> RunResult:
    res = env.runner(argv)
    if res.returncode != 0:
        raise ArmError(f"command failed ({res.returncode}): {_cmd(argv)}: {res.stderr.strip() or res.stdout.strip()}")
    return res


# ------------------------------------------------------------------ commands


def cmd_enable(env: Env, a: argparse.Namespace) -> int:
    models = [m for m in (a.models or "").split(",") if m.strip()] or None
    dry = bool(a.dry_run_shell)
    set_env = [env.kubectl, "-n", a.namespace, "set", "env", f"deploy/{a.deployment}",
               f"TRE_BL_POLICY={a.policy}", f"TRE_BL_DRY_RUN={'true' if dry else 'false'}"]
    scale = [env.kubectl, "-n", a.namespace, "scale", f"deploy/{a.deployment}", "--replicas=1"]
    rollout = [env.kubectl, "-n", a.namespace, "rollout", "status", f"deploy/{a.deployment}",
               f"--timeout={int(a.timeout_s)}s"]
    healthz = _healthz_argv(env.kubectl, a.namespace, a.deployment, a.http_port)
    if not a.execute:
        env.say("# dry print (nothing executed); add --execute to run")
        env.say(f"# precondition: redis GET {CONTROLLER_MODE_KEY} is 'observe' (or missing)")
        env.say(f"# precondition: {_cmd([env.kubectl, 'get', APA_RESOURCE, '-A', '-o', 'json'])} "
                f"has no scaleTargetRef in {models or 'any model'}")
        for argv in (set_env, scale, rollout, healthz):
            env.say(_cmd(argv))
        if not dry:
            env.say(f"# then wait until redis GET {OWNER_KEY} is non-empty (the shell holds the lock)")
        return EXIT_OK
    check_controller_observe(env)
    check_no_apa(env, models)
    _step(env, set_env)
    _step(env, scale)
    _step(env, rollout)
    _poll(env, "/healthz 200", lambda: env.runner(healthz).returncode == 0, a.timeout_s, a.interval_s)
    if not dry:  # a dry-run shell never takes the owner lock
        _poll(env, f"owner lock {OWNER_KEY}", lambda: bool(env.redis.get(OWNER_KEY)), a.timeout_s, a.interval_s)
    env.say(f"enabled: policy={a.policy} dry_run={str(dry).lower()} deployment={a.namespace}/{a.deployment}")
    return EXIT_OK


def _selector(a: argparse.Namespace) -> str:
    return a.selector or f"app.kubernetes.io/name={a.deployment}"


def _list_pods_argv(env: Env, a: argparse.Namespace) -> list[str]:
    return [env.kubectl, "-n", a.namespace, "get", "pods", "-l", _selector(a),
            "-o", "jsonpath={.items[*].metadata.name}"]


def _cp_argv(env: Env, a: argparse.Namespace, pod: str, dest: str) -> list[str]:
    return [env.kubectl, "cp", f"{a.namespace}/{pod}:{a.log_dir}", dest]


def collect_logs(env: Env, a: argparse.Namespace) -> list[str]:
    """Copy the decision-log directory out of every scaler pod; returns the local dirs."""
    pods = _step(env, _list_pods_argv(env, a)).stdout.split()
    if not pods:
        env.say(f"# no pod matches {_selector(a)} in {a.namespace}: no decision logs to collect")
        return []
    os.makedirs(a.collect_dir, exist_ok=True)
    out = []
    for pod in pods:
        dest = os.path.join(a.collect_dir, pod)
        _step(env, _cp_argv(env, a, pod, dest))
        env.say(f"collected {a.namespace}/{pod}:{a.log_dir} -> {dest}")
        out.append(dest)
    return out


def cmd_disable(env: Env, a: argparse.Namespace) -> int:
    scale = [env.kubectl, "-n", a.namespace, "scale", f"deploy/{a.deployment}", "--replicas=0"]
    if not a.execute:
        env.say("# dry print (nothing executed); add --execute to run")
        if a.skip_collect:
            env.say("# --skip-collect: the decision-log files are NOT copied out (Redis stream only)")
        else:
            env.say(_cmd(_list_pods_argv(env, a)))
            env.say("# for each pod listed:")
            env.say(_cmd(_cp_argv(env, a, "<pod>", os.path.join(a.collect_dir or "<collect-dir>", "<pod>"))))
        env.say(_cmd(scale))
        env.say(f"# then wait until redis GET {OWNER_KEY} is empty (released on shutdown, else after the lock TTL)")
        return EXIT_OK
    if not a.skip_collect:
        if not a.collect_dir:
            raise ArmError("refusing: --collect-dir is required with --execute (the decision logs live on an "
                           "emptyDir and are lost with the pod); pass --skip-collect to scale down without "
                           "copying them", EXIT_REFUSED)
        collect_logs(env, a)
        collect_validity(env, a)
    _step(env, scale)
    _poll(env, f"owner lock {OWNER_KEY} to disappear", lambda: not env.redis.get(OWNER_KEY),
          a.timeout_s, a.interval_s)
    env.say(f"disabled: {a.namespace}/{a.deployment} scaled to 0, owner lock gone")
    return EXIT_OK


def replay_marker(t0_ms: int, trace: str, seed: int, gw_dropped0: Optional[float] = None) -> str:
    return json.dumps({"t0_ms": int(t0_ms), "trace_path": trace, "seed": int(seed),
                       "gw_bl_dropped0": gw_dropped0}, sort_keys=True)


# ------------------------------------------------------------------ per-run validity

GW_DROPPED = "tre_gateway_bl_req_events_dropped_total"


def _proxy_metrics_argv(env: Env, ns: str, pod: str, port: int) -> list[str]:
    return [env.kubectl, "get", "--raw", f"/api/v1/namespaces/{ns}/pods/{pod}:{port}/proxy/metrics"]


def _pods_argv(env: Env, ns: str, selector: str) -> list[str]:
    return [env.kubectl, "-n", ns, "get", "pods", "-l", selector, "-o", "jsonpath={.items[*].metadata.name}"]


def _samples(text: str):
    from tre_baselines.sources import parse_prometheus_text

    return parse_prometheus_text(text)


def gateway_bl_dropped(env: Env, a: argparse.Namespace) -> Optional[float]:
    """Summed ``tre_gateway_bl_req_events_dropped_total`` over the gateway plugin pods;
    None when any pod cannot be read (unknown, not 0)."""
    res = env.runner(_pods_argv(env, a.gw_namespace, a.gw_selector))
    pods = res.stdout.split() if res.returncode == 0 else []
    if not pods:
        return None
    total = 0.0
    for pod in pods:
        res = env.runner(_proxy_metrics_argv(env, a.gw_namespace, pod, a.gw_metrics_port))
        if res.returncode != 0:
            return None
        total += sum(s.value for s in _samples(res.stdout) if s.name == GW_DROPPED)
    return total


def run_validity(shell_metrics: dict[str, str], marker: Optional[dict], gw_now: Optional[float]) -> dict:
    """The per-run validity record from the scaler pods' ``/metrics`` and the gateway."""
    by_model: dict[str, dict[str, float]] = {}
    policy_events: dict[str, dict[str, float]] = {}
    totals: dict[str, float] = {}
    for text in shell_metrics.values():
        for smp in _samples(text):
            model = smp.labels.get("model")
            if smp.name in ("tre_bl_arrivals_total", "tre_bl_nonstream_arrivals_total") and model:
                key = smp.name[len("tre_bl_"):-len("_total")]
                by_model.setdefault(model, {})[key] = by_model.get(model, {}).get(key, 0) + smp.value
            elif smp.name == "tre_bl_policy_events_total" and model:
                name = smp.labels.get("name", "?")
                policy_events.setdefault(model, {})[name] = policy_events.get(model, {}).get(name, 0) + smp.value
            elif smp.name in ("tre_bl_sm_calls_total", "tre_bl_sm_failures_total", "tre_bl_sm_dropped_total"):
                totals[smp.name] = totals.get(smp.name, 0) + smp.value
    gw0 = (marker or {}).get("gw_bl_dropped0")
    delta = None if gw0 is None or gw_now is None else gw_now - float(gw0)
    return {
        "replay_marker": marker,
        "gw_bl_dropped_t0": gw0,
        "gw_bl_dropped_end": gw_now,
        "gw_bl_dropped_delta": delta,
        "events_valid": delta == 0,
        "arrivals": by_model,
        "policy_events": policy_events,
        "sm": totals,
    }


def collect_validity(env: Env, a: argparse.Namespace) -> str:
    """Write ``run_validity.json`` (+ raw scaler metrics) into the collect dir."""
    pods = _step(env, _list_pods_argv(env, a)).stdout.split()
    os.makedirs(a.collect_dir, exist_ok=True)
    metrics: dict[str, str] = {}
    for pod in pods:
        res = env.runner(_proxy_metrics_argv(env, a.namespace, pod, a.http_port))
        if res.returncode == 0:
            metrics[pod] = res.stdout
            with open(os.path.join(a.collect_dir, f"{pod}.metrics.txt"), "w", encoding="utf-8") as fh:
                fh.write(res.stdout)
        else:
            env.say(f"# warning: cannot read /metrics of {pod}: {res.stderr.strip()}")
    raw = env.redis.get(REPLAY_T0_KEY)
    try:
        marker = json.loads(raw) if raw else None
    except ValueError:
        marker = None
    doc = run_validity(metrics, marker, gateway_bl_dropped(env, a))
    path = os.path.join(a.collect_dir, "run_validity.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    env.say(f"run validity -> {path} (gw_bl_dropped_delta={doc['gw_bl_dropped_delta']})")
    return path


def cmd_mark_replay(env: Env, a: argparse.Namespace) -> int:
    if not a.execute:
        env.say("# dry print (nothing executed); add --execute to run")
        env.say(f"# gateway dropped events now: {_cmd(_pods_argv(env, a.gw_namespace, a.gw_selector))}, then "
                f"{_cmd(_proxy_metrics_argv(env, a.gw_namespace, '<pod>', a.gw_metrics_port))}")
        env.say(f"# t0_ms = Redis TIME (ms) at execution; SET {REPLAY_T0_KEY} to:")
        env.say(replay_marker(0, a.trace, a.seed).replace('"t0_ms": 0', '"t0_ms": <redis TIME ms>'))
        return EXIT_OK
    gw0 = gateway_bl_dropped(env, a)
    if gw0 is None:
        env.say("# warning: gateway dropped-event counter unreadable; the run's event validity is unknown")
    t0 = env.redis.time_ms()
    value = replay_marker(t0, a.trace, a.seed, gw0)
    env.redis.set(REPLAY_T0_KEY, value)
    env.say(f"{REPLAY_T0_KEY} = {value}")
    return EXIT_OK


# ------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tre_baselines.tools.arm", description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--namespace", default="tre-v2", help="namespace of the scaler deployment / Redis")
    common.add_argument("--deployment", default="tre-v2-baseline-scaler")
    common.add_argument("--kubectl", default=os.environ.get("KUBECTL", "kubectl"))
    common.add_argument("--redis-url", default=os.environ.get("TRE_REDIS_URL"),
                    help="direct Redis URL (default env TRE_REDIS_URL; else kubectl exec into --redis-deploy)")
    common.add_argument("--redis-deploy", default="tre-v2-redis")
    common.add_argument("--execute", action="store_true", help="run the commands (default: print only)")
    common.add_argument("--gw-namespace", default="tre-v2", help="namespace of the gateway plugin pods")
    common.add_argument("--gw-selector", default="app=tre-gateway-plugins", help="label selector of the plugins")
    common.add_argument("--gw-metrics-port", type=int, default=8080, help="the plugins' /metrics port")
    common.add_argument("--timeout-s", type=float, default=120.0)
    common.add_argument("--interval-s", type=float, default=2.0)
    sub = ap.add_subparsers(dest="command", required=True)
    en = sub.add_parser("enable", parents=[common], help="set the policy, scale the shell to 1, wait until healthy")
    en.add_argument("--policy", required=True, choices=POLICIES)
    en.add_argument("--dry-run-shell", action="store_true", help="shell only logs (TRE_BL_DRY_RUN=true)")
    en.add_argument("--models", default="", help="comma list of managed models for the APA check (default: any APA CR)")
    en.add_argument("--http-port", type=int, default=8080)
    dis = sub.add_parser("disable", parents=[common],
                         help="copy the decision logs out, scale the shell to 0, wait until the owner lock is gone")
    dis.add_argument("--collect-dir", default=None,
                     help="local directory receiving <pod>/ copies of the decision logs (required with --execute)")
    dis.add_argument("--skip-collect", action="store_true",
                     help="scale down without copying the decision logs (they remain in the Redis stream)")
    dis.add_argument("--log-dir", default="/var/log/tre-baselines", help="TRE_BL_LOG_DIR inside the pod")
    dis.add_argument("--http-port", type=int, default=8080, help="the scaler's /metrics port")
    dis.add_argument("--selector", default=None,
                     help="label selector of the scaler pods (default app.kubernetes.io/name=<deployment>)")
    mk = sub.add_parser("mark-replay", parents=[common], help=f"write {REPLAY_T0_KEY}")
    mk.add_argument("--trace", required=True)
    mk.add_argument("--seed", required=True, type=int)
    return ap


def main(argv: Optional[Sequence[str]] = None, *, runner: Optional[Runner] = None,
         redis: Optional[RedisOps] = None, sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic, out: Any = None) -> int:
    a = build_parser().parse_args(argv)
    run = runner or subprocess_runner
    if redis is None:
        if not a.execute:
            redis = None  # never contacted in print mode
        elif a.redis_url:
            redis = DirectRedis(a.redis_url)
        else:
            redis = KubectlRedis(run, a.kubectl, a.namespace, a.redis_deploy)
    env = Env(runner=run, redis=redis, kubectl=a.kubectl, sleep=sleep, clock=clock, out=out)  # type: ignore[arg-type]
    handler = {"enable": cmd_enable, "disable": cmd_disable, "mark-replay": cmd_mark_replay}[a.command]
    try:
        return handler(env, a)
    except ArmError as exc:
        print(f"arm: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
