#!/usr/bin/env python3
"""Serial TRE/APA/ablation campaign runner with a fail-closed hygiene gate."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from scripts.analysis.harvest_signal_log import write_signal_csv


DEFAULT_BASELINE = {
    "dsllama-8b": "dsllama-8b-nscc-ds-4a100-node9-gpu-1-5cb98fdbb6-mr5b7",
    "dsqwen-7b": "dsqwen-7b-nscc-ds-4a100-node9-gpu-0-546d5d9f88-f94nf",
    "dsqwen-14b": "dsqwen-14b-nscc-ds-4a100-node9-gpu-2-3-69c86d8db7-vnxtl",
}
GATEWAYS = {
    "tre": "http://192.168.223.76:31094/v1/completions",
    "apa": "http://192.168.223.76:31592/v1/completions",
}
SIGNAL_ARMS = {"zm", "queue_len", "decode_tps", "prefill_tps"}
ALLOWED_ARMS = {"tre", "apa", *SIGNAL_ARMS}
ORPHAN_KEY = "tre:v2:controller:alerts:hidden_orphans"
ORPHAN_WATCH_KEY = "tre:v2:controller:orphan_watch"
PROBES_KEY = "tre:v2:controller:safescale:probes"
SIGNAL_KEY = "tre:v2:controller:signal_log"
MODE_KEY = "tre:v2:controller:mode"
SM_STATE_KEY = "tre:v2:sm:state"
SM_VERSION_KEY = "tre:v2:sm:version"
FIXED_CLEAR_KEYS = {
    "tre:v2:decision:latest",
    "tre:v2:controller:signal_log",
    "tre:v2:controller:safescale:probes",
    "tre:v2:controller:orphan_watch",
    "tre:v2:controller:alerts:hidden_orphans",
    "tre:v2:profile:events",
}
LAYOUT_FIELDS = (
    "ts", "version", "model", "awake_count", "awake_serve_ids", "hidden_serve_ids"
)
ACTUAL_ACTION_FIELDS = ("ts", "model", "action", "serve_id")


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    trace: str
    arm: str
    seed: int


@dataclass(frozen=True)
class Manifest:
    frozen_sha: str
    params_hash: str
    images: dict[str, str]
    baseline: dict[str, str]
    cooldown_s: float
    post_drain_s: float
    runs: tuple[RunSpec, ...]


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    signal_source: str
    disable_eta_gate: bool
    mode: str
    gateway: str
    apa_enabled: bool


def utc_iso(epoch: float | None = None) -> str:
    value = time.time() if epoch is None else epoch
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> Manifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = tuple(
        RunSpec(str(item["id"]), str(item["trace"]), str(item["arm"]), int(item["seed"]))
        for item in raw["runs"]
    )
    ids = [run.run_id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("campaign manifest has duplicate run IDs")
    invalid_arms = sorted({run.arm for run in runs} - ALLOWED_ARMS)
    if invalid_arms:
        raise ValueError(f"unsupported campaign arms: {invalid_arms}")
    baseline = {str(key): str(value) for key, value in raw.get("baseline", DEFAULT_BASELINE).items()}
    if set(baseline) != set(DEFAULT_BASELINE):
        raise ValueError("baseline must name exactly the three registered models")
    images = {str(key): str(value) for key, value in raw["images"].items()}
    if set(images) != {"controller", "service-manager", "ui"}:
        raise ValueError("images must pin controller, service-manager, and ui")
    frozen_sha = str(raw["frozen_sha"])
    if len(frozen_sha) != 40:
        raise ValueError("frozen_sha must be a full 40-character SHA")
    return Manifest(
        frozen_sha=frozen_sha,
        params_hash=str(raw["params_hash"]),
        images=images,
        baseline=baseline,
        cooldown_s=float(raw.get("cooldown_s", 600.0)),
        post_drain_s=float(raw.get("post_drain_s", 30.0)),
        runs=runs,
    )


def arm_config(arm: str) -> ArmConfig:
    if arm == "tre":
        return ArmConfig(arm, "zm", False, "active", GATEWAYS["tre"], False)
    if arm == "apa":
        # APA actuates. TRE loops remain enabled for counterfactual logging, while
        # observe mode makes TRE ActionQueue dispatch impossible.
        return ArmConfig(arm, "zm", False, "observe", GATEWAYS["apa"], True)
    if arm in SIGNAL_ARMS:
        return ArmConfig(arm, arm, True, "active", GATEWAYS["tre"], False)
    raise ValueError(f"unsupported arm: {arm}")


def baseline_errors(state: dict[str, Any], baseline: dict[str, str]) -> list[str]:
    by_id = {item["serve_id"]: item for item in state["bindings"]}
    errors = []
    for model, serve_id in baseline.items():
        binding = by_id.get(serve_id)
        if binding is None:
            errors.append(f"missing baseline binding {serve_id}")
        elif binding["model"] != model:
            errors.append(f"baseline binding {serve_id} belongs to {binding['model']}")
        elif not binding["awake"] or binding["hidden"]:
            errors.append(f"baseline binding {serve_id} is not awake+routable")
    expected = set(baseline.values())
    unexpected = sorted(
        item["serve_id"] for item in state["bindings"]
        if item["awake"] and item["serve_id"] not in expected
    )
    if unexpected:
        errors.append(f"unexpected awake bindings: {unexpected}")
    hidden = sorted(item["serve_id"] for item in state["bindings"] if item["hidden"])
    if hidden:
        errors.append(f"hidden bindings: {hidden}")
    return errors


def pod_is_ready(pod: dict[str, Any]) -> bool:
    if pod.get("metadata", {}).get("deletionTimestamp"):
        return False
    if pod.get("status", {}).get("phase") != "Running":
        return False
    return any(
        item.get("type") == "Ready" and item.get("status") == "True"
        for item in pod.get("status", {}).get("conditions", [])
    )


def redis_keys_to_clear(all_keys: Iterable[str]) -> set[str]:
    selected = set(FIXED_CLEAR_KEYS)
    for key in all_keys:
        if key.startswith("tre:v2:decision:hist:"):
            selected.add(key)
        if key.startswith("tre:v2:controller:safescale:probe:") and key.endswith(":journal"):
            selected.add(key)
    forbidden = {SM_STATE_KEY, SM_VERSION_KEY} & selected
    if forbidden:
        raise ValueError(f"refusing to clear service-manager truth keys: {sorted(forbidden)}")
    return selected


def layout_rows(state: dict[str, Any], epoch: float) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[str]]] = {}
    for binding in state["bindings"]:
        bucket = grouped.setdefault(binding["model"], {"awake": [], "hidden": []})
        if binding["awake"]:
            bucket["awake"].append(binding["serve_id"])
        if binding["hidden"]:
            bucket["hidden"].append(binding["serve_id"])
    return [
        {
            "ts": utc_iso(epoch),
            "version": state["version"],
            "model": model,
            "awake_count": len(values["awake"]),
            "awake_serve_ids": ";".join(sorted(values["awake"])),
            "hidden_serve_ids": ";".join(sorted(values["hidden"])),
        }
        for model, values in sorted(grouped.items())
    ]


def derive_actual_actions(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    previous_awake: dict[str, set[str]] = {}
    previous_hidden: dict[str, set[str]] = {}
    for row in rows:
        model = str(row["model"])
        awake = {value for value in str(row["awake_serve_ids"]).split(";") if value}
        hidden = {value for value in str(row["hidden_serve_ids"]).split(";") if value}
        if model in previous_awake:
            changes = (
                (previous_awake[model] - awake, "sleep"),
                (awake - previous_awake[model], "wake"),
                (hidden - previous_hidden[model], "hide"),
                (previous_hidden[model] - hidden, "unhide"),
            )
            for serve_ids, action in changes:
                for serve_id in sorted(serve_ids):
                    actions.append(
                        {"ts": str(row["ts"]), "model": model, "action": action, "serve_id": serve_id}
                    )
        previous_awake[model] = awake
        previous_hidden[model] = hidden
    return actions


def parse_controller_decisions(
    text: str, *, start_ms: int, end_ms: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = []
    actions = []
    for line in text.splitlines():
        try:
            outer = json.loads(line)
            payload = json.loads(outer.get("message", "{}"))
            ts_ms = int(payload.get("ts_ms", -1))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("event") != "trs_calc_result" or not start_ms <= ts_ms <= end_ms:
            continue
        for key in ("actions", "events", "model_states"):
            if isinstance(payload.get(key), str):
                try:
                    payload[key] = json.loads(payload[key])
                except json.JSONDecodeError:
                    pass
        decisions.append(payload)
        for action in payload.get("actions", []):
            actions.append({"ts_ms": ts_ms, "loop": payload.get("loop"), **action})
    return decisions, actions


def request_health(path: Path) -> dict[str, Any]:
    total = errors_5xx = status_zero = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            total += 1
            status = int(json.loads(line).get("http_status") or 0)
            errors_5xx += status >= 500
            status_zero += status == 0
    return {
        "requests": total,
        "http_5xx": errors_5xx,
        "http_5xx_frac": errors_5xx / total if total else 0.0,
        "status_zero": status_zero,
        "status_zero_frac": status_zero / total if total else 0.0,
    }


def deterministic_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as raw, destination.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            shutil.copyfileobj(raw, compressed)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


class LayoutSampler:
    def __init__(self, state_getter, interval_s: float = 0.5) -> None:
        self._state_getter = state_getter
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                self.rows.extend(layout_rows(self._state_getter(), started))
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(max(0.0, self._interval_s - (time.time() - started)))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        try:
            self.rows.extend(layout_rows(self._state_getter(), time.time()))
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{type(exc).__name__}: {exc}")
class CampaignRunner:
    def __init__(self, *, repo: Path, manifest: Manifest, out_root: Path) -> None:
        self.repo = repo.resolve()
        self.git_root = self.repo.parent
        self.manifest = manifest
        self.out_root = out_root.resolve()
        self.sm_url = self._service_url("tre-v2-service-manager", 8000)
        self.ui_url = "http://127.0.0.1:30812"
        self.redis_url = self._redis_url()
        import redis

        self.redis = redis.Redis.from_url(self.redis_url)

    def _command(
        self, command: Sequence[str], *, stdout=None, stderr=None, check: bool = True
    ):
        return subprocess.run(
            list(command), cwd=self.repo, check=check, text=True,
            stdout=stdout, stderr=stderr,
        )

    def _service_url(self, name: str, port: int) -> str:
        cluster_ip = subprocess.check_output(
            ["kubectl", "-n", "tre-v2", "get", "svc", name,
             "-o", "jsonpath={.spec.clusterIP}"], text=True
        ).strip()
        return f"http://{cluster_ip}:{port}"

    def _redis_url(self) -> str:
        cluster_ip = subprocess.check_output(
            ["kubectl", "-n", "tre-v2", "get", "svc", "tre-v2-redis",
             "-o", "jsonpath={.spec.clusterIP}"], text=True
        ).strip()
        return f"redis://{cluster_ip}:6379/0"

    def http_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(method, url, timeout=kwargs.pop("timeout", 30), **kwargs)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        return self.http_json("GET", f"{self.sm_url}/v2/state")

    def set_mode(self, mode: str) -> None:
        if mode not in {"active", "observe"}:
            raise ValueError(mode)
        self.redis.set(MODE_KEY, mode)
        if self.redis.get(MODE_KEY).decode() != mode:
            raise RuntimeError(f"failed to set controller mode {mode}")

    def toggle(self, arm: str) -> None:
        self._command(["bash", "deploy/scripts/toggle_tre_apa.sh", arm])

    def kubectl_json(self, args: Sequence[str]) -> dict[str, Any]:
        output = subprocess.check_output(["kubectl", *args, "-o", "json"], text=True)
        return json.loads(output)

    def controller_env(self) -> dict[str, str]:
        deployment = self.kubectl_json(
            ["-n", "tre-v2", "get", "deploy", "tre-v2-controller"]
        )
        return {
            item["name"]: str(item.get("value", ""))
            for item in deployment["spec"]["template"]["spec"]["containers"][0].get("env", [])
        }

    def configure_controller(self, config: ArmConfig) -> None:
        values = [
            "ENABLE_TRE_SCALING=true",
            f"TRE_SIGNAL_SOURCE={config.signal_source}",
            f"TRE_DISABLE_ETA_GATE={'true' if config.disable_eta_gate else 'false'}",
        ]
        self._command(
            ["kubectl", "-n", "tre-v2", "set", "env", "deploy/tre-v2-controller", *values]
        )
        self._command(
            ["kubectl", "-n", "tre-v2", "rollout", "restart", "deploy/tre-v2-controller"]
        )
        self._command(
            ["kubectl", "-n", "tre-v2", "rollout", "status", "deploy/tre-v2-controller", "--timeout=180s"]
        )

    def apa_count(self) -> int:
        result = subprocess.run(
            ["kubectl", "-n", "default", "get",
             "podautoscalers.autoscaling.aibrix.ai",
             "-l", "tre.aibrix.io/baseline=apa", "-o", "name"],
            text=True, capture_output=True,
        )
        if result.returncode != 0:
            return 0
        return len([line for line in result.stdout.splitlines() if line.strip()])

    def wait_apa_ready(self, timeout_s: float = 180.0) -> None:
        names = ("dsqwen-7b-apa", "dsllama-8b-apa", "dsqwen-14b-apa")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready = 0
            for name in names:
                result = subprocess.run(
                    ["kubectl", "-n", "default", "get",
                     "podautoscalers.autoscaling.aibrix.ai", name,
                     "-o", "jsonpath={.status.conditions[?(@.type==\"AbleToScale\")].status}"],
                    text=True, capture_output=True,
                )
                ready += result.returncode == 0 and result.stdout.strip() == "True"
            if ready == len(names):
                return
            time.sleep(5)
        raise TimeoutError("APA did not reach AbleToScale=True for all three models")

    def ensure_baseline(self) -> dict[str, Any]:
        state = self.state()
        hidden = [item["serve_id"] for item in state["bindings"] if item["hidden"]]
        if hidden:
            raise RuntimeError(f"refusing to mask hidden bindings during reset: {hidden}")
        targets = set(self.manifest.baseline.values())
        for binding in state["bindings"]:
            if binding["awake"] and binding["serve_id"] not in targets:
                self.http_json(
                    "PUT", f"{self.sm_url}/v2/bindings/{binding['serve_id']}/power",
                    json={"awake": False}, timeout=60,
                )
        state = self.state()
        by_id = {item["serve_id"]: item for item in state["bindings"]}
        for model, serve_id in self.manifest.baseline.items():
            binding = by_id.get(serve_id)
            if binding is None or binding["model"] != model:
                raise RuntimeError(f"missing exact baseline binding {model}={serve_id}")
            if not binding["awake"]:
                self.http_json(
                    "PUT", f"{self.sm_url}/v2/bindings/{serve_id}/power",
                    json={"awake": True}, timeout=60,
                )
        final = self.state()
        errors = baseline_errors(final, self.manifest.baseline)
        if errors:
            raise RuntimeError("baseline reset failed: " + "; ".join(errors))
        return final

    def guard_zero(self, *, include_probes: bool = True) -> None:
        values = {
            "orphan_alerts": self.redis.hlen(ORPHAN_KEY),
            "orphan_watch": self.redis.hlen(ORPHAN_WATCH_KEY),
        }
        if include_probes:
            values["safescale_probes"] = self.redis.hlen(PROBES_KEY)
        nonzero = {key: value for key, value in values.items() if value}
        if nonzero:
            raise RuntimeError(f"controller guard hashes are nonzero: {nonzero}")

    def clear_per_run_redis(self) -> dict[str, Any]:
        self.guard_zero(include_probes=False)
        state_before = self.redis.hgetall(SM_STATE_KEY)
        version_before = self.redis.get(SM_VERSION_KEY)
        all_keys = {
            key.decode() if isinstance(key, bytes) else str(key)
            for key in self.redis.scan_iter(match="tre:v2:*")
        }
        selected = redis_keys_to_clear(all_keys)
        if selected:
            self.redis.delete(*sorted(selected))
        if (
            self.redis.hgetall(SM_STATE_KEY) != state_before
            or self.redis.get(SM_VERSION_KEY) != version_before
        ):
            raise RuntimeError("service-manager state changed while clearing per-run Redis")
        self.guard_zero()
        return {
            "cleared_keys": sorted(selected),
            "preserved_sm_state_fields": len(state_before),
            "preserved_sm_version": version_before.decode() if version_before else None,
        }

    def safescale_snapshot(self) -> dict[str, Any]:
        decode = lambda value: value.decode() if isinstance(value, bytes) else str(value)
        probes = {
            decode(key): decode(value)
            for key, value in self.redis.hgetall(PROBES_KEY).items()
        }
        journals = {}
        for raw_key in self.redis.scan_iter(
            match="tre:v2:controller:safescale:probe:*:journal"
        ):
            key = decode(raw_key)
            journals[key] = [decode(value) for value in self.redis.lrange(key, 0, -1)]
        return {"probes": probes, "journals": journals}

    def assert_ready(self) -> None:
        state = self.state()
        model_pods = [
            self.kubectl_json(["-n", "default", "get", "pod", binding["serve_id"]])
            for binding in state["bindings"]
        ]
        tre_pods = self.kubectl_json(["-n", "tre-v2", "get", "pods"])["items"]
        not_ready = [
            pod["metadata"]["name"]
            for pod in [*model_pods, *tre_pods]
            if not pod_is_ready(pod)
        ]
        if not_ready:
            raise RuntimeError(f"pods not Ready: {sorted(not_ready)}")
        deployments = self.kubectl_json(
            ["-n", "tre-v2", "get", "deployments"]
        )["items"]
        bad_deployments = [
            item["metadata"]["name"]
            for item in deployments
            if item.get("status", {}).get("readyReplicas", 0)
            != item["spec"].get("replicas", 1)
        ]
        if bad_deployments:
            raise RuntimeError(f"deployments not fully Ready: {bad_deployments}")
    def verify_freeze(self, config: ArmConfig) -> dict[str, Any]:
        current_head = subprocess.check_output(
            ["git", "-C", str(self.git_root), "rev-parse", "HEAD"], text=True
        ).strip()
        status_lines = subprocess.check_output(
            ["git", "-C", str(self.git_root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).splitlines()
        unsafe_status = []
        for line in status_lines:
            path = line[3:]
            if line.startswith("?? ") and path.startswith("tre/docs/refactor/p11_evidence/"):
                continue
            unsafe_status.append(line)
        if unsafe_status:
            raise RuntimeError(f"authoritative worktree is not campaign-clean: {unsafe_status}")
        ancestry = subprocess.run(
            ["git", "-C", str(self.git_root), "merge-base", "--is-ancestor",
             self.manifest.frozen_sha, current_head]
        )
        if ancestry.returncode != 0:
            raise RuntimeError("FROZEN_SHA is not an ancestor of current HEAD")
        changed = subprocess.check_output(
            ["git", "-C", str(self.git_root), "diff", "--name-only",
             f"{self.manifest.frozen_sha}..{current_head}"], text=True
        ).splitlines()
        allowed = (
            "tre/deploy/overlays/",
            "tre/deploy/tests/test_kustomize_overlays.py",
            "tre/deploy/campaigns/",
            "tre/docs/",
        )
        runtime_changes = [path for path in changed if not path.startswith(allowed)]
        if runtime_changes:
            raise RuntimeError(f"post-freeze runtime changes detected: {runtime_changes}")
        actual_images = {}
        deployment_names = {
            "controller": "tre-v2-controller",
            "service-manager": "tre-v2-service-manager",
            "ui": "tre-v2-ui",
        }
        for component, deployment_name in deployment_names.items():
            deployment = self.kubectl_json(
                ["-n", "tre-v2", "get", "deploy", deployment_name]
            )
            actual_images[component] = deployment["spec"]["template"]["spec"]["containers"][0]["image"]
        if actual_images != self.manifest.images:
            raise RuntimeError(f"image freeze mismatch: {actual_images}")
        params = self.http_json("GET", f"{self.ui_url}/api/params")
        if (
            params.get("params_hash") != self.manifest.params_hash
            or params.get("applied_hash") != self.manifest.params_hash
        ):
            raise RuntimeError("registry params hash does not match manifest")
        env = self.controller_env()
        expected_env = {
            "ENABLE_TRE_SCALING": "true",
            "TRE_SIGNAL_SOURCE": config.signal_source,
            "TRE_DISABLE_ETA_GATE": "true" if config.disable_eta_gate else "false",
        }
        mismatched = {
            key: (env.get(key), value)
            for key, value in expected_env.items()
            if env.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"controller arm env mismatch: {mismatched}")
        mode = (self.redis.get(MODE_KEY) or b"active").decode()
        if mode != config.mode:
            raise RuntimeError(f"controller mode {mode} != {config.mode}")
        apa_count = self.apa_count()
        if apa_count != (3 if config.apa_enabled else 0):
            raise RuntimeError(f"APA CR count mismatch: {apa_count}")
        return {
            "frozen_sha": self.manifest.frozen_sha,
            "current_head": current_head,
            "allowed_untracked_evidence": status_lines,
            "postfreeze_paths": changed,
            "images": actual_images,
            "params_hash": params.get("params_hash"),
            "applied_hash": params.get("applied_hash"),
            "controller_env": expected_env,
            "controller_mode": mode,
            "apa_cr_count": apa_count,
        }

    def deactivate_and_reset(self, run_dir: Path) -> None:
        self.set_mode("observe")
        self.toggle("tre")
        baseline = self.ensure_baseline()
        self.guard_zero(include_probes=False)
        self.assert_ready()
        atomic_json(run_dir / "reset_state.json", baseline)
        atomic_json(run_dir / "redis_reset.json", self.clear_per_run_redis())
        cooldown_start = time.time()
        time.sleep(self.manifest.cooldown_s)
        baseline_after = self.state()
        errors = baseline_errors(baseline_after, self.manifest.baseline)
        if errors:
            raise RuntimeError("baseline drifted during cooldown: " + "; ".join(errors))
        self.guard_zero()
        self.assert_ready()
        atomic_json(
            run_dir / "cooldown.json",
            {
                "start": utc_iso(cooldown_start),
                "end": utc_iso(),
                "required_s": self.manifest.cooldown_s,
                "actual_s": time.time() - cooldown_start,
                "baseline_verified": True,
            },
        )

    def configure_arm(self, config: ArmConfig) -> dict[str, Any]:
        if config.apa_enabled:
            self.toggle("apa")
            # Re-enable TRE decision loops for counterfactual logging only. Mode remains
            # observe, so APA stays the sole actuator.
            self.configure_controller(config)
            self.set_mode("observe")
            self.wait_apa_ready()
        else:
            self.toggle("tre")
            self.configure_controller(config)
            self.set_mode("active")
        self.assert_ready()
        self.guard_zero()
        errors = baseline_errors(self.state(), self.manifest.baseline)
        if errors:
            raise RuntimeError("arm activation changed baseline: " + "; ".join(errors))
        return self.verify_freeze(config)

    def controller_pod(self) -> str:
        pods = self.kubectl_json(
            ["-n", "tre-v2", "get", "pods",
             "-l", "app.kubernetes.io/name=tre-v2-controller"]
        )["items"]
        ready = [pod for pod in pods if pod_is_ready(pod)]
        if len(ready) != 1:
            raise RuntimeError(f"expected one Ready controller pod, got {len(ready)}")
        return ready[0]["metadata"]["name"]

    def harvest_signals(self, start_ms: int, end_ms: int, output: Path) -> int:
        entries = self.redis.xrange(
            SIGNAL_KEY, min=f"{start_ms}-0", max=f"{end_ms}-999999"
        )
        return write_signal_csv(entries, output)

    def run_one(self, spec: RunSpec) -> dict[str, Any]:
        run_dir = self.out_root / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "STATUS").write_text("PREPARING\n", encoding="utf-8")
        config = arm_config(spec.arm)
        trace_path = (self.repo / spec.trace).resolve()
        if not trace_path.is_file() or self.repo not in trace_path.parents:
            raise ValueError(f"trace path is invalid or outside repo: {trace_path}")
        self.deactivate_and_reset(run_dir)
        freeze = self.configure_arm(config)
        atomic_json(run_dir / "freeze.json", freeze)
        atomic_json(
            run_dir / "params.json",
            self.http_json("GET", f"{self.ui_url}/api/params"),
        )
        controller_pod = self.controller_pod()
        atomic_json(run_dir / "sm_state_start.json", self.state())
        requests_path = run_dir / "requests.jsonl"
        replay_summary = run_dir / "run_trace_summary.json"
        replay_error = run_dir / "run_trace.stderr"
        command = [
            sys.executable, "-m", "tre_replayer.run_trace",
            "--trace", str(trace_path),
            "--gateway-url", config.gateway,
            "--out", str(requests_path),
            "--registry", str(self.repo / "deploy/registry.yaml"),
            "--seed", str(spec.seed),
            "--max-in-flight", "512",
            "--trim-ramp-windows", "1",
        ]
        metadata = {
            "run_id": spec.run_id,
            "arm": spec.arm,
            "seed": spec.seed,
            "trace": str(trace_path.relative_to(self.repo)),
            "trace_sha256": sha256_file(trace_path),
            "gateway": config.gateway,
            "command": command,
            "operator": "root via Codex",
            "controller_pod": controller_pod,
        }
        atomic_json(run_dir / "command.json", metadata)
        start_epoch = time.time()
        start_ms = int(start_epoch * 1000)
        (run_dir / "STATUS").write_text("RUNNING\n", encoding="utf-8")
        sampler = LayoutSampler(self.state)
        sampler.start()
        try:
            with (
                replay_summary.open("w", encoding="utf-8") as stdout,
                replay_error.open("w", encoding="utf-8") as stderr,
            ):
                self._command(command, stdout=stdout, stderr=stderr)
            time.sleep(self.manifest.post_drain_s)
        finally:
            sampler.stop()
        end_epoch = time.time()
        end_ms = int(end_epoch * 1000)
        if sampler.errors:
            raise RuntimeError(f"SM layout sampler errors: {sampler.errors[:3]}")
        if self.controller_pod() != controller_pod:
            raise RuntimeError("controller pod changed during trace")
        if not requests_path.exists() or requests_path.stat().st_size == 0:
            raise RuntimeError("replayer produced no request records")
        health = request_health(requests_path)
        if health["http_5xx_frac"] > 0.75 or health["status_zero_frac"] > 0.10:
            raise RuntimeError(f"infrastructure collapse gate failed: {health}")
        atomic_json(run_dir / "request_health.json", health)
        write_csv(run_dir / "layout_timeline.csv", LAYOUT_FIELDS, sampler.rows)
        actual_actions = derive_actual_actions(sampler.rows)
        write_csv(run_dir / "actual_actions.csv", ACTUAL_ACTION_FIELDS, actual_actions)
        signal_count = self.harvest_signals(
            start_ms, end_ms, run_dir / "timeline_signals.csv"
        )
        if signal_count == 0:
            raise RuntimeError("signal harvest is empty")
        controller_log = subprocess.check_output(
            ["kubectl", "-n", "tre-v2", "logs", controller_pod,
             f"--since-time={utc_iso(start_epoch - 1)}"],
            text=True, errors="replace",
        )
        (run_dir / "controller.log").write_text(controller_log, encoding="utf-8")
        decisions, proposed_actions = parse_controller_decisions(
            controller_log, start_ms=start_ms, end_ms=end_ms
        )
        write_jsonl(run_dir / "controller_decisions.jsonl", decisions)
        write_jsonl(run_dir / "proposed_actions.jsonl", proposed_actions)
        atomic_json(
            run_dir / "pod_events_default.json",
            self.kubectl_json(["-n", "default", "get", "events"]),
        )
        atomic_json(
            run_dir / "pod_events_tre_v2.json",
            self.kubectl_json(["-n", "tre-v2", "get", "events"]),
        )
        atomic_json(run_dir / "safescale.json", self.safescale_snapshot())
        self.guard_zero(include_probes=False)
        atomic_json(run_dir / "sm_state_end.json", self.state())
        score_command = [
            sys.executable,
            "deploy/scripts/analysis/score_request_trace.py",
            "--input", str(requests_path),
            "--output", str(run_dir / "score.json"),
            "--windows-output", str(run_dir / "violation_windows.csv"),
            "--registry", str(self.repo / "deploy/registry.yaml"),
            "--trim-ramp-windows", "1",
        ]
        with (run_dir / "score.log").open("w", encoding="utf-8") as stdout:
            self._command(score_command, stdout=stdout, stderr=subprocess.STDOUT)
        compressed = run_dir / "requests.jsonl.gz"
        deterministic_gzip(requests_path, compressed)
        requests_sha = sha256_file(requests_path)
        requests_path.unlink()
        result = {
            **metadata,
            "start": utc_iso(start_epoch),
            "end": utc_iso(end_epoch),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "signal_rows": signal_count,
            "actual_actions": len(actual_actions),
            "proposed_actions": len(proposed_actions),
            "request_health": health,
            "requests_jsonl_sha256": requests_sha,
            "requests_gzip_sha256": sha256_file(compressed),
            "trim_ramp_windows": 1,
        }
        atomic_json(run_dir / "run.json", result)
        (run_dir / "STATUS").write_text("DONE\n", encoding="utf-8")
        self.set_mode("observe")
        return result

    def restore_safe(self) -> None:
        self.set_mode("observe")
        self.toggle("tre")
        self.configure_controller(arm_config("tre"))
        self.set_mode("observe")
        self.ensure_baseline()
        self.clear_per_run_redis()
        self.guard_zero()
        self.assert_ready()


def select_runs(
    runs: Sequence[RunSpec], *, start_at: str | None, limit: int | None
) -> tuple[RunSpec, ...]:
    selected = list(runs)
    if start_at is not None:
        indices = [index for index, run in enumerate(selected) if run.run_id == start_at]
        if not indices:
            raise ValueError(f"unknown --start-at run: {start_at}")
        selected = selected[indices[0]:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return tuple(selected)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    script = Path(__file__).resolve()
    default_repo = script.parents[2] if len(script.parents) > 2 else Path.cwd()
    parser.add_argument("--repo", default=str(default_repo))
    parser.add_argument("--start-at")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    selected = select_runs(
        manifest.runs, start_at=args.start_at, limit=args.limit
    )
    out_root = Path(args.out_root)
    plan = {
        "manifest": str(Path(args.manifest).resolve()),
        "frozen_sha": manifest.frozen_sha,
        "selected_runs": [asdict(run) for run in selected],
        "execute": args.execute,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_json(out_root / "queue_plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    runner = CampaignRunner(
        repo=Path(args.repo), manifest=manifest, out_root=out_root
    )
    completed = []
    try:
        for spec in selected:
            run_dir = out_root / spec.run_id
            status_path = run_dir / "STATUS"
            if status_path.exists() and status_path.read_text().strip() == "DONE":
                completed.append({"run_id": spec.run_id, "status": "skipped_done"})
                continue
            if run_dir.exists():
                raise RuntimeError(
                    f"run directory already exists and is not DONE: {run_dir}"
                )
            try:
                result = runner.run_one(spec)
            except Exception as exc:
                if run_dir.exists():
                    status_path.write_text("FAILED\n", encoding="utf-8")
                    atomic_json(
                        run_dir / "failure.json",
                        {
                            "type": type(exc).__name__,
                            "error": str(exc),
                            "time": utc_iso(),
                        },
                    )
                raise
            completed.append(
                {"run_id": spec.run_id, "status": "done", "result": result}
            )
            atomic_json(
                out_root / "queue_status.json",
                {"completed": completed, "updated": utc_iso()},
            )
    finally:
        runner.restore_safe()
    atomic_json(
        out_root / "queue_status.json",
        {"completed": completed, "updated": utc_iso(), "status": "DONE"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())