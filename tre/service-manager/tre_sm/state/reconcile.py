from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Protocol

from tre_common.registry import ClusterTopology
from tre_common.registry import NodeSpec
from tre_sm.allocator.slots import Binding, Slot, SlotAllocator, binding_sort_key
from tre_sm.gpu_truth import GpuTruthProvider
from tre_sm.state.store import StateStore


POD_STATE_SLEEPING = "sleeping"
POD_STATE_AWAKE = "awake"
POD_STATE_HIDDEN = "hidden"
_VALID_POD_STATES = {POD_STATE_SLEEPING, POD_STATE_AWAKE, POD_STATE_HIDDEN}


@dataclass(frozen=True)
class PodRecord:
    serve_id: str
    model: str
    node: str
    cuda_visible_devices: str
    state: str = POD_STATE_AWAKE
    # pod_ip lets the physical prober reach the vLLM /is_sleeping endpoint.
    pod_ip: str | None = None
    # routable mirrors the current tre.aibrix.io/routable label (write-through
    # cache) so Layer 1 can patch-on-diff rather than unconditionally.
    routable: bool | None = None
    # Running is not sufficient for routing; all containers must be Ready.
    ready: bool = True

    def to_binding(self) -> Binding:
        if self.state not in _VALID_POD_STATES:
            raise ValueError(f"unknown pod state for {self.serve_id}: {self.state}")
        return Binding(
            serve_id=self.serve_id,
            model=self.model,
            slot=Slot(self.node, _parse_cuda_visible_devices(self.cuda_visible_devices)),
            awake=self.state != POD_STATE_SLEEPING,
            hidden=self.state == POD_STATE_HIDDEN,
        )


class K8sPodClient(Protocol):
    def list_pods(self) -> list[PodRecord]: ...


class PodPhysicalProber(Protocol):
    """Physical /is_sleeping probe: OBSERVED ground truth.

    Returns True if the pod is physically sleeping, False if physically awake,
    and None when the physical state cannot be determined (unreachable pod).
    """

    def is_sleeping(self, pod: PodRecord) -> bool | None: ...


class RoutableLabelWriter(Protocol):
    """Write-path for the tre.aibrix.io/routable label (Layer 1 enforcement)."""

    def set_pod_routable(self, serve_id: str, *, routable: bool) -> None: ...


@dataclass(frozen=True)
class ReconcileResult:
    version: int
    bindings: list[Binding]
    warnings: list[str]
    allocator: SlotAllocator


@dataclass(frozen=True)
class AuditResult:
    version: int
    issues: list[dict[str, object]]

    @property
    def healthy(self) -> bool:
        return not self.issues


def audit_state(
    store: StateStore,
    k8s_client: K8sPodClient,
    *,
    prober: PodPhysicalProber | None = None,
) -> AuditResult:
    """Compare persisted, Kubernetes and physical state without writing any of them."""
    persisted = store.load()
    observed = list(k8s_client.list_pods())
    issues: list[dict[str, object]] = []

    persisted_by_id = _unique_bindings_by_id(
        persisted.bindings, source="persisted", issues=issues
    )
    observed_by_id: dict[str, tuple[PodRecord, Binding]] = {}
    physically_awake_by_gpu: dict[tuple[str, int], str] = {}
    for pod in sorted(observed, key=lambda item: binding_sort_key(item.to_binding())):
        binding = pod.to_binding()
        if binding.binding_id in observed_by_id:
            issues.append(
                {
                    "code": "duplicate_observed_binding_id",
                    "binding_id": binding.binding_id,
                    "serve_id": binding.serve_id,
                }
            )
            continue
        observed_by_id[binding.binding_id] = (pod, binding)

    for binding_id in sorted(set(persisted_by_id) | set(observed_by_id)):
        stored = persisted_by_id.get(binding_id)
        observed_item = observed_by_id.get(binding_id)
        if stored is None:
            pod, live = observed_item
            issues.append(
                {
                    "code": "untracked_pod",
                    "binding_id": binding_id,
                    "serve_id": live.serve_id,
                }
            )
            continue
        if observed_item is None:
            issues.append(
                {
                    "code": "ghost_binding",
                    "binding_id": binding_id,
                    "serve_id": stored.serve_id,
                }
            )
            continue

        pod, live = observed_item
        if stored.serve_id != live.serve_id:
            issues.append(
                {
                    "code": "instance_replaced",
                    "binding_id": binding_id,
                    "persisted_serve_id": stored.serve_id,
                    "observed_serve_id": live.serve_id,
                }
            )

        if not pod.ready:
            issues.append(
                {
                    "code": "pod_not_ready",
                    "binding_id": binding_id,
                    "serve_id": live.serve_id,
                }
            )

        physical_awake: bool | None = live.awake
        if prober is not None:
            sleeping = prober.is_sleeping(pod)
            physical_awake = None if sleeping is None else not sleeping
            if physical_awake is None:
                issues.append(
                    {
                        "code": "physical_state_unknown",
                        "binding_id": binding_id,
                        "serve_id": live.serve_id,
                    }
                )
        if physical_awake is not None and stored.awake != physical_awake:
            issues.append(
                {
                    "code": "power_mismatch",
                    "binding_id": binding_id,
                    "serve_id": live.serve_id,
                    "persisted_awake": stored.awake,
                    "physical_awake": physical_awake,
                }
            )

        if physical_awake:
            for gpu_key in _slot_keys(live.slot):
                occupant = physically_awake_by_gpu.get(gpu_key)
                if occupant is not None:
                    node, gpu_id = gpu_key
                    issues.append(
                        {
                            "code": "awake_gpu_conflict",
                            "binding_id": binding_id,
                            "serve_id": live.serve_id,
                            "node": node,
                            "gpu_id": gpu_id,
                            "conflicts_with": occupant,
                        }
                    )
                else:
                    physically_awake_by_gpu[gpu_key] = live.serve_id

        expected_routable = (
            bool(physical_awake)
            and pod.ready
            and not stored.hidden
            and not live.hidden
        )
        if pod.routable is None or pod.routable != expected_routable:
            issues.append(
                {
                    "code": "routable_mismatch",
                    "binding_id": binding_id,
                    "serve_id": live.serve_id,
                    "observed_routable": pod.routable,
                    "expected_routable": expected_routable,
                }
            )

    return AuditResult(version=persisted.version, issues=issues)


def reconcile_state(
    topology: ClusterTopology,
    store: StateStore,
    k8s_client: K8sPodClient,
    *,
    gpu_truth: GpuTruthProvider | None = None,
    sleep_leak_used_mib: int = 8192,
    prober: PodPhysicalProber | None = None,
    label_writer: RoutableLabelWriter | None = None,
    drop_missing: bool = False,
) -> ReconcileResult:
    persisted = store.load()
    persisted_by_serve = {binding.serve_id: binding for binding in persisted.bindings}
    reconciled_by_serve: dict[str, Binding] = {}
    warnings: list[str] = []

    observed = list(k8s_client.list_pods())
    observed_by_serve = {pod.serve_id: pod for pod in observed}

    for pod in sorted(observed, key=lambda item: binding_sort_key(item.to_binding())):
        binding = pod.to_binding()
        # Physical /is_sleeping is OBSERVED ground truth and wins over the
        # tre.aibrix.io/state annotation, which is only a write-through cache.
        if prober is not None:
            sleeping = prober.is_sleeping(pod)
            if sleeping is not None:
                binding = replace(binding, awake=not sleeping)
            else:
                # Unknown physical state is never eligible for routing. Keeping
                # awake as observed avoids claiming that a physical sleep was
                # performed; hidden is the fail-closed quarantine bit.
                binding = replace(binding, hidden=True)
                warnings.append(
                    f"{binding.serve_id}: physical power state unknown; quarantined unroutable"
                )
        previous = persisted_by_serve.get(binding.serve_id)
        if previous is not None and previous != binding:
            warnings.append(f"{binding.serve_id}: pod reality overrides persisted binding")
        if binding.serve_id in reconciled_by_serve:
            raise ValueError(f"duplicate pod observation: {binding.serve_id}")
        reconciled_by_serve[binding.serve_id] = binding

    observed_slots = {
        slot_key
        for binding in reconciled_by_serve.values()
        for slot_key in _slot_keys(binding.slot)
    }

    for binding in persisted.bindings:
        if binding.serve_id in reconciled_by_serve:
            continue
        if any(slot_key in observed_slots for slot_key in _slot_keys(binding.slot)):
            warnings.append(
                f"{binding.serve_id}: dropped stale persisted binding that overlaps pod observation"
            )
            continue
        if drop_missing:
            warnings.append(
                f"{binding.serve_id}: dropped persisted binding with no pod observation (strict)"
            )
            continue
        warnings.append(f"{binding.serve_id}: persisted binding has no matching pod observation")
        reconciled_by_serve[binding.serve_id] = binding

    bindings = _quarantine_awake_conflicts(
        sorted(reconciled_by_serve.values(), key=binding_sort_key), warnings
    )
    if gpu_truth is not None:
        warnings.extend(_sleep_leak_warnings(topology, bindings, gpu_truth, sleep_leak_used_mib))

    # Layer 1 (SAFETY INVARIANT): re-assert routable = physical-awake AND not
    # hidden onto every observed pod, patch-on-diff (idempotent). This never
    # loops: a pod that refuses to converge (leak) is simply left non-routable
    # and surfaced via the sleep_leak warning above (D8 leak candidate).
    if label_writer is not None:
        _enforce_routable_labels(bindings, observed_by_serve, label_writer)

    allocator = SlotAllocator(topology, bindings, allow_awake_conflicts=True)
    if bindings == persisted.bindings:
        return ReconcileResult(
            version=persisted.version,
            bindings=bindings,
            warnings=warnings,
            allocator=allocator,
        )

    version = store.save(bindings, expected_version=persisted.version)
    return ReconcileResult(version=version, bindings=bindings, warnings=warnings, allocator=allocator)


def _enforce_routable_labels(
    bindings: list[Binding],
    observed_by_serve: dict[str, PodRecord],
    label_writer: RoutableLabelWriter,
) -> None:
    for binding in bindings:
        pod = observed_by_serve.get(binding.serve_id)
        if pod is None:
            # No live pod observation -> nothing to re-assert.
            continue
        desired_routable = binding.awake and pod.ready and not binding.hidden
        if pod.routable == desired_routable:
            continue
        label_writer.set_pod_routable(binding.serve_id, routable=desired_routable)


def _parse_cuda_visible_devices(value: str) -> tuple[int, ...]:
    devices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not devices:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain at least one GPU id")
    return devices


def _slot_keys(slot: Slot) -> tuple[tuple[str, int], ...]:
    return tuple((slot.node, gpu) for gpu in slot.gpu_ids)


def _unique_bindings_by_id(
    bindings: list[Binding],
    *,
    source: str,
    issues: list[dict[str, object]],
) -> dict[str, Binding]:
    indexed: dict[str, Binding] = {}
    for binding in sorted(bindings, key=binding_sort_key):
        if binding.binding_id in indexed:
            issues.append(
                {
                    "code": f"duplicate_{source}_binding_id",
                    "binding_id": binding.binding_id,
                    "serve_id": binding.serve_id,
                }
            )
            continue
        indexed[binding.binding_id] = binding
    return indexed


def _quarantine_awake_conflicts(bindings: list[Binding], warnings: list[str]) -> list[Binding]:
    awake_by_gpu: dict[tuple[str, int], str] = {}
    reconciled: list[Binding] = []
    for binding in bindings:
        conflict_key = None
        if binding.awake:
            for key in _slot_keys(binding.slot):
                if key in awake_by_gpu:
                    conflict_key = key
                    break
        if conflict_key is None:
            reconciled.append(binding)
            if binding.awake:
                for key in _slot_keys(binding.slot):
                    awake_by_gpu[key] = binding.serve_id
            continue

        node, gpu = conflict_key
        warnings.append(
            f"{binding.serve_id}: awake GPU conflict on {node}/{gpu}; quarantined unroutable"
        )
        # Never claim a physical sleep that reconcile did not perform. Hidden
        # keeps the conflicting pod out of Service endpoints while awake=True
        # truthfully records the probed physical state.
        reconciled.append(replace(binding, hidden=True))
    return reconciled


def _sleep_leak_warnings(
    topology: ClusterTopology,
    bindings: list[Binding],
    gpu_truth: GpuTruthProvider,
    threshold_mib: int,
) -> list[str]:
    warnings: list[str] = []
    nodes = {node.name: node for node in topology.nodes}
    warned: set[str] = set()
    for (node_name, gpu_id), gpu_bindings in _bindings_by_gpu(bindings).items():
        if any(binding.awake for binding in gpu_bindings):
            continue
        node = nodes.get(node_name)
        gpu_uuid = _gpu_uuid(node, gpu_id)
        if gpu_uuid is None:
            continue
        used_mib = gpu_truth.used_mib(node=node_name, gpu_id=gpu_id, gpu_uuid=gpu_uuid)
        if used_mib is None or used_mib <= threshold_mib:
            continue
        for binding in gpu_bindings:
            if binding.awake or binding.serve_id in warned:
                continue
            warnings.append(
                f"sleep_leak:{binding.serve_id}: {node_name}/{gpu_uuid} "
                f"used_mib={used_mib} threshold_mib={threshold_mib}"
            )
            warned.add(binding.serve_id)
    return warnings


def _bindings_by_gpu(bindings: list[Binding]) -> dict[tuple[str, int], list[Binding]]:
    grouped: dict[tuple[str, int], list[Binding]] = {}
    for binding in bindings:
        for gpu in binding.slot.gpu_ids:
            grouped.setdefault((binding.slot.node, gpu), []).append(binding)
    return grouped


def _gpu_uuid(node: NodeSpec | None, gpu_id: int) -> str | None:
    if node is None:
        return None
    if gpu_id < 0 or gpu_id >= len(node.gpu_uuids):
        return None
    return node.gpu_uuids[gpu_id]
