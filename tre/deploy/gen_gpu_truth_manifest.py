"""Generate the tre-v2 gpu-truth ConfigMap + DaemonSet manifest.

The gpu-truth DaemonSet mounts the node GPU-truth agent from a ConfigMap whose
data is a byte-for-byte copy of ``deploy/scripts/gpu_truth_agent.py``. Kustomize's
default load restrictor forbids a ``configMapGenerator`` that reads a file outside
the overlay directory, so the script is embedded inline here instead and kept in
sync by ``deploy/tests/test_gpu_truth_daemonset.py``.

Run ``python3 deploy/gen_gpu_truth_manifest.py`` after editing the agent script
(``--interval-s`` / ``--refresh-poll-s`` / ``--ttl-s`` tune the agent's command line;
the guard test expects the defaults).
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEPLOY_ROOT = Path(__file__).resolve().parent
AGENT_SCRIPT = DEPLOY_ROOT / "scripts" / "gpu_truth_agent.py"
MANIFEST = DEPLOY_ROOT / "overlays" / "tre-v2" / "gpu-truth.yaml"

_HEADER = """# ConfigMap data below is a byte-for-byte copy of
# tre/deploy/scripts/gpu_truth_agent.py, kept in sync by
# deploy/tests/test_gpu_truth_daemonset.py (asserts equality). Regenerate with:
#   python3 deploy/gen_gpu_truth_manifest.py
apiVersion: v1
kind: ConfigMap
metadata:
  name: tre-v2-gpu-truth-agent
  namespace: tre-v2
data:
  gpu_truth_agent.py: |
"""

_DAEMONSET = """---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: tre-v2-gpu-truth
  namespace: tre-v2
  labels:
    app.kubernetes.io/name: tre-v2-gpu-truth
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: tre-v2-gpu-truth
  template:
    metadata:
      labels:
        app.kubernetes.io/name: tre-v2-gpu-truth
    spec:
      # hostPID stays false: the agent only reads nvidia-smi inside its own
      # container (GPUs injected via NVIDIA_VISIBLE_DEVICES), never the host PID ns.
      hostPID: false
      # Restrict to the two TRE GPU nodes (registry cluster). The cluster has a
      # third A100 node ("cloud") out of TRE scope; gpu.present matches it too.
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:
                      - nscc-ds-4a100-node9
                      - nscc-ds-4a100-node10
      tolerations:
        - key: node-role.kubernetes.io/control-plane
          operator: Exists
          effect: NoSchedule
      volumes:
        - name: agent
          configMap:
            name: tre-v2-gpu-truth-agent
      containers:
        - name: gpu-truth
          # Reuse the vllm image already present on both GPU nodes (has python3 +
          # nvidia-smi); no new image build/push required. The agent script is
          # mounted from a ConfigMap so the source of truth stays
          # tre/deploy/scripts/gpu_truth_agent.py.
          image: vllm/vllm-openai:0.10.1-sleep
          imagePullPolicy: IfNotPresent
          command:
            - python3
            - /agent/gpu_truth_agent.py
            - --redis-url
            - redis://tre-v2-redis:6379/0
            - --node
            - $(NODE_NAME)
            - --interval-s
            - "{interval_s}"
            - --refresh-poll-s
            - "{refresh_poll_s}"
            - --ttl-s
            - "{ttl_s}"
          env:
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: NVIDIA_VISIBLE_DEVICES
              value: all
          volumeMounts:
            - name: agent
              mountPath: /agent
          resources:
            requests:
              cpu: 50m
              memory: 64Mi
            limits:
              cpu: 250m
              memory: 256Mi
"""


def _indent_script(script: str) -> str:
    lines = []
    for line in script.splitlines():
        lines.append(("    " + line) if line.strip() else "")
    return "\n".join(lines) + "\n"


#: Periodic sample interval (s). The service-manager also asks for a sample on
#: demand before a wake / cold-start headroom gate (tre:gpu_truth_refresh:<node>).
DEFAULT_INTERVAL_S = 10.0
#: How often the agent polls its refresh counter (s); 0 disables refreshes.
DEFAULT_REFRESH_POLL_S = 0.25
#: TTL of tre:gpu_truth:<node>; expiry = truth unavailable (gates fail closed).
DEFAULT_TTL_S = 120


def _num(value: float) -> str:
    return f"{float(value):g}"


def render(
    script_text: str,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    refresh_poll_s: float = DEFAULT_REFRESH_POLL_S,
    ttl_s: int = DEFAULT_TTL_S,
) -> str:
    if interval_s <= 0 or refresh_poll_s < 0 or ttl_s <= interval_s:
        raise ValueError("need interval_s > 0, refresh_poll_s >= 0 and ttl_s > interval_s")
    daemonset = _DAEMONSET.format(
        interval_s=_num(interval_s), refresh_poll_s=_num(refresh_poll_s), ttl_s=int(ttl_s)
    )
    return _HEADER + _indent_script(script_text) + daemonset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--refresh-poll-s", type=float, default=DEFAULT_REFRESH_POLL_S)
    parser.add_argument("--ttl-s", type=int, default=DEFAULT_TTL_S)
    parser.add_argument("--output", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    content = render(
        AGENT_SCRIPT.read_text(encoding="utf-8"),
        interval_s=args.interval_s,
        refresh_poll_s=args.refresh_poll_s,
        ttl_s=args.ttl_s,
    )
    args.output.write_text(content, encoding="utf-8")
    print(f"wrote {args.output} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
