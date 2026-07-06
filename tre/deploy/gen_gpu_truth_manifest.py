"""Generate the tre-v2 gpu-truth ConfigMap + DaemonSet manifest.

The gpu-truth DaemonSet mounts the node GPU-truth agent from a ConfigMap whose
data is a byte-for-byte copy of ``deploy/scripts/gpu_truth_agent.py``. Kustomize's
default load restrictor forbids a ``configMapGenerator`` that reads a file outside
the overlay directory, so the script is embedded inline here instead and kept in
sync by ``deploy/tests/test_gpu_truth_daemonset.py``.

Run ``python3 deploy/gen_gpu_truth_manifest.py`` after editing the agent script.
"""
from __future__ import annotations

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
            - "30"
            - --ttl-s
            - "120"
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


def render(script_text: str) -> str:
    return _HEADER + _indent_script(script_text) + _DAEMONSET


def main() -> int:
    content = render(AGENT_SCRIPT.read_text(encoding="utf-8"))
    MANIFEST.write_text(content, encoding="utf-8")
    print(f"wrote {MANIFEST} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
