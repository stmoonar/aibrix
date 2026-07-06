# TRE v2 Image Lock

Pinned container images for the reproducible AIBrix 0.7.0 + TRE v2 redeploy (D11 / F4).
No `latest` or `nightly` tags are permitted in the tre-v2 overlay (enforced by
`deploy/tests/test_kustomize_overlays.py`). Image IDs are local build digests
(images are built on the GPU nodes with `imagePullPolicy: IfNotPresent`; there is
no external registry).

Captured: 2026-07-06 (F4.0). Source commit for tre-v2 images: see per-image note.

| Component | Image tag | Image ID (sha256) | Source |
|---|---|---|---|
| service-manager | `tre-v2-service-manager:20260706-a1d21c00` | `` | commit `f6dce214` (route guard) |
| controller | `tre-v2-controller:20260705-d795a715` | `6b722a12a4aadb01dd3b485d5d537196deb337c0d4ebd7d63b54269b5eb118d3` | commit `d795a715` (inflight fix) |
| ui | `tre-v2-ui:20260704-669f0381` | `e81b68295f31103c22a24b51f1645e21e4d927b58e11f7127388628110ff06bc` | commit `669f0381` |
| redis | `redis:7.2-alpine` | `dfa18828cbc07b3ae6a95ec7343f6c214fdee2d836197b4be8e9904420762cd8` | upstream |
| vllm (model pods + gpu-truth DaemonSet) | `vllm/vllm-openai:0.10.1-sleep` | `6a3a5efad7779b594bf82dbda62c47efa789786a38963acb869142d9d8406492` | upstream (sleep-mode build) |
| tre-gateway-plugins (ADR-0008 isolated metrics scraper) | `aibrix/gateway-plugins:20260704-0d869b49-nozmq2` | `050845eaeca2beaa1e1357fefe0b0339d0f274ae5fd38a31b2f5c83ccafaf634` | TRE-patched gateway-plugins build (dual redis schema) |

## Rebuild note (F4.3)

The `controller` image above (`d795a715`) predates the F4.0 code change that wires
`TRE_HIST_BASELINE_LOOKBACK_MS` through `ControllerConfig`. The mismatch is benign
at runtime (an image that does not read the env falls back to the same 90s default),
but before the F4.3 clean deploy the controller image MUST be rebuilt from the
post-F4.0 HEAD and this table + `deploy/overlays/tre-v2/controller.yaml` updated to
the new tag/digest. The service-manager (`f6dce214`) and ui images are unchanged by
F4.0 and remain valid.
