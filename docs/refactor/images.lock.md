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
| controller | `tre-v2-controller:20260706-6fd540e6` | `4f315e82427d9c98d7a937a60686900404f16174f0a26e00e63e48002807552c` | commit `6fd540e6` (S1 signal freshness + F-onset warmup guard ADR-0013) |
| ui | `tre-v2-ui:20260704-669f0381` | `e81b68295f31103c22a24b51f1645e21e4d927b58e11f7127388628110ff06bc` | commit `669f0381` |
| redis | `redis:7.2-alpine` | `dfa18828cbc07b3ae6a95ec7343f6c214fdee2d836197b4be8e9904420762cd8` | upstream |
| vllm (model pods + gpu-truth DaemonSet) | `vllm/vllm-openai:0.10.1-sleep` | `6a3a5efad7779b594bf82dbda62c47efa789786a38963acb869142d9d8406492` | upstream (sleep-mode build) |
| tre-gateway-plugins (ADR-0008 isolated metrics scraper) | `aibrix/gateway-plugins:20260704-0d869b49-nozmq2` | `050845eaeca2beaa1e1357fefe0b0339d0f274ae5fd38a31b2f5c83ccafaf634` | TRE-patched gateway-plugins build (dual redis schema) |

## Rebuild note (F4.3) — RESOLVED 2026-07-06

The controller image was rebuilt from post-S1 HEAD `446ce73a` (tag
`tre-v2-controller:20260706-446ce73a`, digest `617607e8…`), superseding the stale
`d795a715` build. It now carries the full F1/F2 read-side fixes **and** the S1
signal-freshness changes (sliding window, 5s refresh decoupled from monitor_interval,
wall-clock time-constant EMA / ADR-0011, N1 min-latency-sample guard, N2 window
invariant guard). `deploy/overlays/tre-v2/controller.yaml`, the images.lock table
above, and `deploy/tests/test_kustomize_overlays.py` are all updated to this tag.
service-manager (`a1d21c00`) and ui images are unchanged and remain valid.

> S1.2 real-machine W-freeze: this image is what the S1.2 acceptance (freeze the
> 30s window, lag P95 ≤ 35s) must run — either on the current cluster or, per D11,
> folded into the F4.3 clean redeploy immediately before R3 (S1.4's hard gate).
