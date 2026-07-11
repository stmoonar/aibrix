# APA fix live canary (2026-07-12)

Validates commit `fix(apa-baseline): scope anchor selectors to awake pods via routable label`.

Procedure: toggle to the APA arm (fixed anchors applied), drive a 180 s burst at dsqwen-7b
(30 RPS x 1024 input x 256 output tokens, replayer, gateway 31592, seed 1, max-in-flight
512), observe the aibrix controller-manager, then verify scale-down and restore the exact
node9 baseline via the toggle.

Observed (see `apa_controller_log_excerpt.txt`):
- Before load: `pods=1` (awake-only scrape now that the selector requires routable=true).
- Under load: `current_value` rose to ~0.50 and APA recommended `DesiredReplicas: 5`;
  service-manager woke 4 additional 7b pods; the metric pool grew to `pods=5` as each
  new pod turned routable. First-ever APA actuation in this system.
- After load: recommendation returned to `DesiredReplicas: 1`; natural serve-id downscale
  landed back on the original baseline pod `dsqwen-7b-...-node9-gpu-0-546d5d9f88-f94nf`.
- Final state: TRE arm restored, controller observe, node9 1/1/1 exact baseline,
  orphan/safescale guards empty.

Both scale directions of the APA baseline are therefore functional post-fix. The
9-run APA rerun campaign (`deploy/campaigns/apa_rerun_e1.json`) is unblocked.
