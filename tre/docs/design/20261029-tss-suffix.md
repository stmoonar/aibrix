# TSS suffix: Z of the current pod set, settle 10 s, scope healthy_and_shrink (2026-10-29)

User decisions 2026-10-29. Branch `feat/tss-suffix-20261029` (on `feat/tss-only-20261029`
@ 82ee9c69). Tests: `controller/tests/test_tss_suffix_20261029.py`. Supersedes the
"whole window after a change" rule of [20261029-tss-only.md](20261029-tss-only.md); the
removals listed there stay removed (no H3, no saturation rescue, no 50 s fallback, no
onset guard, no EMA carry).

## Estimand

TSS answers "how fast does the pod set the gateway routes to **now** serve its queue":
tokens ÷ (running + λ·waiting) over the pods that are awake and not hidden. A window that
spans a change of that set measures two pod sets at once.

**Queue-weighted blend.** Split the window at the change t_c into a before part (time
share f₁, mean queue Q₁, speed TSS₁ = tokens₁ / Q₁ scaled to a whole window) and an
after part (f₂, Q₂, TSS₂). The whole-window value is

    TSS_window = (f₁·Q₁·TSS₁ + f₂·Q₂·TSS₂) / (f₁·Q₁ + f₂·Q₂)

so the old set's speed is weighted by its time share **and its queue**. Around a scale
event the old set usually has the larger queue (that is why the controller acted), so the
window keeps reading the old speed long after the change. A replica factor cannot fix this:
the error is the blend, not the pod count.

## Rules

1. **Pod set** = the pods the SM reports awake and not hidden. Hidden and asleep pods are
   out of the numerator and the denominator (`serving_window`, every reader).
2. **t_c** = the `done_ms` of the controller's own SM call (wake / sleep done, hide
   confirmed, unhide done: `ActionQueue.view_changes` / `last_actions`); for a change the
   controller did not make, the first fleet view that shows the new set. Suffix = the
   store's sub-window from the first evidence boundary at or after t_c
   (`MetricsStore(suffix_change=...)`). Tokens = the step-counter delta scaled to the
   window; Q = the mean of the suffix samples; no replica factor. L3 / coverage /
   token-span rules are relative to the suffix span.
3. **Settle**: Z is decision-valid only when the suffix spans ≥ S = `scaling.settle_ms`
   (10 000; observed `scraped_ms`, 1 s tolerance). A suffix with < 2 samples of a serving
   pod, or unknown tokens, is unknown, never Z = 0. While settling the model is UNKNOWN:
   not a receiver, not a donor, not probed. One `settling_start` / `settling_end` event per
   episode.
4. **EMA** restarts at t_c and seeds on the first decision-valid suffix value.
5. **Donors** (`TRE_SAFESCALE_SCOPE=healthy_and_shrink`; `all`, `receiverless_shrink`,
   `off` unchanged, default `all`): IDLE at once; HIGH → a receiver as an immediate relay
   (reason suffix `_relay`); HEALTHY → only a CRITICAL receiver, only through a probe;
   receiverless HIGH shrink → probe. A probe judges its donor on the **remaining** pods,
   on the post-hide suffix ≥ S, and commits early when Z_remaining ≥ τ_low and the SLO
   holds (the early-commit floor is now S, was 20 s). Direct SLO rollback, deadline and
   "a rolled-back decision waits for new evidence" (`scaling.min_evidence_ms`, the retry
   floor only) stay; after a rollback t_c = unhide done. The receiver wakes at commit.
6. **TP2 block**: two probed donors → both probes in the same plan, one block id; the last
   probe to pass commits the block in **one** SM plan (its partners' pods are
   companions); any failure rolls both back and the receiver is not woken. With ≤ 1
   probed donor the IDLE / HIGH half sleeps at the probe's commit, not at its start.
   No new probe type.
7. Correctness guards unchanged: action in flight, probe open, view-pending.

## Measured settling (offline, 2026-10-29 analysis)

After an up-scale the whole window reads Z −54 % below the post-change steady value at
+10 s; after a down-scale +74 % above it. The suffix value reaches the per-window noise
floor at about +10 s, hence S = 10 s.

## Rejected

- **Projected-Z guard** (predict the donor's Z after the give from the current window and
  refuse a give that would cross τ_crit): offline it blocked 27 of 31 gives, including the
  good ones, because it projects from the blended window it is meant to correct.
- Longer holds / timers: the hold is evidence-based (suffix span), not a clock.

## Expected costs

- HEALTHY probes roll back often in a full cluster (the donor has little spare). The user
  chose them anyway: the probe drains the hidden pod and verifies the donor on its
  remaining pods before anything sleeps.
- A model whose e2e is longer than S still has some transient in its first valid suffix;
  at a 10 s evidence period the suffix starts up to 10 s later and reads less of it
  (period_sim `shrink`: the third probe's start moves).
- Clock: t_c is controller clock, the suffix is cut on gateway stamps. They agree when the
  controller and the gateway plugins run on one node (the overlays pin both); a skew d
  moves the suffix start by d.
