# P5 Paper vs Implementation Notes

Date: 2026-07-04
Environment: remote server 76, `/data/nfs_shared_data/xxy/aibrix`

## TRS Signal Migration

The code keeps the legacy controller name `TRS`; the plan notes this is the same metric as paper `TSS`.

The migrated `TRSComputer` intentionally preserves the frozen upstream implementation's replica correction:

```python
TRS_raw = TRS_raw * assigned_replicas / routable_pods
```

This correction is applied after `Y_m / Q_ctl` and is documented in the frozen upstream `trs.py` as matching the existing `main.py` behavior. It may not appear as a separate term in the paper derivation, so it is recorded here as an implementation contract rather than changed during migration.

The saturation guard formula is preserved as `Gamma_m = (Y_m(t) - Y_m(t-1)) / (Q_ctl(t) - Q_ctl(t-1))`, with saturation only when `Q_ctl >= qsat` and `abs(Gamma_m) <= epsat` for `Hsat` consecutive windows.
