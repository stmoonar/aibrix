"""TRE v2 baseline scaler: runs a published autoscaling policy (Chiron, TokenScale,
PreServe, ...) as an experiment arm against the TRE service manager.

Per tick: gather a :class:`~tre_baselines.snapshot.ClusterSnapshot` -> ``policy.decide``
-> clamp to the registry bounds -> scale-downs before scale-ups -> ``PUT
/v2/models/{m}/target`` (dry-run by default: decisions are only logged).
"""
