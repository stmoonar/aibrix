"""tracegen: v2 workload traces (per-request plans) for the multi-model comparison.

Pipeline (``python3 -m tre_replayer.tracegen <command>``):

1. ``fit``         Azure CSV -> lognormal length fits + 1 s arrival dispersion
                   (committed as ``data/azure_fits.json``; the CSVs are never committed).
2. ``generate``    trace spec (JSON) + seed -> ``design.json`` (no prompt text) + ``manifest.json``.
                   Per-model rate functions (flat / sinusoid / square / pulses / steps /
                   piecewise, in rho or req/s), non-homogeneous Poisson arrivals by thinning,
                   per-request lengths from truncated lognormals, or a real Azure slice.
3. ``materialize`` design -> ``traces_tre.effective.json`` (the file ``tre_loadgen_v1
                   --trace-file`` replays): one natural prompt per request, fitted with the
                   model's own tokenizer to the chat-templated length ``prompt_length``;
                   then ``verify`` runs (effective == design).
4. ``verify``      effective == design, request by request (ids, times, models, the
                   per-request ``max_output_tokens``, prompt token counts); never null.
5. ``audit``       R1/R2/R3/R5 + in-flight per replica (see :mod:`.audit`).

Nothing here hard-codes a node, an IP or a path: dataset CSVs, tokenizers (via the
replayer's ``model_tokenizer`` resolution) and output directories are arguments.
"""
