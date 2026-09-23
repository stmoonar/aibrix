"""TRE v2 request client ported from the v1 (ICSE submission) CustomTraceGenerator.

Behaviour is kept identical to v1 (/root/aibrix-main/CustomTraceGenerator); the only
deliberate changes are ``max_retries`` as a parameter (default 2, as v1), an explicit
``--base-url``, replay of recorded ``traces.json`` files, and extra audit fields appended
to every performance_metrics.json record (attempts / stream_interrupted / stream_error /
finish_reason / attempt_log, and http_status now filled in).  See cli.py for usage.
"""
