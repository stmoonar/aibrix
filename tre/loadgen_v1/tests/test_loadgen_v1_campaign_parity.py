"""The campaign's E1 arms and smoke-E1's run_arm.sh send the same request (2026-10-01).

campaign_queue runs ``tre_replayer.run_trace --client-profile e1_v1``; run_arm.sh runs
``python3 -m tre_loadgen_v1`` (this package). Both must put v1's request on the wire:
here the same requests go through both against the local fake server and the server's
view - method, path, every header, the body and its key order - must be identical.
"""

from __future__ import annotations

import uuid

from loadgen_v1_testlib import write_config
from tre_loadgen_v1.client_dispatcher import ClientDispatcher
from tre_loadgen_v1.config_manager import ConfigManager
from tre_loadgen_v1.trace_types import RequestTrace


def test_the_campaign_e1_client_sends_byte_for_byte_what_run_arm_sh_sends(fake_server, tmp_path, monkeypatch):
    from tre_replayer import run_trace
    from tre_replayer.engine import prompt_store
    from tre_replayer.engine.schedule import ScheduledRequest

    tag = uuid.uuid4().hex[:8]
    ids = [f"{tag}-{i}" for i in range(6)]
    prompts = {rid: f"{tag} prompt {i} 中文 text" for i, rid in enumerate(ids)}
    schedule = [ScheduledRequest(rid, "ok", 0.1 + 0.05 * i, prompt_tokens=8, max_output_tokens=3 + i % 2)
                for i, rid in enumerate(ids)]
    # the campaign builds its prompts from the trace's lengths; here they are given
    monkeypatch.setattr(run_trace, "materialize_prompts",
                        lambda events, **kw: prompt_store.PromptStore(prompts, api=kw.get("api")))

    before = len(fake_server.records())
    rows, _, client = run_trace._run_e1(
        schedule, gateway_url=fake_server.url + "/v1/completions", prompt_path=None, prompt_workers=None,
        routing_strategy="least-gpu-cache", corpus_lang="mix", zh_ratio=0.5, processes=2, max_retries=0,
    )
    campaign = fake_server.records()[before:]

    cfg = write_config(tmp_path / "c.yaml", tmp_path / "out", max_retries=0)
    cm = ConfigManager(str(cfg))
    cm.load_config()
    cm.config.gateway_endpoint = fake_server.url
    traces = [RequestTrace(request_id=e.request_id, timestamp=e.scheduled_offset_s, model_name=e.model,
                           prompt=prompts[e.request_id], prompt_length=8, phase_type="stable",
                           max_output_tokens=e.max_output_tokens) for e in schedule]
    mark = len(fake_server.records())
    ClientDispatcher(cm).dispatch_traces(traces)
    shell = fake_server.records()[mark:]

    def view(records):
        out = {}
        for r in records:
            out[r["body"]["messages"][0]["content"]] = (r["method"], r["path"], r["headers"], r["body"],
                                                         list(r["body"]))
        return out

    a, b = view(campaign), view(shell)
    assert sorted(a) == sorted(b) == sorted(prompts.values())
    for prompt in prompts.values():
        assert a[prompt] == b[prompt], prompt
    sample = a[prompts[ids[1]]][3]
    assert "ignore_eos" not in sample and sample["temperature"] is None and sample["max_tokens"] == 4
    assert sample["stream"] is True and a[prompts[ids[1]]][1] == "/v1/chat/completions"
    # the campaign's rows: the e1 record plus the replay keys on the strict basis
    assert len(rows) == 6 and all(r["client_profile"] == "e1_v1" and r["http_status"] == 200 for r in rows)
    assert all(r["ttft_ms"] == r["ttft_strict_s"] * 1000.0 for r in rows)
    assert client["profile"]["name"] == "e1_v1" and client["processes"] == 2
