"""客户端调度器 - 薄壳（2026-09-30 起）

发送不再在这里实现。全项目只有一个发送客户端：``tre_replayer``（``tre/replayer``）的统一
发送核心，本模块用它的 ``e1_v1`` profile：

* 请求：与 v1 相同 —— OpenAI SDK ``AsyncOpenAI.chat.completions.create``，客户端参数与 v1
  的 ``WorkerProcess.create_client`` 相同（api_key 占位、``<gateway>/v1``、max_retries、
  timeout、``routing-strategy`` 头）；请求参数与 v1 的 ``send_request_streaming`` 相同
  （messages=单条 user、temperature 取模型配置（未配置 = JSON null）、stream +
  include_usage、max_tokens 取 trace 否则取模型配置；默认不加 ignore_eos，
  ``--ignore-eos`` / 配置 ``client.ignore_eos: true`` 时每个请求加 ``ignore_eos: true``；
  ``--send-in-tokens`` / 配置 ``client.send_in_tokens: true`` 时每个请求加
  ``x-tre-bl-in-tokens`` 头 = 套 chat 模板后的 prompt token 数，fork 之前在父进程逐请求
  预先算好（``tre_replayer.engine.in_tokens``），算不出的请求不发该头并计数）。
* 并发模型：与 v1 相同 —— ``process_count`` 个进程 × 每进程一个 asyncio 事件循环 × 每进程
  一个 SDK 客户端（httpx 连接池 1000/100）。不同处：schedule 事先按请求轮转分片给各进程，
  各进程按绝对时间自行发送（``tre_replayer.engine.procpool``），不再有 v1 那个在事件循环
  里阻塞的 ``Queue.get(timeout=0.1)`` 交接。
* 记录：``performance_metrics.json`` 每行仍是 v1 字段（v1 口径、v1 公式）+ v1 移植时加的
  审计字段，之后追加严格口径与发送迟到字段（``tre_replayer.engine.http_sender.E1_EXTRA_FIELDS``）。

差异逐项见 ``tre/replayer/README.md``。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


#: The sending core this shell belongs with: the sibling package ``tre/replayer``.
SIBLING_REPLAYER = Path(__file__).resolve().parents[2] / "replayer"


def _ensure_replayer_importable() -> None:
    """``python3 -m tre_loadgen_v1`` is run with only ``tre/loadgen_v1`` on PYTHONPATH
    (smoke/E1 run_arm.sh); the sending core is the sibling package ``tre/replayer``.
    When some other ``tre_replayer`` is importable already, it is used - and said loudly,
    because then the client is not the one this checkout ships."""
    try:
        import tre_replayer
    except ImportError:
        sys.path.insert(0, str(SIBLING_REPLAYER))
        return
    found = Path(tre_replayer.__file__).resolve().parent.parent
    if found != SIBLING_REPLAYER:
        print(f"WARNING: tre_loadgen_v1 sends through tre_replayer from {found}, not the sibling "
              f"{SIBLING_REPLAYER} (check PYTHONPATH); the run's provenance records which",
              file=sys.stderr)


_ensure_replayer_importable()

from tre_replayer.engine.api import API_CHAT  # noqa: E402
from tre_replayer.engine.in_tokens import RECORD_FIELD as IN_TOKENS_FIELD  # noqa: E402
from tre_replayer.engine.in_tokens import precount_in_tokens  # noqa: E402
from tre_replayer.engine.http_sender import (  # noqa: E402
    E1_EXTRA_FIELDS,
    V1_AUDIT_FIELDS,
    V1_RECORD_FIELDS,
    StreamingHttpSender,
)
from tre_replayer.engine.metrics import summarize_v1_records  # noqa: E402
from tre_replayer.engine.procpool import ProcessPoolRunner, RunnerError  # noqa: E402
from tre_replayer.engine.profiles import PROFILE_E1_V1, V1ChatOptions  # noqa: E402
from tre_replayer.engine.schedule import ScheduledRequest  # noqa: E402

from .config_manager import ConfigManager  # noqa: E402
from .trace_types import RequestTrace  # noqa: E402

#: performance_metrics.json 每行的字段顺序：v1 字段、v1 移植时的审计字段、统一客户端新增字段。
RECORD_FIELDS = V1_RECORD_FIELDS + V1_AUDIT_FIELDS + E1_EXTRA_FIELDS


def v1_options_from_config(config: Any) -> V1ChatOptions:
    """v1 配置（models / client 段）-> e1_v1 请求参数。"""
    client = config.client
    return V1ChatOptions(
        model_params={m.name: {"max_tokens": m.max_tokens, "temperature": m.temperature} for m in config.models},
        api_key=getattr(client, "api_key", "") or "",
        max_retries=int(getattr(client, "max_retries", 2)),
        timeout_s=float(getattr(client, "timeout", 300.0)),
        routing_strategy=getattr(client, "routing_algorithm", None) or None,
        streaming=bool(getattr(client, "enable_streaming", True)),
        ignore_eos=bool(getattr(client, "ignore_eos", False)),
        send_in_tokens=bool(getattr(client, "send_in_tokens", False)),
    )


def scheduled_requests(traces: List[RequestTrace]) -> List[ScheduledRequest]:
    """trace -> 统一客户端的调度请求：时间戳、prompt 原文、per-request max_output_tokens 原样。"""
    return [
        ScheduledRequest(request_id=t.request_id, model=t.model_name, scheduled_offset_s=float(t.timestamp),
                         prompt=t.prompt, max_output_tokens=getattr(t, "max_output_tokens", None))
        for t in traces
    ]


def validate_request(request: ScheduledRequest) -> None:
    """e1_v1 sends the trace's own prompt: a request without one is refused before any
    worker starts (it would otherwise fail inside a worker, mid-run)."""
    if not request.prompt:
        raise ValueError(f"trace request {request.request_id} has no prompt; e1_v1 sends the trace's own text")


def ordered_record(record: Dict[str, Any], phase_type: Optional[str],
                   extra_fields: tuple = ()) -> Dict[str, Any]:
    """按 RECORD_FIELDS 排好字段（opt-in 字段 ``extra_fields`` 接在最后），并填上 trace 的 phase_type。"""
    record = dict(record)
    record["phase_type"] = phase_type if phase_type is not None else "unknown"
    return {key: record.get(key) for key in RECORD_FIELDS + tuple(extra_fields)}


class ClientDispatcher:
    """把 trace 交给统一客户端（e1_v1 profile，``process_count`` 个进程）并落盘结果。"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.logger = self._setup_logger()
        self.process_count = int(getattr(self.config.client, "process_count", 4))
        self.options = v1_options_from_config(self.config)
        self.provenance: Dict[str, Any] = {}
        self.workers: List[Dict[str, Any]] = []
        #: --send-in-tokens 的预计数汇总（关时为 None）
        self.in_tokens_summary: Optional[Dict[str, Any]] = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ClientDispatcher")
        level = getattr(logging, str(getattr(self.config.client, "log_level", "INFO")).upper(), logging.INFO)
        logger.setLevel(level)
        output_paths = self.config_manager.get_output_paths()
        log_dir = os.path.dirname(output_paths.get("performance_metrics", "./output"))
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.abspath(os.path.join(log_dir, "client_dispatcher_main.log"))
        if not any(getattr(h, "baseFilename", None) == log_file for h in logger.handlers):
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def make_sender(self, index: int, in_flight, on_record) -> StreamingHttpSender:
        return StreamingHttpSender(
            self.config.gateway_endpoint, profile=PROFILE_E1_V1, v1_options=self.options,
            in_flight=in_flight, on_record=on_record, process_id=index,
        )

    def dispatch_traces(self, traces: List[RequestTrace]) -> List[Dict[str, Any]]:
        """按 trace 时间戳开环发送全部请求，返回 performance_metrics.json 的各行（完成顺序）。"""
        if not traces:
            self.logger.warning("没有请求需要调度")
            return []
        phase = {t.request_id: t.phase_type for t in traces}
        requests = scheduled_requests(traces)
        extra_fields: tuple = ()
        if self.options.send_in_tokens:
            # e1_v1 发 chat：头的值 = trace 原文套 chat 模板后的 token 数；fork 前一次算完
            requests, self.in_tokens_summary = precount_in_tokens(
                requests, api=API_CHAT, tokenizer_paths=getattr(self.config.client, "tokenizer_paths", None))
            extra_fields = (IN_TOKENS_FIELD,)
            self.logger.info("x-tre-bl-in-tokens 预计数: " + json.dumps(self.in_tokens_summary, ensure_ascii=False))
            if self.in_tokens_summary["omitted"]:
                self.logger.warning(f"{self.in_tokens_summary['omitted']} 个请求算不出 prompt token 数，不发该头")
        self.logger.info(f"开始调度 {len(requests)} 个请求；进程数 {self.process_count}；"
                         f"网关 {self.config.gateway_endpoint}")
        self.provenance = self.make_sender(0, None, None).provenance(processes=self.process_count)
        failure: Optional[RunnerError] = None
        # fork 之前在父进程校验：e1_v1 发送 trace 自带的 prompt，缺了就整体拒绝
        with ProcessPoolRunner(requests, self.make_sender, processes=self.process_count,
                               validate=validate_request) as runner:
            try:
                run = runner.run()
                self.workers = run.workers
                received = run.records
            except RunnerError as exc:
                # 运行中途失败：已收到的记录照样落盘，再把失败抛给调用方（运行标记为失败）
                failure = exc
                self.workers = exc.workers
                received = exc.records
        records = [ordered_record(r, phase.get(r["request_id"]), extra_fields) for r in received]
        missing = len(requests) - len(records)
        if missing:
            self.logger.warning(f"{missing} 个请求没有结果记录")
        metrics_file = self.config_manager.get_output_paths()["performance_metrics"]
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        self.logger.info(f"性能指标已写入: {metrics_file}")
        if failure is not None:
            self.logger.error(f"发送中途失败，已写入 {len(records)}/{len(requests)} 条记录: {failure}")
            raise failure
        summary = summarize_v1_records(records)
        self.logger.info("汇总: " + json.dumps({k: v for k, v in summary.items() if k != "bases"}, ensure_ascii=False))
        self._plots(metrics_file, records)
        return records

    # ------------------------------------------------------------------ plots (v1 输出)

    def _plots(self, metrics_file: str, records: List[Dict[str, Any]]) -> None:
        try:
            from . import plots
        except Exception as exc:  # noqa: BLE001 - 画图失败不影响数据
            self.logger.error(f"画图模块不可用: {exc}")
            return
        output_paths = self.config_manager.get_output_paths()
        plot_dir = output_paths.get("performance_plots")
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            load_plot = os.path.join(plot_dir, "process_load_over_time.png")
        else:
            load_plot = os.path.join(os.path.dirname(metrics_file), "process_load_over_time.png")
        plots.process_load_plot(records, load_plot, self.logger)
        plots.actual_send_rps_plot(metrics_file, self.logger)


def create_client_dispatcher(config_manager: ConfigManager) -> ClientDispatcher:
    """创建客户端调度器的便捷函数"""
    return ClientDispatcher(config_manager)


def per_process_counts(records: List[Dict[str, Any]]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for record in records:
        counts[int(record.get("process_id") or 0)] += 1
    return dict(counts)
