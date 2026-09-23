"""
客户端调度器模块 - 多进程负载均衡版本

基于原始client.py的逻辑，适配CustomTraceGenerator项目
负责读取trace数据，按时间戳精确调度HTTP请求，记录性能指标
支持多进程负载均衡和协程管理
"""

import logging
import time
import asyncio
import openai
import json
import os
import multiprocessing as mp
from multiprocessing import Queue, Manager, Process, Event
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import contextvars
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
import threading
from queue import Empty
import signal
import sys

# 设置matplotlib字体配置 - 优先使用系统可用字体
matplotlib.rcParams['font.sans-serif'] = [
    'DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif'
]
matplotlib.rcParams['axes.unicode_minus'] = False
# 设置字体大小
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9

from .config_manager import ConfigManager
# [v2 port] v1: from trace_generator import RequestTrace（同一个 dataclass，拆到 trace_types 以免分发路径 import transformers）
from .trace_types import RequestTrace


# ---------------------------------------------------------------------------
# [v2 port] 审计字段：逐次尝试（含 SDK 内部重试）记录
#
# OpenAI SDK 的重试循环在调用 create() 的同一个协程里执行，每次尝试都会经过
# httpx.AsyncClient.send()，从而触发 client 级 event hook（request / response）。
# 我们在每个请求协程里把一个 AttemptTracker 放进 ContextVar，hook 读取当前协程
# 上下文里的 tracker 记账。不改动线上请求（不加任何 header），也不依赖 SDK 私有 API。
#   - request hook: 追加一条 {"t": 发出时刻, "status": None}
#   - response hook: 把最近一条的 status 填为 HTTP 状态码（流式时在收到响应头时触发）
# status 仍为 None 的条目 = 该次尝试没有拿到响应头（连接错误/超时）。
# ---------------------------------------------------------------------------
class AttemptTracker:
    """单个请求的逐次尝试记录"""

    def __init__(self):
        self.attempts: List[Dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self.attempts)

    @property
    def last_status(self) -> Optional[int]:
        return self.attempts[-1]["status"] if self.attempts else None


_ATTEMPT_TRACKER: "contextvars.ContextVar[Optional[AttemptTracker]]" = contextvars.ContextVar(
    "tre_loadgen_v1_attempt_tracker", default=None
)


async def _on_request_hook(request):
    tracker = _ATTEMPT_TRACKER.get()
    if tracker is not None:
        tracker.attempts.append({"t": time.time(), "status": None})


async def _on_response_hook(response):
    tracker = _ATTEMPT_TRACKER.get()
    if tracker is not None and tracker.attempts:
        tracker.attempts[-1]["status"] = response.status_code


@dataclass
class RequestRecord:
    """请求记录 - 从trace转换而来"""
    request_id: str
    timestamp: float
    model_name: str
    prompt: str
    prompt_length: int
    phase_type: str
    session_id: Optional[int] = None
    max_output_tokens: Optional[int] = None  # 来自real trace的per-request max_tokens


@dataclass
class ResponseRecord:
    """响应记录 - 性能指标"""
    request_id: str
    model_name: str
    timestamp: float
    start_time: float
    end_time: float
    e2e_latency: float
    ttft: Optional[float]
    tpot: Optional[float]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    success: bool
    error_message: Optional[str] = None
    http_status: Optional[int] = None
    phase_type: str = "unknown"
    target_pod: Optional[str] = None
    process_id: int = 0  # 添加进程ID字段
    # ---- [v2 port] 以下为新增审计字段（追加在 v1 字段之后，不改变 v1 字段及其语义）----
    attempts: int = 0                      # 实际 HTTP 尝试次数（1 + SDK 重试次数）
    stream_interrupted: bool = False       # 流式读取中途抛异常（v1 仍记 success=True）
    stream_error: Optional[str] = None     # 流中途异常文本 "<类型>: <消息>"
    finish_reason: Optional[str] = None    # 最后一个非空 finish_reason（stop/length/...）
    attempt_log: List[Dict[str, Any]] = field(default_factory=list)  # 每次尝试 {"t", "status"}


@dataclass
class ProcessStats:
    """进程统计信息"""
    process_id: int
    active_coroutines: int
    total_processed: int
    timestamp: float


@dataclass
class TaskBatch:
    """任务批次"""
    batch_id: int
    requests: List[RequestRecord]
    start_time: float
    end_time: float
    base_time: float = 0.0 # 新增：用于传递基准时间


class WorkerProcess:
    """工作进程类"""
    
    def __init__(self, process_id: int, config: Any, task_queue: Queue, 
                 result_queue: Queue, stats_queue: Queue, stop_event: Event):
        self.process_id = process_id
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stats_queue = stats_queue
        self.stop_event = stop_event
        self.active_coroutines = 0
        self.total_processed = 0
        self.logger = None
        self.client = None
        
    def setup_logger(self):
        """设置进程专用日志器"""
        logger = logging.getLogger(f"WorkerProcess-{self.process_id}")
        log_level_str = getattr(self.config.client, 'log_level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)
        
        if not logger.handlers:
            # 创建文件处理器 - 放入配置的输出目录下的process_log子目录
            output_dir = self.config.output.get_full_output_dir()
            log_dir = os.path.join(output_dir, "process_log")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"worker_process_{self.process_id}.log")
            
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                f'%(asctime)s - Worker-{self.process_id} - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        self.logger = logger

    def create_client(self) -> openai.AsyncOpenAI:
        """创建OpenAI客户端"""
        api_key = getattr(self.config.client, 'api_key', None)
        timeout_val = getattr(self.config.client, 'timeout', 300.0)
        routing_strategy = getattr(self.config.client, 'routing_algorithm', 'random')
        
        # [v2 port] v1 硬编码 max_retries=2；现从配置/CLI 读取，默认仍为 2
        max_retries = getattr(self.config.client, 'max_retries', 2)

        if not api_key or api_key == "dummy":
            api_key = "dummy-key-for-local-gateway"

        # [v2 port] 传入自建 httpx client 仅为挂 event hook 统计逐次尝试。
        # DefaultAsyncHttpxClient 的 limits / follow_redirects 与 SDK 自建 client 的默认值相同，
        # timeout 同样取 timeout_val（v1 中 SDK 以 timeout=timeout_val 自建 client），因此线上行为不变。
        http_client = openai.DefaultAsyncHttpxClient(
            timeout=timeout_val,
            event_hooks={"request": [_on_request_hook], "response": [_on_response_hook]},
        )

        client = openai.AsyncOpenAI(
            api_key=api_key,  # OpenAI API密钥，用于身份验证
            base_url=f"{self.config.gateway_endpoint}/v1",  # API网关的基础URL
            max_retries=max_retries,  # 最大重试次数，当请求失败时自动重试
            timeout=timeout_val,  # 单个HTTP请求的超时时间（秒），包括连接建立、发送请求和接收响应的总时间
            http_client=http_client,
        )
        
        if routing_strategy:
            client = client.with_options(
                default_headers={"routing-strategy": routing_strategy}
            )
        
        return client

    def get_model_config(self, model_name: str):
        """获取模型配置"""
        for model in self.config.models:
            if model.name == model_name:
                return model
        
        from .config_manager import ModelConfig
        return ModelConfig(name=model_name, modelscope_url="", max_tokens=128, temperature=0.0)
    
    def prepare_prompt(self, prompt: str, session_id: Optional[int] = None) -> List[Dict]:
        """准备prompt消息格式"""
        return [{"role": "user", "content": prompt}]

    async def send_request_streaming(self, request: RequestRecord, target_time: float) -> ResponseRecord:
        """流式请求处理"""
        prompt = self.prepare_prompt(request.prompt)
        task_start_time = time.time()
        first_response_time = None
        target_pod = ""
        response_stream = None  # 初始化为 None，以便在 finally 中检查
        # [v2 port] 审计：本协程专属的尝试记录器（见 AttemptTracker）
        tracker = AttemptTracker()
        tracker_token = _ATTEMPT_TRACKER.set(tracker)
        stream_interrupted = False
        stream_error_text = None
        finish_reason = None

        try:
            # 等待到目标时间
            sleep_time = target_time - task_start_time
            if sleep_time > 0:
                self.logger.info(f"请求 {request.request_id} 需等待发送: sleep={sleep_time:.3f}s")
                await asyncio.sleep(sleep_time)
            else:
                self.logger.error(f"请求 {request.request_id} 调度超时，sleep={sleep_time:.3f}s")
            
            # 记录实际发送时间
            actual_send_time = time.time()
            
            # 获取模型配置
            model_config = self.get_model_config(request.model_name)
            max_tokens = getattr(model_config, 'max_tokens', None)
            # per-request max_output_tokens优先于模型配置
            if request.max_output_tokens is not None:
                max_tokens = request.max_output_tokens
            temperature = getattr(model_config, 'temperature', 0.0)

            # 构造请求参数
            request_kwargs = dict(
                model=request.model_name,
                messages=prompt,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens

            # 创建流式请求
            response_stream = await self.client.chat.completions.create(**request_kwargs)
            
            # 提取响应头信息
            if hasattr(response_stream, 'response') and hasattr(response_stream.response, 'headers'):
                target_pod = response_stream.response.headers.get('target-pod')
            
            # 处理流式响应
            text_chunks = []
            prompt_tokens = 0
            output_tokens = 0
            total_tokens = 0
            
            try:
                async for chunk in response_stream:
                    if chunk.choices:
                        if chunk.choices[0].delta.content is not None:
                            if not first_response_time:
                                first_response_time = time.time()
                            output_text = chunk.choices[0].delta.content
                            text_chunks.append(output_text)
                        # [v2 port] 审计：记录 finish_reason（不参与任何判定）
                        chunk_finish_reason = getattr(chunk.choices[0], 'finish_reason', None)
                        if chunk_finish_reason is not None:
                            finish_reason = chunk_finish_reason

                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        if chunk.usage.prompt_tokens is not None:
                            prompt_tokens = chunk.usage.prompt_tokens
                        if chunk.usage.completion_tokens is not None:
                            output_tokens = chunk.usage.completion_tokens
                        if chunk.usage.total_tokens is not None:
                            total_tokens = chunk.usage.total_tokens

            except Exception as stream_error:
                self.logger.error(f"请求 {request.request_id} 流式处理中断: {stream_error}")
                # [v2 port] 审计：v1 在此吞掉异常并继续记 success=True；我们保持该行为，只额外记录
                stream_interrupted = True
                stream_error_text = f"{type(stream_error).__name__}: {stream_error}"

            response_end_time = time.time()
            
            # 计算性能指标
            e2e_latency = response_end_time - actual_send_time
            ttft = first_response_time - actual_send_time if first_response_time else None
            tpot = (response_end_time - first_response_time) / output_tokens if first_response_time and output_tokens > 0 else None
            
            # 创建响应记录
            result = ResponseRecord(
                request_id=request.request_id,
                model_name=request.model_name,
                timestamp=request.timestamp,
                start_time=actual_send_time,
                end_time=response_end_time,
                e2e_latency=e2e_latency,
                ttft=ttft,
                tpot=tpot,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                success=True,
                http_status=tracker.last_status,  # [v2 port] v1 恒为 None
                phase_type=request.phase_type,
                target_pod=target_pod,
                process_id=self.process_id,
                attempts=tracker.count,
                stream_interrupted=stream_interrupted,
                stream_error=stream_error_text,
                finish_reason=finish_reason,
                attempt_log=list(tracker.attempts),
            )

            self.logger.debug(f"请求 {request.request_id} 完成 - E2E: {e2e_latency:.3f}s")
            
            # 直接提交结果到队列
            self.result_queue.put(result)
            self.total_processed += 1
            
            return result
            
        except Exception as e:
            error_time = time.time()
            self.logger.error(f"请求 {request.request_id} 失败: {e}")
            # [v2 port] 审计：失败时的 HTTP 状态（APIStatusError 带 status_code；超时/连接错误为 None）
            fail_status = getattr(e, 'status_code', None)
            if fail_status is None:
                fail_status = tracker.last_status
            # put失败结果
            fail_result = ResponseRecord(
                request_id=request.request_id,
                model_name=request.model_name,
                timestamp=request.timestamp,
                start_time=actual_send_time if 'actual_send_time' in locals() else task_start_time,
                end_time=error_time,
                e2e_latency=error_time - (actual_send_time if 'actual_send_time' in locals() else task_start_time),
                ttft=None,
                tpot=None,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                success=False,
                error_message=str(e),
                http_status=fail_status,
                phase_type=request.phase_type,
                target_pod=target_pod,
                process_id=self.process_id,
                attempts=tracker.count,
                stream_interrupted=stream_interrupted,
                stream_error=stream_error_text,
                finish_reason=finish_reason,
                attempt_log=list(tracker.attempts),
            )
            try:
                self.result_queue.put(fail_result)
            except Exception as put_e:
                self.logger.error(f"put失败结果到result_queue异常: {put_e}")
            return fail_result
        finally:
            _ATTEMPT_TRACKER.reset(tracker_token)  # [v2 port]
            # 确保在所有情况下都关闭流式连接
            if response_stream is not None:
                try:
                    # 不同版本的 openai SDK 可能使用不同的方法名来关闭连接
                    # 较新版本使用 aclose，较旧版本使用 close
                    close_method = getattr(response_stream, 'aclose', None)
                    if close_method is None:
                        close_method = getattr(response_stream, 'close', None)
                    
                    if close_method and callable(close_method):
                        await close_method() if asyncio.iscoroutinefunction(close_method) else close_method()
                        self.logger.debug(f"请求 {request.request_id} 的流式连接已关闭")
                except Exception as close_error:
                    self.logger.error(f"关闭请求 {request.request_id} 的流式连接时出错: {close_error}")

    async def send_request_batch(self, request: RequestRecord, target_time: float) -> ResponseRecord:
        """批量请求处理"""
        prompt = self.prepare_prompt(request.prompt)
        task_start_time = time.time()
        target_pod = ""
        # [v2 port] 审计：本协程专属的尝试记录器
        tracker = AttemptTracker()
        tracker_token = _ATTEMPT_TRACKER.set(tracker)

        try:
            # 等待到目标时间
            sleep_time = target_time - task_start_time
            if sleep_time > 0:
                self.logger.info(f"请求 {request.request_id} 需等待发送: sleep={sleep_time:.3f}s")
                await asyncio.sleep(sleep_time)
            else:
                self.logger.error(f"请求 {request.request_id} 调度超时，sleep={sleep_time:.3f}s")
            
            # 记录实际发送时间
            actual_send_time = time.time()
            
            # 获取模型配置
            model_config = self.get_model_config(request.model_name)
            max_tokens = getattr(model_config, 'max_tokens', None)
            # per-request max_output_tokens优先于模型配置
            if request.max_output_tokens is not None:
                max_tokens = request.max_output_tokens
            temperature = getattr(model_config, 'temperature', 0.0)

            # 构造请求参数
            request_kwargs = dict(
                model=request.model_name,
                messages=prompt,
                temperature=temperature,
            )
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens

            # 创建批量请求
            response = await self.client.chat.completions.create(**request_kwargs)
            
            # 提取响应头信息
            if hasattr(response, 'response') and hasattr(response.response, 'headers'):
                target_pod = response.response.headers.get('target-pod')
            
            response_end_time = time.time()
            e2e_latency = response_end_time - actual_send_time
            
            # 提取token信息
            prompt_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # 创建响应记录
            result = ResponseRecord(
                request_id=request.request_id,
                model_name=request.model_name,
                timestamp=request.timestamp,
                start_time=actual_send_time,
                end_time=response_end_time,
                e2e_latency=e2e_latency,
                ttft=None,
                tpot=None,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                success=True,
                http_status=tracker.last_status,  # [v2 port] v1 恒为 None
                phase_type=request.phase_type,
                target_pod=target_pod,
                process_id=self.process_id,
                attempts=tracker.count,
                finish_reason=(response.choices[0].finish_reason if getattr(response, 'choices', None) else None),
                attempt_log=list(tracker.attempts),
            )

            self.logger.debug(f"批量请求 {request.request_id} 完成 - E2E: {e2e_latency:.3f}s")
            
            # 直接提交结果到队列
            self.result_queue.put(result)
            self.total_processed += 1
            
            return result
            
        except Exception as e:
            error_time = time.time()
            self.logger.error(f"批量请求 {request.request_id} 失败: {e}")
            fail_status = getattr(e, 'status_code', None)  # [v2 port] 审计
            if fail_status is None:
                fail_status = tracker.last_status
            fail_result = ResponseRecord(
                request_id=request.request_id,
                model_name=request.model_name,
                timestamp=request.timestamp,
                start_time=actual_send_time if 'actual_send_time' in locals() else task_start_time,
                end_time=error_time,
                e2e_latency=error_time - (actual_send_time if 'actual_send_time' in locals() else task_start_time),
                ttft=None,
                tpot=None,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                success=False,
                error_message=str(e),
                http_status=fail_status,
                phase_type=request.phase_type,
                target_pod=target_pod,
                process_id=self.process_id,
                attempts=tracker.count,
                attempt_log=list(tracker.attempts),
            )
            try:
                self.result_queue.put(fail_result)
            except Exception as put_e:
                self.logger.error(f"put失败结果到result_queue异常: {put_e}")
            return fail_result
        finally:
            _ATTEMPT_TRACKER.reset(tracker_token)  # [v2 port]

    async def process_task_batch(self, batch: TaskBatch, base_time: float):
        """处理任务批次 - 不等待任务完成，让协程自然运行
        base_time 必须为dispatch_traces的绝对基准时间
        """
        self.logger.info(f"处理批次 {batch.batch_id}，包含 {len(batch.requests)} 个请求")
        for request in batch.requests:
            # 计算目标发送时间：基准时间 + 请求的相对时间戳
            target_time = base_time + request.timestamp
            if getattr(self.config.client, 'enable_streaming', True):
                asyncio.create_task(self.send_request_streaming(request, target_time))
            else:
                asyncio.create_task(self.send_request_batch(request, target_time))
        self.logger.info(f"批次 {batch.batch_id} 任务已创建，协程将自然运行")

    def get_active_coroutines_count(self) -> int:
        """获取当前活跃协程数量"""
        try:
            # 获取当前事件循环中所有正在运行的任务
            loop = asyncio.get_running_loop()
            tasks = asyncio.all_tasks(loop)
            
            # 过滤掉当前协程本身和已完成的任务
            active_tasks = [task for task in tasks 
                          if not task.done() and not task.cancelled()]
            
            return len(active_tasks)
        except RuntimeError:
            # 如果没有运行中的事件循环，返回0
            return 0

    def report_stats(self):
        """报告进程统计信息"""
        # 获取真实的活跃协程数量
        self.active_coroutines = self.get_active_coroutines_count()
        
        stats = ProcessStats(
            process_id=self.process_id,
            active_coroutines=self.active_coroutines,
            total_processed=self.total_processed,
            timestamp=time.time()
        )
        try:
            self.stats_queue.put_nowait(stats)
        except:
            pass  # 忽略队列满的情况

    async def run(self):
        """运行工作进程"""
        self.setup_logger()
        self.client = self.create_client()
        self.logger.info(f"工作进程 {self.process_id} 启动")
        last_stats_time = time.time()
        stats_interval = getattr(self.config.client, 'load_monitor_interval', 1.0)
        # 新增：worker进程启动时等待主进程传递base_time
        base_time = None
        while base_time is None:
            try:
                # 取第一个batch时，主进程需在batch附带base_time
                batch = self.task_queue.get(timeout=0.1)
                if hasattr(batch, 'base_time'):
                    base_time = batch.base_time
                    self.logger.info(f"收到主进程基准时间: {base_time}")
                    # 重新放回队列，正常流程处理
                    self.task_queue.put(batch)
                    break
                else:
                    # 兼容老逻辑，直接break
                    base_time = time.time()
                    self.logger.warning("未收到主进程基准时间，使用当前时间作为base_time")
                    self.task_queue.put(batch)
                    break
            except Empty:
                await asyncio.sleep(0.01)
        try:
            while not self.stop_event.is_set():
                try:
                    batch = self.task_queue.get(timeout=0.1)
                    if batch is None:
                        break
                    current_coroutines = self.get_active_coroutines_count()
                    max_coroutines = getattr(self.config.client, 'max_coroutines_per_process', 500)
                    if current_coroutines >= max_coroutines:
                        self.logger.warning(f"进程 {self.process_id} 协程数量超过预设限制 {max_coroutines}，当前: {current_coroutines}，继续处理...")
                    else:
                        self.logger.debug(f"进程 {self.process_id} 协程数量: {current_coroutines}/{max_coroutines}")
                    # 只用base_time，不用batch.start_time
                    await self.process_task_batch(batch, base_time)
                except Empty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    self.logger.error(f"工作进程 {self.process_id} 执行错误: {e}")
                    # 发生异常时，尝试将batch内所有请求都put失败结果
                    if 'batch' in locals() and hasattr(batch, 'requests'):
                        for req in batch.requests:
                            fail_result = ResponseRecord(
                                request_id=req.request_id,
                                model_name=req.model_name,
                                timestamp=req.timestamp,
                                start_time=time.time(),
                                end_time=time.time(),
                                e2e_latency=0.0,
                                ttft=None,
                                tpot=None,
                                input_tokens=0,
                                output_tokens=0,
                                total_tokens=0,
                                success=False,
                                error_message=f"worker进程异常: {e}",
                                phase_type=req.phase_type,
                                target_pod=None,
                                process_id=self.process_id
                            )
                            try:
                                self.result_queue.put(fail_result)
                            except Exception as put_e:
                                self.logger.error(f"put失败结果到result_queue异常: {put_e}")
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    self.report_stats()
                    last_stats_time = current_time
        except Exception as e:
            self.logger.error(f"工作进程 {self.process_id} 运行异常: {e}")
            # 进程即将退出时，尝试put所有未完成任务为失败
            # 这里无法直接获取未完成请求，只能依赖上层调度保证
        finally:
            self.logger.info(f"工作进程 {self.process_id} 关闭")


def worker_process_main(process_id: int, config: Any, task_queue: Queue, 
                       result_queue: Queue, stats_queue: Queue, stop_event: Event):
    """工作进程主函数"""
    # 设置信号处理
    def signal_handler(signum, frame):
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建工作进程实例
    worker = WorkerProcess(process_id, config, task_queue, result_queue, stats_queue, stop_event)
    
    # 运行异步主循环
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"工作进程 {process_id} 异常退出: {e}")


class ClientDispatcher:
    """客户端调度器 - 多进程负载均衡版本"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.logger = self._setup_logger()
        
        # 多进程配置
        self.process_count = getattr(self.config.client, 'process_count', 4)
        self.max_coroutines_per_process = getattr(self.config.client, 'max_coroutines_per_process', 500)
        self.load_monitor_interval = getattr(self.config.client, 'load_monitor_interval', 1.0)
        
        # 调度策略配置
        self.dispatch_window_size = getattr(self.config.client, 'task_batch_window', 5.0)  # 从配置读取调度窗口大小
        
        # 进程管理
        self.processes: List[Process] = []
        self.task_queues: List[Queue] = []
        self.result_queue = Queue()
        self.stats_queue = Queue()
        self.stop_event = Event()
        
        # 监控数据
        self.process_stats_history: List[ProcessStats] = []
        self.actual_send_times = []
        
        self.logger.info(f"初始化多进程ClientDispatcher")
        self.logger.info(f"  - 进程数: {self.process_count}")
        self.logger.info(f"  - 单进程协程数限制: {self.max_coroutines_per_process} (仅记录信息)")
        self.logger.info(f"  - 负载监控间隔: {self.load_monitor_interval}s")
        self.logger.info(f"  - 调度策略: 每{self.dispatch_window_size}s选择最小负载进程")

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ClientDispatcher")
        log_level_str = getattr(self.config.client, 'log_level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)
        
        if not logger.handlers:
            output_paths = self.config_manager.get_output_paths()
            log_dir = os.path.dirname(output_paths.get('performance_metrics', './output'))
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, 'client_dispatcher_main.log')
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger

    def _convert_traces_to_requests(self, traces: List[RequestTrace]) -> List[RequestRecord]:
        """将RequestTrace转换为RequestRecord"""
        requests = []
        for trace in traces:
            request = RequestRecord(
                request_id=trace.request_id,
                timestamp=trace.timestamp,
                model_name=trace.model_name,
                prompt=trace.prompt,
                prompt_length=trace.prompt_length,
                phase_type=trace.phase_type,
                session_id=None,
                max_output_tokens=getattr(trace, 'max_output_tokens', None)
            )
            requests.append(request)
        return requests

    # 移除了_create_time_windows方法，因为现在使用滑动窗口调度逻辑

    def _get_least_loaded_process(self) -> int:
        """获取负载最小的进程ID"""
        # 获取所有进程的当前负载
        current_loads = self._get_all_process_loads()
        
        # 如果没有统计信息，随机选择
        if not current_loads:
            return 0
        
        # 过滤掉过载的进程（协程数超过限制的80%）
        overload_threshold = int(self.max_coroutines_per_process * 0.8)
        available_processes = {pid: load for pid, load in current_loads.items() 
                             if load < overload_threshold}
        
        # 如果有可用进程，选择负载最少的
        if available_processes:
            min_load_process = min(available_processes.items(), key=lambda x: x[1])
            return min_load_process[0]
        
        # 如果所有进程都过载，选择负载最少的
        min_load_process = min(current_loads.items(), key=lambda x: x[1])
        return min_load_process[0]

    def _get_all_process_loads(self) -> Dict[int, int]:
        """获取所有进程的当前协程数"""
        current_loads = {}
        
        # 从统计历史中获取最新的负载数据
        if self.process_stats_history:
            # 按进程ID分组，获取每个进程的最新统计
            latest_stats = {}
            for stats in self.process_stats_history:
                if stats.process_id not in latest_stats or stats.timestamp > latest_stats[stats.process_id].timestamp:
                    latest_stats[stats.process_id] = stats
            
            # 使用最新的统计数据
            for process_id, stats in latest_stats.items():
                current_loads[process_id] = stats.active_coroutines
        
        # 从队列中获取最新的统计信息（非阻塞）
        try:
            while True:
                stats = self.stats_queue.get_nowait()
                current_loads[stats.process_id] = stats.active_coroutines
                self.process_stats_history.append(stats)
        except Empty:
            pass
        
        # 确保所有进程都有数据，没有数据的设为0
        for i in range(self.process_count):
            if i not in current_loads:
                current_loads[i] = 0
        
        return current_loads

    def _check_overload(self) -> bool:
        """检查是否所有进程都过载"""
        current_loads = self._get_all_process_loads()
        
        if not current_loads:
            return False
        
        # 检查是否所有进程都达到最大协程数
        overloaded_count = sum(1 for load in current_loads.values() 
                              if load >= self.max_coroutines_per_process)
        
        return overloaded_count == len(current_loads)

    def start_worker_processes(self):
        """启动工作进程"""
        self.logger.info(f"启动 {self.process_count} 个工作进程")
        
        for i in range(self.process_count):
            # 为每个进程创建任务队列
            task_queue = Queue()
            self.task_queues.append(task_queue)
            
            # 创建工作进程
            process = Process(
                target=worker_process_main,
                args=(i, self.config, task_queue, self.result_queue, 
                     self.stats_queue, self.stop_event)
            )
            process.start()
            self.processes.append(process)
            
            self.logger.info(f"工作进程 {i} 已启动，PID: {process.pid}")

    def stop_worker_processes(self):
        """停止工作进程"""
        self.logger.info("停止所有工作进程")
        
        # 设置停止事件
        self.stop_event.set()
        
        # 向所有队列发送停止信号
        for task_queue in self.task_queues:
            try:
                task_queue.put(None, timeout=1.0)
            except:
                pass
        
        # 等待进程结束
        for i, process in enumerate(self.processes):
            try:
                process.join(timeout=5.0)
                if process.is_alive():
                    self.logger.warning(f"进程 {i} 未正常结束，强制终止")
                    process.terminate()
                    process.join(timeout=2.0)
                    if process.is_alive():
                        process.kill()
            except Exception as e:
                self.logger.error(f"停止进程 {i} 时出错: {e}")

    def collect_results(self, expected_count: int) -> List[ResponseRecord]:
        """收集所有结果"""
        results = []
        collected = 0
        
        self.logger.info(f"开始收集结果，预期 {expected_count} 个")
        
        while collected < expected_count:
            try:
                result = self.result_queue.get(timeout=5.0)
                results.append(result)
                collected += 1
                
                if collected % 100 == 0:
                    self.logger.info(f"已收集 {collected}/{expected_count} 个结果")
                    
            except Empty:
                self.logger.warning(f"等待结果超时，已收集 {collected}/{expected_count}")
                # 检查进程状态
                alive_processes = sum(1 for p in self.processes if p.is_alive())
                if alive_processes == 0:
                    self.logger.warning("所有工作进程已退出，停止等待结果")
                    break
                else:
                    # 输出未完成请求数
                    self.logger.warning(f"部分请求可能因worker进程异常丢失，未收集到结果: {expected_count-collected}")
        
        self.logger.info(f"结果收集完成，共收集 {len(results)} 个")
        return results

    def generate_process_load_plot(self, output_file_path: str):
        """生成进程负载变化图"""
        if not self.process_stats_history:
            self.logger.warning("没有进程统计数据，无法生成负载图")
            return
        
        try:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # 按进程ID分组数据
            process_data = defaultdict(list)
            for stats in self.process_stats_history:
                process_data[stats.process_id].append(stats)
            
            # 获取时间范围
            all_timestamps = [stats.timestamp for stats in self.process_stats_history]
            if not all_timestamps:
                self.logger.warning("没有有效的时间戳数据")
                return
                
            start_time = min(all_timestamps)
            
            plt.figure(figsize=(14, 10))
            
            # 绘制每个进程的协程数变化
            plt.subplot(2, 1, 1)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for process_id, stats_list in process_data.items():
                if not stats_list:
                    continue
                
                timestamps = [(s.timestamp - start_time) for s in stats_list]
                coroutine_counts = [s.active_coroutines for s in stats_list]
                
                color = colors[process_id % len(colors)]
                plt.plot(timestamps, coroutine_counts, 
                        color=color, linewidth=2, 
                        label=f'Process {process_id}', marker='o', markersize=3)
            
            plt.axhline(y=self.max_coroutines_per_process, color='red', 
                       linestyle='--', alpha=0.7, label=f'Max Coroutines Limit: {self.max_coroutines_per_process}')
            
            plt.title('Coroutine Count Changes by Process', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Active Coroutines', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制总体负载分布
            plt.subplot(2, 1, 2)
            
            # 计算每个时间点的总协程数
            time_points = sorted(set(all_timestamps))
            total_coroutines = []
            
            for t in time_points:
                total = 0
                for process_id in process_data.keys():
                    # 找到该时间点最近的统计数据
                    process_stats = [s for s in process_data[process_id] if s.timestamp <= t]
                    if process_stats:
                        total += process_stats[-1].active_coroutines
                total_coroutines.append(total)
            
            relative_times = [(t - start_time) for t in time_points]
            plt.plot(relative_times, total_coroutines, 'black', linewidth=2, label='Total Coroutines')
            
            max_total = self.max_coroutines_per_process * self.process_count
            plt.axhline(y=max_total, color='red', linestyle='--', alpha=0.7, 
                       label=f'System Max Capacity: {max_total}')
            
            plt.title('System Overall Load Changes', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Total Active Coroutines', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"进程负载图已保存至: {output_file_path}")
            
            # 输出统计信息
            self.logger.info("进程负载统计:")
            for process_id, stats_list in process_data.items():
                if stats_list:
                    avg_load = np.mean([s.active_coroutines for s in stats_list])
                    max_load = max([s.active_coroutines for s in stats_list])
                    total_processed = stats_list[-1].total_processed if stats_list else 0
                    self.logger.info(f"  进程 {process_id}: 平均负载 {avg_load:.1f}, "
                                   f"峰值负载 {max_load}, 处理请求 {total_processed}")
                
        except Exception as e:
            self.logger.error(f"生成进程负载图失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def generate_actual_send_rps_plot(self, metrics_file: str):
        """根据实际发送时间生成RPS随时间变化图 - 按模型分组"""
        try:
            # 读取性能指标文件
            if not os.path.exists(metrics_file):
                self.logger.warning(f"性能指标文件不存在: {metrics_file}")
                return
            
            # 解析JSONL格式的性能指标，按模型分组
            model_data = defaultdict(list)
            total_send_times = []
            
            with open(metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'start_time' in data and data['start_time'] and 'model_name' in data:
                            send_time = data['start_time']
                            model_name = data['model_name']
                            model_data[model_name].append(send_time)
                            total_send_times.append(send_time)
                    except json.JSONDecodeError:
                        continue
            
            if not total_send_times:
                self.logger.warning("没有找到有效的发送时间数据")
                return
            
            # 按时间排序
            total_send_times.sort()
            
            # 计算总体RPS
            start_time = min(total_send_times)
            end_time = max(total_send_times)
            total_duration = end_time - start_time
            
            # 设置时间窗口大小（秒）
            window_size = 1.0
            
            # 计算时间窗口数量
            num_windows = int(total_duration / window_size) + 1
            
            # 初始化时间序列
            time_points = []
            for i in range(num_windows):
                window_start = start_time + i * window_size
                window_center = window_start + window_size / 2
                time_points.append(window_center - start_time)  # 相对时间
            
            # 创建图像 - 两个子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            
            # 定义颜色列表
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # 绘制每个模型的RPS（上图）
            model_stats = {}
            for i, (model_name, send_times) in enumerate(model_data.items()):
                if not send_times:
                    continue
                
                # 计算该模型的RPS
                rps_values = []
                for j in range(num_windows):
                    window_start = start_time + j * window_size
                    window_end = window_start + window_size
                    
                    # 统计该窗口内该模型的请求数
                    requests_in_window = sum(1 for t in send_times 
                                           if window_start <= t < window_end)
                    
                    # 计算RPS（请求数/窗口大小）
                    rps = requests_in_window / window_size
                    rps_values.append(rps)
                
                # 绘制该模型的RPS曲线（细线）
                color = colors[i % len(colors)]
                ax1.plot(time_points, rps_values, color=color, linewidth=1, 
                        label=f'{model_name}', marker='o', markersize=2, alpha=0.8)
                
                # 计算该模型的统计信息
                avg_rps = np.mean(rps_values)
                max_rps = np.max(rps_values)
                model_stats[model_name] = {
                    'avg_rps': avg_rps,
                    'max_rps': max_rps,
                    'total_requests': len(send_times)
                }
            
            # 设置上图属性
            ax1.set_title('Request Send RPS by Model', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (seconds)', fontsize=10)
            ax1.set_ylabel('RPS (requests/second)', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            
            # 计算总体RPS
            total_rps_values = []
            for j in range(num_windows):
                window_start = start_time + j * window_size
                window_end = window_start + window_size
                
                # 统计该窗口内的总请求数
                total_requests_in_window = sum(1 for t in total_send_times 
                                             if window_start <= t < window_end)
                
                # 计算总RPS
                total_rps = total_requests_in_window / window_size
                total_rps_values.append(total_rps)
            
            # 绘制总体RPS曲线（下图，细线）
            ax2.plot(time_points, total_rps_values, 'black', linewidth=1, 
                    label='Total RPS', alpha=0.9, marker='o', markersize=2)
            
            # 设置下图属性
            ax2.set_title('Total Request Send RPS', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time (seconds)', fontsize=10)
            ax2.set_ylabel('RPS (requests/second)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9)
            
            # 添加统计信息文本框（放在上图）
            stats_text = "Model Statistics:\n"
            total_avg_rps = np.mean(total_rps_values)
            total_max_rps = np.max(total_rps_values)
            
            for model_name, stats in model_stats.items():
                stats_text += f"{model_name}:\n"
                stats_text += f"  Avg: {stats['avg_rps']:.2f} RPS\n"
                stats_text += f"  Peak: {stats['max_rps']:.2f} RPS\n"
                stats_text += f"  Total: {stats['total_requests']} reqs\n\n"
            
            stats_text += f"Overall:\n"
            stats_text += f"  Avg: {total_avg_rps:.2f} RPS\n"
            stats_text += f"  Peak: {total_max_rps:.2f} RPS\n"
            stats_text += f"  Total: {len(total_send_times)} reqs"
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图像
            output_dir = os.path.dirname(metrics_file)
            rps_plot_path = os.path.join(output_dir, 'actual_send_rps_by_model.png')
            plt.savefig(rps_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"按模型分组的RPS图已保存至: {rps_plot_path}")
            self.logger.info(f"总体RPS统计: 总请求 {len(total_send_times)}, 平均RPS {total_avg_rps:.2f}, 峰值RPS {total_max_rps:.2f}")
            
            # 输出各模型的统计信息
            for model_name, stats in model_stats.items():
                self.logger.info(f"模型 {model_name}: 平均RPS {stats['avg_rps']:.2f}, "
                               f"峰值RPS {stats['max_rps']:.2f}, 总请求 {stats['total_requests']}")
                
        except Exception as e:
            self.logger.error(f"生成按模型分组的RPS图失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def dispatch_traces(self, traces: List[RequestTrace], trace_file: str = None) -> List[ResponseRecord]:
        """调度trace请求 - 动态调度：每0.5s查找负载最小进程，调度current_time+5s内的请求"""
        self.logger.info(f"开始调度 {len(traces)} 个trace请求")
        requests = self._convert_traces_to_requests(traces)
        self.start_worker_processes()
        try:
            base_time = time.time()

            look_ahead_window = self.dispatch_window_size
            dispatch_interval = 0.5
            self.logger.info(f"基准时间: {base_time}")
            self.logger.info(f"前瞻窗口: {look_ahead_window}s")
            self.logger.info(f"调度间隔: {dispatch_interval}s")
            sorted_requests = sorted(requests, key=lambda x: x.timestamp)
            if not sorted_requests:
                self.logger.warning("没有请求需要调度")
                return []
            min_timestamp = sorted_requests[0].timestamp
            max_timestamp = sorted_requests[-1].timestamp
            total_duration = max_timestamp - min_timestamp
            self.logger.info(f"请求时间范围: {min_timestamp:.1f}s - {max_timestamp:.1f}s (总时长: {total_duration:.1f}s)")
            batch_count = 0
            overload_warnings = 0
            total_dispatched = 0
            processed_requests = set()
            # 均分首批窗口
            first_window_end = min_timestamp + look_ahead_window
            first_window_requests = [req for req in sorted_requests if min_timestamp <= req.timestamp < first_window_end]
            per_proc = max(1, len(first_window_requests) // self.process_count)
            for i in range(self.process_count):
                start_idx = i * per_proc
                end_idx = (i + 1) * per_proc if i < self.process_count - 1 else len(first_window_requests)
                batch_reqs = first_window_requests[start_idx:end_idx]
                if not batch_reqs:
                    continue
                batch = TaskBatch(
                    batch_id=batch_count,
                    requests=batch_reqs,
                    start_time=min_timestamp,
                    end_time=first_window_end
                )
                batch.base_time = base_time
                self.task_queues[i].put(batch, timeout=1.0)
                batch_count += 1
                total_dispatched += len(batch_reqs)
                for req in batch_reqs:
                    processed_requests.add(req.request_id)
                current_loads = self._get_all_process_loads()
                self.logger.info(f"首批均分分配到进程 {i}，包含 {len(batch_reqs)} 个请求，当前各进程协程数: {list(current_loads.values())}")
            # 动态调度循环，严格滑动窗口推进
            current_trace_time = 0.0
            while total_dispatched < len(sorted_requests):
                # 物理时间推进
                real_time = time.time() - base_time
                if current_trace_time > real_time:
                    await asyncio.sleep(current_trace_time - real_time)
                look_ahead_end = current_trace_time + look_ahead_window
                window_requests = []
                for req in sorted_requests:
                    if (req.request_id not in processed_requests and 
                        current_trace_time <= req.timestamp < look_ahead_end):
                        window_requests.append(req)
                if window_requests:
                    current_loads = self._get_all_process_loads()
                    target_process = self._get_least_loaded_process()
                    if self._check_overload():
                        overload_warnings += 1
                        self.logger.warning(f"系统负载较高！所有进程协程数都超过预设限制 {self.max_coroutines_per_process}")
                    batch = TaskBatch(
                        batch_id=batch_count,
                        requests=window_requests,
                        start_time=current_trace_time,
                        end_time=look_ahead_end
                    )
                    batch.base_time = base_time
                    try:
                        self.task_queues[target_process].put(batch, timeout=1.0)
                        batch_count += 1
                        total_dispatched += len(window_requests)
                        for req in window_requests:
                            processed_requests.add(req.request_id)
                        self.logger.info(f"动态调度批次 {batch_count} (当前时间: {current_trace_time:.1f}s, 前瞻窗口: {current_trace_time:.1f}s-{look_ahead_end:.1f}s) 分配给进程 {target_process}，包含 {len(window_requests)} 个请求，当前各进程协程数: {list(current_loads.values())}")
                    except Exception as e:
                        self.logger.error(f"分发动态调度批次 {batch_count} 失败: {e}")
                else:
                    current_loads = self._get_all_process_loads()
                    self.logger.debug(f"当前时间 {current_trace_time:.1f}s，前瞻窗口 {look_ahead_end:.1f}s 内无新请求，当前各进程协程数: {list(current_loads.values())}")
                current_trace_time += dispatch_interval
                if total_dispatched >= len(sorted_requests):
                    self.logger.info(f"所有 {len(sorted_requests)} 个请求已分发完成")
                    break
                if current_trace_time > max_timestamp + look_ahead_window:
                    self.logger.warning(f"已超过最大trace时间 {max_timestamp}s + 前瞻窗口 {look_ahead_window}s，停止调度")
                    break
            self.logger.info(f"动态调度完成，共分发 {batch_count} 个批次，{total_dispatched} 个请求")
            if overload_warnings > 0:
                self.logger.warning(f"系统负载警告次数: {overload_warnings}")
            results = self.collect_results(len(requests))
            output_paths = self.config_manager.get_output_paths()
            metrics_file = output_paths['performance_metrics']
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            try:
                with open(metrics_file, 'w', encoding='utf-8') as output_file:
                    for result in results:
                        output_file.write(json.dumps(asdict(result)) + "\n")
                self.logger.info(f"性能指标已写入: {metrics_file}")
            except Exception as e:
                self.logger.error(f"写入性能指标文件失败: {e}")
            if 'performance_plots' in output_paths:
                plot_dir = output_paths['performance_plots']
                os.makedirs(plot_dir, exist_ok=True)
                load_plot_path = os.path.join(plot_dir, 'process_load_over_time.png')
            else:
                load_plot_path = os.path.join(os.path.dirname(metrics_file), 'process_load_over_time.png')
            self.generate_process_load_plot(load_plot_path)
            self.generate_actual_send_rps_plot(metrics_file)
            return results
        except Exception as e:
            self.logger.error(f"调度过程中出现错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
        finally:
            self.stop_worker_processes()


def create_client_dispatcher(config_manager: ConfigManager) -> ClientDispatcher:
    """创建客户端调度器的便捷函数"""
    return ClientDispatcher(config_manager)