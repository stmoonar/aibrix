"""Trace数据类型 (v1 trace_generator.py 中的 dataclass 原样拆出)

拆出的唯一原因: 让只做请求分发的路径 (client_dispatcher / cli dispatch)
不必 import data_generator (它在 import 时就要求 modelscope+transformers)。
字段、默认值与 v1 完全相同。
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class LoadPhase:
    """负载阶段"""
    start_time: float
    end_time: float
    base_load: float
    phase_type: str  # "stable" or "transition"
    phase_id: int = 0  # 阶段ID，用于确保稳定期内负载分配一致
    load_changes: Dict[str, float] = None  # 负载变化值（用于过渡期）


@dataclass
class ModelLoadPoint:
    """模型负载时间点"""
    timestamp: float
    total_load: float
    model_loads: Dict[str, float]
    phase_type: str


@dataclass
class RequestTrace:
    """请求trace"""
    request_id: str
    timestamp: float
    model_name: str
    prompt: str
    prompt_length: int
    phase_type: str
    max_output_tokens: Optional[int] = None  # 来自real trace的output_tokens
