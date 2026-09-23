"""
Real Trace适配器模块

将AzurePublicDataset生成的trace.json转换为CustomTraceGenerator的RequestTrace格式。
每个模型独立配置自己的trace文件，支持不同数据集的流量模式。
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config_manager import ConfigManager
from .data_generator import DataGenerator
from .trace_generator import RequestTrace, ModelLoadPoint


class RealTraceAdapter:
    """将AzurePublicDataset的trace.json转换为CustomTraceGenerator的RequestTrace格式"""

    def __init__(self, config_manager: ConfigManager, data_generator: DataGenerator):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.data_generator = data_generator
        self.request_id_counter = 0

        # 读取real_trace全局配置
        self.real_trace_config = getattr(self.config, 'real_trace_config', {}) or {}
        self.time_scale = self.real_trace_config.get('time_scale', 1.0)
        self.max_duration = self.real_trace_config.get('max_duration', None)

        # 随机种子
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    def load_azure_trace(self, trace_path: str) -> List[dict]:
        """加载并校验Azure trace JSON文件"""
        abs_path = os.path.abspath(trace_path)
        if not os.path.exists(abs_path):
            # 尝试相对于配置文件目录解析
            cfg_dir = self.config_manager.config_path.parent
            abs_path = os.path.abspath(os.path.join(cfg_dir, trace_path))

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Real trace文件未找到: {trace_path}")

        with open(abs_path, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)

        if not isinstance(trace_data, list) or len(trace_data) == 0:
            raise ValueError(f"Real trace文件格式错误，应为非空JSON数组: {trace_path}")

        # 校验必需字段
        required_fields = {"start_time", "end_time", "rps", "requests"}
        for i, entry in enumerate(trace_data):
            missing = required_fields - set(entry.keys())
            if missing:
                raise ValueError(f"Real trace第{i}个条目缺少字段: {missing}")
            if not isinstance(entry["requests"], list):
                raise ValueError(f"Real trace第{i}个条目的requests应为数组")

        return trace_data

    def _apply_time_transforms(self, trace_data: List[dict]) -> List[dict]:
        """应用时间缩放和最大时长截取。

        当 max_duration 裁剪尾部窗口时，按比例裁减请求数量，
        避免将整秒的请求塞进半秒窗口导致负载尖峰。
        """
        result = []
        for entry in trace_data:
            scaled_start = entry["start_time"] * self.time_scale
            scaled_end = entry["end_time"] * self.time_scale

            # 截取最大时长
            if self.max_duration is not None and scaled_start >= self.max_duration:
                break

            original_duration = scaled_end - scaled_start
            if self.max_duration is not None and scaled_end > self.max_duration:
                scaled_end = self.max_duration

            actual_duration = scaled_end - scaled_start
            requests = entry["requests"]

            # 如果窗口被裁剪，按比例保留请求以维持原始RPS
            if original_duration > 0 and actual_duration < original_duration:
                keep_ratio = actual_duration / original_duration
                keep_count = max(0, int(round(len(requests) * keep_ratio)))
                requests = requests[:keep_count]

            result.append({
                **entry,
                "start_time": scaled_start,
                "end_time": scaled_end,
                "requests": requests,
            })

        return result

    def _create_request_trace(
        self,
        timestamp: float,
        model_name: str,
        phase_type: str,
        prompt: str,
        prompt_length: int,
        max_output_tokens: Optional[int] = None,
    ) -> RequestTrace:
        """创建单个RequestTrace"""
        self.request_id_counter += 1
        request_id = f"req_{self.request_id_counter:06d}"
        return RequestTrace(
            request_id=request_id,
            timestamp=timestamp,
            model_name=model_name,
            prompt=prompt,
            prompt_length=prompt_length,
            phase_type=phase_type,
            max_output_tokens=max_output_tokens,
        )

    def _convert_single_model(
        self, model_name: str, trace_data: List[dict]
    ) -> Tuple[List[RequestTrace], List[ModelLoadPoint], List[ModelLoadPoint]]:
        """转换单个模型的Azure trace为RequestTrace列表和时间线"""
        traces = []
        rps_timeline = []
        token_rate_timeline = []

        for entry in trace_data:
            start_time = entry["start_time"]
            end_time = entry["end_time"]
            period_type = entry.get("period_type", "stable")
            requests = entry["requests"]
            window_duration = end_time - start_time

            # 一致性检查: rps字段与实际requests数量
            declared_rps = entry.get("rps", None)
            if declared_rps is not None and declared_rps != len(requests):
                print(f"    [警告] 时间窗口 [{start_time}, {end_time}) 声明rps={declared_rps}, "
                      f"实际requests数={len(requests)}, 以实际requests为准")

            total_tokens_at_point = 0
            num_requests = len(requests)

            for req in requests:
                input_tokens = req.get("input_tokens", 100)
                output_tokens = req.get("output_tokens", None)

                # 在时间窗口内分配随机时间戳，确保不越过窗口边界
                jitter_max = max(0, window_duration - 1e-6)
                jitter = random.uniform(0, jitter_max) if jitter_max > 0 else 0.0
                timestamp = start_time + jitter

                # 使用DataGenerator生成对应长度的prompt
                target_length = max(10, int(round(input_tokens)))
                try:
                    prompt_data = self.data_generator.generate_prompt(
                        target_token_length=target_length,
                        model_name=model_name,
                        content_category="mixed",
                        randomness=0.9,
                    )
                    actual_prompt = prompt_data["prompt"]
                    actual_length = prompt_data["actual_token_length"]
                except Exception as e:
                    print(f"    为模型 {model_name} 生成prompt失败(target={target_length}): {e}")
                    actual_prompt = f"fallback_prompt_{self.request_id_counter + 1}"
                    actual_length = target_length

                max_out = int(round(output_tokens)) if output_tokens is not None else None

                trace = self._create_request_trace(
                    timestamp=timestamp,
                    model_name=model_name,
                    phase_type=period_type,
                    prompt=actual_prompt,
                    prompt_length=actual_length,
                    max_output_tokens=max_out,
                )
                traces.append(trace)
                total_tokens_at_point += actual_length

            # 构建时间线点 (单位统一为 /sec)
            rps_value = num_requests / window_duration if window_duration > 0 else num_requests
            token_rate_value = total_tokens_at_point / window_duration if window_duration > 0 else total_tokens_at_point

            rps_timeline.append(ModelLoadPoint(
                timestamp=start_time,
                total_load=rps_value,
                model_loads={model_name: rps_value},
                phase_type=period_type,
            ))
            token_rate_timeline.append(ModelLoadPoint(
                timestamp=start_time,
                total_load=token_rate_value,
                model_loads={model_name: token_rate_value},
                phase_type=period_type,
            ))

        return traces, rps_timeline, token_rate_timeline

    def convert_all_models(
        self,
    ) -> Tuple[List[ModelLoadPoint], List[ModelLoadPoint], List[RequestTrace]]:
        """
        主入口: 遍历每个模型的real_trace_file，合并生成完整trace。

        Returns:
            (rps_timeline, token_rate_timeline, traces)
        """
        all_traces: List[RequestTrace] = []
        all_rps_points: Dict[float, Dict[str, float]] = {}
        all_token_points: Dict[float, Dict[str, float]] = {}
        all_phase_types: Dict[float, str] = {}

        for model_config in self.config.models:
            trace_file = getattr(model_config, 'real_trace_file', None)
            if not trace_file:
                print(f"    模型 {model_config.name} 未配置 real_trace_file，跳过")
                continue

            print(f"  加载模型 {model_config.name} 的trace: {trace_file}")
            raw_trace = self.load_azure_trace(trace_file)
            transformed_trace = self._apply_time_transforms(raw_trace)

            print(f"    trace包含 {len(transformed_trace)} 个时间窗口, "
                  f"共 {sum(len(e['requests']) for e in transformed_trace)} 个请求")

            model_traces, model_rps, model_tokens = self._convert_single_model(
                model_config.name, transformed_trace
            )
            all_traces.extend(model_traces)

            # 合并时间线
            for point in model_rps:
                t = point.timestamp
                if t not in all_rps_points:
                    all_rps_points[t] = {}
                    all_token_points[t] = {}
                    all_phase_types[t] = point.phase_type
                elif point.phase_type == "transition":
                    # 任一模型处于transition，则合并后为transition
                    all_phase_types[t] = "transition"
                all_rps_points[t][model_config.name] = point.model_loads[model_config.name]

            for point in model_tokens:
                t = point.timestamp
                if t not in all_token_points:
                    all_token_points[t] = {}
                all_token_points[t][model_config.name] = point.model_loads[model_config.name]

        # 按时间排序，构建合并后的时间线
        sorted_times = sorted(all_rps_points.keys())
        rps_timeline = []
        token_rate_timeline = []

        for t in sorted_times:
            rps_loads = all_rps_points[t]
            token_loads = all_token_points.get(t, {})
            phase_type = all_phase_types.get(t, "stable")

            rps_timeline.append(ModelLoadPoint(
                timestamp=t,
                total_load=sum(rps_loads.values()),
                model_loads=rps_loads,
                phase_type=phase_type,
            ))
            token_rate_timeline.append(ModelLoadPoint(
                timestamp=t,
                total_load=sum(token_loads.values()),
                model_loads=token_loads,
                phase_type=phase_type,
            ))

        # 按时间排序traces
        all_traces.sort(key=lambda x: x.timestamp)

        print(f"\n  Real Trace转换完成:")
        print(f"    总请求数: {len(all_traces)}")
        print(f"    时间线长度: {len(rps_timeline)} 个时间点")
        if all_traces:
            print(f"    时间范围: {all_traces[0].timestamp:.1f}s - {all_traces[-1].timestamp:.1f}s")

        return rps_timeline, token_rate_timeline, all_traces
