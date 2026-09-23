"""
Trace生成模块

集成负载规划和trace生成功能，支持RPS和Token Rate两种负载模式
根据配置生成负载时间序列，然后生成对应的请求trace，并可视化结果
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .config_manager import ConfigManager
from .data_generator import DataGenerator


# v1 的 LoadPhase / ModelLoadPoint / RequestTrace 原样移至 trace_types.py，此处重新导出以保持 import 兼容
from .trace_types import LoadPhase, ModelLoadPoint, RequestTrace  # noqa: F401


class IntegratedTraceGenerator:
    """集成的Trace生成器"""

    def __init__(self, config_manager: ConfigManager):
        """
        初始化集成trace生成器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.model_names = [model.name for model in self.config.models]
        self.random_seed = self.config.random_seed
        self.request_id_counter = 0
        
        # 初始化数据生成器
        self.data_generator = DataGenerator(config_manager)
        
        # 设置随机种子
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        else:
            # 使用真随机种子
            import time
            true_seed = int(time.time() * 1000000) % (2**31)
            random.seed(true_seed)
            np.random.seed(true_seed)
        
        print(f"🚀 初始化Trace生成器")
        print(f"   负载模式: {self.config.load_mode}")
        print(f"   总负载: {self.config.total_load}")
        print(f"   模型数量: {len(self.model_names)}")
        print(f"   随机种子: {self.random_seed}")

    def generate_load_phases(self) -> List[LoadPhase]:
        """生成负载阶段（稳定期+过渡期）"""
        phases = []
        current_time = 0.0
        total_duration = self.config.duration_seconds
        
        stable_config = self.config.stable_period
        transition_config = self.config.transition_period
        
        # 计算最优阶段数
        phase_count = self._calculate_optimal_phase_count(total_duration, stable_config, transition_config)
        
        current_phase_index = 0
        phase_id = 0
        
        print(f"📊 生成负载阶段:")
        print(f"   测试时长: {total_duration}s")
        print(f"   计划阶段数: {phase_count}")
        
        while current_time < total_duration and current_phase_index < phase_count:
            remaining_time = total_duration - current_time
            remaining_phases = phase_count - current_phase_index
            
            if remaining_phases <= 0 or remaining_time <= 0:
                break
            
            # 确定阶段类型和时长
            if current_phase_index % 2 == 0:
                # 稳定期
                min_duration = stable_config['min_duration']
                max_duration = stable_config['max_duration']
                phase_type = "stable"
            else:
                # 过渡期
                min_duration = transition_config['min_duration']
                max_duration = transition_config['max_duration']
                phase_type = "transition"
            
            # 计算合适的时长
            avg_remaining_duration = remaining_time / remaining_phases
            suggested_duration = min(max_duration, max(min_duration, avg_remaining_duration))
            
            if remaining_phases == 1:
                phase_duration = remaining_time
            else:
                duration_range = min(suggested_duration * 0.3, max_duration - min_duration)
                phase_duration = random.uniform(
                    max(min_duration, suggested_duration - duration_range),
                    min(max_duration, suggested_duration + duration_range, remaining_time)
                )
            
            if phase_duration > 0:
                new_phase = LoadPhase(
                    start_time=current_time,
                    end_time=current_time + phase_duration,
                    base_load=self.config.total_load,
                    phase_type=phase_type,
                    phase_id=phase_id if phase_type == "stable" else phase_id - 1
                )
                phases.append(new_phase)
                current_time += phase_duration
                
                if phase_type == "stable":
                    phase_id += 1
            
            current_phase_index += 1
        
        print(f"✅ 生成了 {len(phases)} 个阶段")
        return phases
    
    def _calculate_optimal_phase_count(self, total_duration: float, 
                                      stable_config, transition_config) -> int:
        """根据总时长计算最优的阶段数"""
        avg_stable_duration = (stable_config['min_duration'] + stable_config['max_duration']) / 2
        avg_transition_duration = (transition_config['min_duration'] + transition_config['max_duration']) / 2
        avg_cycle_duration = avg_stable_duration + avg_transition_duration
        
        estimated_stable_phases = max(1, int(total_duration / avg_cycle_duration))
        stable_phase_count = max(2, estimated_stable_phases)
        transition_phase_count = stable_phase_count - 1
        total_phases = stable_phase_count + transition_phase_count
        
        # 验证时长合理性
        min_total_time = (stable_phase_count * stable_config['min_duration'] + 
                         transition_phase_count * transition_config['min_duration'])
        max_total_time = (stable_phase_count * stable_config['max_duration'] + 
                         transition_phase_count * transition_config['max_duration'])
        
        if min_total_time > total_duration:
            stable_phase_count = max(1, stable_phase_count - 1)
            transition_phase_count = max(0, stable_phase_count - 1)
            total_phases = stable_phase_count + transition_phase_count
        elif max_total_time < total_duration * 0.8:
            stable_phase_count += 1
            transition_phase_count = stable_phase_count - 1
            total_phases = stable_phase_count + transition_phase_count
        
        return total_phases

    def generate_load_timeline(self, time_step: float = 1.0) -> List[ModelLoadPoint]:
        """生成完整的负载时间序列"""
        phases = self.generate_load_phases()
        timeline = []
        
        # 预先计算每个稳定期的模型负载分配
        stable_phase_allocations = {}
        for phase in phases:
            if phase.phase_type == "stable":
                # 为每个稳定期生成固定的负载分配比例
                phase_seed = phase.phase_id * 1000 + (self.random_seed or 0)
                random.seed(phase_seed)
                
                num_models = len(self.model_names)
                random_weights = [random.random() for _ in range(num_models)]
                total_weight = sum(random_weights)
                normalized_weights = [w / total_weight for w in random_weights]
                
                stable_phase_allocations[phase.phase_id] = {
                    self.model_names[i]: normalized_weights[i] 
                    for i in range(num_models)
                }
        
        print(f"⏱️ 生成负载时间序列:")
        print(f"   时间步长: {time_step}s")
        
        current_time = 0.0
        while current_time <= self.config.duration_seconds:
            current_phase = self._find_phase_at_time(phases, current_time)
            if current_phase is None:
                break
                
            # 计算当前时间点的负载
            if current_phase.phase_type == "transition":
                # 过渡期：总负载保持稳定，模型分配比例线性插值
                total_load = current_phase.base_load
                
                phase_duration = current_phase.end_time - current_phase.start_time
                phase_progress = (current_time - current_phase.start_time) / phase_duration
                phase_progress = max(0.0, min(1.0, phase_progress))
                
                # 找到前后稳定期的分配比例
                prev_stable_id, next_stable_id = self._find_adjacent_stable_phases(
                    phases, current_phase
                )
                
                if (prev_stable_id is not None and next_stable_id is not None and
                    prev_stable_id in stable_phase_allocations and 
                    next_stable_id in stable_phase_allocations):
                    
                    prev_allocation = stable_phase_allocations[prev_stable_id]
                    next_allocation = stable_phase_allocations[next_stable_id]
                    
                    model_loads = {}
                    for model_name in self.model_names:
                        prev_ratio = prev_allocation[model_name]
                        next_ratio = next_allocation[model_name]
                        current_ratio = prev_ratio + (next_ratio - prev_ratio) * phase_progress
                        model_loads[model_name] = current_ratio * total_load
                else:
                    # 备用方案
                    model_loads = self._calculate_model_loads(total_load, current_time, current_phase)
                    
            else:
                # 稳定期：使用预计算的固定负载分配
                total_load = current_phase.base_load
                
                if current_phase.phase_id in stable_phase_allocations:
                    allocation_ratios = stable_phase_allocations[current_phase.phase_id]
                    model_loads = {
                        model_name: ratio * total_load 
                        for model_name, ratio in allocation_ratios.items()
                    }
                else:
                    model_loads = self._calculate_model_loads(total_load, current_time, current_phase)
            
            timeline.append(ModelLoadPoint(
                timestamp=current_time,
                total_load=total_load,
                model_loads=model_loads,
                phase_type=current_phase.phase_type
            ))
            
            current_time += time_step
        
        print(f"✅ 生成了 {len(timeline)} 个时间点的负载数据")
        return timeline

    def _find_adjacent_stable_phases(self, phases: List[LoadPhase], 
                                   transition_phase: LoadPhase) -> Tuple[Optional[int], Optional[int]]:
        """找到过渡期前后的稳定期ID"""
        current_index = phases.index(transition_phase)
        
        prev_stable_id = None
        next_stable_id = None
        
        # 向前找稳定期
        for i in range(current_index - 1, -1, -1):
            if phases[i].phase_type == "stable":
                prev_stable_id = phases[i].phase_id
                break
        
        # 向后找稳定期
        for i in range(current_index + 1, len(phases)):
            if phases[i].phase_type == "stable":
                next_stable_id = phases[i].phase_id
                break
        
        return prev_stable_id, next_stable_id

    def _calculate_model_loads(self, total_load: float, timestamp: float, 
                              phase: LoadPhase) -> Dict[str, float]:
        """计算各模型的负载分配"""
        model_loads = {}
        num_models = len(self.model_names)
        
        if num_models == 0:
            return model_loads
        
        if num_models == 1:
            model_loads[self.model_names[0]] = total_load
            return model_loads
        
        # 根据阶段信息设置随机种子
        if phase.phase_type == 'stable':
            seed = phase.phase_id * 1000 + (self.random_seed or 0)
        else:
            seed = int(timestamp * 100) % (2**31)
        
        random.seed(seed)
        
        # 生成随机权重并归一化
        random_weights = [random.random() for _ in range(num_models)]
        total_weight = sum(random_weights)
        normalized_weights = [w / total_weight for w in random_weights]
        
        for i, model_name in enumerate(self.model_names):
            model_loads[model_name] = normalized_weights[i] * total_load
        
        return model_loads

    def apply_noise_to_timeline(self, timeline: List[ModelLoadPoint]) -> List[ModelLoadPoint]:
        """为负载时间序列应用噪音"""
        if not self.config.noise.get('enabled', False):
            return timeline
        
        amplitude = self.config.noise.get('amplitude', 0.1)
        print(f"🌊 应用负载噪音: amplitude={amplitude}")
        
        noisy_timeline = []
        for point in timeline:
            new_model_loads = {}
            new_total_load = 0
            
            for model_name, original_load in point.model_loads.items():
                # 生成噪音乘数
                noise_multiplier = 1.0 + random.normalvariate(0, amplitude)
                noise_multiplier = max(0.1, min(2.0, noise_multiplier))
                
                noisy_load = original_load * noise_multiplier
                new_model_loads[model_name] = noisy_load
                new_total_load += noisy_load
            
            noisy_point = ModelLoadPoint(
                timestamp=point.timestamp,
                total_load=new_total_load,
                model_loads=new_model_loads,
                phase_type=point.phase_type
            )
            noisy_timeline.append(noisy_point)
        
        return noisy_timeline

    def generate_rps_mode_traces(self) -> Tuple[List[ModelLoadPoint], List[ModelLoadPoint], List[RequestTrace]]:
        """
        RPS模式的trace生成
        
        Returns:
            Tuple[rps_timeline, token_rate_timeline, traces]
        """
        print("\n🎯 RPS模式 - 生成trace")
        
        # 1. 生成RPS负载时间序列
        rps_timeline = self.generate_load_timeline()
        rps_timeline = self.apply_noise_to_timeline(rps_timeline)
        
        # 2. 根据RPS时间序列生成prompts和traces
        traces = []
        token_rate_timeline = []
        
        print("📝 生成prompts和traces...")
        
        for point in rps_timeline:
            point_traces = []
            total_tokens_at_point = 0
            model_token_loads = {}
            
            # 为每个模型生成对应数量的请求
            for model_name, rps_load in point.model_loads.items():
                if rps_load <= 0:
                    model_token_loads[model_name] = 0
                    continue
                
                num_requests = max(0, int(round(rps_load)))  # 四舍五入
                model_tokens = 0
                
                # 生成该模型的请求
                for _ in range(num_requests):
                    # 生成符合配置的prompt长度
                    request_overrides = self._get_request_overrides(model_name, point.timestamp)
                    prompt_length = self._generate_prompt_length(
                        request_overrides.get("input_token_config")
                    )
                    
                    # 使用data_generator生成实际的prompt内容
                    try:
                        prompt_data = self.data_generator.generate_prompt(
                            target_token_length=prompt_length,
                            model_name=model_name,
                            content_category="mixed",
                            randomness=0.9
                        )
                        actual_prompt = prompt_data['prompt']
                        actual_length = prompt_data['actual_token_length']
                    except Exception as e:
                        print(f"⚠️ 为模型 {model_name} 生成prompt失败: {e}")
                        actual_prompt = f"fallback_prompt_{self.request_id_counter}"
                        actual_length = prompt_length
                    
                    # 创建trace
                    trace = self._create_request_trace(
                        timestamp=point.timestamp,
                        model_name=model_name,
                        phase_type=point.phase_type,
                        prompt=actual_prompt,
                        prompt_length=actual_length,
                        max_output_tokens=request_overrides.get("max_output_tokens")
                    )
                    point_traces.append(trace)
                    explicit_decode_tokens = request_overrides.get("max_output_tokens")
                    model_tokens += actual_length + (explicit_decode_tokens or 0)
                
                model_token_loads[model_name] = model_tokens
                total_tokens_at_point += model_tokens
            
            traces.extend(point_traces)
            
            # 创建对应的token rate时间点
            token_rate_point = ModelLoadPoint(
                timestamp=point.timestamp,
                total_load=total_tokens_at_point,
                model_loads=model_token_loads,
                phase_type=point.phase_type
            )
            token_rate_timeline.append(token_rate_point)
        
        # 按时间排序
        traces.sort(key=lambda x: x.timestamp)
        
        print(f"✅ RPS模式完成，生成了 {len(traces)} 个请求")
        return rps_timeline, token_rate_timeline, traces

    def generate_token_rate_mode_traces(self) -> Tuple[List[ModelLoadPoint], List[ModelLoadPoint], List[RequestTrace]]:
        """
        Token Rate模式的trace生成
        
        Returns:
            Tuple[rps_timeline, token_rate_timeline, traces]  
        """
        print("\n🎯 Token Rate模式 - 生成trace")
        
        # 1. 生成Token Rate负载时间序列
        token_rate_timeline = self.generate_load_timeline()
        token_rate_timeline = self.apply_noise_to_timeline(token_rate_timeline)
        
        # 2. 根据Token Rate时间序列生成prompts和traces，反推RPS
        traces = []
        rps_timeline = []
        
        print("📝 生成prompts并反推RPS...")
        
        for point in token_rate_timeline:
            point_traces = []
            total_requests_at_point = 0
            model_rps_loads = {}
            
            # 为每个模型生成满足token rate的请求
            for model_name, token_rate_load in point.model_loads.items():
                if token_rate_load <= 0:
                    model_rps_loads[model_name] = 0
                    continue
                
                # 根据目标token数生成请求
                target_tokens = int(round(token_rate_load))
                accumulated_tokens = 0
                model_requests = 0
                
                while accumulated_tokens < target_tokens:
                    # 生成符合配置的prompt长度
                    prompt_length = self._generate_prompt_length()
                    
                    # 检查是否会超出目标token数（允许一定的误差）
                    if accumulated_tokens + prompt_length > target_tokens * 1.1:
                        # 如果超出太多，尝试生成更短的prompt
                        remaining_tokens = target_tokens - accumulated_tokens
                        if remaining_tokens > 10:  # 最小prompt长度
                            prompt_length = min(prompt_length, remaining_tokens)
                        else:
                            break
                    
                    # 使用data_generator生成实际的prompt内容
                    try:
                        prompt_data = self.data_generator.generate_prompt(
                            target_token_length=prompt_length,
                            model_name=model_name,
                            content_category="mixed",
                            randomness=0.9
                        )
                        actual_prompt = prompt_data['prompt']
                        actual_length = prompt_data['actual_token_length']
                    except Exception as e:
                        print(f"⚠️ 为模型 {model_name} 生成prompt失败: {e}")
                        actual_prompt = f"fallback_prompt_{self.request_id_counter}"
                        actual_length = prompt_length
                    
                    # 创建trace
                    trace = self._create_request_trace(
                        timestamp=point.timestamp,
                        model_name=model_name,
                        phase_type=point.phase_type,
                        prompt=actual_prompt,
                        prompt_length=actual_length
                    )
                    point_traces.append(trace)
                    
                    accumulated_tokens += actual_length
                    model_requests += 1
                    
                    # 防止生成过多请求
                    if model_requests > target_tokens:
                        break
                
                model_rps_loads[model_name] = model_requests
                total_requests_at_point += model_requests
            
            traces.extend(point_traces)
            
            # 创建对应的RPS时间点
            rps_point = ModelLoadPoint(
                timestamp=point.timestamp,
                total_load=total_requests_at_point,
                model_loads=model_rps_loads,
                phase_type=point.phase_type
            )
            rps_timeline.append(rps_point)
        
        # 按时间排序
        traces.sort(key=lambda x: x.timestamp)
        
        print(f"✅ Token Rate模式完成，生成了 {len(traces)} 个请求")
        return rps_timeline, token_rate_timeline, traces

    def _generate_prompt_length(self, input_token_config: Optional[Dict[str, any]] = None) -> int:
        """生成prompt长度"""
        config = input_token_config or self.config.input_token_config
        distribution = config.get('distribution', 'normal')
        min_length = config.get('min_length', 50)
        max_length = config.get('max_length', 2000)
        
        if distribution == 'uniform':
            length = random.uniform(min_length, max_length)
        elif distribution == 'normal':
            mean = config.get('mean', 500)
            std = config.get('std', 200)
            length = random.normalvariate(mean, std)
        elif distribution == 'exponential':
            mean = config.get('mean', 500)
            length = random.expovariate(1.0 / mean) + min_length
        else:
            length = config.get('mean', 500)
        
        return int(max(min_length, min(max_length, length)))

    def _get_request_overrides(self, model_name: str, timestamp: float) -> Dict[str, any]:
        return {}

    def _create_request_trace(self, timestamp: float, model_name: str, phase_type: str,
                            prompt: str, prompt_length: int,
                            max_output_tokens: Optional[int] = None) -> RequestTrace:
        """创建单个请求trace"""
        self.request_id_counter += 1
        request_id = f"req_{self.request_id_counter:06d}"
        
        # 添加微小的时间偏移以避免完全同时发送
        max_jitter = 0.9
        jittered_timestamp = timestamp + random.uniform(0, max_jitter)
        
        return RequestTrace(
            request_id=request_id,
            timestamp=jittered_timestamp,
            model_name=model_name,
            prompt=prompt,
            prompt_length=prompt_length,
            phase_type=phase_type,
            max_output_tokens=max_output_tokens
        )

    def _find_phase_at_time(self, phases: List[LoadPhase], timestamp: float) -> LoadPhase:
        """找到指定时间戳对应的阶段"""
        for phase in phases:
            if phase.start_time <= timestamp <= phase.end_time:
                return phase
        return None

    def export_load_timeline(self, timeline: List[ModelLoadPoint], file_path: str, timeline_type: str):
        """导出负载时间序列到文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        data = []
        for point in timeline:
            data.append({
                "timestamp": point.timestamp,
                "total_load": point.total_load,
                "phase_type": point.phase_type,
                "model_loads": point.model_loads
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 {timeline_type}负载时间序列已导出到: {file_path}")

    def export_traces(self, traces: List[RequestTrace], file_path: str):
        """导出traces到文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        trace_data = []
        for trace in traces:
            trace_data.append(asdict(trace))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Traces已导出到: {file_path}")

    def plot_load_timelines(self, rps_timeline: List[ModelLoadPoint], 
                           token_rate_timeline: List[ModelLoadPoint], 
                           output_path: str):
        """绘制RPS和Token Rate随时间变化图"""
        try:
            import matplotlib.pyplot as plt
            
            # 设置字体
            plt.rcParams['font.family'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # 提取时间戳和负载数据
            timestamps = [point.timestamp for point in rps_timeline]
            
            # RPS图
            ax1.set_title('RPS Over Time', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('RPS (requests/sec)', fontsize=12)
            
            total_rps = [point.total_load for point in rps_timeline]
            total_color = '#111111'
            ax1.plot(timestamps, total_rps, color=total_color, linestyle='-',
                    linewidth=2.4, label='Total RPS', alpha=0.9)
            
            # 为每个模型绘制RPS
            # Colorblind-friendly Okabe-Ito palette with distinct line styles.
            colors = [
                '#0072B2',  # blue
                '#E69F00',  # orange
                '#009E73',  # bluish green
                '#D55E00',  # vermillion
                '#CC79A7',  # reddish purple
                '#56B4E9',  # sky blue
            ]
            line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
            for i, model_name in enumerate(self.model_names):
                model_rps = [point.model_loads.get(model_name, 0) for point in rps_timeline]
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)]
                ax1.plot(timestamps, model_rps, color=color, linestyle=line_style,
                        linewidth=1.7, label=f'{model_name} RPS', alpha=0.9)
            
            ax1.legend(fontsize=10, ncol=2, frameon=True)
            ax1.grid(True, alpha=0.25)
            
            # Token Rate图
            ax2.set_title('Token Rate Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Token Rate (tokens/sec)', fontsize=12)
            
            total_token_rate = [point.total_load for point in token_rate_timeline]
            ax2.plot(timestamps, total_token_rate, color=total_color, linestyle='-',
                    linewidth=2.4, label='Total Token Rate', alpha=0.9)
            
            # 为每个模型绘制Token Rate
            for i, model_name in enumerate(self.model_names):
                model_token_rate = [point.model_loads.get(model_name, 0) for point in token_rate_timeline]
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)]
                ax2.plot(timestamps, model_token_rate, color=color, linestyle=line_style,
                        linewidth=1.7, label=f'{model_name} Token Rate', alpha=0.9)
            
            ax2.legend(fontsize=10, ncol=2, frameon=True)
            ax2.grid(True, alpha=0.25)
            
            plt.tight_layout()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 负载变化图已保存到: {output_path}")
            
        except ImportError:
            print("⚠️ 无法导入matplotlib，跳过图表生成")
            print("   请安装: pip install matplotlib")
        except Exception as e:
            print(f"⚠️ 生成图表时出错: {e}")

    def generate_complete_trace_data(self) -> Dict[str, any]:
        """
        生成完整的trace数据
        根据配置的load_mode选择相应的生成策略
        
        Returns:
            包含所有生成数据的字典
        """
        print(f"\n🚀 开始生成完整的trace数据")
        print(f"负载模式: {self.config.load_mode}")
        
        # 根据负载模式选择生成策略
        if self.config.load_mode == "rps":
            rps_timeline, token_rate_timeline, traces = self.generate_rps_mode_traces()
        else:  # token_rate
            rps_timeline, token_rate_timeline, traces = self.generate_token_rate_mode_traces()
        
        # 获取输出路径
        output_paths = self.config_manager.get_output_paths()
        
        # 导出数据
        self.export_load_timeline(rps_timeline, output_paths['load_timeline_rps'], "RPS")
        self.export_load_timeline(token_rate_timeline, output_paths['load_timeline_token_rate'], "Token Rate")
        self.export_traces(traces, output_paths['trace_data'])
        
        # 生成图表
        self.plot_load_timelines(rps_timeline, token_rate_timeline, output_paths['trace_plots'])
        
        # 生成统计信息
        stats = self._get_generation_statistics(rps_timeline, token_rate_timeline, traces)
        
        print(f"\n✅ Trace数据生成完成！")
        print(f"   RPS文件: {output_paths['load_timeline_rps']}")
        print(f"   Token Rate文件: {output_paths['load_timeline_token_rate']}")
        print(f"   Traces文件: {output_paths['trace_data']}")
        print(f"   图表文件: {output_paths['trace_plots']}")
        
        return {
            "rps_timeline": rps_timeline,
            "token_rate_timeline": token_rate_timeline,
            "traces": traces,
            "statistics": stats,
            "output_paths": output_paths
        }

    def _get_generation_statistics(self, rps_timeline: List[ModelLoadPoint], 
                                 token_rate_timeline: List[ModelLoadPoint], 
                                 traces: List[RequestTrace]) -> Dict[str, any]:
        """获取生成数据的统计信息"""
        if not traces:
            return {"error": "没有生成trace数据"}
        
        # 按模型分组统计
        model_stats = {}
        for trace in traces:
            model_name = trace.model_name
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "request_count": 0,
                    "total_tokens": 0,
                    "prompt_lengths": []
                }
            
            model_stats[model_name]["request_count"] += 1
            model_stats[model_name]["total_tokens"] += trace.prompt_length
            model_stats[model_name]["prompt_lengths"].append(trace.prompt_length)
        
        # 计算总体统计
        total_requests = len(traces)
        total_tokens = sum(trace.prompt_length for trace in traces)
        test_duration = max(trace.timestamp for trace in traces) - min(trace.timestamp for trace in traces)
        
        avg_rps = total_requests / test_duration if test_duration > 0 else 0
        avg_token_rate = total_tokens / test_duration if test_duration > 0 else 0
        
        return {
            "总体统计": {
                "总请求数": total_requests,
                "总Token数": total_tokens,
                "测试时长": round(test_duration, 2),
                "平均RPS": round(avg_rps, 2),
                "平均Token Rate": round(avg_token_rate, 2)
            },
            "模型统计": {
                model_name: {
                    "请求数": stats["request_count"],
                    "总Token数": stats["total_tokens"],
                    "平均Prompt长度": round(np.mean(stats["prompt_lengths"]), 1),
                    "Prompt长度标准差": round(np.std(stats["prompt_lengths"]), 1)
                }
                for model_name, stats in model_stats.items()
            }
        }


class IntegratedTraceGenerator2(IntegratedTraceGenerator):
    """
    集成的Trace生成器 - 版本2，使用自定义的json文件生成trace（json文件里是各个模型的稳定期时间和rps）
    """
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self._custom_trace_paths_cache = None
        self._custom_model_segments_cache = None

    def _expand_custom_trace_paths(self) -> List[str]:
        """根据配置展开自定义trace文件路径.
        支持：
        - "all": 使用 config/custom_traces 目录下所有 .json
        - 单个字符串: 文件名或相对/绝对路径
        - 字符串列表
        返回绝对路径列表。
        """
        from pathlib import Path

        cfg = self.config.custom_trace_json
        cfg_dir = self.config_manager.config_path.parent
        traces_dir = cfg_dir / "custom_traces"

        def to_abs(p: str) -> Path:
            path = Path(p)
            if path.is_absolute():
                return path
            # 优先在 custom_traces/ 下找，否则相对于配置文件目录
            cand = traces_dir / p
            return cand if cand.exists() else (cfg_dir / p)

        files: List[Path] = []
        if isinstance(cfg, str):
            if cfg.strip().lower() == "all":
                if traces_dir.exists():
                    files = sorted(traces_dir.glob("*.json"))
                else:
                    raise FileNotFoundError(f"未找到目录: {traces_dir}")
            else:
                files = [to_abs(cfg)]
        elif isinstance(cfg, list):
            files = [to_abs(x) for x in cfg]
        else:
            raise ValueError("custom_trace_json 必须为 str 或 list[str]")

        # 校验存在性
        abs_files = [str(f.resolve()) for f in files if f.exists() and f.suffix == ".json"]
        if not abs_files:
            raise FileNotFoundError(f"未找到任何可用的trace json文件, 配置: {cfg}")
        return abs_files

    def _load_custom_traces(self, file_paths: List[str]) -> Dict[str, List[Dict[str, float]]]:
        """加载多个自定义trace json，合并为 {model: [segments...]}
        每个 segment: {start_time, end_time, rps}
        如果多个文件同一时间段对同一模型有配置，将在时间线上相加。
        """
        merged: Dict[str, List[Dict[str, float]]] = {}
        for fp in file_paths:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            for model, segs in data.items():
                if not isinstance(segs, list):
                    continue
                lst = merged.setdefault(model, [])
                for seg in segs:
                    try:
                        start = float(seg["start_time"])
                        end = float(seg["end_time"])
                        rps = float(seg["rps"])
                    except Exception:
                        continue
                    if end <= start:
                        continue
                    normalized = {"start_time": start, "end_time": end, "rps": rps}
                    for key in (
                        "input_token_config",
                        "input_min_length",
                        "input_max_length",
                        "input_mean",
                        "input_std",
                        "input_distribution",
                        "input_tokens",
                        "prefill_tokens",
                        "prefill_mean",
                        "max_tokens",
                        "decode_tokens",
                        "output_tokens",
                    ):
                        if key in seg:
                            normalized[key] = seg[key]
                    lst.append(normalized)
        # 对每个模型的段按开始时间排序
        for model in merged:
            merged[model].sort(key=lambda s: (s["start_time"], s["end_time"]))
        return merged

    def _get_custom_model_segments(self) -> Tuple[List[str], Dict[str, List[Dict[str, float]]]]:
        if self._custom_trace_paths_cache is None or self._custom_model_segments_cache is None:
            self._custom_trace_paths_cache = self._expand_custom_trace_paths()
            self._custom_model_segments_cache = self._load_custom_traces(self._custom_trace_paths_cache)
        return self._custom_trace_paths_cache, self._custom_model_segments_cache

    def _get_request_overrides(self, model_name: str, timestamp: float) -> Dict[str, any]:
        _, model_segs = self._get_custom_model_segments()
        seg = None
        for candidate in model_segs.get(model_name, []):
            if candidate["start_time"] <= timestamp < candidate["end_time"]:
                seg = candidate
        if seg is None:
            return {}

        # ── mix_with: concurrent heterogeneous requests within a single segment ──
        # When a segment has a "mix_with" dict, each request randomly picks
        # between the primary config (seg) and the secondary config (mix_with)
        # based on mix_with.ratio (default 0.5 = 50/50 split).
        mix_with = seg.get("mix_with")
        if isinstance(mix_with, dict) and len(mix_with) > 0:
            ratio = float(mix_with.get("ratio", 0.5))
            if random.random() < ratio:
                # Overlay mix_with keys onto a copy of seg so we don't mutate the original
                seg = dict(seg)
                for k, v in mix_with.items():
                    if k != "ratio":
                        seg[k] = v

        overrides: Dict[str, any] = {}
        token_config = dict(self.config.input_token_config)
        if isinstance(seg.get("input_token_config"), dict):
            token_config.update(seg["input_token_config"])

        key_map = {
            "input_min_length": "min_length",
            "input_max_length": "max_length",
            "input_mean": "mean",
            "input_std": "std",
            "input_distribution": "distribution",
            "prefill_mean": "mean",
        }
        for source_key, target_key in key_map.items():
            if source_key in seg:
                token_config[target_key] = seg[source_key]

        fixed_input_tokens = seg.get("input_tokens", seg.get("prefill_tokens"))
        if fixed_input_tokens is not None:
            fixed_input_tokens = int(fixed_input_tokens)
            token_config.update({
                "min_length": fixed_input_tokens,
                "max_length": fixed_input_tokens,
                "mean": fixed_input_tokens,
                "std": 0,
                "distribution": "normal",
            })

        if token_config != self.config.input_token_config:
            overrides["input_token_config"] = token_config

        max_output_tokens = seg.get("max_tokens", seg.get("decode_tokens", seg.get("output_tokens")))
        if max_output_tokens is not None:
            overrides["max_output_tokens"] = int(max_output_tokens)

        return overrides

    def _rps_at(self, segs: List[Dict[str, float]], t: float) -> float:
        """返回时间 t 时刻所有覆盖段 rps 的总和，半开区间 [start, end)。"""
        total = 0.0
        for s in segs:
            if s["start_time"] <= t < s["end_time"]:
                total += s["rps"]
        return total

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * max(0.0, min(1.0, t))

    def _build_model_intervals(self, segs: List[Dict[str, float]], duration: float) -> Dict[str, List[float]]:
        """构建单模型的分段常值区间。
        返回 {times: [t0..tn], values: [v0..v_{n-1}]}, 其中 [ti, ti+1) 常值为 vi。
        覆盖段内 vi>0，空隙段 vi==0。
        """
        # 收集边界
        boundaries = {0.0, duration}
        for s in segs:
            boundaries.add(max(0.0, min(duration, float(s["start_time"]))))
            boundaries.add(max(0.0, min(duration, float(s["end_time"]))))
        times = sorted(x for x in boundaries if 0.0 <= x <= duration)
        if len(times) < 2:
            times = [0.0, duration]
        # 计算各区间的稳定值与覆盖标记
        values: List[float] = []
        covered: List[bool] = []
        for i in range(len(times) - 1):
            start = times[i]
            end = times[i + 1]
            if end <= start:
                values.append(0.0)
                covered.append(False)
                continue
            mid = (start + end) / 2.0
            # 是否被任一段覆盖，与具体 rps 大小无关（rps=0 也算覆盖稳定段）
            is_cov = any(s["start_time"] <= mid < s["end_time"] for s in segs)
            covered.append(is_cov)
            values.append(self._rps_at(segs, mid))
        return {"times": times, "values": values, "covered": covered}

    def _model_rps_gap_interpolated(self, intervals: Dict[str, List[float]], t: float) -> Tuple[float, bool]:
        """在空隙上进行整段线性插值。
        返回 (rps, in_transition)。当位于提供的稳定段内时 in_transition=False；
        位于空隙(无稳定段覆盖)时 in_transition=True，并在空隙两端锚点间线性插值。
        """
        times: List[float] = intervals["times"]
        values: List[float] = intervals["values"]
        covered: List[bool] = intervals["covered"]
        n = len(times) - 1
        if n <= 0:
            return 0.0, False

        # 边界修正: 将 t==times[-1] 归入最后一个区间
        if t >= times[-1]:
            idx = n - 1
        else:
            idx = 0
            while idx < n and not (times[idx] <= t < times[idx + 1]):
                idx += 1
            if idx >= n:
                idx = n - 1

        v = values[idx]
        if covered[idx]:
            # 稳定段（包含 rps==0 的稳定）
            return v, False

        # 空隙: 找到两端锚点值
        gap_start = times[idx]
        gap_end = times[idx + 1]
        prev_val = values[idx - 1] if idx - 1 >= 0 else 0.0
        next_val = values[idx + 1] if idx + 1 < n else 0.0
        # 如果前后都是0，仍返回0，但视为空隙(过渡)
        if gap_end > gap_start:
            p = (t - gap_start) / (gap_end - gap_start)
        else:
            p = 0.0
        return self._lerp(prev_val, next_val, p), True

    def generate_load_phases(self) -> List[LoadPhase]:  # override
        """根据自定义 JSON 生成阶段列表：
        - JSON 段落严格视为稳定期(常值RPS)
        - 其它时间段视为过渡期，按空隙全段线性插值
        """
        file_paths = self._expand_custom_trace_paths()
        model_segs = self._load_custom_traces(file_paths)

        duration = float(self.config.duration_seconds)

        # 活跃模型集合
        json_models = set(model_segs.keys())
        cfg_models = set(self.model_names)
        active_models = sorted(list(json_models & cfg_models)) or sorted(list(json_models))

        # 为每个模型预计算分段
        intervals_map: Dict[str, Dict[str, List[float]]] = {
            m: self._build_model_intervals(model_segs.get(m, []), duration) for m in active_models
        }

        # 收集全局边界: 0, duration, 所有模型的 start/end
        boundaries = {0.0, duration}
        for segs in model_segs.values():
            for s in segs:
                boundaries.add(max(0.0, min(duration, float(s["start_time"]))));
                boundaries.add(max(0.0, min(duration, float(s["end_time"]))));
        points = sorted(x for x in boundaries if 0.0 <= x <= duration)

        phases: List[LoadPhase] = []
        phase_id = 0
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            if end <= start:
                continue
            # 使用区间中点判定是否属于稳定或过渡
            mid = (start + end) / 2.0
            total = 0.0
            in_transition_any = False
            for m in active_models:
                rps, in_trans = self._model_rps_gap_interpolated(intervals_map[m], mid)
                total += rps
                in_transition_any = in_transition_any or in_trans

            phases.append(LoadPhase(
                start_time=start,
                end_time=end,
                base_load=total,
                phase_type="transition" if in_transition_any else "stable",
                phase_id=(phase_id if not in_transition_any else max(0, phase_id - 1)),
                load_changes=None
            ))
            if not in_transition_any:
                phase_id += 1

        print(f"📊 生成负载阶段(自定义JSON): 使用文件 {file_paths}")
        print(f"✅ 生成了 {len(phases)} 个阶段(含过渡期)")
        return phases

    def generate_load_timeline(self, time_step: float = 1.0) -> List[ModelLoadPoint]:
        """根据自定义json生成完整的负载时间序列：
        - JSON 段为稳定常值
        - 其它时间按空隙整段线性插值
        """
        file_paths = self._expand_custom_trace_paths()
        model_segs = self._load_custom_traces(file_paths)

        duration = float(self.config.duration_seconds)

        # 活跃模型集合
        json_models = set(model_segs.keys())
        cfg_models = set(self.model_names)
        active_models = sorted(list(json_models & cfg_models)) or sorted(list(json_models))

        # 预计算每个模型的常值区间
        intervals_map: Dict[str, Dict[str, List[float]]] = {
            m: self._build_model_intervals(model_segs.get(m, []), duration) for m in active_models
        }

        t = 0.0
        timeline: List[ModelLoadPoint] = []

        print(f"⏱️ 生成自定义负载时间序列: 时间步长 {time_step}s, 时长 {duration}s")

        while t <= duration + 1e-9:
            model_loads: Dict[str, float] = {}
            total = 0.0
            in_transition_any = False
            for model in active_models:
                r, in_trans = self._model_rps_gap_interpolated(intervals_map[model], t)
                model_loads[model] = r
                total += r
                if in_trans:
                    in_transition_any = True

            timeline.append(ModelLoadPoint(
                timestamp=t,
                total_load=total,
                model_loads=model_loads,
                phase_type="transition" if in_transition_any else "stable"
            ))
            t += time_step

        print(f"✅ 生成了 {len(timeline)} 个时间点的负载数据（自定义JSON+空隙插值）")
        return timeline


class IntegratedTraceGenerator3(IntegratedTraceGenerator):
    """
    集成的Trace生成器 - 版本3，使用AzurePublicDataset生成的真实trace文件
    每个模型独立配置自己的trace文件(real_trace_file)
    """

    def generate_complete_trace_data(self) -> Dict[str, any]:
        """使用RealTraceAdapter加载并转换Azure trace"""
        from .real_trace_adapter import RealTraceAdapter

        print(f"\n  Real Trace模式 - 加载真实trace数据")

        adapter = RealTraceAdapter(self.config_manager, self.data_generator)
        rps_timeline, token_rate_timeline, traces = adapter.convert_all_models()

        # 获取输出路径
        output_paths = self.config_manager.get_output_paths()

        # 导出数据
        self.export_load_timeline(rps_timeline, output_paths['load_timeline_rps'], "RPS")
        self.export_load_timeline(token_rate_timeline, output_paths['load_timeline_token_rate'], "Token Rate")
        self.export_traces(traces, output_paths['trace_data'])

        # 生成图表
        # 临时设置model_names为实际出现的模型
        active_models = list(set(t.model_name for t in traces))
        original_model_names = self.model_names
        self.model_names = active_models
        self.plot_load_timelines(rps_timeline, token_rate_timeline, output_paths['trace_plots'])
        self.model_names = original_model_names

        # 生成统计信息
        stats = self._get_generation_statistics(rps_timeline, token_rate_timeline, traces)

        print(f"\n  Real Trace数据生成完成！")
        print(f"   RPS文件: {output_paths['load_timeline_rps']}")
        print(f"   Token Rate文件: {output_paths['load_timeline_token_rate']}")
        print(f"   Traces文件: {output_paths['trace_data']}")
        print(f"   图表文件: {output_paths['trace_plots']}")

        return {
            "rps_timeline": rps_timeline,
            "token_rate_timeline": token_rate_timeline,
            "traces": traces,
            "statistics": stats,
            "output_paths": output_paths
        }


def create_trace_generator(
    config_path: str = "config/default.yaml",
    config_manager: Optional[ConfigManager] = None
) -> IntegratedTraceGenerator:
    """
    创建trace生成器的便捷函数(工厂函数)
    
    Args:
        config_path: 配置文件路径
        config_manager: 已加载的配置管理器实例，优先使用
        
    Returns:
        IntegratedTraceGenerator实例
    """
    if config_manager is None:
        from .config_manager import ConfigManager

        config_manager = ConfigManager(config_path)

    if config_manager.config is None:
        config_manager.load_config()
    if config_manager.config.generate_mode == "auto":
        return IntegratedTraceGenerator(config_manager)
    elif config_manager.config.generate_mode == "custom":
        return IntegratedTraceGenerator2(config_manager)
    elif config_manager.config.generate_mode == "real_trace":
        return IntegratedTraceGenerator3(config_manager)
    else:
        print("Error: 错误的generate_mode！支持: auto, custom, real_trace")
        return
