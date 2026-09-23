"""配置管理模块

负责加载、验证和管理测试配置
支持统一的输出路径管理和模型配置
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class OutputConfig:
    """输出配置"""
    base_dir: str
    sub_dir_name: str
    files: Dict[str, str]
    
    def get_full_output_dir(self) -> str:
        """获取完整的输出目录路径"""
        return os.path.join(self.base_dir, self.sub_dir_name)
    
    def get_file_path(self, file_key: str) -> str:
        """获取指定文件的完整路径"""
        if file_key not in self.files:
            raise ConfigError(f"未知的文件键: {file_key}")
        
        full_dir = self.get_full_output_dir()
        file_name = self.files[file_key]
        return os.path.join(full_dir, file_name)
    
    def ensure_directories(self):
        """确保所有必需的目录存在"""
        # 创建主输出目录
        os.makedirs(self.get_full_output_dir(), exist_ok=True)
        
        # 创建包含子目录的文件路径的目录
        for file_path in self.files.values():
            full_file_path = os.path.join(self.get_full_output_dir(), file_path)
            file_dir = os.path.dirname(full_file_path)
            if file_dir:  # 如果文件在子目录中
                os.makedirs(file_dir, exist_ok=True)


@dataclass
class ClientConfig:
    """客户端配置"""
    api_key: str
    timeout: float
    enable_streaming: bool
    log_level: str
    routing_algorithm: str
    send_workers: int = 4
    recv_workers: int = 32
    max_connections: int = 64
    max_keepalive_connections: int = 64
    # 新增多进程负载均衡配置
    process_count: int = 4
    max_coroutines_per_process: int = 500
    task_batch_window: float = 5.0
    load_monitor_interval: float = 1.0
    oracle_trace_upload_url: str = None
    # [v2 port] v1 在 client_dispatcher 里硬编码 max_retries=2；这里做成参数，默认值不变
    max_retries: int = 2

    def validate_routing_algorithm(self):
        """验证路由算法配置"""
        valid_algorithms = [
            "round_robin", "random", "weighted_random", "vtc-basic",
            "least_connections", "response_time", "least-gpu-cache"
        ]
        
        if self.routing_algorithm not in valid_algorithms:
            raise ConfigError(
                f"不支持的路由算法: {self.routing_algorithm}. "
                f"支持的算法: {', '.join(valid_algorithms)}"
            )
    
    def validate_thread_config(self):
        """验证线程配置"""
        if self.send_workers < 1:
            raise ConfigError("send_workers 必须至少为 1")
        if self.recv_workers < 1:
            raise ConfigError("recv_workers 必须至少为 1")
    
    def validate_process_config(self):
        """验证多进程配置"""
        if self.process_count < 1:
            raise ConfigError("process_count 必须至少为 1")
        if self.max_coroutines_per_process < 1:
            raise ConfigError("max_coroutines_per_process 必须至少为 1")
        if self.task_batch_window <= 0:
            raise ConfigError("task_batch_window 必须大于 0")
        if self.load_monitor_interval <= 0:
            raise ConfigError("load_monitor_interval 必须大于 0")
        if self.max_retries < 0:
            raise ConfigError("max_retries 不能为负数")
        


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    modelscope_url: str
    temperature: Optional[float] = None  # 只有配置文件中明确设置才不为None
    max_tokens: Optional[int] = None     # 只有配置文件中明确设置才不为None
    real_trace_file: Optional[str] = None  # real_trace模式下的trace文件路径
    
    def get_tokenizer_download_info(self) -> Dict[str, str]:
        """获取tokenizer下载信息"""
        return {
            "url": self.modelscope_url,
            "model_name": self.name
        }
    
    def get_request_params(self) -> Dict[str, any]:
        """获取请求参数（只包含明确设置的参数）"""
        params = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params


@dataclass
class AnalysisConfig:
    """分析配置"""
    percentiles: List[int]


@dataclass
class LoadTestConfig:
    """负载测试配置"""
    duration_seconds: int
    gateway_endpoint: str
    output: OutputConfig
    client: ClientConfig
    models: List[ModelConfig]
    load_mode: str
    total_load: float
    stable_period: Dict[str, int]
    transition_period: Dict[str, int]
    noise: Dict[str, Any]
    input_token_config: Dict[str, Any]
    analysis: AnalysisConfig
    generate_mode: str
    custom_trace_json: Union[List[str], str]
    real_trace_config: Optional[Dict[str, Any]] = None
    random_seed: Optional[int] = None
    
    def validate(self):
        """验证配置的有效性"""
        # 验证负载模式
        if self.load_mode not in ['rps', 'token_rate']:
            raise ConfigError("load_mode 必须是 'rps' 或 'token_rate'")
        
        # 验证客户端配置
        self.client.validate_routing_algorithm()
        self.client.validate_thread_config()
        self.client.validate_process_config()
        
        # 验证模型配置
        if not self.models:
            raise ConfigError("至少需要配置一个模型")
        
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ConfigError("模型名称不能重复")


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Optional[LoadTestConfig] = None

    def load_config(self, sub_dir_name: Optional[str] = None) -> LoadTestConfig:
        """
        加载并验证配置文件
        
        Args:
            sub_dir_name: 可选的输出子目录名称，会覆盖配置文件中的设置
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            # 验证配置格式
            self._validate_config_structure(raw_config)

            # 转换为配置对象
            config_dict = raw_config['custom_load_test']
            
            # 如果指定了子目录名，覆盖配置文件中的设置
            if sub_dir_name:
                config_dict['output']['sub_dir_name'] = sub_dir_name

            self.config = self._parse_config(config_dict)
            
            # 验证配置有效性
            self.config.validate()
            
            # 确保输出目录存在
            self.config.output.ensure_directories()

            return self.config

        except Exception as e:
            raise ConfigError(f"配置加载失败: {e}")

    def _validate_config_structure(self, config: Dict[str, Any]):
        """验证配置文件结构"""
        if 'custom_load_test' not in config:
            raise ConfigError("配置文件缺少 'custom_load_test' 节点")

        required_fields = [
            'duration_seconds', 'gateway_endpoint', 'output', 'client',
            'models', 'load_mode', 'total_load', 'stable_period', 
            'transition_period', 'analysis'
        ]

        test_config = config['custom_load_test']
        for field in required_fields:
            if field not in test_config:
                raise ConfigError(f"配置文件缺少必需字段: {field}")

    def _parse_config(self, config_dict: Dict[str, Any]) -> LoadTestConfig:
        """解析配置字典为配置对象"""
        
        # 解析输出配置
        output_config = self._parse_output_config(config_dict['output'])
        
        # 解析客户端配置
        client_config = self._parse_client_config(config_dict['client'])
        
        # 解析模型配置
        models = self._parse_models_config(config_dict['models'])
        
        # 解析分析配置
        analysis_config = self._parse_analysis_config(config_dict['analysis'])

        return LoadTestConfig(
            duration_seconds=config_dict['duration_seconds'],
            gateway_endpoint=config_dict['gateway_endpoint'],
            output=output_config,
            client=client_config,
            models=models,
            load_mode=config_dict['load_mode'],
            total_load=config_dict['total_load'],
            stable_period=config_dict['stable_period'],
            transition_period=config_dict['transition_period'],
            noise=config_dict.get('noise', {'enabled': False}),
            input_token_config=config_dict.get('input_token_config', {}),
            analysis=analysis_config,
            random_seed=config_dict.get('random_seed', None),
            generate_mode=config_dict.get('generate_mode', 'auto'),
            custom_trace_json=config_dict.get('custom_trace_json', None),
            real_trace_config=config_dict.get('real_trace_config', None)
        )

    def _parse_output_config(self, output_dict: Dict[str, Any]) -> OutputConfig:
        """解析输出配置"""
        required_fields = ['base_dir', 'sub_dir_name', 'files']
        for field in required_fields:
            if field not in output_dict:
                raise ConfigError(f"输出配置缺少必需字段: {field}")

        return OutputConfig(
            base_dir=output_dict['base_dir'],
            sub_dir_name=output_dict['sub_dir_name'],
            files=output_dict['files']
        )

    def _parse_client_config(self, client_dict: Dict[str, Any]) -> ClientConfig:
        """解析客户端配置"""
        required_fields = ['routing_algorithm']
        for field in required_fields:
            if field not in client_dict:
                raise ConfigError(f"客户端配置缺少必需字段: {field}")

        return ClientConfig(
            api_key=client_dict.get('api_key', ''),
            timeout=client_dict.get('timeout', 30.0),
            enable_streaming=client_dict.get('enable_streaming', True),
            log_level=client_dict.get('log_level', 'INFO'),
            routing_algorithm=client_dict['routing_algorithm'],
            send_workers=client_dict.get('send_workers', 4),
            recv_workers=client_dict.get('recv_workers', 32),
            max_connections=client_dict.get('max_connections', 64),
            max_keepalive_connections=client_dict.get('max_keepalive_connections', 64),
            # 多进程负载均衡配置
            process_count=client_dict.get('process_count', 4),
            max_coroutines_per_process=client_dict.get('max_coroutines_per_process', 500),
            task_batch_window=client_dict.get('task_batch_window', 5.0),
            load_monitor_interval=client_dict.get('load_monitor_interval', 1.0),
            oracle_trace_upload_url=client_dict.get('oracle_trace_upload_url', None),
            max_retries=client_dict.get('max_retries', 2)
        )

    def _parse_models_config(self, models_list: List[Dict[str, Any]]) -> List[ModelConfig]:
        """解析模型配置"""
        models = []
        for model_dict in models_list:
            required_fields = ['name', 'modelscope_url']
            for field in required_fields:
                if field not in model_dict:
                    raise ConfigError(f"模型配置缺少必需字段: {field}")

            # 只有在配置文件中明确设置了才传递参数，否则为None
            temperature = model_dict.get('temperature') if 'temperature' in model_dict else None
            max_tokens = model_dict.get('max_tokens') if 'max_tokens' in model_dict else None
            real_trace_file = model_dict.get('real_trace_file', None)

            models.append(ModelConfig(
                name=model_dict['name'],
                modelscope_url=model_dict['modelscope_url'],
                temperature=temperature,
                max_tokens=max_tokens,
                real_trace_file=real_trace_file
            ))

        return models

    def _parse_analysis_config(self, analysis_dict: Dict[str, Any]) -> AnalysisConfig:
        """解析分析配置"""
        return AnalysisConfig(
            percentiles=analysis_dict.get('percentiles', [50, 90, 99])
        )

    def get_model_names(self) -> List[str]:
        """获取模型名称列表"""
        if not self.config:
            raise RuntimeError("配置未加载")
        return [model.name for model in self.config.models]

    def get_model_urls(self) -> Dict[str, str]:
        """获取所有模型的URL"""
        if not self.config:
            raise RuntimeError("配置未加载")
        
        return {
            model.name: model.modelscope_url 
            for model in self.config.models
        }

    def get_output_paths(self) -> Dict[str, str]:
        """获取所有输出文件的完整路径"""
        if not self.config:
            raise RuntimeError("配置未加载")
        
        paths = {}
        # 添加所有配置的文件路径
        for file_key in self.config.output.files.keys():
            paths[file_key] = self.config.output.get_file_path(file_key)
        
        return paths

    def save_config_summary(self, output_path: str):
        """保存配置摘要"""
        if not self.config:
            raise RuntimeError("配置未加载")

        summary = {
            "测试配置摘要": {
                "测试时长": f"{self.config.duration_seconds}秒",
                "负载模式": self.config.load_mode,
                "总负载": self.config.total_load,
                "网关端点": self.config.gateway_endpoint,
                "输出目录": self.config.output.get_full_output_dir()
            },
            "模型配置": [
                {
                    "名称": model.name,
                    "ModelScope URL": model.modelscope_url
                }
                for model in self.config.models
            ],
            "客户端配置": {
                "路由算法": self.config.client.routing_algorithm,
                "启用流式": self.config.client.enable_streaming,
                "超时时间": f"{self.config.client.timeout}秒",
                "日志级别": self.config.client.log_level,
                "发送工作线程": self.config.client.send_workers,
                "接收工作线程": self.config.client.recv_workers,
                "最大连接数": self.config.client.max_connections,
                "最大保持连接数": self.config.client.max_keepalive_connections,
                "工作进程数": self.config.client.process_count,
                "单进程最大协程数": self.config.client.max_coroutines_per_process,
                "任务批次时间窗口": f"{self.config.client.task_batch_window}秒",
                "负载监控间隔": f"{self.config.client.load_monitor_interval}秒"
            },
            "负载配置": {
                "稳定期时长": f"{self.config.stable_period['min_duration']}-{self.config.stable_period['max_duration']}秒",
                "过渡期时长": f"{self.config.transition_period['min_duration']}-{self.config.transition_period['max_duration']}秒",
                "噪音配置": self.config.noise
            },
            "分析配置": {
                "百分位数": self.config.analysis.percentiles
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

    def update_sub_dir_name(self, sub_dir_name: str):
        """更新输出子目录名称"""
        if not self.config:
            raise RuntimeError("配置未加载")
        
        self.config.output.sub_dir_name = sub_dir_name
        # 重新创建目录
        self.config.output.ensure_directories()


class ConfigError(Exception):
    """配置错误异常"""
    pass 