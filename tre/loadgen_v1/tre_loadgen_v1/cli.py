#!/usr/bin/env python3
"""
CustomTraceGenerator - 增强版负载测试系统

支持分阶段执行的主程序：
1. trace构建 - 生成负载时间序列和请求trace
2. 客户端分发 - 发送HTTP请求并记录性能指标  
3. 指标分析 - 计算百分位数并生成分析报告

作者: CustomTraceGenerator Team
版本: 1.0.0

[v2 port] 由 v1 main.py 移植（tre/loadgen_v1）。与 v1 的差异仅限：
  --base-url       目标网关（必须显式给出，不再读配置里的 gateway_endpoint）
  --max-retries    OpenAI SDK 重试次数（默认 2，与 v1 硬编码值相同）
  --routing-strategy  routing-strategy 请求头（默认取配置，v1 配置均为 least-gpu-cache；传 "" 不发该头）
  --trace-file     直接重放 v1 记录的逐请求计划 traces.json（跳过阶段1生成）
以及 performance_metrics.json 每行追加的审计字段（见 client_dispatcher.ResponseRecord）。

用法（重放 v1 投稿时记录的 trace）:
  python3 -m tre_loadgen_v1 --stage all \\
      --config loadgen_v1/configs/traces_v14/Decode_heavy_burst/config.yaml \\
      --trace-file /data/nfs_shared_data/xxy/trace_output/output_traces_v14/tre/Decode_heavy_burst/traces.json \\
      --base-url http://<gateway-host>:<port> \\
      --output /abs/path/out/tre/Decode_heavy_burst
"""

import sys
import os
import json
import shutil
import hashlib
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

from .config_manager import ConfigManager
# [v2 port] v1 在模块顶部 import trace_generator（会连带 import modelscope/transformers）；
# 改为阶段1内部再 import，让纯分发路径不依赖 tokenizer 栈。行为不变。
from .client_dispatcher import ClientDispatcher
from .metrics_analysis import MetricsAnalysis


# [v2 port] 记录 v1 trace 旁边一并复制到输出目录的文件（v1 输出目录布局即如此）
V1_TIMELINE_KEYS = ('load_timeline_rps', 'load_timeline_token_rate')
RUN_META_FILE = 'loadgen_run_meta.json'


def normalize_gateway_endpoint(base_url: str) -> str:
    """[v2 port] --base-url 允许写成 http://host:port 或 http://host:port/v1；
    统一成 v1 的 gateway_endpoint 语义（client 内部再拼 /v1）。"""
    url = (base_url or '').strip().rstrip('/')
    if url.endswith('/v1'):
        url = url[:-3]
    return url


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class CustomTraceGenerator:
    """CustomTraceGenerator主类"""

    def __init__(self, config_path: str, output_dir: str = None, verbose: bool = False,
                 base_url: str = None, max_retries: int = None,
                 routing_strategy: str = None, trace_file: str = None):
        """
        初始化CustomTraceGenerator

        Args:
            config_path: 配置文件路径
            output_dir: 输出目录名称
            verbose: 是否显示详细信息
            base_url: [v2 port] 目标网关地址（分发阶段必填）
            max_retries: [v2 port] OpenAI SDK 最大重试次数（None=配置值，默认 2）
            routing_strategy: [v2 port] routing-strategy 头（None=配置值；""=不发送）
            trace_file: [v2 port] 要重放的 v1 traces.json（None=用阶段1生成的）
        """
        self.config_path = config_path
        self.trace_file = trace_file
        
        # 如果没有指定输出目录，使用配置文件中的设置，否则使用时间戳
        if output_dir is None:
            # 先加载配置以获取默认的sub_dir_name
            temp_config_manager = ConfigManager(config_path)
            temp_config = temp_config_manager.load_config()
            self.output_dir = temp_config.output.sub_dir_name or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = output_dir
            
        self.verbose = verbose
        
        # 初始化配置管理器
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config(self.output_dir)

        # [v2 port] 目标地址必须显式给出：丢弃配置文件里的 gateway_endpoint（v1 配置里是 localhost:8888 端口转发）
        self.config.gateway_endpoint = normalize_gateway_endpoint(base_url) if base_url else ""
        if max_retries is not None:
            self.config.client.max_retries = int(max_retries)
        if routing_strategy is not None:
            self.config.client.routing_algorithm = routing_strategy
        self.config.client.validate_process_config()

        if self.verbose:
            print(f"📋 配置已加载:")
            print(f"   测试模式: {self.config.load_mode}")
            print(f"   测试时长: {self.config.duration_seconds}s")
            print(f"   总负载: {self.config.total_load}")
            print(f"   输出目录: {self.config.output.get_full_output_dir()}")
            print(f"   模型数量: {len(self.config.models)}")
            print(f"   百分位数: {self.config.analysis.percentiles}")

    def stage1_trace_generation(self) -> bool:
        """
        阶段1: Trace构建
        生成负载时间序列和请求trace数据
        
        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("🚀 阶段1: Trace构建")
        print("="*60)
        
        try:
            # 创建trace生成器
            # trace_generator = IntegratedTraceGenerator(self.config_manager)
            from .trace_generator import create_trace_generator  # [v2 port] 延迟 import

            trace_generator = create_trace_generator(
                config_path=str(self.config_path),
                config_manager=self.config_manager
            )
            
            # 生成完整的trace数据
            result = trace_generator.generate_complete_trace_data()
            
            traces = result["traces"]
            rps_timeline = result["rps_timeline"]  
            token_rate_timeline = result["token_rate_timeline"]
            statistics = result["statistics"]
            
            print(f"✅ Trace构建完成")
            print(f"   生成请求数: {len(traces)}")
            print(f"   RPS时间点数: {len(rps_timeline)}")
            print(f"   Token Rate时间点数: {len(token_rate_timeline)}")
            
            if self.verbose and statistics:
                if "总体统计" in statistics:
                    stats = statistics["总体统计"]
                    print(f"   平均RPS: {stats.get('平均RPS', 'N/A')}")
                    print(f"   平均Token Rate: {stats.get('平均Token Rate', 'N/A')}")
            
            # 显示输出文件
            output_paths = self.config_manager.get_output_paths()
            print(f"\n📁 生成的文件:")
            for key in ['load_timeline_rps', 'load_timeline_token_rate', 'trace_data', 'trace_plots']:
                if key in output_paths:
                    print(f"   {key}: {output_paths[key]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Trace构建失败: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def stage_import_recorded_trace(self) -> bool:
        """
        [v2 port] 阶段1替代: 导入 v1 记录的逐请求计划
        把 --trace-file 指向的 traces.json 复制到输出目录的 trace_data 路径，
        同目录下的 load_timeline_*.json 若存在也一并复制（与 v1 输出目录布局一致，
        v1 的分析/画图脚本可直接复用）。请求内容、时间戳、max_output_tokens 原样不变。
        """
        print("\n" + "="*60)
        print("📥 阶段1(替代): 导入已记录的 trace")
        print("="*60)
        try:
            src = os.path.abspath(self.trace_file)
            if not os.path.exists(src):
                print(f"❌ Trace文件不存在: {src}")
                return False
            output_paths = self.config_manager.get_output_paths()
            dst = os.path.abspath(output_paths['trace_data'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if src != dst:
                shutil.copyfile(src, dst)
            print(f"   traces: {src} -> {dst}")
            src_dir = os.path.dirname(src)
            for key in V1_TIMELINE_KEYS:
                if key not in output_paths:
                    continue
                cand = os.path.join(src_dir, os.path.basename(output_paths[key]))
                tgt = os.path.abspath(output_paths[key])
                if os.path.exists(cand) and os.path.abspath(cand) != tgt:
                    shutil.copyfile(cand, tgt)
                    print(f"   {key}: {cand} -> {tgt}")
            return True
        except Exception as e:
            print(f"❌ 导入trace失败: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def _write_run_meta(self, trace_file: str, output_paths: dict, extra: dict = None):
        """[v2 port] 审计: 记录本次分发的参数与输入指纹（新文件，不影响 v1 文件）"""
        try:
            import openai
            import httpx
            meta = {
                "tool": "tre_loadgen_v1",
                "written_at": datetime.now().isoformat(),
                "config_path": os.path.abspath(str(self.config_path)),
                "trace_source": os.path.abspath(self.trace_file) if self.trace_file else None,
                "trace_data": os.path.abspath(trace_file),
                "trace_sha256": _sha256(trace_file),
                "gateway_endpoint": self.config.gateway_endpoint,
                "openai_base_url": f"{self.config.gateway_endpoint}/v1",
                "max_retries": self.config.client.max_retries,
                "timeout": self.config.client.timeout,
                "routing_strategy_header": self.config.client.routing_algorithm or None,
                "enable_streaming": self.config.client.enable_streaming,
                "process_count": self.config.client.process_count,
                "max_coroutines_per_process": self.config.client.max_coroutines_per_process,
                "task_batch_window": self.config.client.task_batch_window,
                "load_monitor_interval": self.config.client.load_monitor_interval,
                "models": [{"name": m.name, "max_tokens": m.max_tokens, "temperature": m.temperature}
                           for m in self.config.models],
                "openai_version": openai.__version__,
                "httpx_version": httpx.__version__,
                "python": sys.version.split()[0],
            }
            if extra:
                meta.update(extra)
            meta_path = os.path.join(os.path.dirname(output_paths['performance_metrics']), RUN_META_FILE)
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 写入运行元数据失败: {e}")

    def stage2_client_dispatch(self) -> bool:
        """
        阶段2: 客户端分发
        发送HTTP请求并记录性能指标

        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("🚦 阶段2: 客户端分发")
        print("="*60)

        # [v2 port] 目标地址必须显式给出
        if not self.config.gateway_endpoint:
            print("❌ 未指定目标网关: 分发阶段必须显式传入 --base-url")
            return False

        try:
            # 检查trace文件是否存在
            output_paths = self.config_manager.get_output_paths()
            trace_file = output_paths['trace_data']
            
            if not os.path.exists(trace_file):
                print(f"❌ Trace文件不存在: {trace_file}")
                print("   请先运行阶段1: trace构建")
                return False
            
            # 加载trace数据
            import json
            with open(trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
            
            if not trace_data:
                print(f"❌ Trace文件为空")
                return False
            
            # 转换为RequestTrace对象
            from .trace_types import RequestTrace  # [v2 port] 同一 dataclass，见 trace_types.py
            traces = []
            for data in trace_data:
                trace = RequestTrace(
                    request_id=data['request_id'],
                    timestamp=data['timestamp'],
                    model_name=data['model_name'],
                    prompt=data['prompt'],
                    prompt_length=data['prompt_length'],
                    phase_type=data.get('phase_type', 'unknown'),
                    max_output_tokens=data.get('max_output_tokens')
                )
                traces.append(trace)
            
            print(f"📥 加载了 {len(traces)} 个请求trace")
            print(f"🎯 目标端点: {self.config.gateway_endpoint}")
            print(f"⚙️ 路由算法: {self.config.client.routing_algorithm}")
            print(f"🔄 流式请求: {self.config.client.enable_streaming}")
            
            # 创建客户端调度器
            dispatcher = ClientDispatcher(self.config_manager)
            
            # 执行请求调度 - 使用asyncio.run调用异步方法
            print(f"\n🚀 开始发送请求...")
            
            # 使用asyncio运行异步dispatch_traces方法
            response_records = asyncio.run(dispatcher.dispatch_traces(traces))
            
            # 显示基础统计
            success_count = sum(1 for r in response_records if r.success)
            error_count = len(response_records) - success_count
            success_rate = success_count / len(response_records) * 100 if response_records else 0
            
            print(f"\n✅ 客户端分发完成")
            print(f"   执行请求数: {len(response_records)}")
            print(f"   成功请求数: {success_count}")
            print(f"   失败请求数: {error_count}")
            print(f"   成功率: {success_rate:.1f}%")
            print(f"   性能指标文件: {output_paths['performance_metrics']}")

            # [v2 port] 审计汇总（v1 口径的 success 里有多少其实是流中途断开/经过重试）
            interrupted = sum(1 for r in response_records if getattr(r, 'stream_interrupted', False))
            retried = sum(1 for r in response_records if getattr(r, 'attempts', 0) > 1)
            print(f"   [审计] 流中途断开但记为成功: {interrupted}")
            print(f"   [审计] 发生过重试的请求: {retried}")
            self._write_run_meta(trace_file, output_paths, extra={
                "summary": {
                    "expected_requests": len(traces),
                    "collected_records": len(response_records),
                    "success": success_count,
                    "failed": error_count,
                    "success_but_stream_interrupted": interrupted,
                    "retried_requests": retried,
                    "total_attempts": sum(getattr(r, 'attempts', 0) for r in response_records),
                }
            })

            return True
            
        except Exception as e:
            print(f"❌ 客户端分发失败: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def stage3_metrics_analysis(self) -> bool:
        """
        阶段3: 指标分析
        生成CDF图表
        
        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("📈 阶段3: CDF图表生成")
        print("="*60)
        
        try:
            # 检查性能指标文件是否存在
            output_paths = self.config_manager.get_output_paths()
            metrics_file = output_paths['performance_metrics']
            
            if not os.path.exists(metrics_file):
                print(f"❌ 性能指标文件不存在: {metrics_file}")
                print("   请先运行阶段2: 客户端分发")
                return False
            
            # 创建性能分析器
            analyzer = MetricsAnalysis(self.config_manager)
            
            # 加载性能指标数据
            if not analyzer.load_metrics_data(metrics_file):
                return False
            
            print(f"📊 分析配置:")
            print(f"   百分位数: {self.config.analysis.percentiles}")
            
            # 生成CDF图表
            plots_output_dir = output_paths['performance_plots']
            # 如果performance_plots指向一个文件而不是目录，获取其目录
            if not plots_output_dir.endswith('/'):
                plots_output_dir = os.path.dirname(plots_output_dir)
            
            analyzer.generate_cdf_plots(plots_output_dir)
            # 新增：生成prompt/input长度CDF图
            analyzer.generate_length_cdf_plots(metrics_file, plots_output_dir)

            # 新增：不区分模型，所有请求整体CDF图（E2E/TTFT/TPOT）
            analyzer.generate_overall_cdf_plot(plots_output_dir, file_name='all_requests_cdf.png')

            # 新增：将所有CDF图汇总到一个PDF中
            analyzer.merge_cdf_pngs_to_pdf(
                input_dir=plots_output_dir,
                output_pdf=os.path.join(plots_output_dir, 'cdf_summary.pdf')
            )
            
            print(f"\n✅ CDF图表生成完成")
            print(f"   图表目录: {plots_output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ CDF图表生成失败: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def run_all_stages(self) -> bool:
        """
        运行所有阶段
        
        Returns:
            是否全部成功
        """
        print(f"🚀 CustomTraceGenerator - 完整负载测试流程")
        print(f"📁 输出目录: {self.config.output.get_full_output_dir()}")
        
        # 阶段1: Trace构建（[v2 port] 给了 --trace-file 时改为导入已记录的 trace）
        if self.trace_file:
            if not self.stage_import_recorded_trace():
                return False
        elif not self.stage1_trace_generation():
            return False

        # 阶段2: 客户端分发
        if not self.stage2_client_dispatch():
            return False
        
        # 阶段3: 指标分析
        if not self.stage3_metrics_analysis():
            return False
        
        print(f"\n🎉 所有阶段执行完成！")
        print(f"\n📁 查看结果:")
        output_paths = self.config_manager.get_output_paths()
        for file_key, file_path in output_paths.items():
            if os.path.exists(file_path):
                print(f"   ✅ {file_key}: {file_path}")
        
        return True


def main(argv=None):
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CustomTraceGenerator - 负载测试系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行所有阶段（生成新 trace -> 分发 -> 分析）
  python3 -m tre_loadgen_v1 --config config.yaml --base-url http://GW:PORT

  # 仅运行trace构建
  python3 -m tre_loadgen_v1 --stage trace --config config.yaml

  # 重放 v1 记录的 traces.json（导入 -> 分发 -> 分析）
  python3 -m tre_loadgen_v1 --config config.yaml --trace-file .../traces.json \\
      --base-url http://GW:PORT --output /abs/out/dir
        """
    )
    
    parser.add_argument('--config', '-c', 
                       default='config/default.yaml',
                       help='配置文件路径 (默认: config/default.yaml)')
    
    parser.add_argument('--output', '-o',
                       help='输出目录名称 (默认: 自动生成时间戳)')
    
    parser.add_argument('--stage', '-s',
                       choices=['trace', 'dispatch', 'analysis', 'all'],
                       default='all',
                       help='执行阶段: trace(构建), dispatch(分发), analysis(分析), all(全部)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='显示详细信息')

    # ---- [v2 port] 新增参数 ----
    parser.add_argument('--base-url', default='',
                       help='目标网关地址，如 http://host:port（可带 /v1）。分发阶段必填，无默认值；'
                            '配置文件里的 gateway_endpoint 被忽略')
    parser.add_argument('--max-retries', type=int, default=None,
                       help='OpenAI SDK 最大重试次数（默认取配置 client.max_retries，缺省 2 = v1 硬编码值）')
    parser.add_argument('--routing-strategy', default=None,
                       help='routing-strategy 请求头取值（默认取配置 client.routing_algorithm，'
                            'v1 配置均为 least-gpu-cache；传空串则不发送该头）')
    parser.add_argument('--trace-file', default=None,
                       help='重放 v1 记录的逐请求计划 traces.json（跳过阶段1生成；'
                            '同目录 load_timeline_*.json 一并复制到输出目录）')

    args = parser.parse_args(argv)

    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return 1
    if args.stage in ('dispatch', 'all') and not args.base_url.strip():
        print("❌ 分发阶段必须显式指定 --base-url（不再使用配置文件中的 gateway_endpoint）")
        return 2
    if args.trace_file and args.stage == 'trace':
        print("❌ --trace-file 与 --stage trace 互斥（导入已记录的 trace 不需要生成）")
        return 2

    try:
        # 创建CustomTraceGenerator实例
        ctg = CustomTraceGenerator(
            config_path=args.config,
            output_dir=args.output,
            verbose=args.verbose,
            base_url=args.base_url,
            max_retries=args.max_retries,
            routing_strategy=args.routing_strategy,
            trace_file=args.trace_file,
        )

        success = False

        # 根据选择的阶段执行
        if args.stage == 'trace':
            success = ctg.stage1_trace_generation()

        elif args.stage == 'dispatch':
            if args.trace_file and not ctg.stage_import_recorded_trace():
                return 1
            success = ctg.stage2_client_dispatch()
            
        elif args.stage == 'analysis':
            success = ctg.stage3_metrics_analysis()
            
        else:  # all
            success = ctg.run_all_stages()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
        return 130
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 