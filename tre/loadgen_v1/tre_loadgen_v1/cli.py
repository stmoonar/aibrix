#!/usr/bin/env python3
"""
CustomTraceGenerator - 增强版负载测试系统

支持分阶段执行的主程序：
1. trace构建 - 生成负载时间序列和请求trace
2. 客户端分发 - 发送HTTP请求并记录性能指标  
3. 指标分析 - 计算百分位数并生成分析报告

作者: CustomTraceGenerator Team
版本: 1.0.0
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# 添加src目录到Python路径
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from .config_manager import ConfigManager
from .trace_generator import IntegratedTraceGenerator, create_trace_generator
from .client_dispatcher import ClientDispatcher
from .metrics_analysis import MetricsAnalysis


class CustomTraceGenerator:
    """CustomTraceGenerator主类"""
    
    def __init__(self, config_path: str, output_dir: str = None, verbose: bool = False):
        """
        初始化CustomTraceGenerator
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录名称
            verbose: 是否显示详细信息
        """
        self.config_path = config_path
        
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
            from .trace_generator import RequestTrace
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
        
        # 阶段1: Trace构建
        if not self.stage1_trace_generation():
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CustomTraceGenerator - 负载测试系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行所有阶段
  python3 main.py --config config/default.yaml
  
  # 仅运行trace构建
  python3 main.py --stage trace --config config/default.yaml
  
  # 详细输出模式
  python3 main.py --verbose --output detailed_test
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
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return 1
    
    try:
        # 创建CustomTraceGenerator实例
        ctg = CustomTraceGenerator(
            config_path=args.config,
            output_dir=args.output,
            verbose=args.verbose
        )
        
        success = False
        
        # 根据选择的阶段执行
        if args.stage == 'trace':
            success = ctg.stage1_trace_generation()
            
        elif args.stage == 'dispatch':
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