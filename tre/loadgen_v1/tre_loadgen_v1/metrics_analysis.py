"""
性能指标分析模块 - CDF绘图专用版本

专门用于根据performance_metrics.json生成CDF图表
为每个模型分别绘制E2E、TTFT、TPOT的CDF图
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging
from datetime import datetime

from .config_manager import ConfigManager


class MetricsAnalysis:
    """性能指标分析器 - 专注于CDF图表生成"""

    def __init__(self, config_manager: ConfigManager):
        """
        初始化性能分析器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.metrics_data: List[Dict[str, Any]] = []
        self.logger = self._setup_logger()
        
        print(f"📊 初始化CDF图表生成器")
        print(f"   分析百分位数: {self.config.analysis.percentiles}")

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("MetricsAnalysis")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def load_metrics_data(self, metrics_file_path: str) -> bool:
        """
        加载性能指标数据
        
        Args:
            metrics_file_path: 性能指标文件路径
            
        Returns:
            是否加载成功
        """
        try:
            with open(metrics_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.metrics_data = []
                for line in lines:
                    line = line.strip()
                    if line:
                        self.metrics_data.append(json.loads(line))
            
            print(f"📥 成功加载性能数据: {len(self.metrics_data)} 条记录")
            return True
            
        except FileNotFoundError:
            self.logger.error(f"性能指标文件不存在: {metrics_file_path}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"性能指标文件格式错误: {e}")
            return False
        except Exception as e:
            self.logger.error(f"加载性能指标失败: {e}")
            return False

    def _calculate_cdf(self, data: List[float]) -> tuple:
        """
        计算累积分布函数
        
        Args:
            data: 数据列表
            
        Returns:
            (x_values, y_values) CDF的x和y坐标
        """
        if not data:
            return [], []
        
        sorted_data = np.sort(data)
        n = len(sorted_data)
        y_values = np.arange(1, n + 1) / n
        
        return sorted_data, y_values

    def _get_percentile_value(self, data: List[float], percentile: int) -> float:
        """获取指定百分位数的值"""
        if not data:
            return 0
        return np.percentile(data, percentile)

    def _plot_model_cdf(self, model_name: str, data: Dict[str, List[float]], output_dir: str):
        """
        为单个模型绘制CDF图
        
        Args:
            model_name: 模型名称
            data: 包含e2e_latencies, ttfts, tpots的字典
            output_dir: 输出目录
        """
        # 设置matplotlib使用非交互式后端
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{model_name} - Performance Metrics CDF', fontsize=16)
        
        percentiles = self.config.analysis.percentiles
        key_percentiles = [50, 90, 99]  # 主要显示的百分位数
        
        # E2E延迟CDF
        if data['e2e_latencies']:
            x_e2e, y_e2e = self._calculate_cdf(data['e2e_latencies'])
            axes[0].plot(x_e2e, y_e2e, 'b-', linewidth=2, label='E2E Latency')
            axes[0].set_xlabel('Latency (ms)')
            axes[0].set_ylabel('Cumulative Probability')
            axes[0].set_title('E2E Latency CDF')
            axes[0].grid(True, alpha=0.3)
            
            # 添加百分位数标注和平均值
            text_lines = []
            mean_value = np.mean(data['e2e_latencies'])
            text_lines.append(f'Mean: {mean_value:.1f}ms')
            
            for p in key_percentiles:
                if p in percentiles:
                    value = self._get_percentile_value(data['e2e_latencies'], p)
                    # 在图上标记百分位数点
                    axes[0].axvline(value, color='red', linestyle='--', alpha=0.7)
                    axes[0].axhline(p/100, color='red', linestyle='--', alpha=0.7)
                    text_lines.append(f'P{p}: {value:.1f}ms')
            
            # 在图下方添加文本
            axes[0].text(0.02, 0.98, '\n'.join(text_lines), transform=axes[0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            axes[0].text(0.5, 0.5, 'No E2E Data', transform=axes[0].transAxes, 
                        ha='center', va='center', fontsize=12)
            axes[0].set_title('E2E Latency CDF')
        
        # TTFT CDF
        if data['ttfts']:
            x_ttft, y_ttft = self._calculate_cdf(data['ttfts'])
            axes[1].plot(x_ttft, y_ttft, 'g-', linewidth=2, label='TTFT')
            axes[1].set_xlabel('Time (ms)')
            axes[1].set_ylabel('Cumulative Probability')
            axes[1].set_title('TTFT CDF')
            axes[1].grid(True, alpha=0.3)
            
            # 添加百分位数标注和平均值
            text_lines = []
            mean_value = np.mean(data['ttfts'])
            text_lines.append(f'Mean: {mean_value:.1f}ms')
            
            for p in key_percentiles:
                if p in percentiles:
                    value = self._get_percentile_value(data['ttfts'], p)
                    axes[1].axvline(value, color='red', linestyle='--', alpha=0.7)
                    axes[1].axhline(p/100, color='red', linestyle='--', alpha=0.7)
                    text_lines.append(f'P{p}: {value:.1f}ms')
            
            axes[1].text(0.02, 0.98, '\n'.join(text_lines), transform=axes[1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            axes[1].text(0.5, 0.5, 'No TTFT Data', transform=axes[1].transAxes, 
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('TTFT CDF')
        
        # TPOT CDF
        if data['tpots']:
            x_tpot, y_tpot = self._calculate_cdf(data['tpots'])
            axes[2].plot(x_tpot, y_tpot, 'orange', linewidth=2, label='TPOT')
            axes[2].set_xlabel('Time (ms)')
            axes[2].set_ylabel('Cumulative Probability')
            axes[2].set_title('TPOT CDF')
            axes[2].grid(True, alpha=0.3)
            
            # 添加百分位数标注和平均值
            text_lines = []
            mean_value = np.mean(data['tpots'])
            text_lines.append(f'Mean: {mean_value:.1f}ms')
            
            for p in key_percentiles:
                if p in percentiles:
                    value = self._get_percentile_value(data['tpots'], p)
                    axes[2].axvline(value, color='red', linestyle='--', alpha=0.7)
                    axes[2].axhline(p/100, color='red', linestyle='--', alpha=0.7)
                    text_lines.append(f'P{p}: {value:.1f}ms')
            
            axes[2].text(0.02, 0.98, '\n'.join(text_lines), transform=axes[2].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            axes[2].text(0.5, 0.5, 'No TPOT Data', transform=axes[2].transAxes, 
                        ha='center', va='center', fontsize=12)
            axes[2].set_title('TPOT CDF')
        
        plt.tight_layout()
        
        # 保存图表
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        output_file = os.path.join(output_dir, f'{safe_model_name}_cdf.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Generated {model_name} CDF: {output_file}")

    def generate_cdf_plots(self, output_dir: str):
        """
        生成CDF图表
        
        Args:
            output_dir: 输出目录
        """
        if not self.metrics_data:
            print("⚠️ 没有性能数据，跳过图表生成")
            return
        
        try:
            # 设置matplotlib使用非交互式后端
            import matplotlib
            matplotlib.use('Agg')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 过滤成功的请求
            successful_metrics = [m for m in self.metrics_data if m.get('success', False)]
            
            if not successful_metrics:
                print("⚠️ 没有成功的请求，跳过图表生成")
                return
            
            # 按模型分组数据
            model_data = defaultdict(lambda: {
                'e2e_latencies': [],
                'ttfts': [],
                'tpots': []
            })
            
            for metric in successful_metrics:
                model_name = metric.get('model_name', 'unknown')
                
                # E2E延迟 (转换为毫秒)
                e2e_latency = metric.get('e2e_latency', 0) * 1000
                model_data[model_name]['e2e_latencies'].append(e2e_latency)
                
                # TTFT (转换为毫秒)
                ttft = metric.get('ttft')
                if ttft is not None:
                    model_data[model_name]['ttfts'].append(ttft * 1000)
                
                # TPOT (转换为毫秒)
                tpot = metric.get('tpot')
                if tpot is not None:
                    model_data[model_name]['tpots'].append(tpot * 1000)
            
            # 为每个模型生成CDF图
            for model_name, data in model_data.items():
                self._plot_model_cdf(model_name, data, output_dir)
            
            print(f"📈 CDF图表已生成到: {output_dir}")
            
        except ImportError:
            print("⚠️ 无法导入matplotlib，跳过图表生成")
            print("   请安装: pip install matplotlib")
        except Exception as e:
            self.logger.error(f"生成图表时出错: {e}")

    def _extract_latency_arrays_ms(self, metrics_records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """从metrics记录提取 E2E/TTFT/TPOT 三组延迟（毫秒）。"""
        arrays = {
            'e2e_latencies': [],
            'ttfts': [],
            'tpots': [],
        }
        for metric in metrics_records:
            if not metric.get('success', False):
                continue

            e2e_latency = metric.get('e2e_latency', 0)
            if e2e_latency is not None:
                arrays['e2e_latencies'].append(float(e2e_latency) * 1000)

            ttft = metric.get('ttft')
            if ttft is not None:
                arrays['ttfts'].append(float(ttft) * 1000)

            tpot = metric.get('tpot')
            if tpot is not None:
                arrays['tpots'].append(float(tpot) * 1000)
        return arrays

    def _plot_overall_cdf(self, data: Dict[str, List[float]], output_dir: str, file_name: str = 'all_requests_cdf.png') -> Optional[str]:
        """绘制不分模型的整体CDF图（E2E/TTFT/TPOT 三子图）。"""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except Exception:
            pass

        if not (data.get('e2e_latencies') or data.get('ttfts') or data.get('tpots')):
            print("⚠️ 没有可用的整体性能数据，跳过整体CDF图")
            return None

        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('All Requests - Performance Metrics CDF', fontsize=16)

        percentiles = self.config.analysis.percentiles
        key_percentiles = [50, 90, 99]

        # E2E
        if data.get('e2e_latencies'):
            x, y = self._calculate_cdf(data['e2e_latencies'])
            axes[0].plot(x, y, 'b-', linewidth=2, label='E2E Latency')
            axes[0].set_xlabel('Latency (ms)')
            axes[0].set_ylabel('Cumulative Probability')
            axes[0].set_title('E2E Latency CDF')
            axes[0].grid(True, alpha=0.3)
            text_lines = [f"Mean: {np.mean(data['e2e_latencies']):.1f}ms"]
            for p in key_percentiles:
                if p in percentiles:
                    value = self._get_percentile_value(data['e2e_latencies'], p)
                    axes[0].axvline(value, color='red', linestyle='--', alpha=0.7)
                    axes[0].axhline(p / 100, color='red', linestyle='--', alpha=0.7)
                    text_lines.append(f'P{p}: {value:.1f}ms')
            axes[0].text(0.02, 0.98, '\n'.join(text_lines), transform=axes[0].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            axes[0].text(0.5, 0.5, 'No E2E Data', transform=axes[0].transAxes,
                         ha='center', va='center', fontsize=12)
            axes[0].set_title('E2E Latency CDF')

        # TTFT
        if data.get('ttfts'):
            x, y = self._calculate_cdf(data['ttfts'])
            axes[1].plot(x, y, 'g-', linewidth=2, label='TTFT')
            axes[1].set_xlabel('Time (ms)')
            axes[1].set_ylabel('Cumulative Probability')
            axes[1].set_title('TTFT CDF')
            axes[1].grid(True, alpha=0.3)
            text_lines = [f"Mean: {np.mean(data['ttfts']):.1f}ms"]
            for p in key_percentiles:
                if p in percentiles:
                    value = self._get_percentile_value(data['ttfts'], p)
                    axes[1].axvline(value, color='red', linestyle='--', alpha=0.7)
                    axes[1].axhline(p / 100, color='red', linestyle='--', alpha=0.7)
                    text_lines.append(f'P{p}: {value:.1f}ms')
            axes[1].text(0.02, 0.98, '\n'.join(text_lines), transform=axes[1].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            axes[1].text(0.5, 0.5, 'No TTFT Data', transform=axes[1].transAxes,
                         ha='center', va='center', fontsize=12)
            axes[1].set_title('TTFT CDF')

        # TPOT
        if data.get('tpots'):
            x, y = self._calculate_cdf(data['tpots'])
            axes[2].plot(x, y, 'orange', linewidth=2, label='TPOT')
            axes[2].set_xlabel('Time (ms)')
            axes[2].set_ylabel('Cumulative Probability')
            axes[2].set_title('TPOT CDF')
            axes[2].grid(True, alpha=0.3)
            text_lines = [f"Mean: {np.mean(data['tpots']):.1f}ms"]
            for p in key_percentiles:
                if p in percentiles:
                    value = self._get_percentile_value(data['tpots'], p)
                    axes[2].axvline(value, color='red', linestyle='--', alpha=0.7)
                    axes[2].axhline(p / 100, color='red', linestyle='--', alpha=0.7)
                    text_lines.append(f'P{p}: {value:.1f}ms')
            axes[2].text(0.02, 0.98, '\n'.join(text_lines), transform=axes[2].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            axes[2].text(0.5, 0.5, 'No TPOT Data', transform=axes[2].transAxes,
                         ha='center', va='center', fontsize=12)
            axes[2].set_title('TPOT CDF')

        plt.tight_layout()
        output_file = os.path.join(output_dir, file_name)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Generated overall CDF: {output_file}")
        return output_file

    def generate_overall_cdf_plot(self, output_dir: str, file_name: str = 'all_requests_cdf.png') -> Optional[str]:
        """对所有成功请求（不区分模型）生成一张CDF图。"""
        if not self.metrics_data:
            print("⚠️ 没有性能数据，跳过整体CDF图")
            return None
        data = self._extract_latency_arrays_ms(self.metrics_data)
        return self._plot_overall_cdf(data, output_dir, file_name=file_name)

    def merge_cdf_pngs_to_pdf(self, input_dir: str, output_pdf: str, include_patterns: Optional[List[str]] = None) -> Optional[str]:
        """把目录下CDF相关PNG合并成一个多页PDF，并按指定顺序排序。

        页面顺序：
        1) all_requests_cdf.png
        2) 各模型 *_cdf.png
        3) prompt_input_length_cdf.png
        4) output_length_cdf.png
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib.backends.backend_pdf import PdfPages
        except Exception as e:
            self.logger.error(f"无法生成PDF（matplotlib缺失或后端不可用）: {e}")
            return None

        if not os.path.isdir(input_dir):
            print(f"⚠️ 输入目录不存在，跳过PDF汇总: {input_dir}")
            return None

        include_patterns = include_patterns or ['_cdf.png', 'cdf.png', 'length_cdf.png']

        png_files = []
        for name in os.listdir(input_dir):
            lower = name.lower()
            if not lower.endswith('.png'):
                continue
            if any(pat in lower for pat in include_patterns):
                png_files.append(os.path.join(input_dir, name))

        def _sort_key(path: str) -> tuple:
            name = os.path.basename(path).lower()
            if name == 'all_requests_cdf.png':
                return (0, name)
            if name == 'prompt_input_length_cdf.png':
                return (2, name)
            if name == 'output_length_cdf.png':
                return (3, name)
            if name.endswith('_cdf.png'):
                return (1, name)
            return (4, name)

        png_files.sort(key=_sort_key)
        if not png_files:
            print(f"⚠️ 未找到CDF PNG文件，跳过PDF汇总: {input_dir}")
            return None

        os.makedirs(os.path.dirname(output_pdf) or '.', exist_ok=True)

        with PdfPages(output_pdf) as pdf:
            for png_path in png_files:
                try:
                    img = plt.imread(png_path)
                    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape-ish
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.axis('off')
                    ax.imshow(img)
                    title = os.path.basename(png_path)
                    fig.suptitle(title, fontsize=10, y=0.99)
                    pdf.savefig(fig, dpi=300)
                    plt.close(fig)
                except Exception as e:
                    self.logger.error(f"合并PNG到PDF失败: {png_path}, err={e}")
                    continue

            meta = pdf.infodict()
            meta['Title'] = 'CDF Summary'
            meta['Author'] = 'CustomTraceGenerator'
            meta['CreationDate'] = datetime.now()

        print(f"📄 CDF汇总PDF已生成: {output_pdf}")
        return output_pdf

    def _plot_length_cdf(self, lengths: list, title: str, output_file: str):
        """
        绘制长度的CDF图（英文字符，防止格式问题）
        Args:
            lengths: 长度列表
            title: 图标题
            output_file: 输出文件路径
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        if not lengths:
            print(f"⚠️ No data for {title}, skip plot.")
            return
        sorted_data = np.sort(lengths)
        n = len(sorted_data)
        y_values = np.arange(1, n + 1) / n

        plt.figure(figsize=(8, 6))
        plt.plot(sorted_data, y_values, 'b-', linewidth=2)
        plt.xlabel('Length (tokens or chars)')
        plt.ylabel('Cumulative Probability')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        # 标注均值和常用百分位
        mean_value = np.mean(lengths)
        p50 = np.percentile(lengths, 50)
        p90 = np.percentile(lengths, 90)
        p99 = np.percentile(lengths, 99)
        text = f"Mean: {mean_value:.1f}\nP50: {p50:.1f}\nP90: {p90:.1f}\nP99: {p99:.1f}"
        plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Generated {title}: {output_file}")

    def generate_length_cdf_plots(self, metrics_file_path: str, output_dir: str):
        """
        生成 prompt input 和 output 长度的CDF图（英文字符，防止格式问题）
        Args:
            metrics_file_path: 性能指标文件路径
            output_dir: 输出目录
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            os.makedirs(output_dir, exist_ok=True)
            # 读取数据
            prompt_lengths = []
            output_lengths = []
            with open(metrics_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        # 字段名修正：input_tokens/output_tokens
                        if 'input_tokens' in record:
                            prompt_lengths.append(record['input_tokens'])
                        if 'output_tokens' in record:
                            output_lengths.append(record['output_tokens'])
                    except Exception as e:
                        continue
            # 绘制
            self._plot_length_cdf(prompt_lengths, 'Prompt Input Length CDF', os.path.join(output_dir, 'prompt_input_length_cdf.png'))
            self._plot_length_cdf(output_lengths, 'Output Length CDF', os.path.join(output_dir, 'output_length_cdf.png'))
            print(f"📈 Prompt/Input Length CDF 图表已生成到: {output_dir}")
        except Exception as e:
            self.logger.error(f"生成长度CDF图表时出错: {e}")

    def _generate_rps_plot(self, file_path: str):
        """生成RPS图像"""
        if not self.send_log_records:
            return
        
        try:
            # 设置matplotlib使用非交互式后端
            import matplotlib
            matplotlib.use('Agg')
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 按模型分组
            model_data = defaultdict(list)
            base_time = min(record['actual_time'] for record in self.send_log_records)
            
            for record in self.send_log_records:
                relative_time = record['actual_time'] - base_time
                model_data[record['model_name']].append(relative_time)
            
            # 计算每个模型的RPS
            plt.figure(figsize=(12, 8))
            
            for model_name, times in model_data.items():
                if not times:
                    continue
                    
                # 使用滑动窗口计算RPS
                window_size = 1.0  # 1秒窗口
                max_time = max(times)
                time_points = np.arange(0, max_time + window_size, 0.1)  # 每0.1秒一个点
                rps_values = []
                
                for t in time_points:
                    # 计算在时间窗口[t, t+window_size]内的请求数
                    count = sum(1 for req_time in times if t <= req_time < t + window_size)
                    rps = count / window_size
                    rps_values.append(rps)
                
                plt.plot(time_points, rps_values, label=f'{model_name}', linewidth=2, alpha=0.8)
            
                plt.title('Actual Send RPS Over Time', fontsize=14)
                plt.xlabel('Time (seconds)', fontsize=12)
                plt.ylabel('RPS (requests/sec)', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.error(f"生成RPS图表时出错: {e}")


def create_metrics_analysis(config_path: str = "config/default.yaml") -> MetricsAnalysis:
    """
    创建性能分析器的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        MetricsAnalysis实例
    """
    config_manager = ConfigManager(config_path)
    config_manager.load_config()
    return MetricsAnalysis(config_manager) 