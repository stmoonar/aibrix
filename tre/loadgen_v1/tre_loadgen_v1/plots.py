"""v1 客户端阶段的两张图（从原 client_dispatcher 移出，画法不变；与发送无关）。

* ``actual_send_rps_by_model.png``：按实际发送时间（``start_time``）统计的每模型 / 总 RPS，
  代码与 v1 ``ClientDispatcher.generate_actual_send_rps_plot`` 相同。
* ``process_load_over_time.png``：v1 画的是每进程每秒上报的活跃协程数；统一客户端没有
  这个上报，改为由逐请求的 ``start_time`` / ``end_time`` 重建每进程在飞请求数（1 s 采样）。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10


def process_load_plot(records: List[Dict[str, Any]], output_file_path: str, logger) -> None:
    """每进程在飞请求数随时间变化（由逐请求起止时间重建）。"""
    spans = [(int(r.get("process_id") or 0), float(r["start_time"]), float(r["end_time"]))
             for r in records if r.get("start_time") and r.get("end_time")]
    if not spans:
        logger.warning("没有请求起止时间，跳过进程负载图")
        return
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        t0 = min(s for _, s, _ in spans)
        t1 = max(e for _, _, e in spans)
        grid = np.arange(t0, t1 + 1.0, 1.0)
        by_proc: Dict[int, np.ndarray] = {}
        for pid in sorted({p for p, _, _ in spans}):
            starts = np.array(sorted(s for p, s, _ in spans if p == pid))
            ends = np.array(sorted(e for p, _, e in spans if p == pid))
            by_proc[pid] = np.searchsorted(starts, grid, side="right") - np.searchsorted(ends, grid, side="right")
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        for pid, series in by_proc.items():
            plt.plot(grid - t0, series, linewidth=1.5, label=f'Process {pid}')
        plt.title('In-flight Requests by Process', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('In-flight requests')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        plt.plot(grid - t0, sum(by_proc.values()), 'black', linewidth=2, label='Total in-flight')
        plt.title('System Overall Load Changes', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Total in-flight requests')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"进程负载图已保存至: {output_file_path}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"生成进程负载图失败: {exc}")


def actual_send_rps_plot(metrics_file: str, logger) -> None:
    """根据实际发送时间生成RPS随时间变化图 - 按模型分组（v1 原样）"""
    try:
        if not os.path.exists(metrics_file):
            logger.warning(f"性能指标文件不存在: {metrics_file}")
            return
        model_data = defaultdict(list)
        total_send_times = []
        with open(metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if 'start_time' in data and data['start_time'] and 'model_name' in data:
                    model_data[data['model_name']].append(data['start_time'])
                    total_send_times.append(data['start_time'])
        if not total_send_times:
            logger.warning("没有找到有效的发送时间数据")
            return
        total_send_times.sort()
        start_time = min(total_send_times)
        total_duration = max(total_send_times) - start_time
        window_size = 1.0
        num_windows = int(total_duration / window_size) + 1
        time_points = [i * window_size + window_size / 2 for i in range(num_windows)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        model_stats = {}
        for i, (model_name, send_times) in enumerate(model_data.items()):
            rps_values = []
            for j in range(num_windows):
                ws = start_time + j * window_size
                rps_values.append(sum(1 for t in send_times if ws <= t < ws + window_size) / window_size)
            ax1.plot(time_points, rps_values, color=colors[i % len(colors)], linewidth=1, label=f'{model_name}',
                     marker='o', markersize=2, alpha=0.8)
            model_stats[model_name] = {'avg_rps': np.mean(rps_values), 'max_rps': np.max(rps_values),
                                       'total_requests': len(send_times)}
        ax1.set_title('Request Send RPS by Model', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=10)
        ax1.set_ylabel('RPS (requests/second)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        total_rps_values = []
        for j in range(num_windows):
            ws = start_time + j * window_size
            total_rps_values.append(sum(1 for t in total_send_times if ws <= t < ws + window_size) / window_size)
        ax2.plot(time_points, total_rps_values, 'black', linewidth=1, label='Total RPS', alpha=0.9, marker='o',
                 markersize=2)
        ax2.set_title('Total Request Send RPS', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=10)
        ax2.set_ylabel('RPS (requests/second)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        stats_text = "Model Statistics:\n"
        for model_name, stats in model_stats.items():
            stats_text += (f"{model_name}:\n  Avg: {stats['avg_rps']:.2f} RPS\n  Peak: {stats['max_rps']:.2f} RPS\n"
                           f"  Total: {stats['total_requests']} reqs\n\n")
        stats_text += (f"Overall:\n  Avg: {np.mean(total_rps_values):.2f} RPS\n"
                       f"  Peak: {np.max(total_rps_values):.2f} RPS\n  Total: {len(total_send_times)} reqs")
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        rps_plot_path = os.path.join(os.path.dirname(metrics_file), 'actual_send_rps_by_model.png')
        plt.savefig(rps_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"按模型分组的RPS图已保存至: {rps_plot_path}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"生成按模型分组的RPS图失败: {exc}")
