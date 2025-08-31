#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule by Law vs Rule of Law - 频率趋势重绘工具

功能：
1. 重新绘制"法制"与"法治"词频趋势图
2. 支持移动平均线计算
3. 解决中文显示问题
4. 自定义窗口长度和间隔

输出目录：output/rule_by-of_law_freq/
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 将上级目录添加到sys.path以导入utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import download_chinese_font

def setup_chinese_font():
    """下载并设置中文字体"""
    font_path = download_chinese_font()
    if font_path:
        try:
            # 查找字体名称
            font_prop = plt.matplotlib.font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"成功设置字体: {font_prop.get_name()}")
        except Exception as e:
            print(f"设置字体时出错: {e}")
            # 如果失败，尝试使用备用字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
            print("尝试使用备用字体: SimHei, Microsoft YaHei")
    else:
        print("字体下载失败，尝试使用系统默认中文字体")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

def calculate_moving_average(series: pd.Series, window_length: int, window_gap: int):
    """
    计算滑动平均值
    窗口根据gap进行滑动，而不是逐年滚动
    """
    years = series.index
    start_year = years.min()
    end_year = years.max()
    
    avg_years = []
    avg_values = []
    
    current_year = start_year
    while current_year + window_length -1 <= end_year:
        window_end_year = current_year + window_length - 1
        window_data = series.loc[current_year:window_end_year]
        
        avg_values.append(window_data.mean())
        # 将均值点放在窗口的中心位置
        avg_years.append(current_year + (window_length - 1) / 2)
        
        current_year += window_gap
        
    return pd.Series(avg_values, index=avg_years)


def replot_trends_from_csv(csv_file: Path, output_dir: Path, plot_moving_average: bool, window_length: int, window_gap: int):
    """
    从CSV文件读取数据并重新绘制趋势图

    Args:
        csv_file: 包含词频数据的CSV文件路径
        output_dir: 图片保存目录
    """
    if not csv_file.exists():
        print(f"错误: CSV文件未找到于 {csv_file}")
        return

    # 读取数据
    try:
        df = pd.read_csv(csv_file)
        print(f"成功从 {csv_file} 读取数据")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # 数据准备
    years = sorted(df['年份'].unique())
    fazi = df[df['词语'] == '法制'].set_index('年份')['每百万词出现次数']
    fazhi = df[df['词语'] == '法治'].set_index('年份')['每百万词出现次数']
    
    # 确保数据对齐
    fazi_data = fazi.reindex(years, fill_value=0)
    fazhi_data = fazhi.reindex(years, fill_value=0)

    # 创建图表
    plt.figure(figsize=(16, 9))

    # 绘制原始曲线
    plt.plot(years, fazi_data, color='#1f77b4', marker='o', linestyle='-', linewidth=1.0, markersize=5, label='法制 (年度)', alpha=0.6)
    plt.plot(years, fazhi_data, color='#ff7f0e', marker='s', linestyle='-', linewidth=1.0, markersize=5, label='法治 (年度)', alpha=0.6)

    # 如果启用了滑动平均，则计算并绘制
    if plot_moving_average:
        fazi_ma = calculate_moving_average(fazi_data, window_length, window_gap)
        fazhi_ma = calculate_moving_average(fazhi_data, window_length, window_gap)
        
        plt.plot(fazi_ma.index, fazi_ma, color='#1f77b4', linestyle='--', linewidth=3, label=f'法制 ({window_length}年滑动平均)')
        plt.plot(fazhi_ma.index, fazhi_ma, color='#ff7f0e', linestyle='--', linewidth=3, label=f'法治 ({window_length}年滑动平均)')
        
        # 为滑动平均曲线添加数据点
        # for year, val in fazi_ma.items():
        #     plt.annotate(f'{val:.1f}', (year, val),
        #                textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold', color='#1f77b4')
        # for year, val in fazhi_ma.items():
        #     plt.annotate(f'{val:.1f}', (year, val),
        #                textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold', color='#ff7f0e')


    # 设置图表属性
    plt.title('"法制"与"法治"词频年度变化趋势 (每百万词)', fontsize=30, fontweight='bold', pad=20)
    plt.xlabel('年份', fontsize=24)
    plt.ylabel('每百万词出现次数', fontsize=24)
    plt.legend(fontsize=30, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)

    # 优化坐标轴
    plt.xticks(years[::2], rotation=45, fontsize=20) # 每隔两年显示
    plt.yticks(fontsize=20)
    
    # 原始数据的标签只在非滑动平均模式下显示，避免图表过于拥挤
    if not plot_moving_average:
        for i, year in enumerate(years):
            if fazi_data.iloc[i] > 0:
                plt.annotate(f'{fazi_data.iloc[i]:.1f}', (year, fazi_data.iloc[i]),
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            if fazhi_data.iloc[i] > 0:
                plt.annotate(f'{fazhi_data.iloc[i]:.1f}', (year, fazhi_data.iloc[i]),
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()

    # 保存图片
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if plot_moving_average:
        filename = f"法制法治词频趋势图_MA_len{window_length}_gap{window_gap}.png"
    else:
        filename = "法制法治词频趋势图_replot.png"
        
    img_file = output_dir / filename
    plt.savefig(img_file, dpi=300, bbox_inches='tight')
    print(f"新的趋势图已保存到: {img_file}")

    # 显示图表
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='重新绘制“法制”与“法治”词频趋势图，并可选择添加滑动平均线。')
    parser.add_argument('--moving-average', action='store_true', help='启用滑动平均计算和绘图')
    parser.add_argument('--window-length', type=int, default=10, help='滑动平均的窗口长度 (年)')
    parser.add_argument('--window-gap', type=int, default=5, help='滑动平均的窗口滑动步长 (年)')
    args = parser.parse_args()

    if args.moving_average:
        print(f"开始重新绘制词频趋势图 (滑动平均模式: 窗口长度={args.window_length}, 步长={args.window_gap})...")
    else:
        print("开始重新绘制词频趋势图 (年度数据模式)...")
    
    # 设置字体
    setup_chinese_font()
    
    # 定义路径
    project_root = Path(__file__).parent.parent.parent
    csv_file = project_root / "output" / "rule_by-of_law_freq" / "法制法治词频统计.csv"
    output_dir = project_root / "output" / "rule_by-of_law_freq"
    
    # 重新绘图
    replot_trends_from_csv(csv_file, output_dir, args.moving_average, args.window_length, args.window_gap)
    
    print("\n绘图完成！")

if __name__ == "__main__":
    main() 