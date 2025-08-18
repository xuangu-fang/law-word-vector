#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新绘制“法制”与“法治”词频趋势图
解决中文显示问题
"""

import sys
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

def replot_trends_from_csv(csv_file: Path, output_dir: Path):
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

    # 绘制曲线
    plt.plot(years, fazi_data, color='#1f77b4', marker='o', linewidth=2.5, markersize=7, label='法制')
    plt.plot(years, fazhi_data, color='#ff7f0e', marker='s', linewidth=2.5, markersize=7, label='法治')

    # 设置图表属性
    plt.title('"法制"与"法治"词频年度变化趋势 (每百万词)', fontsize=30, fontweight='bold', pad=20)
    plt.xlabel('年份', fontsize=24)
    plt.ylabel('每百万词出现次数', fontsize=24)
    plt.legend(fontsize=34, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)

    # 优化坐标轴
    plt.xticks(years[::2], rotation=45, fontsize=12) # 每隔两年显示
    plt.yticks(fontsize=12)
    
    # 添加数据标签
    # for i, year in enumerate(years):
    #     if fazi_data.iloc[i] > 0:
    #         plt.annotate(f'{fazi_data.iloc[i]:.1f}', (year, fazi_data.iloc[i]),
    #                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    #     if fazhi_data.iloc[i] > 0:
    #         plt.annotate(f'{fazhi_data.iloc[i]:.1f}', (year, fazhi_data.iloc[i]),
    #                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()

    # 保存图片
    output_dir.mkdir(parents=True, exist_ok=True)
    img_file = output_dir / "法制法治词频趋势图_replot.png"
    plt.savefig(img_file, dpi=300, bbox_inches='tight')
    print(f"新的趋势图已保存到: {img_file}")

    # 显示图表
    plt.show()

def main():
    """主函数"""
    print("开始重新绘制词频趋势图...")
    
    # 设置字体
    setup_chinese_font()
    
    # 定义路径
    project_root = Path(__file__).parent.parent.parent
    csv_file = project_root / "output" / "法制法治词频统计.csv"
    output_dir = project_root / "output"
    
    # 重新绘图
    replot_trends_from_csv(csv_file, output_dir)
    
    print("\n绘图完成！")

if __name__ == "__main__":
    main() 