#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算两个词在不同时期词向量模型中的余弦相似度，并绘制变化趋势图
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import numpy as np

# --- 动态调整 sys.path 以支持从项目根目录导入 ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT_DIR))
from src.utils import download_chinese_font

def setup_chinese_font():
    """下载并设置中文字体"""
    font_path = download_chinese_font()
    if font_path:
        try:
            font_prop = plt.matplotlib.font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"成功设置字体: {font_prop.get_name()}")
        except Exception as e:
            print(f"设置字体时出错: {e}, 尝试使用备用字体。")
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
    else:
        print("字体下载失败，尝试使用系统默认中文字体。")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

def analyze_similarity_trends(model_dir: Path, word1: str, word2: str) -> pd.DataFrame:
    """
    分析指定目录下所有模型的词语相似度
    """
    model_files = sorted(model_dir.glob('*.kv'))
    
    if not model_files:
        print(f"错误: 在目录 {model_dir} 中未找到任何 .kv 文件。")
        return pd.DataFrame()

    print(f"在 {model_dir} 中找到 {len(model_files)} 个模型文件，将按字母顺序处理。")

    results = []
    for model_path in model_files:
        print(f"--- 正在处理模型: {model_path.name} ---")
        try:
            kv = KeyedVectors.load(str(model_path))
            
            # 检查两个词是否都存在于词汇表中
            if word1 in kv and word2 in kv:
                similarity = kv.similarity(word1, word2)
                print(f"  '{word1}' vs '{word2}' 相似度: {similarity:.4f}")
            else:
                missing = [word for word in [word1, word2] if word not in kv]
                print(f"  警告: 词 '{', '.join(missing)}' 在模型中不存在，跳过计算。")
                similarity = np.nan

            period_name = model_path.stem.replace('_wordvectors', '')
            results.append({'period': period_name, 'similarity': similarity})

        except Exception as e:
            print(f"  处理模型 {model_path.name} 时出错: {e}")
            results.append({'period': model_path.stem, 'similarity': np.nan})
            
    return pd.DataFrame(results)

def plot_similarity_trends(df: pd.DataFrame, word1: str, word2: str, output_dir: Path, model_dir_name: str):
    """
    绘制并保存相似度趋势图
    """
    if df.empty:
        print("没有数据可供绘图。")
        return

    plt.figure(figsize=(16, 10))
    
    plt.plot(df['period'], df['similarity'], marker='o', linestyle='-', markersize=10, linewidth=2.5)
    
    # 为每个数据点添加标签
    for i, row in df.iterrows():
        if pd.notna(row['similarity']):
            plt.annotate(f"{row['similarity']:.3f}", 
                         (row['period'], row['similarity']),
                         textcoords="offset points",
                         xytext=(0,20),
                         ha='center',
                         fontsize=14,
                         fontweight='bold')

    plt.title(f'"{word1}"与"{word2}"的余弦相似度变化趋势', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('时期 / 模型', fontsize=16)
    plt.ylabel('余弦相似度', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    
    # 动态调整Y轴范围
    if df['similarity'].notna().any():
        min_val = df['similarity'].min()
        max_val = df['similarity'].max()
        y_margin = (max_val - min_val) * 0.15 # 上下留出15%的边距
        plt.ylim(bottom=min_val - y_margin, top=max_val + y_margin)

    plt.tight_layout()
    
    # 保存图片
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{word1}_{word2}_similarity_trend_{model_dir_name}.png"
    img_path = output_dir / filename
    plt.savefig(img_path, dpi=300)
    
    print(f"\n趋势图已保存到: {img_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='计算并绘制两个词在不同时期模型中的余弦相似度。')
    parser.add_argument(
        '--model-dir', 
        type=str, 
        required=True,
        help='包含 .kv 词向量模型文件的目录路径。'
    )
    parser.add_argument('--word1', type=str, default='法制', help='要比较的第一个词。')
    parser.add_argument('--word2', type=str, default='法治', help='要比较的第二个词。')
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='output/similarity_analysis',
        help='保存输出图表的目录。'
    )
    args = parser.parse_args()

    print("--- 开始计算相似度趋势 ---")
    setup_chinese_font()
    
    model_dir_path = Path(args.model_dir)
    output_dir_path = Path(args.output_dir)
    
    results_df = analyze_similarity_trends(model_dir_path, args.word1, args.word2)
    
    # 获取模型目录的名称用于文件名
    model_dir_name = model_dir_path.name
    plot_similarity_trends(results_df, args.word1, args.word2, output_dir_path, model_dir_name)
    
    print("\n--- 分析完成 ---")

if __name__ == "__main__":
    main() 