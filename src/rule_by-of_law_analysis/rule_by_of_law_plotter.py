#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule by Law vs Rule of Law - 绘图工具

功能：
1. 读取 similarity_analysis_results.json 文件
2. 根据两种setting (exactly/combined) 绘制图表
3. 柱状图: 显示每个时期法治/法制与label-1/label-2词语的相似度
4. 趋势图: 显示每个时期法治/法制相似度的gap指数变化

输出目录：output/rule_by-of_law_index/
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体，确保图表能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class RuleByOfLawPlotter:
    def __init__(self, results_path, plot_output_dir):
        self.results_path = Path(results_path)
        self.plot_output_dir = Path(plot_output_dir)
        self.plot_output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def load_data(self):
        """加载分析结果JSON文件"""
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"成功加载结果文件: {self.results_path}")
            return True
        except FileNotFoundError:
            print(f"错误: 找不到结果文件 at {self.results_path}")
            return False
        except json.JSONDecodeError:
            print(f"错误: 结果文件格式不正确 at {self.results_path}")
            return False

    def plot_similarity_bar_chart(self, setting, output_dir):
        """绘制相似度柱状图"""
        setting_data = self.data[setting]
        eras = sorted(setting_data['法治'].keys())
        
        # 准备数据
        plot_data = []
        for era in eras:
            for target_word in ['法治', '法制']:
                sim1 = setting_data[target_word][era].get('label_1', {}).get('avg_similarity', 0)
                sim2 = setting_data[target_word][era].get('label_2', {}).get('avg_similarity', 0)
                # 将约束性语义 (label 1) 设为负值
                plot_data.append({'era': era, 'word': f'{target_word}-label1', 'similarity': -sim1})
                # 将发展性语义 (label 2) 设为正值
                plot_data.append({'era': era, 'word': f'{target_word}-label2', 'similarity': sim2})
        
        df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bar_width = 0.35
        x = np.arange(len(eras))
        
        # 绘制法治的柱子
        fazhi_label1 = df[(df['word'] == '法治-label1')]['similarity']
        fazhi_label2 = df[(df['word'] == '法治-label2')]['similarity']
        ax.bar(x - bar_width/2, fazhi_label2, bar_width, label='法治 vs 发展性语义', color='#1f77b4')
        ax.bar(x - bar_width/2, fazhi_label1, bar_width, label='法治 vs 约束性语义', color='#aec7e8')
        
        # 绘制法制的柱子
        fazhi_system_label1 = df[(df['word'] == '法制-label1')]['similarity']
        fazhi_system_label2 = df[(df['word'] == '法制-label2')]['similarity']
        ax.bar(x + bar_width/2, fazhi_system_label2, bar_width, label='法制 vs 发展性语义', color='#ff7f0e')
        ax.bar(x + bar_width/2, fazhi_system_label1, bar_width, label='法制 vs 约束性语义', color='#ffbb78')

        # 添加图表标题和标签
        ax.set_title(f'“法治”与“法制”的语义相似度分析 ({setting.capitalize()} Setting)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('时期', fontsize=12)
        ax.set_ylabel('平均余弦相似度\n(发展性语义: 正向, 约束性语义: 负向)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(eras)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在y=0处画一条横线
        ax.axhline(0, color='black', linewidth=0.8)
        
        # 调整y轴刻度标签，显示为正值
        ticks = ax.get_yticks()
        ax.set_yticklabels([f'{abs(tick):.2f}' for tick in ticks])
        
        plt.tight_layout()
        
        # 保存图片
        img_file = output_dir / f'similarity_bar_chart_{setting}.png'
        plt.savefig(img_file, dpi=300, bbox_inches='tight')
        print(f"  柱状图已保存到: {img_file}")
        plt.close(fig)

    def plot_gap_trend_chart(self, setting, output_dir):
        """绘制gap指数趋势图"""
        setting_data = self.data[setting]
        eras = sorted(setting_data['法治'].keys())
        
        # 准备数据
        gap_data = {
            '法治': [setting_data['法治'][era].get('gap_index', 0) for era in eras],
            '法制': [setting_data['法制'][era].get('gap_index', 0) for era in eras]
        }
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {'法治': '#1f77b4', '法制': '#ff7f0e'}
        markers = {'法治': 'o', '法制': 's'}
        
        for word in ['法治', '法制']:
            ax.plot(eras, gap_data[word], 
                    label=word, 
                    color=colors[word],
                    marker=markers[word],
                    linewidth=2.5,
                    markersize=8)
        
        # 添加图表标题和标签
        ax.set_title(f'“法治”与“法制”的语义动态指数 (Semantic Dynamism) 趋势 ({setting.capitalize()} Setting)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('时期', fontsize=12)
        ax.set_ylabel('语义动态指数\n(倾向发展性语义为正, 倾向约束性语义为负)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 在y=0处画一条横线
        ax.axhline(0, color='grey', linewidth=1.0, linestyle='--')
        
        # 添加数据标签
        for word in ['法治', '法制']:
            for i, era in enumerate(eras):
                ax.annotate(f'{gap_data[word][i]:.3f}',
                           (era, gap_data[word][i]),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           fontsize=9)
                           
        plt.tight_layout()
        
        # 保存图片
        img_file = output_dir / f'semantic_dynamism_trend_chart_{setting}.png'
        plt.savefig(img_file, dpi=300, bbox_inches='tight')
        print(f"  趋势图已保存到: {img_file}")
        plt.close(fig)

    def plot_all(self):
        """为每种setting绘制所有图表"""
        if not self.data:
            print("没有数据可供绘图")
            return
            
        for setting in self.data.keys():
            print(f"\n正在为 '{setting}' setting 绘制图表...")
            setting_dir = self.plot_output_dir / setting
            setting_dir.mkdir(parents=True, exist_ok=True)
            
            self.plot_similarity_bar_chart(setting, setting_dir)
            self.plot_gap_trend_chart(setting, setting_dir)
            
def main():
    """主函数"""
    print("Rule by Law vs Rule of Law - 绘图工具")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent.parent
    results_path = base_dir / "output" / "rule_by-of_law_index" / "similarity_analysis_results.json"
    plot_output_dir = base_dir / "output" / "rule_by-of_law_index"
    
    plotter = RuleByOfLawPlotter(results_path, plot_output_dir)
    if plotter.load_data():
        plotter.plot_all()

if __name__ == "__main__":
    main() 