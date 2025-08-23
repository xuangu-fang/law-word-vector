#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule by Law vs Rule of Law - 多模型配置绘图工具

功能：
1. 读取 similarity_analysis_results.json 文件
2. 为每个模型配置创建独立的图表
3. 根据两种setting (exactly/combined) 绘制图表
4. 柱状图: 显示每个时期法治/法制与label-1/label-2词语的相似度
5. 趋势图: 显示每个时期法治/法制相似度的gap指数变化
6. 支持不同时期数量的模型配置（如sliding_window的8个时期）

输出目录：output/rule_by-of_law_index/plots/
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
        
        # 定义时期到时间段的映射
        self.era_time_mapping = {
            'flexible': {
                'era1': '1978-1996',
                'era2': '1997-2013',
                'era3': '2014-2024'
            },
            'sliding_window': {
                'era1': '1978-1988',
                'era2': '1983-1993',
                'era3': '1988-1998',
                'era4': '1993-2003',
                'era5': '1998-2008',
                'era6': '2003-2013',
                'era7': '2008-2018',
                'era8': '2013-2024'
            }
        }

    def get_time_labels(self, config_name, eras):
        """获取时间段标签"""
        if config_name in self.era_time_mapping:
            return [self.era_time_mapping[config_name].get(era, era) for era in eras]
        else:
            return eras

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

    def plot_similarity_bar_chart(self, setting, config_name, output_dir):
        """绘制相似度柱状图"""
        if config_name not in self.data[setting]:
            print(f"  跳过 {config_name} 配置，因为数据不存在")
            return
            
        setting_data = self.data[setting][config_name]
        eras = sorted(setting_data['法治'].keys())
        time_labels = self.get_time_labels(config_name, eras)
        
        # 准备数据
        plot_data = []
        for i, era in enumerate(eras):
            for target_word in ['法治', '法制']:
                sim1 = setting_data[target_word][era].get('label_1', {}).get('avg_similarity', 0)
                sim2 = setting_data[target_word][era].get('label_2', {}).get('avg_similarity', 0)
                # 将约束性语义 (label 1) 设为负值
                plot_data.append({'era': time_labels[i], 'word': f'{target_word}-label1', 'similarity': -sim1})
                # 将发展性语义 (label 2) 设为正值
                plot_data.append({'era': time_labels[i], 'word': f'{target_word}-label2', 'similarity': sim2})
        
        df = pd.DataFrame(plot_data)
        
        # 根据时期数量调整图表大小
        fig_width = max(14, len(eras) * 2.0)  # 增加宽度以容纳时间段标签
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        bar_width = 0.35
        x = np.arange(len(time_labels))
        
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
        title = f'"{config_name}"模型配置的语义相似度分析 ({setting.capitalize()} Setting)'
        if config_name == 'sliding_window':
            title += '\n(8个时期: 1978-1988, 1983-1993, 1988-1998, 1993-2003, 1998-2008, 2003-2013, 2008-2018, 2013-2024)'
        elif config_name == 'flexible':
            title += '\n(3个时期: 1978-1996, 1997-2013, 2014-2024)'
            
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('时期', fontsize=12)
        ax.set_ylabel('平均余弦相似度\n(发展性语义: 正向, 约束性语义: 负向)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(time_labels, rotation=45 if len(time_labels) > 3 else 0)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在y=0处画一条横线
        ax.axhline(0, color='black', linewidth=0.8)
        
        # 调整y轴刻度标签，显示为正值
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{abs(tick):.2f}' for tick in ticks])
        
        plt.tight_layout()
        
        # 保存图片
        img_file = output_dir / f'{config_name}_similarity_bar_chart_{setting}.png'
        plt.savefig(img_file, dpi=300, bbox_inches='tight')
        print(f"    {config_name} 柱状图已保存到: {img_file}")
        plt.close(fig)

    def plot_gap_trend_chart(self, setting, config_name, output_dir):
        """绘制gap指数趋势图"""
        if config_name not in self.data[setting]:
            print(f"  跳过 {config_name} 配置，因为数据不存在")
            return
            
        setting_data = self.data[setting][config_name]
        eras = sorted(setting_data['法治'].keys())
        time_labels = self.get_time_labels(config_name, eras)
        
        # 准备数据
        gap_data = {
            '法治': [setting_data['法治'][era].get('gap_index', 0) for era in eras],
            '法制': [setting_data['法制'][era].get('gap_index', 0) for era in eras]
        }
        
        # 根据时期数量调整图表大小
        fig_width = max(12, len(eras) * 1.5)  # 增加宽度以容纳时间段标签
        fig, ax = plt.subplots(figsize=(fig_width, 7))
        
        colors = {'法治': '#1f77b4', '法制': '#ff7f0e'}
        markers = {'法治': 'o', '法制': 's'}
        
        for word in ['法治', '法制']:
            ax.plot(time_labels, gap_data[word], 
                    label=word, 
                    color=colors[word],
                    marker=markers[word],
                    linewidth=2.5,
                    markersize=8)
        
        # 添加图表标题和标签
        title = f'"{config_name}"模型配置的语义动态指数趋势 ({setting.capitalize()} Setting)'
        if config_name == 'sliding_window':
            title += '\n(8个时期: 1978-1988, 1983-1993, 1988-1998, 1993-2003, 1998-2008, 2003-2013, 2008-2018, 2013-2024)'
        elif config_name == 'flexible':
            title += '\n(3个时期: 1978-1996, 1997-2013, 2014-2024)'
            
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('时期', fontsize=12)
        ax.set_ylabel('语义动态指数\n(倾向发展性语义为正, 倾向约束性语义为负)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 在y=0处画一条横线
        ax.axhline(0, color='grey', linewidth=1.0, linestyle='--')
        
        # 添加数据标签
        for word in ['法治', '法制']:
            for i, time_label in enumerate(time_labels):
                ax.annotate(f'{gap_data[word][i]:.3f}',
                           (time_label, gap_data[word][i]),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           fontsize=9)
        
        # 如果时期数量多，调整x轴标签角度
        if len(time_labels) > 3:
            ax.set_xticks(range(len(time_labels)))
            ax.set_xticklabels(time_labels, rotation=45)
                           
        plt.tight_layout()
        
        # 保存图片
        img_file = output_dir / f'{config_name}_semantic_dynamism_trend_chart_{setting}.png'
        plt.savefig(img_file, dpi=300, bbox_inches='tight')
        print(f"    {config_name} 趋势图已保存到: {img_file}")
        plt.close(fig)

    def plot_config_comparison(self, setting, output_dir):
        """绘制不同模型配置的对比图"""
        if not self.data[setting]:
            return
            
        configs = list(self.data[setting].keys())
        if len(configs) < 2:
            return
            
        # 准备数据
        comparison_data = {}
        for config_name in configs:
            if config_name in self.data[setting]:
                eras = sorted(self.data[setting][config_name]['法治'].keys())
                time_labels = self.get_time_labels(config_name, eras)
                comparison_data[config_name] = {
                    'eras': eras,
                    'time_labels': time_labels,
                    '法治': [self.data[setting][config_name]['法治'][era].get('gap_index', 0) for era in eras],
                    '法制': [self.data[setting][config_name]['法制'][era].get('gap_index', 0) for era in eras]
                }
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        colors = {'flexible': '#1f77b4', 'sliding_window': '#ff7f0e'}
        markers = {'flexible': 'o', 'sliding_window': 's'}
        
        # 法治对比图
        for config_name in configs:
            if config_name in comparison_data:
                ax1.plot(comparison_data[config_name]['time_labels'], 
                        comparison_data[config_name]['法治'],
                        label=f'{config_name} - 法治',
                        color=colors.get(config_name, '#2ca02c'),
                        marker=markers.get(config_name, '^'),
                        linewidth=2.5,
                        markersize=8)
        
        ax1.set_title(f'不同模型配置下"法治"的语义动态指数对比 ({setting.capitalize()} Setting)', 
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('语义动态指数', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.axhline(0, color='grey', linewidth=1.0, linestyle='--')
        
        # 法制对比图
        for config_name in configs:
            if config_name in comparison_data:
                ax2.plot(comparison_data[config_name]['time_labels'], 
                        comparison_data[config_name]['法制'],
                        label=f'{config_name} - 法制',
                        color=colors.get(config_name, '#2ca02c'),
                        marker=markers.get(config_name, '^'),
                        linewidth=2.5,
                        markersize=8)
        
        ax2.set_title(f'不同模型配置下"法制"的语义动态指数对比 ({setting.capitalize()} Setting)', 
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('时期', fontsize=12)
        ax2.set_ylabel('语义动态指数', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.axhline(0, color='grey', linewidth=1.0, linestyle='--')
        
        plt.tight_layout()
        
        # 保存对比图
        img_file = output_dir / f'config_comparison_{setting}.png'
        plt.savefig(img_file, dpi=300, bbox_inches='tight')
        print(f"  配置对比图已保存到: {img_file}")
        plt.close(fig)

    def plot_all(self):
        """为每种setting和每个模型配置绘制所有图表"""
        if not self.data:
            print("没有数据可供绘图")
            return
            
        # 创建plots目录
        plots_dir = self.plot_output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for setting in self.data.keys():
            print(f"\n正在为 '{setting}' setting 绘制图表...")
            setting_dir = plots_dir / setting
            setting_dir.mkdir(parents=True, exist_ok=True)
            
            # 为每个模型配置绘制图表
            for config_name in self.data[setting].keys():
                print(f"  正在绘制 {config_name} 配置的图表...")
                config_dir = setting_dir / config_name
                config_dir.mkdir(parents=True, exist_ok=True)
                
                self.plot_similarity_bar_chart(setting, config_name, config_dir)
                self.plot_gap_trend_chart(setting, config_name, config_dir)
            
            # 绘制配置对比图
            self.plot_config_comparison(setting, setting_dir)
            
def main():
    """主函数"""
    print("Rule by Law vs Rule of Law - 多模型配置绘图工具")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent.parent
    results_path = base_dir / "output" / "rule_by-of_law_index" / "similarity_analysis_results.json"
    plot_output_dir = base_dir / "output" / "rule_by-of_law_index"
    
    plotter = RuleByOfLawPlotter(results_path, plot_output_dir)
    if plotter.load_data():
        plotter.plot_all()

if __name__ == "__main__":
    main() 