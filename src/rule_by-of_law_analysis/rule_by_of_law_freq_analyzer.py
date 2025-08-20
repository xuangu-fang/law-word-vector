#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule by Law vs Rule of Law - 频率分析器

功能：
1. 统计"法制"和"法治"两个词的年度词频
2. 计算每百万词中的出现次数
3. 绘制词频趋势图（包括移动平均线）
4. 输出统计报告和数据文件

输出目录：output/rule_by-of_law_freq/
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class RuleByOfLawFreqAnalyzer:
    def __init__(self, corpus_dir: str = None):
        """
        初始化语料统计分析器
        
        Args:
            corpus_dir: 年度语料目录路径，如果为None则使用默认路径
        """
        if corpus_dir is None:
            # 获取脚本所在目录的上级目录的上级目录
            script_dir = Path(__file__).parent
            self.corpus_dir = script_dir.parent.parent / "processed_data" / "yearly_corpus"
        else:
            self.corpus_dir = Path(corpus_dir)
        self.target_words = ["法制", "法治"]
        self.results = defaultdict(dict)
        
    def count_words_in_file(self, file_path: Path) -> Tuple[int, Dict[str, int]]:
        """
        统计单个文件中的词频
        
        Args:
            file_path: 文件路径
            
        Returns:
            (总词数, {词: 出现次数})
        """
        word_counts = defaultdict(int)
        total_words = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 使用正则表达式匹配目标词（避免部分匹配）
            for word in self.target_words:
                # 使用词边界确保完整匹配
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = re.findall(pattern, content)
                word_counts[word] = len(matches)
                
            # 计算总词数（简单按空格和标点符号分割）
            words = re.findall(r'\b\w+\b', content)
            total_words = len(words)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return 0, defaultdict(int)
            
        return total_words, word_counts
    
    def analyze_yearly_corpus(self) -> Dict:
        """
        分析年度语料，统计目标词的词频
        
        Returns:
            包含年度统计结果的字典
        """
        print("开始分析年度语料...")
        
        # 获取所有年度语料文件
        corpus_files = sorted([f for f in self.corpus_dir.glob("*.txt")])
        
        if not corpus_files:
            print(f"在目录 {self.corpus_dir} 中未找到语料文件")
            return {}
        
        for file_path in corpus_files:
            # 从文件名提取年份
            year = file_path.stem
            print(f"正在处理 {year} 年的语料...")
            
            total_words, word_counts = self.count_words_in_file(file_path)
            
            if total_words > 0:
                # 计算每百万词中的出现次数
                for word in self.target_words:
                    count = word_counts[word]
                    per_million = (count / total_words) * 1_000_000
                    self.results[year][word] = {
                        'count': count,
                        'total_words': total_words,
                        'per_million': per_million
                    }
                    
                print(f"  {year}: 总词数={total_words:,}, 法制={word_counts['法制']}, 法治={word_counts['法治']}")
            else:
                print(f"  {year}: 文件为空或读取失败")
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        """
        保存统计结果到文件
        
        Args:
            output_dir: 输出目录，如果为None则使用默认路径
        """
        if output_dir is None:
            script_dir = Path(__file__).parent
            output_dir = str(script_dir.parent.parent / "output" / "rule_by-of_law_freq")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_file = output_path / "法制法治词频统计.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式的汇总数据
        summary_data = []
        for year in sorted(self.results.keys()):
            for word in self.target_words:
                if word in self.results[year]:
                    data = self.results[year][word]
                    summary_data.append({
                        '年份': year,
                        '词语': word,
                        '出现次数': data['count'],
                        '总词数': data['total_words'],
                        '每百万词出现次数': round(data['per_million'], 2)
                    })
        
        df = pd.DataFrame(summary_data)
        csv_file = output_path / "法制法治词频统计.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"统计结果已保存到:")
        print(f"  JSON: {results_file}")
        print(f"  CSV: {csv_file}")
        
        return df
    
    def plot_trends(self, output_dir: str = None):
        """
        绘制词频趋势图
        
        Args:
            output_dir: 输出目录，如果为None则使用默认路径
        """
        if output_dir is None:
            script_dir = Path(__file__).parent
            output_dir = str(script_dir.parent.parent / "output" / "rule_by-of_law_freq")
        
        if not self.results:
            print("没有数据可以绘图")
            return
        
        # 准备绘图数据
        years = sorted(self.results.keys())
        data = {word: [] for word in self.target_words}
        
        for year in years:
            for word in self.target_words:
                if word in self.results[year]:
                    data[word].append(self.results[year][word]['per_million'])
                else:
                    data[word].append(0)
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 绘制两条曲线
        colors = ['#1f77b4', '#ff7f0e']
        markers = ['o', 's']
        
        for i, word in enumerate(self.target_words):
            plt.plot(years, data[word], 
                    color=colors[i], 
                    marker=markers[i], 
                    linewidth=2, 
                    markersize=6,
                    label=word)
        
        # 设置图表属性
        plt.title('"法制"与"法治"词频年度变化趋势\n(每百万词中的出现次数)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('每百万词出现次数', fontsize=12)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 设置x轴标签
        plt.xticks(years[::2], rotation=45)  # 每隔一年显示一个标签
        
        # 添加数据标签
        for word in self.target_words:
            for i, year in enumerate(years):
                if i < len(data[word]) and data[word][i] > 0:
                    plt.annotate(f'{data[word][i]:.1f}', 
                               (year, data[word][i]),
                               textcoords="offset points",
                               xytext=(0,10),
                               ha='center',
                               fontsize=8)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        img_file = output_path / "法制法治词频趋势图.png"
        plt.savefig(img_file, dpi=300, bbox_inches='tight')
        print(f"趋势图已保存到: {img_file}")
        
        # 显示图表
        plt.show()
        
        return img_file
    
    def generate_summary_report(self) -> str:
        """
        生成统计摘要报告
        
        Returns:
            报告文本
        """
        if not self.results:
            return "没有数据可以生成报告"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("法制与法治词频统计报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 总体统计
        total_stats = {word: {'total_count': 0, 'total_words': 0} for word in self.target_words}
        
        for year in self.results:
            for word in self.target_words:
                if word in self.results[year]:
                    total_stats[word]['total_count'] += self.results[year][word]['count']
                    total_stats[word]['total_words'] += self.results[year][word]['total_words']
        
        report_lines.append("总体统计:")
        for word in self.target_words:
            stats = total_stats[word]
            if stats['total_words'] > 0:
                overall_per_million = (stats['total_count'] / stats['total_words']) * 1_000_000
                report_lines.append(f"  {word}: 总计出现 {stats['total_count']:,} 次")
                report_lines.append(f"       总体频率: {overall_per_million:.2f} 次/百万词")
        report_lines.append("")
        
        # 年度统计
        report_lines.append("年度详细统计:")
        report_lines.append("-" * 60)
        report_lines.append(f"{'年份':<6} {'法制(次/百万词)':<15} {'法治(次/百万词)':<15}")
        report_lines.append("-" * 60)
        
        for year in sorted(self.results.keys()):
            line = f"{year:<6}"
            for word in self.target_words:
                if word in self.results[year]:
                    line += f" {self.results[year][word]['per_million']:<15.2f}"
                else:
                    line += f" {'0.00':<15}"
            report_lines.append(line)
        
        report_lines.append("")
        
        # 趋势分析
        report_lines.append("趋势分析:")
        years = sorted(self.results.keys())
        if len(years) >= 2:
            for word in self.target_words:
                first_year = years[0]
                last_year = years[-1]
                
                if (word in self.results[first_year] and 
                    word in self.results[last_year]):
                    
                    first_freq = self.results[first_year][word]['per_million']
                    last_freq = self.results[last_year][word]['per_million']
                    change = last_freq - first_freq
                    change_pct = (change / first_freq * 100) if first_freq > 0 else 0
                    
                    report_lines.append(f"  {word}: {first_year}年 {first_freq:.2f} → {last_year}年 {last_freq:.2f}")
                    report_lines.append(f"        变化: {change:+.2f} ({change_pct:+.1f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    print("Rule by Law vs Rule of Law - 频率分析")
    print("=" * 60)
    
    # 创建分析器
    analyzer = RuleByOfLawFreqAnalyzer()
    
    # 分析语料
    results = analyzer.analyze_yearly_corpus()
    
    if not results:
        print("分析完成，但没有找到有效数据")
        return
    
    # 保存结果
    df = analyzer.save_results()
    
    # 绘制趋势图
    analyzer.plot_trends()
    
    # 生成并显示报告
    report = analyzer.generate_summary_report()
    print("\n" + report)
    
    # 保存报告到文件
    output_path = Path(__file__).parent.parent.parent / "output" / "rule_by-of_law_freq"
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / "法制法治词频统计报告.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n详细报告已保存到: {report_file}")
    print("\n分析完成！")


if __name__ == "__main__":
    main()
