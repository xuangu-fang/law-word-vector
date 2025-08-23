#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule by Law vs Rule of Law - 指数分析器

功能：
1. 读取 expert_labeled_data/法制与法治比较词包.xlsx，提取6个sheet的数据
2. 将sheet内容转化为结构化的JSON文件
3. 读取各时期的词向量模型（支持多种模型配置）
4. 计算"法治"和"法制"与特定词语集合的平均余弦相似度
5. 计算gap指数（语义动态性指数）
6. 生成柱状图和趋势图
7. 将结果以JSON格式存储

输出目录：output/rule_by-of_law_index/
"""

import pandas as pd
import json
from pathlib import Path
from gensim.models import KeyedVectors
import numpy as np
from collections import defaultdict
import itertools

class RuleByOfLawIndexAnalyzer:
    def __init__(self, excel_path, models_config, output_dir):
        self.excel_path = Path(excel_path)
        self.models_config = models_config  # 新的模型配置字典
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个模型配置创建专门的输出目录
        self.config_output_dirs = {}
        for config_name in models_config.keys():
            config_output_dir = self.output_dir / config_name
            config_output_dir.mkdir(parents=True, exist_ok=True)
            self.config_output_dirs[config_name] = config_output_dir
        
        self.sheet_names = [
            "法治era1", "法治era2", "法治era3",
            "法制era1", "法制era2", "法制era3"
        ]
        self.target_words = ["法治", "法制"]
        self.word_sets = defaultdict(dict)
        self.models = {}  # 将存储所有模型配置的模型
        self.results = {}

    def process_excel(self):
        """读取Excel文件，提取词语并按要求构建词语集合"""
        temp_word_sets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        try:
            xls = pd.ExcelFile(self.excel_path)
            for sheet_name in self.sheet_names:
                if sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    
                    # 提取关键词、era和目标词
                    target_word = "法治" if "法治" in sheet_name else "法制"
                    era = "era" + sheet_name[-1]
                    
                    # 提取所需列
                    if "word" in df.columns and "class-label" in df.columns:
                        for _, row in df.iterrows():
                            word = row["word"]
                            label = row["class-label"]
                            if pd.notna(word) and pd.notna(label):
                                temp_word_sets[target_word][era][f"label_{int(label)}"].append(str(word))
                    else:
                        print(f"警告: 在sheet '{sheet_name}' 中找不到 'word' 或 'class-label' 列")
                else:
                    print(f"警告: 在Excel文件中找不到名为 '{sheet_name}' 的sheet")
        except FileNotFoundError:
            print(f"错误: 找不到Excel文件 at {self.excel_path}")
            return
        except Exception as e:
            print(f"读取Excel文件时发生错误: {e}")
            return
            
        self.word_sets = temp_word_sets
        
        # 保存为JSON文件
        output_file = self.output_dir / "word_sets.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.word_sets, f, ensure_ascii=False, indent=2)
        print(f"处理后的词语集合已保存到: {output_file}")

    def load_models(self):
        """加载所有配置的词向量模型"""
        for config_name, config in self.models_config.items():
            print(f"\n加载模型配置: {config_name}")
            self.models[config_name] = {}
            
            for era, filename in config['model_files'].items():
                model_path = config['base_dir'] / filename
                try:
                    self.models[config_name][era] = KeyedVectors.load(str(model_path))
                    print(f"  成功加载模型: {filename}")
                except FileNotFoundError:
                    print(f"  错误: 找不到模型文件 at {model_path}")
                except Exception as e:
                    print(f"  加载模型 {filename} 时发生错误: {e}")

    def _calculate_avg_similarity(self, model, word1, word_list):
        """计算一个词与一个词语列表的平均余弦相似度"""
        if word1 not in model.key_to_index:
            print(f"警告: '{word1}' 不在词向量模型中")
            return 0
            
        valid_words = [w for w in word_list if w in model.key_to_index]
        if not valid_words:
            return 0
            
        similarities = [model.similarity(word1, w) for w in valid_words]
        return np.mean(similarities) if similarities else 0

    def calculate_all_similarities(self, setting='exactly'):
        """
        根据两种不同的setting计算所有目标词在所有时期的相似度
        setting='exactly': 法治/法制 分别使用自己的词包
        setting='combined': 法治/法制 使用合并后的词包
        """
        if setting not in self.results:
            self.results[setting] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for config_name in self.models:
            print(f"\n计算 {config_name} 配置的相似度...")
            
            # 获取当前配置的所有时期
            config_eras = list(self.models[config_name].keys())
            
            for era in config_eras:
                model = self.models[config_name][era]
                
                # 准备词语集合
                era_word_sets = {}
                if setting == 'exactly':
                    era_word_sets = self.word_sets
                elif setting == 'combined':
                    combined_set = defaultdict(list)
                    
                    # 对于sliding_window模型，需要为每个时期分配适当的词语集合
                    if config_name == 'sliding_window':
                        # 根据时期范围分配词语集合
                        if era == 'era1':  # 1978-1988
                            source_eras = ['era1']
                        elif era == 'era2':  # 1983-1993
                            source_eras = ['era1', 'era2']
                        elif era == 'era3':  # 1988-1998
                            source_eras = ['era2', 'era3']
                        elif era == 'era4':  # 1993-2003
                            source_eras = ['era2', 'era3']
                        elif era == 'era5':  # 1998-2008
                            source_eras = ['era2', 'era3']
                        elif era == 'era6':  # 2003-2013
                            source_eras = ['era2', 'era3']
                        elif era == 'era7':  # 2008-2018
                            source_eras = ['era3']
                        elif era == 'era8':  # 2013-2024
                            source_eras = ['era3']
                        else:
                            source_eras = ['era1', 'era2', 'era3']
                    else:
                        # 对于flexible模型，使用原有的逻辑
                        source_eras = ['era1', 'era2', 'era3']
                    
                    # 从相关时期收集词语
                    for tw in self.target_words:
                        for source_era in source_eras:
                            if tw in self.word_sets and source_era in self.word_sets[tw]:
                                for label in self.word_sets[tw][source_era]:
                                    combined_set[label].extend(self.word_sets[tw][source_era][label])
                    
                    # 去重
                    for label in combined_set:
                        combined_set[label] = list(set(combined_set[label]))
                    
                    # 为法治和法制都分配这个合并后的集合
                    for tw in self.target_words:
                        era_word_sets[tw] = {era: combined_set}

                # 计算相似度
                for target_word in self.target_words:
                    if target_word in era_word_sets and era in era_word_sets[target_word]:
                        for label in ["label_1", "label_2"]:
                            word_list = era_word_sets[target_word][era].get(label, [])
                            
                            avg_sim = self._calculate_avg_similarity(model, target_word, word_list)
                            
                            self.results[setting][config_name][target_word][era][label] = {
                                'avg_similarity': float(avg_sim),
                                'word_count': len(word_list)
                            }
                            
                            print(f"    [{setting.upper()}] {era} - {target_word} vs {label}: "
                                  f"Avg Similarity = {avg_sim:.4f} (基于 {len(word_list)} 个词)")

    def calculate_gap_index(self):
        """计算每个时期、每个目标词的语义动态指数"""
        for setting in self.results:
            for config_name in self.results[setting]:
                for target_word in self.results[setting][config_name]:
                    for era in self.results[setting][config_name][target_word]:
                        sim_label1 = self.results[setting][config_name][target_word][era].get('label_1', {}).get('avg_similarity', 0)
                        sim_label2 = self.results[setting][config_name][target_word][era].get('label_2', {}).get('avg_similarity', 0)
                        
                        # 新公式: 发展性语义 (label 2) - 约束性语义 (label 1)
                        gap = sim_label2 - sim_label1
                        self.results[setting][config_name][target_word][era]['gap_index'] = gap
                        print(f"  [{setting.upper()}] {config_name} - {era} - {target_word}: 语义动态指数 = {gap:.4f}")

    def save_all_results(self):
        """将所有分析结果保存到JSON文件中，为每个配置创建单独的文件"""
        # 保存总体结果
        overall_output_file = self.output_dir / "similarity_analysis_results.json"
        with open(overall_output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"总体分析结果已保存到: {overall_output_file}")
        
        # 为每个配置保存单独的结果文件
        for config_name in self.models_config.keys():
            if config_name in self.results.get('exactly', {}):
                config_output_dir = self.config_output_dirs[config_name]
                
                # 保存exactly setting的结果
                exactly_file = config_output_dir / f"{config_name}_exactly_results.json"
                with open(exactly_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results['exactly'][config_name], f, ensure_ascii=False, indent=2)
                print(f"{config_name} exactly结果已保存到: {exactly_file}")
                
                # 保存combined setting的结果
                if 'combined' in self.results and config_name in self.results['combined']:
                    combined_file = config_output_dir / f"{config_name}_combined_results.json"
                    with open(combined_file, 'w', encoding='utf-8') as f:
                        json.dump(self.results['combined'][config_name], f, ensure_ascii=False, indent=2)
                    print(f"{config_name} combined结果已保存到: {combined_file}")

    def run_analysis(self):
        """执行完整的分析流程"""
        print("步骤 1: 读取并处理Excel文件...")
        self.process_excel()
        
        print("\n步骤 2: 加载词向量模型...")
        self.load_models()
        
        print("\n步骤 3: 计算相似度 (Exactly 对应)...")
        self.calculate_all_similarities(setting='exactly')
        
        print("\n步骤 4: 计算相似度 (合并相同时期)...")
        self.calculate_all_similarities(setting='combined')
        
        print("\n步骤 5: 计算Gap指数...")
        self.calculate_gap_index()
        
        print("\n步骤 6: 保存所有结果...")
        self.save_all_results()
        
        print("\n分析完成！")

def main():
    """主函数"""
    print("Rule by Law vs Rule of Law - 指数分析")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent.parent
    
    # 定义多种模型配置
    models_config = {
        "flexible": {
            "base_dir": base_dir / "models" / "fine_tuned_vectors_flexible",
            "model_files": {
                "era1": "Era1_1978-1996_wordvectors.kv",
                "era2": "Era2_1997-2013_wordvectors.kv",
                "era3": "Era3_2014-2024_wordvectors.kv"
            }
        },
        "sliding_window": {
            "base_dir": base_dir / "models" / "fine_tuned_vectors_sliding_window" / "Year1978-2024_10_5",
            "model_files": {
                "era1": "Era1_1978-1988_wordvectors.kv",
                "era2": "Era2_1983-1993_wordvectors.kv",
                "era3": "Era3_1988-1998_wordvectors.kv",
                "era4": "Era4_1993-2003_wordvectors.kv",
                "era5": "Era5_1998-2008_wordvectors.kv",
                "era6": "Era6_2003-2013_wordvectors.kv",
                "era7": "Era7_2008-2018_wordvectors.kv",
                "era8": "Era8_2013-2024_wordvectors.kv"
            }
        }
    }
    
    excel_path = base_dir / "expert_labeled_data" / "法制与法治比较词包.xlsx"
    output_dir = base_dir / "output" / "rule_by-of_law_index"
    
    analyzer = RuleByOfLawIndexAnalyzer(excel_path, models_config, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 