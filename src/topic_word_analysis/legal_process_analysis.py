
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Analysis - 法律流程分析器

功能：
1. 分析"立法、司法、执法、守法"四个法律流程维度  
2. 计算"法治"/"法制"与各维度的相似度
3. 支持多种era-keyword组合和归一化模式
4. 生成雷达图、趋势图、热力图
5. 使用General Union模式确保词包一致性

输出目录：output/topic_analysis/legal_process/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gensim.models import KeyedVectors
import json
from itertools import chain
import warnings
warnings.filterwarnings('ignore')

# Setup plotting style and fonts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置matplotlib不显示图形，只保存
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "fine_tuned_vectors_flexible"
OUTPUT_DIR = PROJECT_ROOT / "output" / "topic_analysis" / "legal_process"
DATA_PATH = PROJECT_ROOT / "output" / "topic_analysis" / "legal_process" / "topic_word_sets_legal_process.json"

class LegalProcessAnalyzer:
    def __init__(self, models):
        if not models:
            raise ValueError("No models provided.")
        self.models = models
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            self.topic_word_sets = json.load(f)
        
        # 创建general union wordset (跨关键词+跨时期)
        self.general_union_wordset = self._create_general_union_wordset()

    def _get_word_set(self, keyword, era, use_union=False, use_general_union=False):
        """Helper to retrieve word sets for a given keyword and era."""
        if use_general_union:
            # 跨关键词+跨时期的完全并集
            return self.general_union_wordset
        elif use_union:
            # 仅跨时期的并集（针对特定关键词）
            all_words = {}
            for era_data in self.topic_word_sets.get(keyword, {}).values():
                for topic, words in era_data.items():
                    if topic not in all_words:
                        all_words[topic] = set()
                    all_words[topic].update(words)
            return {topic: list(words) for topic, words in all_words.items()}
        
        return self.topic_word_sets.get(keyword, {}).get(era, {})

    def _create_general_union_wordset(self):
        """创建跨关键词+跨时期的完全并集词包"""
        try:
            # 直接读取现有的JSON文件
            general_union_path = self.output_dir / "general_union_wordset_legal_process.json"
            with open(general_union_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            print(f"General Union Wordset 统计 (从文件读取):")
            for topic, words in result.items():
                print(f"  {topic}: {len(words)} 个词")
            
            return result
        except FileNotFoundError:
            print("警告: 未找到general_union_wordset_legal_process.json文件，将创建新的")
        general_union = {}
        
        # 遍历所有关键词（法治、法制等）
        for keyword, keyword_data in self.topic_word_sets.items():
            # 遍历所有时期
            for era, era_data in keyword_data.items():
                # 遍历所有topic
                for topic, words in era_data.items():
                    if topic not in general_union:
                        general_union[topic] = set()
                    general_union[topic].update(words)
        
        # 转换为list并排序
        result = {topic: sorted(list(word_set)) for topic, word_set in general_union.items()}
        
        print(f"General Union Wordset 统计:")
        for topic, words in result.items():
            print(f"  {topic}: {len(words)} 个词")
        
        # 保存到JSON文件
        general_union_path = self.output_dir / "general_union_wordset_legal_process.json"
        with open(general_union_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"General Union Wordset 已保存到: {general_union_path}")
        
        return result

    def calculate_similarities(self, era_keyword_map, use_union=False, use_general_union=False, normalize=None):
        """
        计算特定时期-特定维度的相似度
        
        对于每个era-topic组合：
        - 使用该era的词向量模型
        - 使用指定的关键词（法治/法制等）
        - 与该era该topic的词包计算相似度
        
        Args:
            era_keyword_map (dict): Maps eras to keywords (e.g., {'era1': '法制', 'era2': '法治'}).
                                  Supports mixed mode: {'era2': ['法制', '法治']}.
            use_union (bool): If True, use the union of word sets across all eras.
            use_general_union (bool): If True, use complete union across keywords and eras.
            normalize (str): Normalization method ('cross_era', 'same_era', None).

        Returns:
            pd.DataFrame: A DataFrame with similarity scores.
        """
        similarity_data = []
        eras = sorted(era_keyword_map.keys())
        
        # 获取所有可能的topics（应该是：发展、秩序、规范、权力限制）
        all_topics = set()
        for keyword_data in self.topic_word_sets.values():
            for era_data in keyword_data.values():
                all_topics.update(era_data.keys())
        all_topics = sorted(list(all_topics))
        
        print(f"发现的topics: {all_topics}")

        for era in eras:
            if era not in self.models:
                print(f"警告: 没有找到era {era}的模型")
                continue
                
            model = self.models[era]
            keywords = era_keyword_map[era]
            if isinstance(keywords, str):
                keywords = [keywords]

            era_similarities = {"era": era}
            
            for topic in all_topics:
                topic_similarities = []
                
                for keyword in keywords:
                    # 获取该era该keyword的topic词包
                    word_set = self._get_word_set(keyword, era, use_union=use_union, use_general_union=use_general_union)
                    
                    topic_words = word_set.get(topic, [])
                    
                    if not topic_words:
                        print(f"警告: {era}-{keyword}-{topic} 没有找到词包")
                        continue
                    
                    # 计算该关键词与该topic词包的相似度
                    if keyword not in model:
                        print(f"警告: 关键词 '{keyword}' 不在 {era} 模型中")
                        continue
                    
                    valid_sims = []
                    for word in topic_words:
                        if word in model and word != keyword:
                            try:
                                sim = model.similarity(keyword, word)
                                valid_sims.append(sim)
                            except KeyError:
                                pass
                    
                    if valid_sims:
                        avg_sim = np.mean(valid_sims)
                        topic_similarities.append(avg_sim)
                        print(f"{era}-{keyword}-{topic}: {len(valid_sims)}个有效词, 平均相似度={avg_sim:.3f}")

                # 如果是混合模式（多个关键词），取平均
                if topic_similarities:
                    era_similarities[topic] = np.mean(topic_similarities)
                else:
                    era_similarities[topic] = 0.0
            
            similarity_data.append(era_similarities)

        df = pd.DataFrame(similarity_data)
        
        if normalize and not df.empty:
            if normalize == 'same_era':
                # 同一era内的各topic相似度归一化（和为1）
                df.iloc[:, 1:] = df.iloc[:, 1:].div(df.iloc[:, 1:].sum(axis=1), axis=0).fillna(0)
            elif normalize == 'cross_era':
                # 跨era标准化
                for col in df.columns[1:]:
                    col_data = df[col]
                    if col_data.max() > col_data.min():
                        df[col] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                    else:
                        df[col] = 0
        
        return df

    def _create_output_path(self, settings):
        """Creates a descriptive output path based on analysis settings."""
        path_parts = []
        for key, value in settings.items():
            if isinstance(value, bool) and value:
                path_parts.append(key)
            elif isinstance(value, str) and value is not None:
                path_parts.append(f"{key}_{value}")
            elif isinstance(value, list):
                str_value = '_'.join(map(str, value))
                path_parts.append(f"{key}_{str_value}")
        
        setting_str = "-".join(filter(None, path_parts))
        path = self.output_dir / setting_str
        path.mkdir(parents=True, exist_ok=True)
        return path

    def plot_radar(self, df, path, title):
        """Generates and saves a radar plot."""
        labels = df.columns[1:]
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, row in df.iterrows():
            data = row.drop('era').tolist()
            data += data[:1]
            ax.plot(angles, data, label=row['era'])
            ax.fill(angles, data, alpha=0.25)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title(title)
        plt.savefig(path / "radar_chart.png", dpi=300)
        plt.close()

    def plot_trend(self, df, path, title):
        """Generates and saves a trend plot."""
        plt.figure(figsize=(12, 7))
        for column in df.columns[1:]:
            sns.lineplot(data=df, x='era', y=column, marker='o', label=column)
        plt.title(title)
        plt.ylabel("Similarity")
        plt.xlabel("Era")
        plt.legend(title="Topic")
        plt.tight_layout()
        plt.savefig(path / "trend_chart.png", dpi=300)
        plt.close()

    def plot_heatmap(self, df, path, title):
        """Generates and saves a heatmap with era on x-axis."""
        # 转置数据，使era在x轴，topics在y轴
        df_transposed = df.set_index('era').T
        plt.figure(figsize=(10, 8))
        # 如果需要自定义x轴和y轴的ticks名称，可以通过设置xticklabels和yticklabels参数
        # 例如，假设你想自定义x轴为["时期一", "时期二", "时期三"]，y轴为["维度A", "维度B", "维度C", ...]
        custom_xticklabels = ["1978-1996", "1997-2013", "2014-2024"]  # 根据实际era数量自定义
        custom_yticklabels = [ "司法", "守法","执法","立法"]  # 根据实际topic数量自定义

        ax = sns.heatmap(
            df_transposed,
            annot=True,
            fmt=".3f",
            cmap="Greys",  # 使用黑白灰色调，适合黑白打印
            xticklabels=custom_xticklabels,
            yticklabels=custom_yticklabels,
            annot_kws={"fontsize": 20}  # 设置热力图数字的字体大小
        )
        # 设置x轴和y轴label的字体大小
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=25)
        # plt.title(title)
        plt.xlabel("时期", fontsize=24)
        plt.ylabel("类别", fontsize=24)
        plt.tight_layout()
        plt.savefig(path / "heatmap.png", dpi=300)
        plt.close()

    def run_analysis(self, era_keyword_map, use_union=False, use_general_union=False, normalize=None):
        """
        运行完整的分析流程：计算相似度并生成所有图表
        """
        
        print(f"\n{'='*50}")
        print(f"开始分析: {era_keyword_map}")
        print(f"Union模式: {use_union}, General Union模式: {use_general_union}, 归一化: {normalize}")
        print(f"{'='*50}")
        
        keyword_strs = []
        for era, keywords in sorted(era_keyword_map.items()):
            if isinstance(keywords, list):
                keyword_strs.append(f"{era}-[{'+'.join(keywords)}]")
            else:
                keyword_strs.append(f"{era}-{keywords}")
        
        settings = {
            "keywords": keyword_strs,
            "union": use_union,
            "general_union": use_general_union,
            "normalize": normalize
        }
        
        output_path = self._create_output_path(settings)
        print(f"输出路径: {output_path}")
        
        df = self.calculate_similarities(era_keyword_map, use_union, use_general_union, normalize)
        
        if df.empty:
            print(f"无相似度数据: {settings}")
            return
        
        print(f"\n相似度矩阵:")
        print(df)

        title_suffix = f" (Union: {use_union}, General Union: {use_general_union}, Normalize: {normalize})"
        
        try:
            self.plot_radar(df, output_path, "Topic Similarity Radar Chart" + title_suffix)
            print(f"雷达图已保存")
        except Exception as e:
            print(f"雷达图生成失败: {e}")
            
        try:
            self.plot_trend(df, output_path, "Topic Similarity Trend Chart" + title_suffix)
            print(f"趋势图已保存")
        except Exception as e:
            print(f"趋势图生成失败: {e}")
            
        try:
            self.plot_heatmap(df, output_path, "Topic Similarity Heatmap" + title_suffix)
            print(f"热力图已保存")
        except Exception as e:
            print(f"热力图生成失败: {e}")
            
        print(f"分析完成. 图表保存到: {output_path}")


def load_models():
    """Loads word vector models for each era."""
    models = {}
    model_files = {
        'era1': 'Era1_1978-1996_wordvectors.kv',
        'era2': 'Era2_1997-2013_wordvectors.kv',
        'era3': 'Era3_2014-2024_wordvectors.kv'
    }
    for era, filename in model_files.items():
        try:
            models[era] = KeyedVectors.load(str(MODELS_DIR / filename), mmap='r')
        except FileNotFoundError:
            print(f"Warning: Model for {era} not found at {MODELS_DIR / filename}")
    return models

if __name__ == '__main__':
    print("脚本开始执行...")
    
    try:
        print("加载模型中...")
        models = load_models()
        
        if not models:
            print("错误: 没有加载到任何模型。退出。")
            exit()
        
        print(f"成功加载了 {len(models)} 个模型: {list(models.keys())}")
            
        print("初始化分析器...")
        analyzer = LegalProcessAnalyzer(models)
        
        print("数据加载成功，开始分析...")

        print("\n" + "="*80)
        print("⚖️ 法律流程分析: 立法、司法、执法、守法")
        print("📊 默认使用 General Union + Same Era 归一化")
        print("="*80)
        
        # 测试不同的era-keyword组合
        

        
        # 2. 混合模式: era1-法制, era2-[法制+法治], era3-法治
        mixed_keywords = {
            'era1': '法制',
            'era2': ['法制', '法治'],
            'era3': '法治'
        }
        print("\n--- 混合模式: era1-法制, era2-[法制+法治], era3-法治 ---")
        analyzer.run_analysis(mixed_keywords, use_general_union=True, normalize='same_era')
        # analyzer.run_analysis(mixed_keywords, use_general_union=True, normalize="none")
        

        
        print("\n🎉 法律流程分析完成！")

    except Exception as e:
        import traceback
        print(f"发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
