"""
语义场变迁分析 (Gainers and Losers)
计算并可视化在不同时期之间，与目标词（如"法治"）语义关系发生显著变化的词汇。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gensim.models import KeyedVectors
import matplotlib.font_manager as fm
import sys
import warnings
import re

warnings.filterwarnings('ignore')

# --- 环境与样式设置 ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'font.family': 'SimHei'})

project_root = Path.cwd().parent
sys.path.append(str(project_root))
from src.utils import download_chinese_font
import src.utils as utils

font_path = download_chinese_font()
if font_path:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"成功设置中文字体: {font_prop.get_name()}")
else:
    print("无法设置中文字体，将使用替代方案")
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path.cwd().parent
MODELS_DIR = PROJECT_ROOT / "models"


class SemanticChangeAnalyzer:
    """一个用于分析语义变迁（获益者与失势者）的类"""

    def __init__(self, models, exclude_words_path=None):
        self.models = models
        if not models:
            raise ValueError("模型字典不能为空。")
        self.periods = sorted(models.keys())
        self.exclude_words = self._load_words_from_simple_list(exclude_words_path) or set()
        print(f"语义变迁分析器已初始化，包含 {len(self.periods)} 个时期。")
        if self.exclude_words:
            print(f"已加载 {len(self.exclude_words)} 个排除词。")

    def _load_words_from_simple_list(self, file_path):
        if not file_path: return set()
        words = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    word = line.split()[0]
                    words.add(word)
            return words
        except FileNotFoundError: return set()

    def _load_words_from_cluster_file(self, file_path):
        vocab = set()
        current_label = None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    match = re.match(r'^([^:( ]+)', line)
                    if line.endswith(':') or (match and ':' in line):
                        current_label = match.group(1)
                    elif current_label:
                        vocab.update(line.split())
            return vocab
        except Exception: return set()

    def _get_vocabulary_sources(self, source_strategy='top_n', target_word="法治", **kwargs):
        """
        步骤1: 根据策略获取不同时期的词汇来源 (返回一个字典)
        - top_n: 从各时期模型动态计算
        - from_cluster_file: 从单个聚类/维度文件加载，应用于所有时期
        - from_period_specific_files: 从各时期对应的独立文件加载
        """
        period_vocabs = {p: set() for p in self.periods}

        if source_strategy == 'top_n':
            top_n = kwargs.get('top_n', 200)
            for period in self.periods:
                model = self.models[period]
                if target_word in model:
                    similar_words = {word for word, sim in model.most_similar(target_word, topn=top_n)}
                    period_vocabs[period].update(similar_words)
        
        elif source_strategy == 'from_cluster_file':
            file_path = kwargs.get('file_path')
            if file_path:
                words = self._load_words_from_cluster_file(file_path)
                for period in self.periods:
                    period_vocabs[period].update(words)

        elif source_strategy == 'from_period_specific_files':
            file_map = kwargs.get('file_map', {})
            for period in self.periods:
                p_file = file_map.get(period)
                if p_file:
                    period_vocabs[period].update(self._load_words_from_simple_list(p_file))
        
        return period_vocabs

    def _process_vocabulary_pool(self, period_vocabs, process_mode='union'):
        """
        步骤2: 处理词汇来源字典，生成最终的分析词汇池
        - union: 取并集
        - intersection: 取交集
        - no_process: 不做处理，直接合并所有词（不推荐，但保留选项）
        """
        # 先对每个时期的词汇进行排除词过滤
        for period, words in period_vocabs.items():
            period_vocabs[period] = words - self.exclude_words

        if process_mode == 'union':
            full_vocab = set()
            for words in period_vocabs.values():
                full_vocab.update(words)
            return full_vocab
        
        elif process_mode == 'intersection':
            if not period_vocabs: return set()
            # 以第一个时期的词汇为基础开始求交集
            all_sets = list(period_vocabs.values())
            intersect_vocab = all_sets[0].copy()
            for i in range(1, len(all_sets)):
                intersect_vocab.intersection_update(all_sets[i])
            return intersect_vocab

        elif process_mode == 'no_process':
            full_vocab = set()
            for words in period_vocabs.values():
                full_vocab.update(words)
            return full_vocab
        
        return set()


    def analyze_similarity_change(self, target_word="法治", source_strategy='top_n', process_mode='union', **kwargs):
        """主分析函数，整合了词汇获取和处理"""
        print(f"\n开始分析“{target_word}”的语义变迁...")
        print(f"来源策略: {source_strategy}, 处理模式: {process_mode}, 其他参数: {kwargs}")

        # 1. 获取词汇来源
        period_vocabs = self._get_vocabulary_sources(source_strategy=source_strategy, target_word=target_word, **kwargs)
        
        # 2. 处理成最终的词汇池
        vocab_pool = self._process_vocabulary_pool(period_vocabs, process_mode=process_mode)
        
        if not vocab_pool:
            print("错误：无法获取到用于分析的词汇池。")
            return pd.DataFrame()
        print(f"最终分析词汇池大小: {len(vocab_pool)}")

        # 3. 计算相似度矩阵
        similarity_data = {}
        for period in self.periods:
            model = self.models[period]
            sims = {}
            if target_word in model:
                for word in vocab_pool:
                    if word in model and word != target_word:
                        sims[word] = float(model.similarity(target_word, word))
                    else:
                        sims[word] = np.nan
            similarity_data[period] = pd.Series(sims)
        
        similarity_df = pd.DataFrame(similarity_data).fillna(0)
        
        # 4. 计算变化
        change_df = pd.DataFrame()
        if len(self.periods) > 1:
            for i in range(len(self.periods) - 1):
                period1, period2 = self.periods[i], self.periods[i+1]
                change_df[f'{period1}_to_{period2}'] = similarity_df[period2] - similarity_df[period1]
        
        print("分析完成。")
        return change_df

    def plot_gainers_losers(self, change_series, top_n=20, output_dir=".", filename=""):
        """可视化获益者与失势者"""
        if change_series.empty:
            print("数据为空，无法绘图。")
            return

        gainers = change_series.nlargest(top_n)
        losers = change_series.nsmallest(top_n).sort_values(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # 获益者
        sns.barplot(x=gainers.values, y=gainers.index, ax=axes[0], palette='Greens_r')
        axes[0].set_title(f'Top {top_n} Gainers (关系增强)', fontproperties=font_prop, size=16)
        axes[0].set_xlabel("相似度变化量", fontproperties=font_prop)
        axes[0].tick_params(axis='y', labelsize=12)

        # 失势者
        sns.barplot(x=losers.values, y=losers.index, ax=axes[1], palette='Reds_r')
        axes[1].set_title(f'Top {top_n} Losers (关系减弱)', fontproperties=font_prop, size=16)
        axes[1].set_xlabel("相似度变化量", fontproperties=font_prop)
        axes[1].tick_params(axis='y', labelsize=12)
        
        fig.suptitle(f'语义场变迁: {change_series.name}', fontproperties=font_prop, size=24)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图片
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"{filename}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存图表到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    print("=== 语义场变迁分析 (v2) ===")
    
    models = utils.load_models(MODELS_DIR)
    exclude_file = PROJECT_ROOT / "notebooks" / "exclude_words_network.txt"
    if not models:
        return

    output_root = PROJECT_ROOT / "output" / "change_analysis_v2"
    analyzer = SemanticChangeAnalyzer(models, exclude_words_path=exclude_file)

    # --- 场景1: 来源(top_n) + 处理(并集) ---
    print("\n\n--- 场景1: 来源(top_n) + 处理(并集) ---")
    change_df_1 = analyzer.analyze_similarity_change(
        target_word="法治", 
        source_strategy='top_n', 
        process_mode='union',
        top_n=200
    )
    if not change_df_1.empty:
        for transition in change_df_1.columns:
            analyzer.plot_gainers_losers(change_df_1[transition], top_n=25, output_dir=output_root / "S1_top_n_union", filename=f"change_{transition}")

    # --- 场景2: 来源(维度文件) + 处理(并集) ---
    print("\n\n--- 场景2: 来源(维度文件) + 处理(并集) ---")
    dim_file = PROJECT_ROOT / "notebooks" / "topic_word" / "dimension_words_4d.txt"
    change_df_2 = analyzer.analyze_similarity_change(
        target_word="法治",
        source_strategy='from_cluster_file',
        process_mode='union', # 在此场景下，并集/交集/无处理 结果相同
        file_path=dim_file
    )
    if not change_df_2.empty:
        for transition in change_df_2.columns:
            analyzer.plot_gainers_losers(change_df_2[transition], top_n=15, output_dir=output_root / "S2_dims_file", filename=f"change_{transition}_4d")
            
    # --- 场景3: 来源(时期专家词表) + 处理(交集) ---
    print("\n\n--- 场景3: 来源(时期专家词表) + 处理(交集) ---")
    base_path = PROJECT_ROOT / "notebooks" / "similar_words"
    file_map_sc3 = {
        'Era1_1978-1996': base_path / 'Era1_1978-1996_final.txt',
        'Era2_1997-2013': base_path / 'Era2_1997-2013_final.txt',
        'Era3_2014-2024': base_path / 'Era3_2014-2024_final.txt'
    }
    change_df_3 = analyzer.analyze_similarity_change(
        target_word="法治",
        source_strategy='from_period_specific_files',
        process_mode='intersection',
        file_map=file_map_sc3
    )
    if not change_df_3.empty:
        for transition in change_df_3.columns:
            analyzer.plot_gainers_losers(change_df_3[transition], top_n=15, output_dir=output_root / "S3_expert_intersect", filename=f"change_{transition}_expert")


if __name__ == "__main__":
    main()