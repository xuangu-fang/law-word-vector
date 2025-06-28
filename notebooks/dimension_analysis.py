"""
通过专家定义的维度词表，分析维度词表的语义相似度变化趋势
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gensim.models import KeyedVectors
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置更好的可视化风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# 假设notebooks目录在项目根目录下
project_root = Path.cwd().parent
sys.path.append(str(project_root))
from src.utils import download_chinese_font
import src.utils as utils


# 下载并安装字体
font_path = download_chinese_font()

# 设置matplotlib使用下载的字体
if font_path:
    plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
    print("成功设置中文字体")
else:
    print("无法设置中文字体，将使用替代方案")

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义项目根目录和模型目录
PROJECT_ROOT = Path.cwd().parent
MODELS_DIR = PROJECT_ROOT / "models"


class DimensionAnalyzer:
    """封装维度分析相关的功能"""
    
    def __init__(self, models):
        """
        初始化分析器
        
        Args:
            models (dict): 已加载的词向量模型字典
        """
        if not models:
            raise ValueError("没有成功加载任何模型，请提供有效的模型字典")
            
        self.models = models
        print(f"\nDimensionAnalyzer 初始化成功，共加载 {len(self.models)} 个模型。")
        self.output_dir = PROJECT_ROOT / "output" / "dimension_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _load_dimension_words(self, file_path):
        """
        加载维度词表文件
        
        Args:
            file_path (str or Path): 维度词表文件路径
            
        Returns:
            dict: 维度名称到词列表的映射
        """
        dimensions = {}
        current_dimension = None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    if line.endswith(':'):
                        current_dimension = line[:-1]
                        dimensions[current_dimension] = []
                    elif current_dimension:
                        words = line.split()
                        dimensions[current_dimension].extend(words)
                        
            return dimensions
        except Exception as e:
            print(f"加载维度词表 '{file_path}' 时出错: {e}")
            return {}

    def calculate_dimension_similarities(self, dimension_words, target_word="法治", normalize=False):
        """
        计算目标词与各维度的相似度
        
        Args:
            dimension_words (dict): 维度词表字典
            target_word (str): 目标词
            normalize (bool): 是否归一化 (确保每个period的各个维度的相似度总和为1)
            
        Returns:
            DataFrame: 各时期各维度的相似度矩阵
        """
        periods = sorted(self.models.keys())
        dimensions = list(dimension_words.keys())
        
        similarity_data = []
        
        for period in periods:
            model = self.models[period]
            if target_word not in model:
                print(f"警告: '{target_word}' 在 {period} 模型中不存在")
                continue
                
            period_similarities = {"时期": period}
            
            for dim in dimensions:
                dim_words = dimension_words[dim]
                similarities = []
                
                for word in dim_words:
                    if word in model and word != target_word:
                        try:
                            sim = model.similarity(target_word, word)
                            similarities.append(sim)
                        except KeyError:
                            pass
                
                if similarities:
                    period_similarities[dim] = np.mean(similarities)
                else:
                    period_similarities[dim] = 0
            
            if normalize:
                sum_sim = sum(period_similarities.get(dim, 0) for dim in dimensions)
                if sum_sim > 0:
                    for dim in dimensions:
                        period_similarities[dim] = period_similarities.get(dim, 0) / sum_sim
                else:
                    for dim in dimensions:
                        period_similarities[dim] = 0
                        
            similarity_data.append(period_similarities)
        
        return pd.DataFrame(similarity_data)

    def expand_dimension_words_by_similarity(self, dimension_words, target_word="法治", 
                                           similarity_threshold=0.3, max_words_per_dim=50,
                                           candidate_source='model', candidate_files=None):
        """
        基于词向量相似度扩展维度词表
        
        Args:
            dimension_words (dict): 初始维度词表
            target_word (str): 目标词
            similarity_threshold (float): 相似度阈值
            max_words_per_dim (int): 每个维度最大词数
            candidate_source (str): 候选词来源，'model' 或 'files'
            candidate_files (dict): 各时期对应的候选词文件路径字典
            
        Returns:
            dict: 扩展后的维度词表
        """
        expanded_words = {dim: set(words) for dim, words in dimension_words.items()}
        print(f"使用所有 {len(self.models)} 个模型的平均相似度进行词表扩展")
        
        candidate_words = set()
        
        if candidate_source == 'model':
            # 原有逻辑：从模型动态计算候选词
            for period, model in self.models.items():
                if target_word in model:
                    similar_words = model.most_similar(target_word, topn=500)
                    for word, similarity in similar_words:
                        if similarity >= similarity_threshold:
                            candidate_words.add(word)
        elif candidate_source == 'files':
            # 新增逻辑：从专家定义的文件中读取候选词
            if not candidate_files:
                print("警告: 未提供候选词文件路径，将使用模型动态计算")
                candidate_source = 'model'
                for period, model in self.models.items():
                    if target_word in model:
                        similar_words = model.most_similar(target_word, topn=500)
                        for word, similarity in similar_words:
                            if similarity >= similarity_threshold:
                                candidate_words.add(word)
            else: # NOTE: not finished
                for period, file_path in candidate_files.items():
                    if Path(file_path).exists():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if line and not line.startswith('#'):
                                        words = line.split()
                                        candidate_words.update(words)
                        except Exception as e:
                            print(f"读取文件 {file_path} 时出错: {e}")
                    else:
                        print(f"警告: 文件 {file_path} 不存在")
        
        print(f"候选词数量: {len(candidate_words)}")
        
        for word in candidate_words:
            target_similarities = []
            for model in self.models.values():
                if target_word in model and word in model:
                    try:
                        sim = model.similarity(target_word, word)
                        target_similarities.append(sim)
                    except KeyError:
                        pass
            
            if not target_similarities or np.mean(target_similarities) < similarity_threshold:
                continue
                
            dim_similarities = {}
            for dim, core_words in dimension_words.items():
                all_dim_similarities = []
                for core_word in core_words:
                    period_similarities = []
                    for model in self.models.values():
                        if core_word in model and word in model:
                            try:
                                sim = model.similarity(word, core_word)
                                period_similarities.append(sim)
                            except KeyError:
                                pass
                    if period_similarities:
                        all_dim_similarities.append(np.mean(period_similarities))
                
                if all_dim_similarities:
                    dim_similarities[dim] = np.mean(all_dim_similarities)
            
            if dim_similarities:
                best_dim = max(dim_similarities, key=dim_similarities.get)
                if (dim_similarities[best_dim] > similarity_threshold and 
                    len(expanded_words[best_dim]) < max_words_per_dim):
                    expanded_words[best_dim].add(word)
        
        result = {dim: sorted(list(words)) for dim, words in expanded_words.items()}
        
        print("\n扩展后的词表统计:")
        for dim, words in result.items():
            original_count = len(dimension_words.get(dim, []))
            expanded_count = len(words)
            print(f"{dim}: {expanded_count} 个词 (原有 {original_count} + 新增 {expanded_count - original_count})")
        
        return result

    def save_expanded_dimension_words(self, expanded_words, output_path):
        """保存扩展后的维度词表"""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 扩展后的法治维度词表\n\n")
            for dim, words in expanded_words.items():
                f.write(f"# {dim} ({len(words)}个词)\n")
                f.write(f"{dim}:\n")
                for i in range(0, len(words), 10):
                    line_words = words[i:i+10]
                    f.write(" ".join(line_words) + "\n")
                f.write("\n")
        print(f"已保存扩展词表到: {output_path}")

    def plot_dimension_trends(self, similarity_df, title="法治维度语义相似度变化趋势",filename=None):
        """绘制维度趋势图"""
        plt.figure(figsize=(12, 6))
        periods = similarity_df["时期"]
        dimensions = [col for col in similarity_df.columns if col != "时期"]
        for dim in dimensions:
            plt.plot(periods, similarity_df[dim], marker='o', linewidth=2, label=dim)
        plt.title(title)
        plt.xlabel("时期")
        plt.ylabel("平均相似度")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if filename:
            plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
            print(f"已保存维度趋势图到: {self.output_dir / filename}.png") 
        plt.show()

    def plot_dimension_radar(self, similarity_df, title="法治维度语义结构雷达图",filename=None):
        """绘制雷达图"""
        periods = similarity_df["时期"].tolist()
        dimensions = [col for col in similarity_df.columns if col != "时期"]
        N = len(dimensions)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.get_cmap('Set1', len(periods))
        
        for i, period in enumerate(periods):
            values = similarity_df.loc[similarity_df['时期'] == period, dimensions].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=period, color=colors(i))
            ax.fill(angles, values, alpha=0.1, color=colors(i))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title(title, size=15, pad=20)
        if filename:
            plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
            print(f"已保存维度雷达图到: {self.output_dir / filename}.png")
        plt.show()

    def plot_dimension_heatmap(self, similarity_df, title="法治维度语义相似度热力图",filename=None):
        """绘制热力图"""
        similarity_df = similarity_df.set_index('时期')
        plt.figure(figsize=(10, 6))
        sns.heatmap(similarity_df, annot=True, fmt='.3f', cmap="YlOrRd", linewidths=.5)
        plt.title(title)
        plt.ylabel("时期")
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
            print(f"已保存维度热力图到: {self.output_dir / filename}.png")
        plt.show()
    
    def run_analysis(self, dimension_words_path, target_word="法治", normalize=False):
        """
        加载维度词表、计算相似度并进行可视化
        """
        print(f"\n{'='*20}\n分析维度文件: {dimension_words_path}\n{'='*20}")
        dimension_words = self._load_dimension_words(dimension_words_path)
        
        if not dimension_words:
            print(f"无法加载或维度词表为空: {dimension_words_path}")
            return
            
        print("已加载维度词表:")
        for dim, words in dimension_words.items():
            print(f"  {dim}: {len(words)} 个词")
            
        similarity_df = self.calculate_dimension_similarities(
            dimension_words, target_word=target_word, normalize=normalize
        )
        
        if similarity_df.empty:
            print("计算相似度失败，无法继续分析。")
            return
            
        print(f"\n{len(dimension_words)}维度相似度矩阵:")
        print(similarity_df)
        
        dim_count = len(dimension_words)
        base_title = f"法治{dim_count}维度"
        
        self.plot_dimension_trends(similarity_df.copy(), f"{base_title}语义相似度变化趋势",filename=f"{base_title}_trends")
        self.plot_dimension_radar(similarity_df.copy(), f"{base_title}语义结构雷达图",filename=f"{base_title}_radar")
        self.plot_dimension_heatmap(similarity_df.copy(), f"{base_title}语义相似度热力图",filename=f"{base_title}_heatmap")


def main():
    """主函数"""
    print("=== 法治维度语义分析 ===")
    
    # 1. 加载模型
    try:
        models = utils.load_models(MODELS_DIR)
        analyzer = DimensionAnalyzer(models)
    except (ValueError, FileNotFoundError) as e:
        print(f"初始化分析器失败: {e}")
        return

    # 创建输出目录
    output_dir = PROJECT_ROOT / "output" / "dimension_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    topic_word_dir = PROJECT_ROOT / "topic_word"
    
    # 2. 分析专家定义的维度词表
    analyzer.run_analysis(topic_word_dir / "dimension_words_3d.txt")
    analyzer.run_analysis(topic_word_dir / "dimension_words_4d.txt")
    
    # 3. 扩展4D词表并保存
    print("\n=== 扩展4D词表 ===")
    dimension_words_4d = analyzer._load_dimension_words(topic_word_dir / "dimension_words_4d.txt")
    if dimension_words_4d:
        expanded_4d_words = analyzer.expand_dimension_words_by_similarity(
            dimension_words_4d, 
            target_word="法治",
            similarity_threshold=0.3,
            max_words_per_dim=50
        )
        
        # 保存扩展后的词表
        expanded_4d_output_path = output_dir / "expanded_dimension_words_4d.txt"
        analyzer.save_expanded_dimension_words(expanded_4d_words, expanded_4d_output_path)
        
        # (可选) 分析扩展后的词表
        print("\n=== 分析扩展后的4D词表 ===")
        analyzer.run_analysis(expanded_4d_output_path)

    print("\n=== 分析完成 ===")

    # 4. 分析基于聚类结果的维度词表
    print("\n=== 分析基于聚类结果的维度词表 ===")
    analyzer.run_analysis(topic_word_dir / "cluster_results_3d.txt")
    analyzer.run_analysis(topic_word_dir / "cluster_results_4d.txt")
    # analyzer.run_analysis(topic_word_dir / "cluster_results_5d.txt")

if __name__ == "__main__":
    main() 