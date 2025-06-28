"""
法治维度词表聚类扩展
基于专家定义的核心词汇，通过聚类算法扩展各维度词表
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
from notebooks.dimension_analysis import DimensionAnalyzer


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


class ClusteringAnalyzer:
    """封装聚类分析相关功能"""

    def __init__(self, models, output_dir):
        if not models:
            raise ValueError("没有成功加载任何模型，请提供有效的模型字典")
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dimension_analyzer = DimensionAnalyzer(models)
        print(f"ClusteringAnalyzer 初始化成功，输出目录: {self.output_dir}")

    def run_clustering(self, similar_words_by_period, n_clusters=3, 
                       exclude_words_path=None, top_n=150):
        """对相似词进行聚类分析"""
        intersection, union, _ = utils.get_word_sets(
            similar_words_by_period, top_n=top_n, exclude_words_path=exclude_words_path
        )
        print(f"\n--- 开始对 {len(union)} 个词进行 {n_clusters}-聚类 ---")
        
        latest_period = max(self.models.keys())
        model = self.models[latest_period]
        print(f"使用 {latest_period} 模型进行聚类")
        
        valid_words, word_vectors = [], []
        for word in union:
            if word in model:
                valid_words.append(word)
                word_vectors.append(model[word])
        
        if len(valid_words) < n_clusters:
            print(f"有效词汇数量({len(valid_words)})少于聚类数量({n_clusters})，无法聚类")
            return {}, np.array([]), [], []

        word_vectors = np.array(word_vectors)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(word_vectors)
        
        clusters = {}
        for i in range(n_clusters):
            cluster_words = [valid_words[j] for j, label in enumerate(cluster_labels) if label == i]
            clusters[f"聚类{i+1}"] = cluster_words
            print(f"聚类{i+1}: {len(cluster_words)} 个词, 前10个: {cluster_words[:10]}")
            
        return clusters, word_vectors, cluster_labels, valid_words

    def visualize_clusters(self, word_vectors, cluster_labels, valid_words, n_clusters, method='tsne',filename=None):
        """可视化聚类结果"""
        if len(word_vectors) == 0:
            return
        
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(word_vectors)-1)) if method == 'tsne' else PCA(n_components=2, random_state=42)
        reduced_vectors = reducer.fit_transform(word_vectors)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            plt.scatter(reduced_vectors[mask, 0], reduced_vectors[mask, 1], 
                       c=[colors[i]], label=f'聚类{i+1}', alpha=0.8, s=60)
        
        indices = np.random.choice(len(valid_words), min(50, len(valid_words)), replace=False)
        for idx in indices:
            plt.annotate(valid_words[idx], (reduced_vectors[idx, 0], reduced_vectors[idx, 1]), fontsize=8, alpha=0.7)
        
        plt.title(f'{n_clusters}-聚类结果可视化 ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        if filename:
            plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
            print(f"已保存聚类结果到: {self.output_dir / filename}.png")
        
        plt.show()

    def save_cluster_results(self, clusters, filename):
        """保存聚类结果到文件"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# 法治相关词汇 {len(clusters)}-聚类结果\n\n")
            for cluster_name, words in clusters.items():
                f.write(f"## {cluster_name} ({len(words)}个词)\n")
                # f.write(f"{cluster_name}:\n") # 兼容格式
                for i in range(0, len(words), 10):
                    f.write(" ".join(words[i:i+10]) + "\n")
                f.write("\n")
        print(f"已保存聚类结果到: {output_path}")

    def analyze_cluster_quality(self, word_vectors, cluster_labels):
        """分析聚类质量"""
        if len(set(cluster_labels)) > 1:
            silhouette = silhouette_score(word_vectors, cluster_labels)
            calinski = calinski_harabasz_score(word_vectors, cluster_labels)
            print(f"轮廓系数 (Silhouette Score): {silhouette:.3f}")
            print(f"Calinski-Harabasz 指数: {calinski:.3f}")
            return silhouette, calinski
        print("只有一个聚类，无法计算质量指标")
        return None, None

    def analyze(self, era_files, cluster_numbers, top_n, exclude_words_path):
        """执行完整的聚类和维度分析流程"""
        # 1. 加载各时期相似词
        similar_words_by_period = {}
        for era, file_path in era_files.items():
            word_list = utils.load_expert_word_list(file_path)
            if word_list:
                similar_words_by_period[era] = word_list
                print(f"加载 {era}: {len(word_list)} 个词")

        if not similar_words_by_period:
            print("未能加载任何相似词数据，聚类分析中止。")
            return

        # 2. 执行聚类、可视化、保存和质量分析
        for n in cluster_numbers:
            clusters, vectors, labels, words = self.run_clustering(
                similar_words_by_period, n_clusters=n, 
                exclude_words_path=exclude_words_path, top_n=top_n
            )
            if not clusters: continue

            self.visualize_clusters(vectors, labels, words, n_clusters=n,filename=f"cluster_results_{n}d")
            self.save_cluster_results(clusters, f"cluster_results_{n}d.txt")
            self.analyze_cluster_quality(vectors, labels)
            



def main():
    """主函数"""
    print("=== 法治维度词表聚类扩展与分析 ===")
    
    # 1. 初始化
    try:
        models = utils.load_models(MODELS_DIR)
        output_dir = PROJECT_ROOT / "output" / "clustering_analysis"
        analyzer = ClusteringAnalyzer(models, output_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"初始化分析器失败: {e}")
        return

    # 2. 定义分析参数
    era_files = {
        'Era1_1978-1996': 'similar_words/Era1_1978-1996_final.txt',
        'Era2_1997-2013': 'similar_words/Era2_1997-2013_final.txt',
        'Era3_2014-2024': 'similar_words/Era3_2014-2024_final.txt'
    }
    
    # 3. 运行分析
    analyzer.analyze(
        era_files=era_files,
        cluster_numbers=[3, 4],
        top_n=150,
        exclude_words_path="exclude_words.txt"
    )
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main() 