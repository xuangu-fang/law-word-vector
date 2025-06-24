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

# 可能的模型目录
FINE_TUNED_MODELS_DIR = MODELS_DIR / "fine_tuned_vectors_flexible"
SLIDING_WINDOW_MODELS_DIR = MODELS_DIR / "fine_tuned_vectors_sliding_window"


def load_dimension_words(file_path):
    """
    加载维度词表文件
    
    Args:
        file_path: 维度词表文件路径
        
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
        print(f"加载维度词表时出错: {e}")
        return {}

def cluster_similar_words(models, similar_words_by_period, n_clusters=3, 
                         exclude_words_path=None, top_n=150):
    """
    对相似词进行聚类分析
    
    Args:
        models: 词向量模型字典
        similar_words_by_period: 各时期相似词字典
        n_clusters: 聚类数量
        exclude_words_path: 排除词文件路径
        top_n: 每个时期取前n个词
        
    Returns:
        tuple: (聚类结果字典, 词向量矩阵, 聚类标签, 有效词列表)
    """
    # 获取所有时期的词汇并集
    intersection, union, filtered_similar_words = utils.get_word_sets(
        similar_words_by_period, top_n=top_n, exclude_words_path=exclude_words_path
    )
    
    print(f"聚类分析使用词汇数量: {len(union)}")
    
    # 使用最新时期的模型进行聚类
    latest_period = max(models.keys())
    model = models[latest_period]
    print(f"使用 {latest_period} 模型进行聚类")
    
    # 提取词向量
    valid_words = []
    word_vectors = []
    
    for word in union:
        if word in model:
            valid_words.append(word)
            word_vectors.append(model[word])
    
    if len(valid_words) < n_clusters:
        print(f"有效词汇数量({len(valid_words)})少于聚类数量({n_clusters})")
        return {}, np.array([]), [], []
    
    word_vectors = np.array(word_vectors)
    print(f"有效词汇数量: {len(valid_words)}")
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(word_vectors)
    
    # 整理聚类结果
    clusters = {}
    for i in range(n_clusters):
        cluster_words = [valid_words[j] for j in range(len(valid_words)) if cluster_labels[j] == i]
        clusters[f"聚类{i+1}"] = cluster_words
        print(f"聚类{i+1}: {len(cluster_words)} 个词")
        print(f"  前10个词: {cluster_words[:10]}")
    
    return clusters, word_vectors, cluster_labels, valid_words

def visualize_clusters(word_vectors, cluster_labels, valid_words, method='tsne'):
    """
    可视化聚类结果
    
    Args:
        word_vectors: 词向量矩阵
        cluster_labels: 聚类标签
        valid_words: 词汇列表
        method: 降维方法 ('tsne' 或 'pca')
    """
    if len(word_vectors) == 0:
        print("没有有效的词向量数据，无法可视化")
        return
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(word_vectors)-1))
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_vectors = reducer.fit_transform(word_vectors)
    
    # 绘制散点图
    plt.figure(figsize=(12, 8))
    
    # 为每个聚类使用不同颜色
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        plt.scatter(reduced_vectors[mask, 0], reduced_vectors[mask, 1], 
                   c=[colors[i]], label=f'聚类{label+1}', alpha=0.7, s=50)
    
    # 添加词汇标签（只显示部分，避免过于拥挤）
    n_labels = min(50, len(valid_words))
    indices = np.random.choice(len(valid_words), n_labels, replace=False)
    
    for idx in indices:
        plt.annotate(valid_words[idx], 
                    (reduced_vectors[idx, 0], reduced_vectors[idx, 1]),
                    fontsize=8, alpha=0.7)
    
    plt.title(f'聚类结果可视化 ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_cluster_results(clusters, output_path):
    """保存聚类结果到文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 法治相关词汇聚类结果\n\n")
        
        for cluster_name, words in clusters.items():
            f.write(f"# {cluster_name} ({len(words)}个词)\n")
            f.write(f"{cluster_name}:\n")
            
            # 每行写10个词
            for i in range(0, len(words), 10):
                line_words = words[i:i+10]
                f.write(" ".join(line_words) + "\n")
            f.write("\n")
    
    print(f"已保存聚类结果到: {output_path}")

def analyze_cluster_quality(word_vectors, cluster_labels):
    """分析聚类质量"""
    if len(set(cluster_labels)) > 1:
        silhouette = silhouette_score(word_vectors, cluster_labels)
        calinski = calinski_harabasz_score(word_vectors, cluster_labels)
        
        print(f"轮廓系数 (Silhouette Score): {silhouette:.3f}")
        print(f"Calinski-Harabasz指数: {calinski:.3f}")
        
        return silhouette, calinski
    else:
        print("只有一个聚类，无法计算质量指标")
        return None, None

def calculate_dimension_similarities(models, dimension_words, target_word="法治", normalize=False):
    """
    计算目标词与各维度的相似度
    
    Args:
        models: 词向量模型字典
        dimension_words: 维度词表字典
        target_word: 目标词
        normalize: 是否归一化 (默认False, 确保每个period的各个维度的相似度总和为1)
        
    Returns:
        DataFrame: 各时期各维度的相似度矩阵
    """
    periods = sorted(models.keys())
    dimensions = list(dimension_words.keys())
    
    # 创建结果DataFrame
    similarity_data = []
    
    for period in periods:
        model = models[period]
        if target_word not in model:
            print(f"警告: '{target_word}'在{period}模型中不存在")
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
                    except:
                        pass
            
            if similarities:
                avg_sim = np.mean(similarities)
                period_similarities[dim] = avg_sim
            else:
                period_similarities[dim] = 0
        
        if normalize:
            sum_sim = sum(sim for sim in period_similarities.values())
            period_similarities = {dim: sim / sum_sim for dim, sim in period_similarities.items()}
            
        similarity_data.append(period_similarities)
    
    return pd.DataFrame(similarity_data)

def plot_dimension_trends(similarity_df, title="法治维度语义相似度变化趋势"):
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
    plt.show()
    
    return plt.gcf()

def plot_dimension_radar(similarity_df, title="法治维度语义结构雷达图"):
    """绘制雷达图"""
    periods = similarity_df["时期"].tolist()
    dimensions = [col for col in similarity_df.columns if col != "时期"]
    N = len(dimensions)
    
    # 设置角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 为每个时期绘制一条线
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, period in enumerate(periods):
        values = similarity_df.iloc[i][dimensions].tolist()
        values += values[:1]  # 闭合雷达图
        
        # 绘制线条
        ax.plot(angles, values, linewidth=2, label=period, color=colors[i % len(colors)])
        # 填充区域
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    
    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title(title, size=15, pad=20)
    plt.show()
    
    return fig

def plot_dimension_heatmap(similarity_df, title="法治维度语义相似度热力图"):
    """绘制热力图"""
    # 准备数据
    periods = similarity_df["时期"].tolist()
    dimensions = [col for col in similarity_df.columns if col != "时期"]
    
    # 创建矩阵
    matrix_data = similarity_df[dimensions].values
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix_data, 
                xticklabels=dimensions, 
                yticklabels=periods,
                annot=True, 
                fmt='.3f', 
                cmap="YlOrRd", 
                linewidths=0.5)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def main():
    """主函数"""
    print("=== 法治维度词表聚类扩展分析 ===")
    
    # 1. 加载模型
    models = utils.load_models(MODELS_DIR)
    
    # 检查模型是否成功加载
    if not models:
        print("没有成功加载任何模型，请检查模型路径")
        return
    else:
        print(f"\n成功加载了 {len(models)} 个模型:")
        for period_name, model in models.items():
            print(f"  {period_name}: 词汇量 {len(model.index_to_key)}")
    
    # 创建输出目录
    topic_word_dir = Path("topic_word")
    topic_word_dir.mkdir(exist_ok=True)
    
    # 2. 加载相似词数据
    try:
        era_files = {
            'Era1_1978-1996': 'similar_words/Era1_1978-1996_final.txt',
            'Era2_1997-2013': 'similar_words/Era2_1997-2013_final.txt',
            'Era3_2014-2024': 'similar_words/Era3_2014-2024_final.txt'
        }
        
        similar_words_by_period = {}
        for era, file_path in era_files.items():
            word_list = utils.load_expert_word_list(file_path)
            if word_list:
                similar_words_by_period[era] = word_list
                print(f"加载 {era}: {len(word_list)} 个词")
    except Exception as e:
        print(f"加载相似词数据时出错: {e}")
        return
    
    # 3. 执行聚类分析
    print("\n=== 执行4聚类分析 ===")
    clusters_4, word_vectors_4, cluster_labels_4, valid_words_4 = cluster_similar_words(
        models, similar_words_by_period, n_clusters=4, 
        exclude_words_path="exclude_words.txt", top_n=150
    )
    
    print("\n=== 执行3聚类分析 ===")
    clusters_3, word_vectors_3, cluster_labels_3, valid_words_3 = cluster_similar_words(
        models, similar_words_by_period, n_clusters=3, 
        exclude_words_path="exclude_words.txt", top_n=150
    )
    
    # 4. 可视化聚类结果
    if len(word_vectors_4) > 0:
        print("\n=== 可视化4聚类结果 ===")
        visualize_clusters(word_vectors_4, cluster_labels_4, valid_words_4, method='tsne')
    
    if len(word_vectors_3) > 0:
        print("\n=== 可视化3聚类结果 ===")
        visualize_clusters(word_vectors_3, cluster_labels_3, valid_words_3, method='tsne')
    
    # 5. 保存聚类结果
    if clusters_3:
        save_cluster_results(clusters_3, topic_word_dir / "cluster_results_3.txt")
    if clusters_4:
        save_cluster_results(clusters_4, topic_word_dir / "cluster_results_4.txt")
    
    # 6. 分析聚类质量
    if len(word_vectors_3) > 0:
        print("\n=== 3聚类质量分析 ===")
        analyze_cluster_quality(word_vectors_3, cluster_labels_3)
    
    if len(word_vectors_4) > 0:
        print("\n=== 4聚类质量分析 ===")
        analyze_cluster_quality(word_vectors_4, cluster_labels_4)
    
    # 7. 加载专家定义的维度词表并进行分析
    print("\n=== 维度相似度分析 ===")
    
    # 加载3维度词表
    dimension_words_3d = load_dimension_words(topic_word_dir / "dimension_words_3d.txt")
    if dimension_words_3d:
        print("已加载3维度词表:")
        for dim, words in dimension_words_3d.items():
            print(f"  {dim}: {len(words)} 个词")
        
        # 计算相似度
        similarity_df_3d = calculate_dimension_similarities(models, dimension_words_3d)
        print("\n3维度相似度矩阵:")
        print(similarity_df_3d)
        
        # 绘制可视化
        if not similarity_df_3d.empty:
            plot_dimension_trends(similarity_df_3d, "法治3维度语义相似度变化趋势")
            plot_dimension_radar(similarity_df_3d, "法治3维度语义结构雷达图")
            plot_dimension_heatmap(similarity_df_3d, "法治3维度语义相似度热力图")
    
    # 加载4维度词表
    dimension_words_4d = load_dimension_words(topic_word_dir / "dimension_words_4d.txt")
    if dimension_words_4d:
        print("\n已加载4维度词表:")
        for dim, words in dimension_words_4d.items():
            print(f"  {dim}: {len(words)} 个词")
        
        # 计算相似度
        similarity_df_4d = calculate_dimension_similarities(models, dimension_words_4d)
        print("\n4维度相似度矩阵:")
        print(similarity_df_4d)
        
        # 绘制可视化
        if not similarity_df_4d.empty:
            plot_dimension_trends(similarity_df_4d, "法治4维度语义相似度变化趋势")
            plot_dimension_radar(similarity_df_4d, "法治4维度语义结构雷达图")
            plot_dimension_heatmap(similarity_df_4d, "法治4维度语义相似度热力图")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main() 