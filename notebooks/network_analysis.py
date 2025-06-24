"""
基于词向量模型，生成和可视化"法治"的语义网络
此脚本提供一个 SemanticNetworkAnalyzer 类，支持多种词汇构建和可视化策略。
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
import networkx as nx
import re
from pyvis.network import Network

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
    #plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
    # 使用绝对路径以确保找到字体
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"成功设置中文字体: {font_prop.get_name()}")
else:
    print("无法设置中文字体，将使用替代方案")

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义项目根目录和模型目录
PROJECT_ROOT = Path.cwd().parent
MODELS_DIR = PROJECT_ROOT / "models"


class SemanticNetworkAnalyzer:
    """一个用于生成和分析语义网络的类"""
    
    def __init__(self, models, exclude_words_path=None):
        self.models = models
        self.periods = sorted(models.keys())
        self.exclude_words = self._load_words_from_simple_list(exclude_words_path)
        self.node_labels = {} # 用于存储节点的标签（如维度）
        print(f"分析器已初始化，包含 {len(self.periods)} 个时期。")
        if self.exclude_words:
            print(f"已加载 {len(self.exclude_words)} 个排除词。")

    def _load_words_from_simple_list(self, file_path):
        """从简单文本文件加载词汇列表 (兼容排除词和时期特定词表)"""
        if not file_path:
            return set()
        words = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 兼容 "word <tab> score" 或 "word" 格式
                    word = line.split()[0]
                    words.add(word)
            return words
        except FileNotFoundError:
            print(f"警告: 文件 {file_path} 未找到。")
            return set()

    def _load_labeled_words(self, file_path):
        """从文件加载带标签的词汇，兼容维度和聚类文件格式"""
        labeled_words = {}
        current_label = None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 匹配 "标签:" 或 "标签 (xx个词)"
                    match = re.match(r'^([^:( ]+)', line)
                    if line.endswith(':'):
                        current_label = line[:-1]
                        labeled_words[current_label] = []
                    elif match and ':' in line: # 处理聚类文件格式
                        current_label = match.group(1)
                        labeled_words[current_label] = []
                    elif current_label:
                        words = line.split()
                        labeled_words[current_label].extend(words)
            return labeled_words
        except Exception as e:
            print(f"加载标签词表时出错: {e}")
            return {}

    def get_vocabulary(self, strategy="target_centric", **kwargs):
        """
        根据不同策略获取网络词汇表
        
        策略:
        - 'target_centric': 以目标词为中心，取top_n相似词 (默认)
        - 'union': 取所有时期top_n相似词的并集，确保可比性
        - 'from_file': 从指定的标签词表文件加载
        - 'from_period_specific_files': 使用时期特定的词表文件
        """
        vocab = set()
        self.node_labels = {}
        target_word = kwargs.get("target_word", "法治")

        if strategy == "target_centric":
            period = kwargs.get("period")
            model = self.models.get(period)
            top_n = kwargs.get("top_n", 50)
            if target_word in model:
                similar_words = {word for word, sim in model.most_similar(target_word, topn=top_n)}
                vocab.update(similar_words)
        
        elif strategy == "union":
            top_n = kwargs.get("top_n", 50)
            for period in self.periods:
                model = self.models[period]
                if target_word in model:
                    vocab.update({word for word, sim in model.most_similar(target_word, topn=top_n)})
        
        elif strategy == "from_file":
            file_path = kwargs.get("file_path")
            labeled_words = self._load_labeled_words(file_path)
            for label, words in labeled_words.items():
                vocab.update(words)
                for word in words:
                    self.node_labels[word] = label

        elif strategy == "from_period_specific_files":
            period = kwargs.get("period")
            file_map = kwargs.get("file_map", {})
            file_path = file_map.get(period)
            if file_path:
                vocab.update(self._load_words_from_simple_list(file_path))

        # 确保中心词在词汇表中
        if target_word:
            vocab.add(target_word)
        
        # 过滤排除词
        vocab = {word for word in vocab if word not in self.exclude_words}
        return list(vocab)

    def create_semantic_network(self, model, vocabulary, similarity_quantile=0.7):
        """基于给定的词汇表和相似度分位数创建网络"""
        G = nx.Graph()
        valid_vocab = [word for word in vocabulary if word in model]
        
        # 计算所有可能的边的权重
        edge_weights = []
        for i in range(len(valid_vocab)):
            for j in range(i + 1, len(valid_vocab)):
                word1, word2 = valid_vocab[i], valid_vocab[j]
                try:
                    sim = model.similarity(word1, word2)
                    edge_weights.append(sim)
                except KeyError:
                    continue
        
        if not edge_weights:
            return G # 如果没有边，返回空图

        # 计算分位数阈值
        threshold = np.quantile(edge_weights, similarity_quantile)

        # 添加节点和边
        for word in valid_vocab:
            G.add_node(word)
        for i in range(len(valid_vocab)):
            for j in range(i + 1, len(valid_vocab)):
                word1, word2 = valid_vocab[i], valid_vocab[j]
                try:
                    similarity = model.similarity(word1, word2)
                    if similarity >= threshold:
                        G.add_edge(word1, word2, weight=float(similarity))
                except KeyError:
                    continue
        return G

    def _visualize_pyvis(self, G, file_path):
        """使用pyvis保存交互式网络图"""
        net = Network(height="800px", width="100%", notebook=True, cdn_resources='in_line')
        net.from_nx(G)
        net.show(str(file_path))
        print(f"已保存Pyvis交互式网络图到: {file_path}")

    def visualize_network(self, G, title_info, output_dir, size_by_degree=False, save_pyvis=False):
        """使用matplotlib进行静态可视化并保存为图片"""
        if not G.nodes():
            print(f"网络图为空，无法为 {title_info['period']} 生成可视化。")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(25, 25))
        
        pos = nx.spring_layout(G, k=0.8, iterations=70, seed=42)
        
        # 节点颜色
        target_word = title_info.get('target_word')
        node_colors_map = {}
        if self.node_labels:
            # 1.1. 按标签分配基础颜色
            labels_set = sorted(list(set(self.node_labels.values())))
            colors = plt.cm.get_cmap('tab20', len(labels_set))
            color_map = {label: colors(i) for i, label in enumerate(labels_set)}
            for node in G.nodes():
                node_colors_map[node] = color_map.get(self.node_labels.get(node, ''), '#cccccc')
        else:
            # 1.2. 分配默认颜色
            for node in G.nodes():
                node_colors_map[node] = '#1f78b4'
        
        # 2. 无论何种情况，都用特殊颜色覆盖中心词
        if target_word and target_word in node_colors_map:
            node_colors_map[target_word] = '#ff7f0e' # 亮橙色

        node_colors = [node_colors_map[node] for node in G.nodes()]

        # 节点大小
        if size_by_degree:
            degrees = dict(G.degree())
            min_degree, max_degree = min(degrees.values()) or 1, max(degrees.values()) or 1
            node_sizes = [50 + (degrees.get(node, 0) - min_degree) / (max_degree - min_degree) * 1500 for node in G.nodes()]
        else:
            node_sizes = [600 if node == target_word else 150 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[(w-0.3) * 8 for w in edge_weights], alpha=0.5, edge_color='gray')
        
        font_prop = fm.FontProperties(fname=font_path, size=14)
        for node, (x, y) in pos.items():
            plt.text(x, y, node, fontproperties=font_prop, ha='center', va='center', zorder=10)
        
        if self.node_labels:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=15) for label, color in color_map.items()]
            plt.legend(handles=legend_elements, prop={'fname': font_path, 'size': 16}, bbox_to_anchor=(1.05, 1), loc='upper left')

        # 动态标题
        period = title_info.get('period', '')
        params = title_info.get('params', {})
        param_str = ', '.join([f'{k}={v}' for k,v in params.items()])
        plt.title(f"语义网络 - {period}\n({param_str})", fontproperties=fm.FontProperties(fname=font_path, size=30))
        plt.axis('off')
        
        # 将所有参数值转为字符串以便用于文件名
        param_values_str = [str(v) for v in params.values()]
        filename = output_path / f"net_{period}_{'_'.join(param_values_str)}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存网络图到: {filename}")
        plt.show()

        if save_pyvis:
            pyvis_filename = output_path / f"net_{period}_{'_'.join(param_values_str)}.html"
            self._visualize_pyvis(G, pyvis_filename)

def main():
    """主函数"""
    print('=== "法治"语义网络分析 (v3) ===')
    
    models = utils.load_models(MODELS_DIR)
    if not models:
        print("模型加载失败。")
        return

    output_root = PROJECT_ROOT / "output" / "network_plots_v3"
    exclude_file = PROJECT_ROOT / "notebooks" / "exclude_words_network.txt"
    
    analyzer = SemanticNetworkAnalyzer(models, exclude_words_path=exclude_file)

    # --- 场景1: 分析单个时期，节点大小由degree决定，并保存pyvis图 ---
    print("\n\n--- 场景1: 目标词中心网络 (size by degree, with Pyvis) ---")
    period = 'Era3_2014-2024'
    params_sc1 = {'target': '法治', 'top_n': 80, 'quantile': 0.75}
    vocab_sc1 = analyzer.get_vocabulary(strategy='target_centric', period=period, target_word=params_sc1['target'], top_n=params_sc1['top_n'])
    G_sc1 = analyzer.create_semantic_network(analyzer.models[period], vocab_sc1, similarity_quantile=params_sc1['quantile'])
    title_info_sc1 = {'period': period, 'params': params_sc1}
    analyzer.visualize_network(G_sc1, title_info_sc1, output_root / 'target_centric', size_by_degree=True, save_pyvis=True)

    # --- 场景2: 从维度文件加载，并对每个时期生成网络 ---
    print("\n\n--- 场景2: 从维度文件加载并着色 ---")
    dim_file = PROJECT_ROOT / "notebooks" / "topic_word" / "dimension_words_4d.txt"
    params_sc2 = {'source': '4d_dims', 'quantile': 0.3}
    vocab_sc2 = analyzer.get_vocabulary(strategy='from_file', file_path=dim_file, target_word='法治')
    for period in analyzer.periods:
        G_sc2 = analyzer.create_semantic_network(analyzer.models[period], vocab_sc2, similarity_quantile=params_sc2['quantile'])
        title_info_sc2 = {'period': period, 'params': params_sc2}
        analyzer.visualize_network(G_sc2, title_info_sc2, output_root / 'from_dims_file')

    # --- 场景3: 使用时期特定的词表文件 ---
    print("\n\n--- 场景3: 使用时期特定的专家词表 ---")
    base_path = PROJECT_ROOT / "notebooks" / "similar_words"
    file_map_sc3 = {
        'Era1_1978-1996': base_path / 'Era1_1978-1996_final.txt',
        'Era2_1997-2013': base_path / 'Era2_1997-2013_final.txt',
        'Era3_2014-2024': base_path / 'Era3_2014-2024_final.txt'
    }
    params_sc3 = {'source': 'expert_list', 'quantile': 0.6}
    target_sc3 = '法治'
    for period in analyzer.periods:
        vocab_sc3 = analyzer.get_vocabulary(strategy='from_period_specific_files', period=period, file_map=file_map_sc3, target_word=target_sc3)
        G_sc3 = analyzer.create_semantic_network(analyzer.models[period], vocab_sc3, similarity_quantile=params_sc3['quantile'])
        title_info_sc3 = {'period': period, 'params': params_sc3, 'target_word': target_sc3}
        analyzer.visualize_network(G_sc3, title_info_sc3, output_root / 'period_specific_files')


if __name__ == "__main__":
    main() 