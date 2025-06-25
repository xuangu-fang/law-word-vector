# 法治分析项目 - 可视化实现方案

## 1. 概览

本文档为"法治"概念语义变迁研究提供具体的可视化任务设计与实现规划。这里的每个方案都旨在从不同角度揭示词向量分析的结果，并直接回应`manuscript.md`中提出的研究问题。

## 2. 可视化任务详述

### 2.1 语义网络可视化 (Semantic Network Visualization)

- **目标 (Goal)**:
  - 已有的趋势图和热力图展示了"法治"与不同"维度"的聚合关系强度，但无法揭示语义场内部的*结构*。
  - 本图旨在直观展示在特定时期，"法治"与其语义邻近词构成的网络拓扑。通过网络图，我们可以清晰地看到：
    1.  哪些词是"法治"语义网络中的核心节点（连接度高）？
    2.  语义网络是否呈现出不同的"社区"（即聚类），这些社区是否与我们定义的维度相符？
    3.  网络结构如何随时间演变？（例如，某个时期的网络更分散，另一个时期更集中）

- **技术栈 (Tech Stack)**:
  - `networkx`: 用于创建、操作和分析图形结构。
  - `matplotlib` 或 `pyvis`: 用于绘制网络图。`pyvis`可以生成交互式的HTML文件，便于探索。
  - `pandas`: 数据处理。

- **数据准备 (Data Preparation)**:
  1.  输入：分时期的`gensim`模型、目标词（"法治"）、要分析的近义词数量（`top_n`，如100）。
  2.  流程：
      - 对每个时期的模型，提取与"法治"最相似的`top_n`个词。
      - 计算这`top_n`个词之间的两两相似度，形成一个邻接矩阵。
      - 将这个邻接矩阵作为图的输入。

- **核心代码结构 (Core Code Structure)**:
  ```python
  import networkx as nx
  import matplotlib.pyplot as plt
  from pyvis.network import Network
  
  def create_semantic_network(model, target_word="法治", top_n=100, similarity_threshold=0.4):
      """为单个模型创建语义网络"""
      # 1. 获取top_n相似词
      similar_words = [word for word, sim in model.wv.most_similar(target_word, top_n=top_n)]
      all_words = [target_word] + similar_words
      
      G = nx.Graph()
      
      # 2. 添加节点
      for word in all_words:
          G.add_node(word)
          
      # 3. 添加边（基于相似度）
      for i in range(len(all_words)):
          for j in range(i + 1, len(all_words)):
              word1 = all_words[i]
              word2 = all_words[j]
              if word1 in model.wv and word2 in model.wv:
                  similarity = model.wv.similarity(word1, word2)
                  if similarity > similarity_threshold:
                      G.add_edge(word1, word2, weight=similarity)
                      
      return G

  def visualize_network_matplotlib(G, period_name, target_word="法治"):
      """使用matplotlib进行静态可视化并保存为图片"""
      plt.figure(figsize=(20, 20))
      
      # 使用 spring_layout 布局算法
      pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
      
      # 绘制节点
      node_colors = ['#1f78b4' if node != target_word else '#ff7f0e' for node in G.nodes()]
      node_sizes = [100 if node != target_word else 500 for node in G.nodes()]
      nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
      
      # 绘制边
      edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
      nx.draw_networkx_edges(G, pos, width=[w * 5 for w in edge_weights], alpha=0.5, edge_color='gray')
      
      # 绘制标签
      nx.draw_networkx_labels(G, pos, font_size=10, font_family='SimHei')
      
      plt.title(f"“{target_word}”语义网络 - {period_name}", size=20)
      plt.axis('off')
      plt.savefig(f"semantic_network_{period_name}.png", dpi=300, bbox_inches='tight')
      plt.show()

  def visualize_network_pyvis(G, period_name):
      """使用pyvis进行交互式可视化"""
      net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
      net.from_nx(G)
      
      # 可以增加社区发现算法，并为不同社区的节点着色
      # communities = nx.community.greedy_modularity_communities(G)
      # ...
      
      net.show(f"semantic_network_{period_name}.html")
  
  # 主流程
  # for period, model in models.items():
  #     G = create_semantic_network(model)
  #     visualize_network_matplotlib(G, period) # 主要方法
  #     # visualize_network_pyvis(G, period) # 可选的交互式方法
  ```

- **产出 (Output)**:
  - 为每个时期生成一个独立的、高分辨率的PNG格式静态网络图，可以直接用于论文或报告。
  - (可选) 为每个时期生成一个可交互的HTML格式网络图，用于数据探索。
  - 节点大小和颜色可以区分核心词与周边词。

### 2.2 语义场变迁分析: 获益者与失势者 (Gainers and Losers)

- **目标 (Goal)**:
  - 直接回应手稿中的核心问题："哪些概念被逐渐纳入或排除出法治语义场？"
  - 量化并识别在两个连续时期之间，与"法治"的语义关系发生最大变化（最显著拉近或疏远）的词汇。
  - 通过聚焦于一个**特定的、有意义的词汇池**（如所有时期的Top-N相似词，或专家定义的维度词表），而非整个模型的数万词汇，来确保分析的聚焦性和解释力。
  - 这为叙述"法治"内涵的动态演变提供了非常具体和有力的证据。例如，"'治理'一词在2013年后与'法治'的联系显著增强"。

- **技术栈 (Tech Stack)**:
  - `pandas`: 用于数据处理、排序和差值计算。
  - `matplotlib`/`seaborn`: 用于绘制水平条形图，直观地展示变化最大的词汇。

- **数据准备 (Data Preparation)**:
  1.  **定义词汇池 (Vocabulary Pool)**：这是分析的核心。我们需要先确定要追踪哪些词的变化。策略可以包括：
      - **策略A (并集策略)**：获取"法治"在所有时期Top-N相似词的并集。这是最常用的策略，能捕捉到所有曾与"法治"紧密相关的词。
      - **策略B (文件策略)**：从一个指定的词表文件（如 `dimension_words_4d.txt`）加载词汇。这允许我们观察特定维度内部的词汇是如何相对"法治"变化的。
  2.  **计算相似度矩阵**：
      - 输入：分时期的`gensim`模型、目标词（"法治"）、以及上一步确定的词汇池。
      - 流程：构建一个DataFrame，其中行为词汇池中的每个词，列为每个时期，值为该词与"法治"在该时期的余弦相似度。如果词不存在，则记为NaN。
  3.  **计算变化**：计算相邻时期之间相似度的差值，得到一个新的DataFrame。

- **核心代码结构 (Core Code Structure)**:
  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  class SemanticChangeAnalyzer:
      def __init__(self, models):
          self.models = models
          self.periods = sorted(models.keys())

      def _get_vocabulary_pool(self, strategy='union', **kwargs):
          # ... 实现获取词汇池的逻辑 ...
          pass

      def analyze_similarity_change(self, target_word="法治", strategy='union', **kwargs):
          """计算词汇与目标词在不同时期的相似度变化"""
          # 1. 获取词汇池
          vocab_pool = self._get_vocabulary_pool(strategy, target_word=target_word, **kwargs)
          
          # 2. 计算各时期相似度
          similarity_data = {}
          for period in self.periods:
              model = self.models[period]
              sims = {}
              if target_word in model:
                  for word in vocab_pool:
                      if word in model and word != target_word:
                          sims[word] = model.similarity(target_word, word)
                      else:
                          sims[word] = float('nan')
              similarity_data[period] = pd.Series(sims)
          
          similarity_df = pd.DataFrame(similarity_data).fillna(0) # 用0填充缺失值
          
          # 3. 计算变化
          change_df = pd.DataFrame()
          for i in range(len(self.periods) - 1):
              period1, period2 = self.periods[i], self.periods[i+1]
              change_df[f'{period1}_to_{period2}'] = similarity_df[period2] - similarity_df[period1]
              
          return change_df.sort_values(by=f'{period1}_to_{period2}', ascending=False)

      def plot_gainers_losers(self, change_series, top_n=20, title=""):
          """可视化获益者与失势者"""
          gainers = change_series.nlargest(top_n)
          losers = change_series.nsmallest(top_n).sort_values(ascending=False)
          
          fig, axes = plt.subplots(1, 2, figsize=(18, 10))
          sns.barplot(x=gainers.values, y=gainers.index, ax=axes[0], palette='Greens_r')
          axes[0].set_title(f'Top {top_n} Gainers')
          
          sns.barplot(x=losers.values, y=losers.index, ax=axes[1], palette='Reds_r')
          axes[1].set_title(f'Top {top_n} Losers')
          
          fig.suptitle(title, size=20)
          plt.tight_layout(rect=[0, 0, 1, 0.96])
          plt.show()

  # 主流程
  # analyzer = SemanticChangeAnalyzer(models)
  # change_df = analyzer.analyze_similarity_change(target_word="法治", strategy='union', top_n=200)
  # for transition in change_df.columns:
  #     analyzer.plot_gainers_losers(change_df[transition], title=f'Semantic Change: {transition}')
  ```

- **产出 (Output)**:
  - 每个时期过渡阶段（如 Era1->Era2, Era2->Era3）的"获益者"和"失势者"的 `pandas.DataFrame`。
  - 可视化为并排的两个水平条形图：
    - 左侧为"获益者"(Gainers)，即与"法治"相似度增加最多的Top-N词汇，用绿色表示。
    - 右侧为"失势者"(Losers)，即与"法治"相似度减少最多的Top-N词汇，用红色表示。

---
*此方案专注于比较相邻时期的语义距离变化，是一种直接且有效的历时分析方法。*

### 2.3 词义邻域分析 (Word Neighborhood Analysis)

- **目标 (Goal)**:
  - 替代原有的复杂"词义演变轨迹图"，通过一种更简单、更稳健的方式来追踪一个核心概念（如"权利"、"民主"）的内涵演变。
  - 其核心假设是：一个词的意义由其最相似的邻居词来定义。通过观察这个"邻居集合"随时间的变化，我们可以揭示词义的变迁。
  - 量化特定概念的语义稳定性，并识别出在不同时期，哪些新词成为了它的核心关联词。

- **技术栈 (Tech Stack)**:
  - `pandas`, `numpy`: 用于数据处理和计算。
  - `matplotlib`/`seaborn`: 用于绘制量化指标图。

- **数据准备 (Data Preparation)**:
  1.  输入：分时期的`gensim`模型、要追踪的目标词列表（如 `["权利", "民主"]`）。
  2.  流程：
      - 对每个目标词，在每个时期模型中，获取其Top-N（如 N=50）的相似词，形成一个词集。
      - 比较相邻两个时期的词集，计算 Jaccard 相似度（交集/并集），作为一个量化的"语义稳定指数"。
      - 识别出"新晋邻居"（仅在后期出现）、"退隐邻居"（仅在前期出现）和"稳定邻居"（持续存在）。

- **核心代码结构 (Core Code Structure)**:
  ```python
  import pandas as pd

  def analyze_word_neighborhood(models, target_word, top_n=50):
      """分析单个词在所有时期的邻域变化"""
      periods = sorted(models.keys())
      neighborhoods = {}
      for period in periods:
          model = models[period]
          if target_word in model.wv:
              similar_words = {word for word, sim in model.wv.most_similar(target_word, top_n=top_n)}
              neighborhoods[period] = similar_words
          else:
              neighborhoods[period] = set()
      
      # 计算变化
      results = []
      for i in range(len(periods) - 1):
          period1, period2 = periods[i], periods[i+1]
          set1, set2 = neighborhoods[period1], neighborhoods[period2]
          
          if not set1 or not set2:
              jaccard_sim = 0
          else:
              jaccard_sim = len(set1.intersection(set2)) / len(set1.union(set2))
              
          new_neighbors = list(set2 - set1)
          lost_neighbors = list(set1 - set2)
          
          results.append({
              "transition": f"{period1}_to_{period2}",
              "stability_score": jaccard_sim,
              "new_neighbors": new_neighbors[:10], # 展示前10个
              "lost_neighbors": lost_neighbors[:10] # 展示前10个
          })
          
      return pd.DataFrame(results)

  # 主流程
  # for word_to_track in ["权利", "民主"]:
  #     change_df = analyze_word_neighborhood(models, word_to_track)
  #     print(f"--- 追踪 '{word_to_track}' 的语义邻域变化 ---")
  #     print(change_df)
  #     # 可以进一步用matplotlib绘制 stability_score 的变化趋势图
  ```

- **产出 (Output)**:
  - 对每个追踪的词，生成一个DataFrame，展示其在不同时期过渡阶段的：
    - 语义稳定指数（Jaccard相似度）。
    - Top 10 新晋邻居词列表。
    - Top 10 退隐邻居词列表。
  - 可以为每个追踪词绘制一个线图，展示其"语义稳定指数"随时间的变化，指数越低，说明词义变化越剧烈。

---
*注：本方案取代了原计划中需要向量空间对齐的"词义演变轨迹图"，实现了在不对齐模型上的历时语义分析。*
