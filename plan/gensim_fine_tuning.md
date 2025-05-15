# Gensim 预训练中文词向量微调技术方案

## 1. 概述

本文档详细介绍如何使用 Gensim 在特定语料上微调预训练的中文词向量模型。微调预训练词向量相比从头训练有以下优势：

1. **训练速度快**：利用已有知识，显著减少训练时间
2. **语义基础扎实**：预训练模型已捕获通用语义关系
3. **处理小语料效果更佳**：特别适合专业/特定领域语料
4. **对资源要求低**：相比从头训练，需要更少的计算资源

对于"法治"词向量分析项目，微调方法特别合适，因为我们需要在预训练的通用语义基础上，捕捉不同历史时期的语义变化。

## 2. 实现方法

### 2.1 基本微调流程

1. 加载预训练词向量模型
2. 创建新模型并构建词汇表
3. 将预训练向量权重传递给新模型
4. 在目标语料上微调模型
5. 保存微调后的模型

### 2.2 方法一：使用 Word2Vec 微调

```python
from gensim.models import KeyedVectors, Word2Vec
import numpy as np

def fine_tune_word2vec(pretrained_path, corpus, output_path, epochs=5):
    """
    微调预训练词向量
    
    Args:
        pretrained_path: 预训练词向量路径
        corpus: 训练语料 (分词后的句子列表)
        output_path: 输出模型路径
        epochs: 训练轮数
    """
    # 1. 加载预训练词向量
    print(f"加载预训练词向量: {pretrained_path}")
    pretrained_wv = KeyedVectors.load_word2vec_format(pretrained_path)
    vector_size = pretrained_wv.vector_size
    
    # 2. 创建新的Word2Vec模型
    print("创建新模型并构建词汇表...")
    model = Word2Vec(vector_size=vector_size, 
                     window=5, 
                     min_count=5,
                     workers=4)
    
    # 3. 在新语料上构建词汇表
    model.build_vocab(corpus)
    
    # 4. 将预训练词向量权重传递给新模型
    # 对于词汇表中已有的词
    print("初始化模型权重...")
    intersected_words = set(model.wv.index_to_key) & set(pretrained_wv.index_to_key)
    for word in intersected_words:
        model.wv.vectors[model.wv.key_to_index[word]] = pretrained_wv[word]
    
    # 5. 解锁所有向量以便微调
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
    
    # 6. 在新语料上微调
    print(f"在目标语料上微调 ({epochs} epochs)...")
    model.train(corpus,
                total_examples=len(corpus),
                epochs=epochs,
                start_alpha=0.03,
                end_alpha=0.007)
    
    # 7. 保存微调后的模型
    print(f"保存微调后的模型到: {output_path}")
    model.wv.save(output_path)
    
    return model
```

### 2.3 方法二：使用 FastText 微调（支持未登录词）

```python
from gensim.models import FastText, KeyedVectors

def fine_tune_fasttext(pretrained_path, corpus, output_path, epochs=5):
    """
    微调FastText预训练词向量
    
    Args:
        pretrained_path: 预训练词向量或模型路径
        corpus: 训练语料 (分词后的句子列表)
        output_path: 输出模型路径
        epochs: 训练轮数
    """
    # 1. 加载预训练模型
    try:
        # 尝试直接加载完整模型
        print(f"尝试加载FastText模型: {pretrained_path}")
        model = FastText.load(pretrained_path)
    except:
        # 如果只有词向量文件，则创建新模型并初始化
        print(f"加载词向量文件并创建新FastText模型")
        wv = KeyedVectors.load_word2vec_format(pretrained_path)
        model = FastText(vector_size=wv.vector_size, 
                         window=5, 
                         min_count=5,
                         workers=4)
        model.build_vocab(corpus)
        
        # 复制词向量
        print("初始化模型权重...")
        for word in wv.index_to_key:
            if word in model.wv:
                model.wv.vectors[model.wv.key_to_index[word]] = wv[word]
    
    # 2. 在当前语料上继续训练
    print(f"在目标语料上微调 ({epochs} epochs)...")
    model.train(corpus, 
                total_examples=len(corpus),
                epochs=epochs)
    
    # 3. 保存微调后的模型
    print(f"保存微调后的模型到: {output_path}")
    model.save(output_path)
    
    return model
```

### 2.4 针对"法治"项目的时间序列微调方案

对于不同历史时期的语料，我们可以采用递进式微调策略：

```python
def time_sensitive_fine_tuning(pretrained_path, period_corpus_dict, base_output_dir, model_type="word2vec"):
    """
    针对不同时期进行词向量微调
    
    Args:
        pretrained_path: 预训练词向量路径
        period_corpus_dict: 不同时期的语料字典 {'1980-1985': corpus1, ...}
        base_output_dir: 模型输出目录
        model_type: 使用的模型类型 ("word2vec" 或 "fasttext")
    """
    # 创建输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 选择微调函数
    fine_tune_fn = fine_tune_word2vec if model_type == "word2vec" else fine_tune_fasttext
    
    # 先在最早时期微调
    periods = sorted(period_corpus_dict.keys())
    first_period = periods[0]
    
    print(f"=== 开始处理第一个时期: {first_period} ===")
    # 微调第一个时期
    first_model = fine_tune_fn(
        pretrained_path, 
        period_corpus_dict[first_period],
        f"{base_output_dir}/{first_period}.model"
    )
    
    # 连续微调后续时期
    prev_model_path = f"{base_output_dir}/{first_period}.model"
    for period in periods[1:]:
        print(f"=== 开始处理时期: {period} ===")
        curr_model = fine_tune_fn(
            prev_model_path,
            period_corpus_dict[period],
            f"{base_output_dir}/{period}.model"
        )
        prev_model_path = f"{base_output_dir}/{period}.model"
    
    # 返回所有时期模型路径的字典
    return {period: f"{base_output_dir}/{period}.model" for period in periods}
```

## 3. 资源需求评估

### 3.1 内存需求

1. **基础内存占用**：
   - 典型的中文预训练词向量（如腾讯AI Lab的词向量，800万词，200维）约需要 **6-8GB 内存**
   - 运行时还需额外内存用于语料和计算，建议至少 **16GB 系统内存**

2. **批处理优化**：
   - 如果内存有限，可使用流式处理器减少内存占用
   - 可以减小 `workers` 参数来降低并行程度

### 3.2 计算资源

1. **CPU 需求**：
   - 微调过程主要依赖CPU计算
   - 4-8核处理器足以高效处理
   - 多线程能显著加速训练（通过workers参数控制）

2. **GPU 需求**：
   - 标准Gensim库不使用GPU
   - 词向量微调通常不需要GPU加速
   - 若需GPU加速，可考虑使用PyTorch或TensorFlow实现的词向量模型

### 3.3 训练时间

1. **微调时间估算**：
   - 对于每个时期约50MB文本数据（约1000万词）：
     - 5个epoch：约 **15-30分钟**（多核CPU）
   - 整个项目（10个时期）：约 **2-5小时**

2. **影响训练时间的因素**：
   - 语料大小：线性影响训练时间
   - Epoch数量：线性影响
   - 向量维度：略微影响
   - CPU核心数：影响并行加速效率

### 3.4 存储需求

1. **模型文件大小**：
   - 每个时期的模型文件：约 **200-500MB**
   - 全部时期（假设10个）：约 **2-5GB**

2. **中间文件**：
   - 预处理后的语料：原始数据的约 **60-80%**
   - 临时文件和日志：约 **100-200MB**

## 4. 优化策略

### 4.1 内存优化

如果系统内存有限，可以采用以下策略降低内存占用：

```python
# 使用迭代器减少内存使用
from gensim.models.word2vec import LineSentence
corpus = LineSentence('preprocessed_corpus.txt')  # 从文件流式读取

# 减少同时加载的语料大小
def batch_generator(file_path, batch_size=10000):
    """分批生成语料"""
    sentences = []
    for i, line in enumerate(open(file_path, 'r', encoding='utf-8')):
        sentences.append(line.strip().split())
        if i % batch_size == 0 and i > 0:
            yield sentences
            sentences = []
    if sentences:
        yield sentences

# 使用
for batch in batch_generator('corpus.txt'):
    model.train(batch, epochs=1)
```

### 4.2 选择性微调

对于特定的研究焦点（如"法治"），可以采用选择性微调，只对相关词汇进行完全微调：

```python
def selective_fine_tuning(pretrained_path, corpus, focus_words, output_path):
    """选择性微调，重点调整与研究焦点相关的词向量"""
    # 加载预训练词向量
    pretrained_wv = KeyedVectors.load_word2vec_format(pretrained_path)
    
    # 创建新模型
    model = Word2Vec(vector_size=pretrained_wv.vector_size, 
                     window=5, min_count=5, workers=4)
    model.build_vocab(corpus)
    
    # 初始化词向量
    for word in model.wv.index_to_key:
        if word in pretrained_wv:
            model.wv.vectors[model.wv.key_to_index[word]] = pretrained_wv[word]
    
    # 设置选择性锁定因子
    # focus_words中的词完全微调，其他词部分微调
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32) * 0.1  # 默认小幅调整
    for word in model.wv.index_to_key:
        if word in focus_words:  # 对焦点词完全放开微调
            model.wv.vectors_lockf[model.wv.key_to_index[word]] = 1.0
    
    # 微调
    model.train(corpus, total_examples=len(corpus), epochs=5)
    model.wv.save(output_path)
    
    return model
```

### 4.3 微调参数优化

针对不同应用场景，可以调整以下参数来优化微调效果：

```python
def optimized_fine_tuning(pretrained_path, corpus, output_path, 
                         alpha=0.01,       # 学习率
                         epochs=5,         # 训练轮数
                         window=5,         # 上下文窗口大小
                         negative=5,       # 负采样数量
                         ns_exponent=0.75, # 负采样分布指数
                         lockf=0.1         # 非目标词锁定因子
                        ):
    """使用优化参数的微调函数"""
    
    # 加载预训练词向量
    pretrained_wv = KeyedVectors.load_word2vec_format(pretrained_path)
    
    # 创建新模型
    model = Word2Vec(vector_size=pretrained_wv.vector_size, 
                     window=window, 
                     min_count=1,  # 保留所有词
                     workers=4,
                     negative=negative,
                     ns_exponent=ns_exponent,
                     alpha=alpha)
    
    # 构建词汇表并初始化
    model.build_vocab(corpus)
    
    # 初始化词向量
    for word in model.wv.index_to_key:
        if word in pretrained_wv:
            model.wv.vectors[model.wv.key_to_index[word]] = pretrained_wv[word]
    
    # 设置锁定因子
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32) * lockf
    
    # 微调
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    model.wv.save(output_path)
    
    return model
```

## 5. 实用代码片段

### 5.1 模型评估

```python
def evaluate_fine_tuned_model(model, test_pairs, period=None):
    """评估微调模型的语义表示效果"""
    results = {}
    
    # 1. 测试词语相似度
    similarities = []
    for word1, word2 in test_pairs:
        if word1 in model.wv and word2 in model.wv:
            sim = model.wv.similarity(word1, word2)
            similarities.append((word1, word2, sim))
    
    # 2. 测试特定关键词的最相似词变化
    key_words = ["法治", "法制", "宪法", "权利"]
    similar_words = {}
    for word in key_words:
        if word in model.wv:
            similar_words[word] = model.wv.most_similar(word, topn=20)
    
    results = {
        "period": period,
        "similarities": similarities,
        "similar_words": similar_words
    }
    
    return results
```

### 5.2 跨时期模型比较

```python
def compare_periods(period_models, target_word="法治"):
    """比较不同时期目标词的语义变化"""
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    
    periods = sorted(period_models.keys())
    results = []
    
    # 1. 检查目标词在各时期模型中是否存在
    for period in periods:
        model = period_models[period]
        if target_word not in model.wv:
            print(f"警告: '{target_word}' 在 {period} 时期模型中不存在")
            continue
            
        # 2. 提取目标词向量
        vector = model.wv[target_word]
        
        # 3. 找出最相似的词
        most_similar = model.wv.most_similar(target_word, topn=10)
        
        results.append({
            "period": period,
            "vector": vector,
            "most_similar": most_similar
        })
    
    # 4. 计算时期间的相似度矩阵
    if len(results) > 1:
        similarity_matrix = pd.DataFrame(index=periods, columns=periods)
        for i, period1 in enumerate(periods):
            for j, period2 in enumerate(periods):
                if i <= j and period1 in [r["period"] for r in results] and period2 in [r["period"] for r in results]:
                    vec1 = next(r["vector"] for r in results if r["period"] == period1)
                    vec2 = next(r["vector"] for r in results if r["period"] == period2)
                    sim = cosine_similarity([vec1], [vec2])[0][0]
                    similarity_matrix.loc[period1, period2] = sim
                    similarity_matrix.loc[period2, period1] = sim
        
        print("时期间目标词语义相似度矩阵:")
        print(similarity_matrix)
    
    return results
```

### 5.3 完整处理流程示例

```python
def complete_workflow(pretrained_path, raw_corpus_dir, periods, output_dir):
    """完整的预训练词向量微调工作流"""
    import os
    import json
    from datetime import datetime
    
    # 1. 准备输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 记录配置信息
    config = {
        "pretrained_model": pretrained_path,
        "periods": periods,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 3. 加载并预处理语料
    period_corpora = {}
    for period in periods:
        corpus_path = os.path.join(raw_corpus_dir, f"{period}.txt")
        if os.path.exists(corpus_path):
            # 假设文本已经分词，每行一个句子，词之间用空格分隔
            corpus = list(LineSentence(corpus_path))
            period_corpora[period] = corpus
            print(f"加载{period}语料: {len(corpus)}句")
        else:
            print(f"警告: 找不到{period}的语料文件")
    
    # 4. 执行时间序列微调
    print("\n=== 开始微调过程 ===")
    model_paths = time_sensitive_fine_tuning(
        pretrained_path,
        period_corpora,
        os.path.join(output_dir, "models")
    )
    
    # 5. 评估模型
    print("\n=== 模型评估 ===")
    evaluation_results = {}
    for period, model_path in model_paths.items():
        model = KeyedVectors.load(model_path)
        results = evaluate_fine_tuned_model(model, test_pairs=[
            ("法治", "秩序"), 
            ("法治", "民主"), 
            ("法治", "权利"),
            ("法治", "发展")
        ], period=period)
        evaluation_results[period] = results
    
    # 6. 保存评估结果
    with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        # 需要先将numpy数组转换为列表
        processed_results = {}
        for period, result in evaluation_results.items():
            processed_results[period] = {
                "similarities": result["similarities"],
                "similar_words": {w: [(t[0], float(t[1])) for t in pairs] 
                                  for w, pairs in result["similar_words"].items()}
            }
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 微调完成! 结果保存在: {output_dir} ===")
    return model_paths, evaluation_results
```

## 6. 最低配置建议

基于以上资源需求分析，对于"法治"词向量分析项目，推荐的最低配置是：

- **内存**：16GB RAM
- **处理器**：4核以上CPU
- **存储**：20GB可用空间
- **操作系统**：任何支持Python/Gensim的系统

## 7. 总结与建议

1. **分阶段实施**：
   - 先在小语料上测试微调效果
   - 确认参数有效后再处理完整语料

2. **关注特定词语**：
   - 选择性微调"法治"相关核心词汇
   - 定期检查这些词的语义变化

3. **组合多种模型**：
   - 可同时使用Word2Vec和FastText，取长补短
   - 使用不同技术路线的结果交叉验证

通过本文提供的技术方案和代码实现，可以高效地在特定语料上微调预训练中文词向量，为"法治"语义演变研究提供坚实的技术支持。 