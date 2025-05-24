import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gensim.models import KeyedVectors
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import requests


def download_chinese_font():
    """下载中文字体并安装到matplotlib"""
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    font_path = os.path.join(os.path.expanduser("~"), ".fonts", "SimHei.ttf")
    
    # 创建目录
    os.makedirs(os.path.dirname(font_path), exist_ok=True)
    
    # 如果字体已存在，则不下载
    if os.path.exists(font_path):
        print(f"字体已存在: {font_path}")
    else:
        print(f"下载中文字体到: {font_path}")
        try:
            response = requests.get(font_url)
            with open(font_path, 'wb') as f:
                f.write(response.content)
            print("字体下载成功")
        except Exception as e:
            print(f"下载字体时出错: {e}")
            return None
    
    # 刷新matplotlib字体缓存
    print("刷新字体缓存...")
    fm.fontManager.addfont(font_path)
    fm._load_fontmanager(try_read_cache=False)
    
    return font_path

# 加载所有可用的模型
def load_models(MODELS_DIR):
    """加载目录中所有的词向量模型"""
    models = {}
    model_files = list(MODELS_DIR.glob("*_wordvectors.kv"))
    
    if not model_files:
        print(f"在 {MODELS_DIR} 中没有找到模型文件")
        return models
    
    print(f"找到 {len(model_files)} 个模型文件:")
    for model_file in sorted(model_files):
        period_name = model_file.stem.replace("_wordvectors", "")
        print(f"  加载模型: {period_name}")
        try:
            models[period_name] = KeyedVectors.load(str(model_file))
            print(f"  成功加载 {period_name}, 词汇量: {len(models[period_name].index_to_key)}")
        except Exception as e:
            print(f"  加载 {period_name} 失败: {e}")
    
    return models


def save_similar_words(models, keyword="法治", topn=20, output_dir="similar_words"):
    """
    保存每个时期模型中与关键词最相似的词
    
    Args:
        models: 词向量模型字典
        keyword: 要查询的关键词，默认为"法治"
        topn: 返回最相似词的数量，默认20个
        output_dir: 输出文件保存的目录名，默认为"similar_words"
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for period_name, model in models.items():
        # 检查关键词是否在词汇表中
        if keyword not in model:
            print(f"警告: 关键词'{keyword}'在{period_name}模型中不存在")
            continue
            
        try:
            # 获取最相似的词
            similar_words = model.most_similar(keyword, topn=topn)
            
            # 保存到文件
            output_file = output_path / f"{period_name}_similar_words_top{topn}_{keyword}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"与'{keyword}'最相似的{topn}个词 ({period_name}):\n\n")
                for word, score in similar_words:
                    f.write(f"{word}\t{score:.4f}\n")
                    
            print(f"已保存{period_name}的相似词到: {output_file}")
            
        except Exception as e:
            print(f"处理{period_name}时出错: {e}")
            
    print("\n完成所有时期相似词的保存")


def load_expert_word_list(file_path):
    """
    加载专家选取的关键词和相似度
    
    Args:
        file_path: 专家词列表文件路径
        
    Returns:
        list: 专家词列表，值为(word, similarity)元组列表
    """
    result = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # skip the first line
            next(f)
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        # 尝试将最后一个部分转换为浮点数作为相似度
                        try:
                            similarity = float(parts[-1])
                            result.append((word, similarity))
                        except ValueError:
                            # 如果转换失败，假设没有相似度值，设为0
                            result.append((word, 0.0))
                    elif len(parts) == 1:
                        # 只有词没有相似度
                        result.append((parts[0], 0.0))
        return result
    except Exception as e:
        print(f"读取专家词表出错: {e}")
        return []


def update_sim(models, similar_words_by_period, focus_word):
    """
    根据模型更新similar_words_by_period中的相似度值
    
    参数:
    models: 词向量模型字典，键为时期名称，值为KeyedVectors模型
    similar_words_by_period: 字典，键为时期名称，值为(word, similarity)元组列表
    focus_word: 中心词，用于计算相似度的参考词
    
    返回:
    updated_similar_words: 更新后的相似词字典
    """
    updated_similar_words = {}
    
    for period, word_list in similar_words_by_period.items():
        if period not in models:
            print(f"警告: 找不到{period}的模型，跳过更新")
            updated_similar_words[period] = word_list
            continue
            
        model = models[period]
        
        # 检查中心词是否在模型中
        if focus_word not in model:
            print(f"警告: 中心词'{focus_word}'在{period}模型中不存在，跳过更新")
            updated_similar_words[period] = word_list
            continue
        
        # 更新相似度
        updated_list = []
        for word, _ in word_list:
            if word in model:
                # 计算与中心词的相似度
                similarity = model.similarity(focus_word, word)
                updated_list.append((word, similarity))
            else:
                print(f"警告: 词'{word}'在{period}模型中不存在，保留原始相似度")
                # 找到原始相似度
                original_sim = next((sim for w, sim in word_list if w == word), 0.0)
                updated_list.append((word, original_sim))
        
        updated_similar_words[period] = updated_list
        print(f"已更新 {period} 的相似度值，共 {len(updated_list)} 个词")
    
    return updated_similar_words


def build_similar_words_by_period_from_word_list(models, focus_word, target_word_list, ):
    """
    根据目标词列表构建各时期的相似词字典
    
    参数:
    models: 词向量模型字典，键为时期名称，值为KeyedVectors模型
    focus_word: 中心词，用于计算相似度的参考词
    target_word_list: 目标词列表，将计算这些词与中心词的相似度
    返回:
    similar_words_by_period: 字典，键为时期名称，值为(word, similarity)元组列表
    """
    similar_words_by_period = {}
    
    for period, model in models.items():
        # 检查中心词是否在模型中
        if focus_word not in model:
            print(f"警告: 中心词'{focus_word}'在{period}模型中不存在，跳过该时期")
            similar_words_by_period[period] = []
            continue
        
        # 计算目标词列表中每个词与中心词的相似度
        period_similar_words = []
        for word in target_word_list:
            # 跳过关键词自己
            if word == focus_word:
                continue
            if word in model:
                similarity = model.similarity(focus_word, word)
                period_similar_words.append((word, similarity))
            else:
                print(f"警告: 词'{word}'在{period}模型中不存在，设置为0")
                period_similar_words.append((word, 0))
        
        # 按相似度降序排序
        period_similar_words.sort(key=lambda x: x[1], reverse=True)
        # if normalize:
        #     sum_sim = sum(sim for word, sim in period_similar_words)
        #     period_similar_words = [(word, sim / sum_sim) for word, sim in period_similar_words]
        similar_words_by_period[period] = period_similar_words
        print(f"已为 {period} 构建相似词列表，共 {len(period_similar_words)} 个词")
    
    return similar_words_by_period

def get_word_sets(similar_words_by_period, top_n=50, exclude_words_path=None):
    """
    获取不同时期词表的交集和并集
    
    参数:
    similar_words_by_period: 字典，键为时期名称，值为(word, similarity)元组列表
    top_n: 每个时期要考虑的前N个相似词
    exclude_words_path: 排除词库的文件路径，文件中每行一个词

    返回:
    intersection: 所有时期前N个相似词的交集
    union: 所有时期前N个相似词的并集
    filtered_similar_words: 过滤后的相似词字典
    """
    if not similar_words_by_period:
        print("没有数据可供分析")
        return set(), set(), {}
    
    # 加载排除词库
    exclude_words = set()
    if exclude_words_path:
        try:
            with open(exclude_words_path, 'r', encoding='utf-8') as f:
                exclude_words = set(line.strip() for line in f if line.strip())
            print(f"已加载 {len(exclude_words)} 个排除词")
        except Exception as e:
            print(f"加载排除词库时出错: {e}")
    
    # 过滤并获取每个时期的前N个词
    filtered_similar_words = {}
    word_sets = []
    
    for period, word_list in similar_words_by_period.items():
        # 按相似度排序
        sorted_words = sorted(word_list, key=lambda x: x[1], reverse=True)
        
        # 过滤掉排除词
        filtered_words = [(word, sim) for word, sim in sorted_words if word not in exclude_words]
        
        # 取前N个
        top_words = filtered_words[:top_n]
        filtered_similar_words[period] = top_words
        
        # 提取词汇集合
        word_set = set(word for word, _ in top_words)
        word_sets.append(word_set)
        
        print(f"{period}: 过滤后保留 {len(top_words)} 个词")
    
    # 计算交集和并集
    if word_sets:
        intersection = set.intersection(*word_sets)
        union = set.union(*word_sets)
        print(f"所有时期共有词: {len(intersection)} 个")
        print(f"所有时期词汇并集: {len(union)} 个")
    else:
        intersection = set()
        union = set()
        print("没有可用的词集合")
    
    return list(intersection), list(union), filtered_similar_words


def visualize_similar_words_across_periods(similar_words_by_period, focus_word, top_n=20, exclude_words_path=None, normalize=False):
    """
    可视化不同时期相似词的变化
    
    参数:
    similar_words_by_period: 字典，键为时期名称，值为(word, similarity)元组列表
    focus_word: 焦点词
    top_n: 每个时期要显示的前N个相似词
    exclude_words_path: 排除词库的文件路径，文件中每行一个词
    """
    if not similar_words_by_period:
        print("没有数据可供可视化")
        return
    
    # 加载排除词库
    exclude_words = set()
    if exclude_words_path:
        try:
            with open(exclude_words_path, 'r', encoding='utf-8') as f:
                exclude_words = set(line.strip() for line in f if line.strip())
            print(f"已加载 {len(exclude_words)} 个排除词")
        except Exception as e:
            print(f"加载排除词库时出错: {e}")
    
    # 准备数据
    periods = list(similar_words_by_period.keys())
    
    # 创建一个包含所有时期前top_n个相似词的列表（排除指定词）
    all_top_words = []
    
    # 首先，为每个时期过滤掉排除词
    filtered_similar_words = {}
    for period in periods:
        # 过滤掉排除词
        filtered_words = [(word, sim) for word, sim in similar_words_by_period[period] 
                         if word not in exclude_words]
        filtered_similar_words[period] = filtered_words
    
    # 然后，从过滤后的列表中获取前top_n个词
    for period in periods:
        top_words = [word for word, _ in filtered_similar_words[period][:top_n]]
        for word in top_words:
            if word not in all_top_words:  # 避免重复
                all_top_words.append(word)
    
    # 如果没有词语可以显示，则返回
    if not all_top_words:
        print("过滤后没有词语可以显示")
        return None
    
    # 创建一个DataFrame来存储每个时期每个词的相似度
    df = pd.DataFrame(index=all_top_words, columns=periods)
    
    # 填充DataFrame
    for period in periods:
        word_sim_dict = dict(filtered_similar_words[period])
        for word in all_top_words:
            df.loc[word, period] = word_sim_dict.get(word, 0)
        if normalize:
            sum_sim = sum(sim for word, sim in filtered_similar_words[period])
            filtered_similar_words[period] = [(word, sim / sum_sim) for word, sim in filtered_similar_words[period]]
    
    # 确保所有值都是浮点数
    df = df.astype(float)
    
    # 按照在最后一个时期的相似度排序
    if len(periods) > 0:
        last_period = periods[-1]
        df = df.sort_values(by=last_period, ascending=False)
    
    # 绘制热力图
    plt.figure(figsize=(12, len(all_top_words) * 0.3 + 2))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5)
    plt.title(f"与'{focus_word}'相似度最高的{top_n}个词在不同时期的变化（已排除{len(exclude_words)}个词）")
    plt.tight_layout()
    plt.show()
    
    return df

def load_exclude_words(file_path):
    """
    加载排除词库
    """
    exclude_words = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        exclude_words = set(line.strip() for line in f if line.strip())
    return exclude_words