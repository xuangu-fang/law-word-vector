#!/usr/bin/env python
# coding: utf-8

"""
生成法治和法制关键词在三个历史时期的top5000词汇文件的交集和并集
包含相似度平均值，按相似度排序

输入文件：
- Era1_1978-1996_similar_words_top5000_法治.txt
- Era1_1978-1996_similar_words_top5000_法制.txt
- Era2_1997-2013_similar_words_top5000_法治.txt
- Era2_1997-2013_similar_words_top5000_法制.txt
- Era3_2014-2024_similar_words_top5000_法治.txt
- Era3_2014-2024_similar_words_top5000_法制.txt

输出文件：
- intersection_with_scores.txt (交集，包含相似度平均值)
- union_with_scores.txt (并集，包含相似度平均值)
"""

from pathlib import Path
from collections import defaultdict
import re

# 文件路径配置
SIMILAR_WORDS_DIR = Path("similar_words")
OUTPUT_DIR = Path("similar_words_analysis")

# 定义要分析的关键词和时期
KEYWORDS = ["法治", "法制"]
PERIODS = ["Era1_1978-1996", "Era2_1997-2013", "Era3_2014-2024"]

def load_similar_words_file(file_path: Path):
    """加载相似词汇文件，返回词汇和相似度的字典"""
    try:
        word_scores = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行、注释行和标题行
                if line and not line.startswith('#') and not line.startswith('与') and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        word = parts[0].strip()
                        try:
                            score = float(parts[1].strip())
                            word_scores[word] = score
                        except ValueError:
                            continue
        
        print(f"成功加载文件 {file_path.name}，包含 {len(word_scores)} 个有效词汇")
        return word_scores
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return {}

def calculate_average_scores(all_word_scores):
    """计算每个词汇的平均相似度分数"""
    word_avg_scores = {}
    
    for word, scores in all_word_scores.items():
        if scores:  # 确保有分数
            avg_score = sum(scores) / len(scores)
            word_avg_scores[word] = avg_score
    
    return word_avg_scores

def main():
    """主函数"""
    print("开始分析法治和法制关键词的词汇文件...")
    
    # 检查输入目录
    if not SIMILAR_WORDS_DIR.exists():
        print(f"输入目录不存在: {SIMILAR_WORDS_DIR}")
        return
    
    # 收集所有词汇的相似度分数
    all_fazhi_scores = defaultdict(list)  # 所有法治词汇的分数
    all_fazhi_2_scores = defaultdict(list)  # 所有法制词汇的分数
    
    for period in PERIODS:
        for keyword in KEYWORDS:
            filename = f"{period}_similar_words_top5000_{keyword}.txt"
            file_path = SIMILAR_WORDS_DIR / filename
            if file_path.exists():
                word_scores = load_similar_words_file(file_path)
                for word, score in word_scores.items():
                    if keyword == "法治":
                        all_fazhi_scores[word].append(score)
                    else:  # 法制
                        all_fazhi_2_scores[word].append(score)
            else:
                print(f"文件不存在: {file_path}")
    
    # 计算平均分数
    fazhi_avg_scores = calculate_average_scores(all_fazhi_scores)
    fazhi_2_avg_scores = calculate_average_scores(all_fazhi_2_scores)
    
    print(f"\n=== 统计结果 ===")
    print(f"法治词汇总数: {len(fazhi_avg_scores)}")
    print(f"法制词汇总数: {len(fazhi_2_avg_scores)}")
    
    # 计算交集和并集
    intersection_words = set(fazhi_avg_scores.keys()) & set(fazhi_2_avg_scores.keys())
    union_words = set(fazhi_avg_scores.keys()) | set(fazhi_2_avg_scores.keys())
    
    print(f"交集大小: {len(intersection_words)}")
    print(f"并集大小: {len(union_words)}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 生成交集文件（包含平均相似度）
    intersection_file = OUTPUT_DIR / "intersection_with_scores.txt"
    with open(intersection_file, 'w', encoding='utf-8') as f:
        # 计算交集词汇的平均相似度（法治和法制的平均）
        intersection_scores = {}
        for word in intersection_words:
            fazhi_score = fazhi_avg_scores.get(word, 0)
            fazhi_2_score = fazhi_2_avg_scores.get(word, 0)
            # 取两个关键词的平均相似度的平均值
            avg_score = (fazhi_score + fazhi_2_score) / 2
            intersection_scores[word] = avg_score
        
        # 按相似度降序排序
        sorted_intersection = sorted(intersection_scores.items(), key=lambda x: x[1], reverse=True)
        
        for word, score in sorted_intersection:
            f.write(f"{word}\t{score:.4f}\n")
    
    print(f"交集已保存到: {intersection_file}")
    
    # 生成并集文件（包含平均相似度）
    union_file = OUTPUT_DIR / "union_with_scores.txt"
    with open(union_file, 'w', encoding='utf-8') as f:
        # 计算并集词汇的平均相似度
        union_scores = {}
        for word in union_words:
            if word in intersection_words:
                # 交集词汇：取两个关键词的平均相似度的平均值
                fazhi_score = fazhi_avg_scores.get(word, 0)
                fazhi_2_score = fazhi_2_avg_scores.get(word, 0)
                avg_score = (fazhi_score + fazhi_2_score) / 2
            else:
                # 只在其中一个关键词中的词汇：取该关键词的相似度
                if word in fazhi_avg_scores:
                    avg_score = fazhi_avg_scores[word]
                else:
                    avg_score = fazhi_2_avg_scores[word]
            union_scores[word] = avg_score
        
        # 按相似度降序排序
        sorted_union = sorted(union_scores.items(), key=lambda x: x[1], reverse=True)
        
        for word, score in sorted_union:
            f.write(f"{word}\t{score:.4f}\n")
    
    print(f"并集已保存到: {union_file}")
    
    print(f"\n分析完成！结果文件保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 