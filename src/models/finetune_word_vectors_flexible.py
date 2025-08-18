#!/usr/bin/env python
# coding: utf-8

"""
Flexible Word Vector Fine-tuning Script for People's Daily Corpus.

This script:
1. Loads a pre-trained Chinese word vector model.
2. Takes a list of period definitions (name, start_year, end_year).
3. Uses CorpusManager to get or create period-specific corpora.
4. Fine-tunes Word2Vec models for each defined period.
5. Supports both incremental fine-tuning (default) and independent fine-tuning from the base pre-trained model.
6. Saves the fine-tuned models and performs basic similarity tests.
"""

import logging
from pathlib import Path
import os
import sys
import numpy as np

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import RULE_DISCARD, RULE_KEEP
import tqdm

# --- 调整 sys.path 以便从兄弟目录导入 CorpusManager ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_SRC_DIR = SCRIPT_DIR.parent
DATA_MODULE_DIR = PROJECT_SRC_DIR / "data"
sys.path.append(str(PROJECT_SRC_DIR))

from data.corpus_manager import CorpusManager # type: ignore

# --- 配置信息 ---
PRETRAINED_VECTORS_PATH = Path.home() / "gensim-data" / "vectors" / "chinese_vectors.kv" # 预训练词向量路径

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # 项目根目录

# --- 输出目录结构说明 ---
# 脚本会根据配置自动创建以下目录结构：
# models/
# ├── fine_tuned_vectors_flexible/
# │   ├── incremental/           # 增量微调策略
# │   │   ├── top5000/          # 聚焦微调，TOP_N_COOCCURRING=5000
# │   │   └── no_focus/         # 非聚焦微调
# │   └── independent/          # 独立微调策略
# │       ├── top5000/          # 聚焦微调，TOP_N_COOCCURRING=5000
# │       └── no_focus/         # 非聚焦微调
# └── debug_vocabs/
#     ├── incremental/
#     │   ├── top5000/
#     │   └── no_focus/
#     └── independent/
#         ├── top5000/
#         └── no_focus/
# --- 时期定义 (可自定义此列表) ---
# 每个字典应包含 "name", "start_year", 和 "end_year"
PERIODS_TO_FINETUNE = [
    {"name": "Era1_1978-1996", "start_year": 1978, "end_year": 1996},
    {"name": "Era2_1997-2013", "start_year": 1997, "end_year": 2013},
    {"name": "Era3_2014-2024", "start_year": 2014, "end_year": 2024},
    # {"name": "Era4_1977-1979", "start_year": 1977, "end_year": 1979},
    # Example for later periods if data is available:
    # {"name": "Era5_1980-1989", "start_year": 1980, "end_year": 1989},
    # {"name": "FullRange_1949-2023", "start_year": 1949, "end_year": 2023} 
]

# --- 微调策略 ---
INCREMENTAL_FINETUNING = False # 若为True，时期N的模型将作为时期N+1的基础模型进行增量微调
FORCE_CREATE_PERIOD_CORPORA = False # 若为True，即使时期语料已存在也将重新创建

# --- Word2Vec 微调参数 ---
# VECTOR_SIZE 将从预训练模型动态获取
WINDOW_SIZE = 50 # 上下文窗口大小
MIN_COUNT = 5  # 词频阈值：在聚焦微调模式下，这也是聚焦词必须达到的最低频率
WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1 # 使用的CPU核心数
EPOCHS = 10        # 训练轮数
SG = 1 # 0 表示 CBOW, 1 表示 Skip-gram

# --- 聚焦微调控制 ---
FOCUSED_FINETUNING = True  # 是否启用聚焦微调
FOCUS_WORD = "法治"  # 聚焦词
TOP_N_COOCCURRING = 10000  # 与聚焦词共现频率最高的N个词
 
# 根据微调策略创建不同的输出目录
if INCREMENTAL_FINETUNING:
    strategy_suffix = "incremental"
else:
    strategy_suffix = "independent"

# 在文件名中添加TOP_N_COOCCURRING参数
if FOCUSED_FINETUNING:
    top_n_suffix = f"top{TOP_N_COOCCURRING}"
else:
    top_n_suffix = "no_focus"

FINETUNED_MODELS_OUTPUT_DIR = PROJECT_ROOT / "models" / "fine_tuned_vectors_flexible" / strategy_suffix / top_n_suffix # 微调后模型输出目录
DEBUG_VOCAB_OUTPUT_DIR = PROJECT_ROOT / "models" / "debug_vocabs" / strategy_suffix / top_n_suffix # 调试用词汇表输出目录



# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 辅助类和函数 ---

class EpochLogger(CallbackAny2Vec):
    """记录每个epoch结束时的loss。"""
    def __init__(self, period_name):
        self.epoch = 0
        self.period_name = period_name
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logging.info(f"时期 '{self.period_name}' - Epoch {self.epoch+1}/{model.epochs} 完成. Loss: {loss}")
        self.epoch += 1

class PeriodCorpusSentenceIterator:
    """时期语料句子的迭代器。"""
    def __init__(self, file_path):
        self.file_path = file_path
    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip().split()

def get_pretrained_vectors_info(kv_path: Path):
    """加载预训练词向量并获取信息。"""
    if not kv_path.exists():
        logging.error(f"预训练词向量未找到: {kv_path}")
        return None, 0
    try:
        kv = KeyedVectors.load(str(kv_path), mmap='r')
        vocab_size = len(kv.index_to_key)
        vector_dim = kv.vector_size
        logging.info(f"从 {kv_path} 加载预训练词向量. 词汇量: {vocab_size}, 维度: {vector_dim}")
        return kv, vector_dim
    except Exception as e:
        logging.error(f"从 {kv_path} 加载预训练词向量时出错: {e}")
        return None, 0

def save_vocab_to_file(vocab_set, filename, output_dir):
    """将词汇表集合 (set) 保存到文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        for word in sorted(list(vocab_set)):
            f.write(word + '\n')
    logging.info(f"词汇表已保存到: {filepath}")

def get_top_cooccurring_words(corpus_file, focus_word, top_n=5000, window_size=10):
    """计算与焦点词共现频率最高的N个词。"""
    logging.info(f"计算与'{focus_word}'在窗口 {window_size} 内共现频率最高的{top_n}个词...")
    cooccurrence_counts = {}
    focus_word_corpus_count = 0 # 焦点词在整个语料中出现的总次数 (用于日志)
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0 and line_num > 0: # 每处理10万行日志一次
                logging.info(f"已处理 {line_num} 行文本用于共现统计...")
            tokens = line.strip().split()
            if focus_word not in tokens:
                continue
            focus_word_corpus_count += tokens.count(focus_word)
            for i, token in enumerate(tokens):
                if token == focus_word:
                    start = max(0, i - window_size)
                    end = min(len(tokens), i + window_size + 1)
                    for j in range(start, end):
                        if j != i: # 排除焦点词自身
                            coword = tokens[j]
                            cooccurrence_counts[coword] = cooccurrence_counts.get(coword, 0) + 1
    sorted_cooccurring_words = sorted(cooccurrence_counts.items(), key=lambda x: x[1], reverse=True)
    top_words_set = {word for word, count in sorted_cooccurring_words[:top_n]}
    top_words_set.add(focus_word) # 确保焦点词本身在集合中
    logging.info(f"找到 {len(top_words_set)} 个与'{focus_word}'高频共现的词 (焦点词在语料中出现 {focus_word_corpus_count} 次)")
    top_20_for_log = sorted_cooccurring_words[:20] if len(sorted_cooccurring_words) >= 20 else sorted_cooccurring_words
    logging.info(f"前 {len(top_20_for_log)} 个最高频共现词 (词, 次数): {top_20_for_log}")
    return top_words_set

# --- 主要微调逻辑 ---
def main():
    FINETUNED_MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_VOCAB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"微调后的模型将保存到: {FINETUNED_MODELS_OUTPUT_DIR}")
    logging.info(f"调试用词汇表将保存到: {DEBUG_VOCAB_OUTPUT_DIR}")

    corpus_manager = CorpusManager()
    base_pretrained_kv, initial_vector_size = get_pretrained_vectors_info(PRETRAINED_VECTORS_PATH)
    if not base_pretrained_kv or initial_vector_size == 0:
        logging.error("无法加载基础预训练词向量，程序退出。")
        return
    
    current_base_kv_for_tuning = base_pretrained_kv # 用于增量微调时，追踪上一个时期的模型
    
    for period_config in PERIODS_TO_FINETUNE:
        period_name = period_config["name"]
        start_year = period_config["start_year"]
        end_year = period_config["end_year"]
        
        logging.info(f"\n========== 正在处理时期: {period_name} ({start_year}-{end_year}) ==========")

        try:
            period_corpus_file = corpus_manager.get_or_create_period_corpus(
                period_name=period_name, 
                start_year=start_year, 
                end_year=end_year,
                force_create=FORCE_CREATE_PERIOD_CORPORA
            )
            logging.info(f"使用时期语料文件: {period_corpus_file}")
        except FileNotFoundError as e:
            logging.error(f"无法处理时期 {period_name}: {e}。请确保年度数据存在且 preprocess_yearly_data.py 已运行。")
            continue 
        except Exception as e:
            logging.error(f"获取时期 {period_name} 的语料时发生未知错误: {e}。跳过此时期。")
            continue
            
        sentences = PeriodCorpusSentenceIterator(period_corpus_file)
        
        focused_vocab_set = None # 存储聚焦词汇的集合
        # 为 Word2Vec 的 min_count 参数确定一个有效值
        # 在聚焦微调模式下，我们希望 gensim 的 build_vocab 初步扫描时保留尽可能多的词（min_count=1），
        # 然后由我们的 trim_rule 进行更精细的筛选。
        # 在非聚焦模式下，直接使用全局的 MIN_COUNT。
        effective_model_init_min_count = MIN_COUNT 

        if FOCUSED_FINETUNING:
            focused_vocab_set = get_top_cooccurring_words(
                period_corpus_file, 
                FOCUS_WORD, 
                top_n=TOP_N_COOCCURRING,
                window_size=WINDOW_SIZE # 使用与模型训练相同的窗口大小计算共现
            )
            logging.info(f"聚焦微调模式：将主要针对与'{FOCUS_WORD}'相关的 {len(focused_vocab_set)} 个词进行处理。")
            save_vocab_to_file(focused_vocab_set, f"focused_vocab_set_{period_name}.txt", DEBUG_VOCAB_OUTPUT_DIR)
            effective_model_init_min_count = 1 # 聚焦微调时，让 build_vocab 先用 min_count=1 保留更多词给 trim_rule

        # 初始化 Word2Vec 模型
        model = Word2Vec(
            vector_size=initial_vector_size, # 使用预训练模型的维度
            window=WINDOW_SIZE,
            min_count=effective_model_init_min_count, # 使用上面确定的 min_count 初始化模型
            workers=WORKERS,
            sg=SG,
            epochs=EPOCHS, 
            callbacks=[EpochLogger(period_name)]
        )

        logging.info(f"正在为时期 '{period_name}' 构建词汇表 (模型初始化时 min_count={effective_model_init_min_count})...")
        
        # 定义 trim_rule (词汇修剪规则)
        trim_rule_to_apply = None
        if FOCUSED_FINETUNING and focused_vocab_set:
            # 这个 trim_rule 在 gensim 使用 effective_model_init_min_count (此处为1) 完成初步扫描后应用。
            # 它确保只有在 focused_vocab_set 中，并且实际词频达到全局 MIN_COUNT 的词才被保留。
            def focused_trim_rule(word, count, min_count_from_build_vocab): 
                # min_count_from_build_vocab 是 effective_model_init_min_count (即1)
                if word in focused_vocab_set and count >= MIN_COUNT: # 使用全局 MIN_COUNT 作为真实词频门槛
                    return RULE_KEEP # 保留此词
                return RULE_DISCARD # 丢弃此词
            trim_rule_to_apply = focused_trim_rule
        
        # 从语料构建模型词汇表，应用 trim_rule (如果已定义)
        model.build_vocab(
            sentences, # 在此扫描语料
            trim_rule=trim_rule_to_apply 
        )
        
        actual_model_vocab = set(model.wv.index_to_key) # 模型实际构建出的词汇表
        logging.info(f"时期 '{period_name}' 的词汇表构建完成。实际模型词汇量: {len(actual_model_vocab)}")
        save_vocab_to_file(actual_model_vocab, f"actual_model_vocab_{period_name}.txt", DEBUG_VOCAB_OUTPUT_DIR)

        # 如果模型词汇表为空，这是一个严重问题，记录错误并跳过此时期
        if not actual_model_vocab:
            logging.error(f"严重错误: 时期 '{period_name}' 的模型词汇表在 build_vocab 后为空！跳过此时期处理。")
            logging.error(f"  请检查语料文件: {period_corpus_file}")
            logging.error(f"  使用的全局 MIN_COUNT (用于聚焦微调时的真实频率过滤): {MIN_COUNT}")
            if FOCUSED_FINETUNING:
                logging.error(f"  聚焦词汇表大小: {len(focused_vocab_set) if focused_vocab_set else '未定义'}")
                logging.error(f"  模型初始化时 min_count 为 1, trim_rule 已应用。")
            else:
                logging.error(f"  模型初始化时 min_count 为 {MIN_COUNT}, 未应用 trim_rule。")
            continue # 跳到下一个时期

        # 获取用于权重初始化的源 KeyedVectors (预训练模型或上一个时期的模型)
        source_kv_for_init = base_pretrained_kv if not INCREMENTAL_FINETUNING else current_base_kv_for_tuning
        source_kv_name = "原始预训练模型" if not INCREMENTAL_FINETUNING or current_base_kv_for_tuning == base_pretrained_kv else "上一个时期的模型"
        
        # --- 记录词汇表相关统计信息，用于调试 ---
        logging.info(f"--- 时期 {period_name} 的词汇表统计信息 ---")
        logging.info(f"  实际模型词汇量: {len(actual_model_vocab)}")
        if FOCUSED_FINETUNING and focused_vocab_set:
            logging.info(f"  聚焦词汇表大小 (TopN + 焦点词): {len(focused_vocab_set)}")
            # 计算模型实际词汇表与聚焦词汇表的交集
            model_focused_intersect = actual_model_vocab.intersection(focused_vocab_set)
            logging.info(f"  交集 (实际模型词汇表 & 聚焦词汇表): {len(model_focused_intersect)}")
            save_vocab_to_file(model_focused_intersect, f"model_focused_intersection_{period_name}.txt", DEBUG_VOCAB_OUTPUT_DIR)
            if FOCUS_WORD in model_focused_intersect:
                logging.info(f"  成功: 焦点词 '{FOCUS_WORD}' 存在于 (实际模型 & 聚焦词汇表) 的交集中！")
            else:
                logging.warning(f"  警告: 焦点词 '{FOCUS_WORD}' 不在 (实际模型 & 聚焦词汇表) 的交集中。")
                if FOCUS_WORD in actual_model_vocab:
                    # 这通常不应该发生，如果发生了，说明 get_top_cooccurring_words 没有把 FOCUS_WORD 加到 focused_vocab_set
                    logging.warning(f"    '{FOCUS_WORD}' 在实际模型词汇表中，但不在聚焦词汇表中 (检查 get_top_cooccurring_words)。聚焦词汇表是否包含: {FOCUS_WORD in focused_vocab_set}")
                elif FOCUS_WORD in focused_vocab_set:
                    # 这意味着它在聚焦词汇表中，但没有通过 trim_rule (通常是因为 count < 全局 MIN_COUNT)
                    logging.warning(f"    '{FOCUS_WORD}' 在聚焦词汇表中，但不在实际模型词汇表中 (可能因为词频低于 MIN_COUNT={MIN_COUNT}而被 trim_rule 过滤)。")
                else:
                    logging.warning(f"    '{FOCUS_WORD}' 既不在实际模型词汇表中，也不在聚焦词汇表中。")
        
        source_vocab_set = set(source_kv_for_init.index_to_key) # 源预训练模型的词汇表
        logging.info(f"  源预训练模型词汇量: {len(source_vocab_set)}")
        logging.info(f"  焦点词 '{FOCUS_WORD}' 是否在实际模型词汇表中: {FOCUS_WORD in actual_model_vocab}")
        logging.info(f"  焦点词 '{FOCUS_WORD}' 是否在源预训练模型词汇表中: {FOCUS_WORD in source_vocab_set}")

        # 确定最终需要从源模型初始化向量的词集合
        words_to_init = set()
        if FOCUSED_FINETUNING and focused_vocab_set:
            # 聚焦模式下: (实际模型词汇表 INTERSECT 聚焦词汇表) INTERSECT 源预训练词汇表
            words_to_init = actual_model_vocab.intersection(focused_vocab_set).intersection(source_vocab_set)
            # 记录聚焦词汇表与源预训练词汇表的交集 (用于了解哪些聚焦词有预训练向量)
            focused_source_intersect = focused_vocab_set.intersection(source_vocab_set)
            logging.info(f"  交集 (聚焦词汇表 & 源预训练词汇表): {len(focused_source_intersect)}")
            save_vocab_to_file(focused_source_intersect, f"focused_source_intersection_{period_name}.txt", DEBUG_VOCAB_OUTPUT_DIR)
        else:
            # 非聚焦模式下: 实际模型词汇表 INTERSECT 源预训练词汇表
            words_to_init = actual_model_vocab.intersection(source_vocab_set)
        
        intersecting_words_count = len(words_to_init)
        logging.info(f"  将从源模型初始化向量的词数量: {intersecting_words_count}")
        save_vocab_to_file(words_to_init, f"words_to_init_{period_name}.txt", DEBUG_VOCAB_OUTPUT_DIR)
        logging.info(f"--- 结束时期 {period_name} 的词汇表统计信息 ---")

        # 如果在聚焦模式下，模型中有聚焦词，但这些词在源预训练模型中都找不到，发出警告
        if FOCUSED_FINETUNING and intersecting_words_count == 0 and actual_model_vocab.intersection(focused_vocab_set):
            num_focused_in_model = len(actual_model_vocab.intersection(focused_vocab_set))
            logging.warning(f"警告 '{period_name}': 模型中有 {num_focused_in_model} 个聚焦词，但它们均未在预训练向量中找到！这些词将从头开始学习。")
        
        logging.info(f"正在为时期 '{period_name}' 初始化模型权重 (来自 {source_kv_name})...")
        if intersecting_words_count > 0:
            # 高效复制词向量
            target_model_wv = model.wv
            source_model_kv = source_kv_for_init

            # 获取待复制词在目标模型和源模型中的索引列表
            # 使用列表推导式确保只处理存在于两个模型key_to_index中的词，尽管words_to_init已经是交集
            valid_words_to_init = [word for word in words_to_init if word in target_model_wv.key_to_index and word in source_model_kv.key_to_index]
            if len(valid_words_to_init) != intersecting_words_count:
                logging.warning(f"  初始化时发现 {intersecting_words_count - len(valid_words_to_init)} 个词在一方的key_to_index中缺失，实际初始化 {len(valid_words_to_init)} 个。")
            
            if valid_words_to_init: # 仅当有有效词时才进行复制
                target_indices = [target_model_wv.key_to_index[word] for word in valid_words_to_init]
                source_indices = [source_model_kv.key_to_index[word] for word in valid_words_to_init]

                # 使用NumPy切片直接高效地赋值向量
                target_model_wv.vectors[target_indices] = source_model_kv.vectors[source_indices]
                logging.info(f"已使用直接赋值方式从 {source_kv_name} 为 '{period_name}' 的新模型复制了 {len(valid_words_to_init)} 个词向量。")
            else:
                logging.info(f"  没有有效的词可以从 {source_kv_name} 初始化到 '{period_name}' (可能由于key_to_index不一致)。") 
        else:
            logging.info(f"没有词需要从 {source_kv_name} 为 '{period_name}' 初始化。")

        # 开始模型训练
        logging.info(f"开始为时期 '{period_name}' 进行微调，共 {model.epochs} 个训练轮次...")
        model.train(
            sentences, 
            total_examples=model.corpus_count, 
            epochs=model.epochs,
            start_alpha=model.alpha, # 使用模型当前的alpha值
            end_alpha=model.min_alpha # 使用模型当前的min_alpha值
        )
        logging.info(f"时期 '{period_name}' 的微调完成。")

        # 保存微调后的 KeyedVectors
        if FOCUSED_FINETUNING:
            fine_tuned_kv_path = FINETUNED_MODELS_OUTPUT_DIR / f"{period_name}_wordvectors_top{TOP_N_COOCCURRING}.kv"
        else:
            fine_tuned_kv_path = FINETUNED_MODELS_OUTPUT_DIR / f"{period_name}_wordvectors_no_focus.kv"
        model.wv.save(str(fine_tuned_kv_path))
        logging.info(f"时期 '{period_name}' 的微调后 KeyedVectors 已保存到: {fine_tuned_kv_path}")

        # 如果是增量微调，更新 current_base_kv_for_tuning 以便下一个时期使用
        if INCREMENTAL_FINETUNING:
            current_base_kv_for_tuning = model.wv 
            logging.info(f"时期 '{period_name}' 的模型将作为下一个时期 (如果存在) 的基础模型。词汇量: {len(current_base_kv_for_tuning.index_to_key)}")

    logging.info("\n========== 所有指定时期的微调已全部完成。 ==========")

    # --- 对本轮运行生成的所有微调模型进行基础测试 ---
    test_keywords = ["法治", "法制", "宪法", "改革", "开放", "民主", "社会主义", "市场经济"]
    logging.info(f"\n--- 使用关键词测试微调后的模型: {test_keywords} ---")
    for period_config in PERIODS_TO_FINETUNE:
        period_name = period_config["name"]
        if FOCUSED_FINETUNING:
            model_path = FINETUNED_MODELS_OUTPUT_DIR / f"{period_name}_wordvectors_top{TOP_N_COOCCURRING}.kv"
        else:
            model_path = FINETUNED_MODELS_OUTPUT_DIR / f"{period_name}_wordvectors_no_focus.kv"
        if model_path.exists():
            logging.info(f"\n--- 时期: {period_name} 的测试结果 ---")
            try:
                period_model_kv = KeyedVectors.load(str(model_path))
                for word in test_keywords:
                    if word in period_model_kv:
                        similar_words = period_model_kv.most_similar(word, topn=5)
                        logging.info(f"  与 '{word}' 最相似的词: {similar_words}")
                    else:
                        logging.info(f"  词 '{word}' 不在时期 {period_name} 的词汇表中。")
            except Exception as e:
                logging.error(f"测试时期 {period_name} 的模型时出错: {e}")
        else:
            # 如果某个时期的模型因为之前的错误（如语料获取失败）没有生成，这里会警告
            logging.warning(f"未找到时期 {period_name} 在 {model_path} 的微调模型用于测试。可能已被跳过。")

if __name__ == "__main__":
    # Before running, ensure:
    # 1. `src/data/by_year_preprocess.py` has been run to generate yearly corpus files.
    # 2. `PRETRAINED_VECTORS_PATH` points to your base `chinese_vectors.kv`.
    # 3. `PERIODS_TO_FINETUNE` list at the top of this script is configured as desired.
    main() 