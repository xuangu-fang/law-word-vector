#!/usr/bin/env python
# coding: utf-8

"""
Data preprocessing script for the People's Daily corpus.
Processes raw data on a YEAR-BY-YEAR basis from 1949 to 2024 (configurable).

This script performs the following for each year:
1. Loads raw data from the Excel file for that year.
2. Cleans text: removes irrelevant symbols, extra spaces.
3. Performs Chinese word segmentation using jieba, with added custom 
   legal and political vocabulary.
4. Filters out stopwords and very short words.
5. Saves the processed corpus for the year into a separate .txt file 
   (e.g., processed_data/yearly_corpus/1949.txt).
6. Generates and saves basic statistics about the processed yearly corpus.
"""

import pandas as pd
import numpy as np
import os
import re
import jieba
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# --- Configuration ---
DATA_ROOT = Path.home() / "data" / "rmrb-year" / "分年份保存数据"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
YEARLY_PROCESSED_DIR = PROJECT_ROOT / "processed_data" / "yearly_corpus"
STATS_DIR = PROJECT_ROOT / "processed_data" / "stats"

# Define the range of years to process
START_YEAR_CONFIG = 1978
END_YEAR_CONFIG = 2024 # Adjust as needed

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Stopwords Set ---
stopwords = set()

# --- Helper Functions & Setup ---

def setup_environment():
    """Creates output directories and configures jieba."""
    YEARLY_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Yearly processed data will be saved to: {YEARLY_PROCESSED_DIR}")
    logging.info(f"Statistics will be saved to: {STATS_DIR}")

    core_legal_political_terms = [
        "法治", "法制", "依法治国", "宪法", "法律制度", "司法公正", "司法独立", 
        "法治国家", "法治政府", "法治社会", "权利保障", "程序正义", "实质正义",
        "人权保障", "公民权利", "法律监督", "违宪审查", "司法改革", "法治建设",
        "法治化", "法制化", "依宪治国", "依法行政", "人治", "法治思维", "法治理念",
        "司法公开", "阳光司法", "司法为民", "司法责任制", "司法权力", "司法权威",
        "党的领导", "政治协商", "民主党派", "多党合作", "人民代表大会", "政治体制改革",
        "国家治理", "治理能力", "治理体系", "群众路线", "统一战线", "阶级斗争",
        "无产阶级专政", "社会主义改造", "大跃进", "人民公社", "文化大革命", "拨乱反正", "改革开放",
        "社会主义", "社会主义初级阶段", "社会主义市场经济", "社会主义法治国家", "社会主义法治体系", "社会主义法治理论",
    ]
    for term in core_legal_political_terms:
        jieba.add_word(term, freq=10000, tag='nz') # High frequency and custom tag
    logging.info(f"Added {len(core_legal_political_terms)} custom terms to jieba dictionary.")

    global stopwords
    stopwords_list = [
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '或', '个', '也', '不', '之', '其', '此', '彼', '斯', '者', '所', 
        '自', '从', '给', '向', '以', '在', '对', '但', '使', '让', '被', '将', '把', '着', '等', '很', '最', '更', '又', '再', 
        '于', '由', '为', '则', '因', '至', '而', '然', '且', '虽', '乎', '哉', '矣', '也', '兮', '曰', '云', '一些', '一种', 
        '一样', '许多', '这样', '那样', '任何', '所有', '我们', '你们', '他们', '自己', '大家', '以及', '还有', '例如', 
        '根据', '对于', '关于', '通过', '作为', '为了', '得到', '进行', '实现', '发生', '发展', '问题', '情况', '方面', 
        '过程', '时候', '地方', '人们', '同志', '先生', '女士', 
        # '年', '月', '日', # Keep date related words if they are part of analysis
        '时', '分', '秒', '元', '角', '亿', '万', '千', '百', '十', 
        '〇', '一', '二', '三', '四', '五', '六', '七', '八', '九', '零' 
    ]
    stopwords = set(stopwords_list)
    logging.info(f"Using {len(stopwords)} stopwords.")

def load_year_data(year: int) -> pd.DataFrame:
    filename = f"人民日报{year}年文本数据.xlsx"
    file_path = DATA_ROOT / filename
    if file_path.exists():
        try:
            df = pd.read_excel(file_path)
            if '标题' not in df.columns: df['标题'] = ""
            if '文本内容' not in df.columns: df['文本内容'] = ""
            return df[['标题', '文本内容']].fillna('')
        except Exception as e:
            logging.error(f"Error loading or processing {file_path}: {e}")
            return pd.DataFrame(columns=['标题', '文本内容'])
    else:
        logging.warning(f"File not found for year {year}: {file_path}")
        return pd.DataFrame(columns=['标题', '文本内容'])

def preprocess_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    words = jieba.lcut(text)
    processed_words = []
    for word in words:
        if word not in stopwords and len(word) > 0: # Allow single char words if not stopwords
             # Further filter to keep primarily Chinese words, can be adjusted
            if re.fullmatch(r'[\u4e00-\u9fa5]+', word) or word in jieba.dt.FREQ: # Keep custom dictionary words
                if len(word) > 1 or word in jieba.dt.FREQ: # Keep single char custom words
                     processed_words.append(word)
    return processed_words

# --- Main Processing Logic ---
def main():
    setup_environment()
    
    yearly_stats_summary = []

    for year_to_process in range(START_YEAR_CONFIG, END_YEAR_CONFIG + 1):
        logging.info(f"\n--- Processing year: {year_to_process} ---")
        year_corpus_lines = []
        total_docs_in_year_raw = 0
        processed_docs_in_year = 0
        
        df_year = load_year_data(year_to_process)
        if df_year.empty:
            logging.warning(f"No data found for year {year_to_process}. Skipping.")
            yearly_stats_summary.append({
                "Year": year_to_process,
                "Original Documents (Raw)": 0,
                "Processed Documents (Non-empty)": 0,
                "Total Tokens": 0,
                "Avg Tokens/Doc": 0,
                "Vocabulary Size": 0
            })
            continue # Skip to next year if no data
            
        df_year['full_text'] = df_year['标题'].astype(str) + " " + df_year['文本内容'].astype(str)
        total_docs_in_year_raw = len(df_year)
        
        for text_content in tqdm(df_year['full_text'], desc=f"Docs in {year_to_process}"):
            processed_tokens = preprocess_text(text_content)
            if processed_tokens:
                year_corpus_lines.append(" ".join(processed_tokens))
                processed_docs_in_year += 1
                    
        output_file_path = YEARLY_PROCESSED_DIR / f"{year_to_process}.txt"
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in year_corpus_lines:
                f.write(line + '\n')
        logging.info(f"Year {year_to_process}: {processed_docs_in_year} documents saved to {output_file_path}")
        logging.info(f"Originally {total_docs_in_year_raw} documents, after processing {processed_docs_in_year} remain.")
        
        if processed_docs_in_year > 0:
            all_tokens_in_year = [token for doc_line in year_corpus_lines for token in doc_line.split()]
            total_tokens = len(all_tokens_in_year)
            avg_tokens_per_doc = total_tokens / processed_docs_in_year
            vocabulary_size = len(set(all_tokens_in_year))
        else:
            total_tokens = 0
            avg_tokens_per_doc = 0
            vocabulary_size = 0
            
        yearly_stats_summary.append({
            "Year": year_to_process,
            "Original Documents (Raw)": total_docs_in_year_raw,
            "Processed Documents (Non-empty)": processed_docs_in_year,
            "Total Tokens": total_tokens,
            "Avg Tokens/Doc": round(avg_tokens_per_doc, 2),
            "Vocabulary Size": vocabulary_size
        })

    logging.info("\nAll configured years processed.")

    stats_df = pd.DataFrame(yearly_stats_summary)
    stats_df.set_index("Year", inplace=True)
    stats_file_path = STATS_DIR / f"yearly_corpus_stats_{START_YEAR_CONFIG}-{END_YEAR_CONFIG}.csv"
    stats_df.to_csv(stats_file_path)
    logging.info(f"\nYearly Corpus Statistics ({START_YEAR_CONFIG}-{END_YEAR_CONFIG}) saved to {stats_file_path}")
    print(f"\nYearly Corpus Statistics ({START_YEAR_CONFIG}-{END_YEAR_CONFIG}):")
    print(stats_df.head())

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False
        if not stats_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle(f'Yearly Corpus Statistics ({START_YEAR_CONFIG}-{END_YEAR_CONFIG})', fontsize=16)

            stats_df['Processed Documents (Non-empty)'].plot(kind='line', ax=axes[0,0], title='Processed Documents Over Years', marker='o')
            stats_df['Total Tokens'].plot(kind='line', ax=axes[0,1], title='Total Tokens Over Years', marker='o')
            stats_df['Avg Tokens/Doc'].plot(kind='line', ax=axes[1,0], title='Avg Tokens/Document Over Years', marker='o')
            stats_df['Vocabulary Size'].plot(kind='line', ax=axes[1,1], title='Vocabulary Size Over Years', marker='o')
            
            for ax_row in axes:
                for ax in ax_row:
                    ax.grid(True)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plot_file_path = STATS_DIR / f"yearly_corpus_stats_{START_YEAR_CONFIG}-{END_YEAR_CONFIG}_plot.png"
            plt.savefig(plot_file_path)
            logging.info(f"Statistics plot saved to {plot_file_path}")
        else:
            logging.info("Statistics DataFrame is empty. No plot generated.")
    except Exception as e:
        logging.warning(f"Could not generate plots: {e}. Matplotlib or a Chinese font might be missing or stats_df is empty.")

if __name__ == "__main__":
    main() 