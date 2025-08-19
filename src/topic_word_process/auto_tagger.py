#!/usr/-bin/env python3
# -*- coding: utf-8 -*-
"""
自动为词表进行多维度标签的脚本
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import yaml
from gensim.models import KeyedVectors

class BaseTagger:
    """标签器基类，为不同的标签方法提供统一接口"""
    def tag(self, word: str, tag_options: List[str]) -> str:
        raise NotImplementedError

    def batch_tag(self, words: List[str], tag_options: List[str]) -> List[str]:
        return [self.tag(word, tag_options) for word in words]

class SimilarityTagger(BaseTagger):
    """
    基于词向量相似度的标签器
    """
    def __init__(self, model: KeyedVectors):
        self.model = model

    def tag(self, word: str, tag_options: List[str]) -> str:
        """
        为单个词语在给定的标签选项中选择最相似的一个
        """
        if word not in self.model.key_to_index:
            return np.nan

        max_similarity = -1.0
        best_tag = np.nan

        for tag in tag_options:
            if tag in self.model.key_to_index:
                similarity = self.model.similarity(word, tag)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_tag = tag
        
        return best_tag

class AutoTaggerApp:
    """
    自动标签应用的主控制器
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.tagger = self._initialize_tagger()

    def _load_config(self, path: str) -> Dict[str, Any]:
        print(f"Loading configuration from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _initialize_tagger(self) -> BaseTagger:
        strategy = self.config.get('tagging_strategy', 'similarity')
        print(f"Initializing tagger with strategy: {strategy}")
        if strategy == 'similarity':
            model_path = self.config['word_vector_model_path']
            print(f"Loading word vector model from: {model_path}")
            model = KeyedVectors.load(str(model_path))
            return SimilarityTagger(model)
        # elif strategy == 'llm':
        #     # 预留LLM Tagger的接口
        #     # from .llm_tagger import LLMTagger
        #     # return LLMTagger(self.config['model_params'])
        else:
            raise ValueError(f"Unsupported tagging strategy: {strategy}")

    def load_word_list(self) -> pd.DataFrame:
        input_conf = self.config['input_file']
        print(f"Loading word list from: {input_conf['path']}")
        df = pd.read_excel(
            input_conf['path'], 
            sheet_name=input_conf['sheet_name']
        )
        
        word_col = input_conf['column_name']
        columns_to_keep = [word_col]
        
        # 检查 "similarity" 列是否存在，如果存在则一并读取
        if 'similarity' in df.columns:
            columns_to_keep.append('similarity')
            print("Found 'similarity' column, it will be included in the output.")
        else:
            print("Warning: 'similarity' column not found in the input file. It will not be included.")
            
        return df[columns_to_keep].rename(columns={word_col: 'word'})

    def load_tag_definitions(self) -> Dict[str, List[str]]:
        path = self.config['tag_definitions_path']
        print(f"Loading tag definitions from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run(self):
        """执行自动标签流程"""
        df = self.load_word_list()
        tag_definitions = self.load_tag_definitions()
        
        words_to_tag = df['word'].tolist()

        for tag_category, tag_options in tag_definitions.items():
            print(f"Tagging for category: '{tag_category}'...")
            df[tag_category] = self.tagger.batch_tag(words_to_tag, tag_options)
        
        self._save_results(df)

    def _save_results(self, df: pd.DataFrame):
        output_conf = self.config['output_file']
        
        # 从模型路径中提取基础名称作为子目录
        model_path = Path(self.config['word_vector_model_path'])
        model_name = model_path.stem.replace('_wordvectors', '') # 移除后缀
        
        output_dir = Path(output_conf['directory']) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = output_dir / output_conf['filename_prefix']
        
        if 'csv' in output_conf.get('formats', []):
            csv_path = base_filename.with_suffix('.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Results saved to: {csv_path}")

        if 'xlsx' in output_conf.get('formats', []):
            xlsx_path = base_filename.with_suffix('.xlsx')
            df.to_excel(xlsx_path, index=False)
            print(f"Results saved to: {xlsx_path}")

def main():
    parser = argparse.ArgumentParser(description="Automated Word Tagger")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/tagging_config.yaml',
        help='Path to the configuration YAML file.'
    )
    args = parser.parse_args()
    
    app = AutoTaggerApp(args.config)
    app.run()

if __name__ == "__main__":
    main() 