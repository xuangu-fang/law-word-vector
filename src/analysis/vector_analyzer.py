import numpy as np
from typing import List, Dict, Tuple
from gensim.models import KeyedVectors

class VectorAnalyzer:
    def __init__(self, model_path: str = None):
        """
        初始化词向量分析器
        Args:
            model_path: 预训练词向量模型路径
        """
        self.model = KeyedVectors.load_word2vec_format(model_path) if model_path else None
        
    def find_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """查找最相似的词"""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.most_similar(word, topn=topn)
    
    def calculate_word_similarity(self, word1: str, word2: str) -> float:
        """计算两个词的相似度"""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.similarity(word1, word2) 