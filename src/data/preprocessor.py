import jieba
import re
from typing import List, Set

class TextPreprocessor:
    def __init__(self, stopwords_path: str = None):
        """
        初始化文本预处理器
        Args:
            stopwords_path: 停用词文件路径
        """
        self.stopwords = self._load_stopwords(stopwords_path) if stopwords_path else set()
        
    def _load_stopwords(self, path: str) -> Set[str]:
        """加载停用词"""
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def segment(self, text: str) -> List[str]:
        """分词"""
        text = self.clean_text(text)
        words = jieba.lcut(text)
        # 移除停用词
        words = [w for w in words if w not in self.stopwords]
        return words 