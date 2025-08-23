# Rule by Law vs Rule of Law - 多模型配置分析器

## 功能概述

这个分析器能够同时支持多种词向量模型配置，对"法治"和"法制"的语义变化进行深入分析。

## 支持的模型配置

### 1. Flexible 配置 (3个时期)
- **era1**: 1978-1996
- **era2**: 1997-2013  
- **era3**: 2014-2024

### 2. Sliding Window 配置 (8个时期)
- **era1**: 1978-1988
- **era2**: 1983-1993
- **era3**: 1988-1998
- **era4**: 1993-2003
- **era5**: 1998-2008
- **era6**: 2003-2013
- **era7**: 2008-2018
- **era8**: 2013-2024

## 输出文件结构

```
output/rule_by-of_law_index/
├── word_sets.json                           # 词语集合
├── similarity_analysis_results.json         # 总体分析结果
├── flexible/                                # Flexible模型结果
│   ├── flexible_exactly_results.json       # Exactly模式结果
│   └── flexible_combined_results.json      # Combined模式结果
└── sliding_window/                          # Sliding Window模型结果
    ├── sliding_window_exactly_results.json  # Exactly模式结果
    └── sliding_window_combined_results.json # Combined模式结果
```

## 分析模式

### Exactly 模式
- 法治和法制分别使用自己的词包
- 每个时期使用对应的词语集合

### Combined 模式  
- 法治和法制使用合并后的词包
- 对于sliding_window模型，根据时期范围智能分配词语集合：
  - era1 (1978-1988): 使用era1的词语
  - era2 (1983-1993): 使用era1+era2的词语
  - era3 (1988-1998): 使用era2+era3的词语
  - era4-6 (1993-2013): 使用era2+era3的词语
  - era7-8 (2008-2024): 使用era3的词语

## 关键指标

### 相似度指标
- **avg_similarity**: 目标词与词语集合的平均余弦相似度
- **word_count**: 参与计算的词语数量

### 语义动态指数 (Gap Index)
- **公式**: 发展性语义(label_2) - 约束性语义(label_1)
- **正值**: 表示发展性语义更强
- **负值**: 表示约束性语义更强

## 使用示例

```python
from src.rule_by-of_law_analysis.rule_by_of_law_index_analyzer import RuleByOfLawIndexAnalyzer

# 定义模型配置
models_config = {
    "flexible": {...},
    "sliding_window": {...}
}

# 创建分析器
analyzer = RuleByOfLawIndexAnalyzer(excel_path, models_config, output_dir)

# 运行分析
analyzer.run_analysis()
```

## 优势特点

1. **多模型支持**: 同时分析不同时期划分方式下的语义变化
2. **智能词语分配**: 根据时期范围自动分配最合适的词语集合
3. **分层输出**: 为每个模型配置创建独立的输出目录和文件
4. **完整覆盖**: sliding_window模型提供8个时期的详细分析
5. **灵活配置**: 易于添加新的模型配置和时期划分

## 注意事项

- Excel文件需要包含6个sheet: 法治era1, 法治era2, 法治era3, 法制era1, 法制era2, 法制era3
- 每个sheet需要包含"word"和"class-label"列
- 确保所有词向量模型文件都存在且可访问
