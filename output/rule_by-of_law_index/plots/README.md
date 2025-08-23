# 多模型配置图表说明

## 图表结构

本目录包含了为不同模型配置生成的图表，每个模型配置都有独立的图表文件。

## 目录结构

```
plots/
├── exactly/                           # Exactly模式图表
│   ├── config_comparison_exactly.png  # 模型配置对比图
│   ├── flexible/                      # Flexible模型图表
│   │   ├── flexible_similarity_bar_chart_exactly.png      # 相似度柱状图
│   │   └── flexible_semantic_dynamism_trend_chart_exactly.png  # 语义动态指数趋势图
│   └── sliding_window/                # Sliding Window模型图表
│       ├── sliding_window_similarity_bar_chart_exactly.png      # 相似度柱状图
│       └── sliding_window_semantic_dynamism_trend_chart_exactly.png  # 语义动态指数趋势图
└── combined/                          # Combined模式图表
    ├── config_comparison_combined.png # 模型配置对比图
    ├── flexible/                      # Flexible模型图表
    │   ├── flexible_similarity_bar_chart_combined.png      # 相似度柱状图
    │   └── flexible_semantic_dynamism_trend_chart_combined.png  # 语义动态指数趋势图
    └── sliding_window/                # Sliding Window模型图表
        ├── sliding_window_similarity_bar_chart_combined.png      # 相似度柱状图
        └── sliding_window_semantic_dynamism_trend_chart_combined.png  # 语义动态指数趋势图
```

## 图表类型说明

### 1. 相似度柱状图 (Similarity Bar Chart)
- **功能**: 显示每个时期"法治"和"法制"与不同语义标签的相似度
- **特点**: 
  - 正向柱子：发展性语义(label_2)的相似度
  - 负向柱子：约束性语义(label_1)的相似度
  - 不同颜色区分法治和法制
- **时期覆盖**:
  - Flexible: 3个时期 (1978-1996, 1997-2013, 2014-2024)
  - Sliding Window: 8个时期 (1978-1988, 1983-1993, 1988-1998, 1993-2003, 1998-2008, 2003-2013, 2008-2018, 2013-2024)

### 2. 语义动态指数趋势图 (Semantic Dynamism Trend Chart)
- **功能**: 显示每个时期"法治"和"法制"的语义动态指数变化趋势
- **指标**: Gap Index = 发展性语义(label_2) - 约束性语义(label_1)
- **解读**:
  - 正值：倾向发展性语义
  - 负值：倾向约束性语义
  - 数值大小：语义变化的强度

### 3. 模型配置对比图 (Config Comparison Chart)
- **功能**: 在同一图表中对比不同模型配置的结果
- **布局**: 上下两个子图，分别显示"法治"和"法制"的对比
- **优势**: 直观比较不同时期划分方式对结果的影响

## 分析模式

### Exactly 模式
- 法治和法制分别使用自己的词包
- 每个时期使用对应的词语集合
- 适合分析目标词在不同时期的独立语义特征

### Combined 模式
- 法治和法制使用合并后的词包
- 对于sliding_window模型，智能分配词语集合
- 适合分析目标词在更广泛语义背景下的变化

## 图表特色

1. **自适应尺寸**: 根据时期数量自动调整图表宽度
2. **中文支持**: 完整的中文标题和标签
3. **时期标注**: 在标题中明确标注时期范围
4. **时间段标签**: X轴显示具体的时间段（如"1978-1988"）而非简单的"era1"
5. **数据标签**: 在趋势图中显示具体的数值
6. **网格线**: 便于读取数值的辅助线
7. **高分辨率**: 300 DPI输出，适合学术用途
8. **智能旋转**: 时期较多时自动旋转X轴标签以避免重叠

## 使用建议

1. **对比分析**: 使用config_comparison图表比较不同模型配置
2. **详细分析**: 使用单独的模型图表进行深入分析
3. **模式选择**: 根据研究目的选择exactly或combined模式
4. **时期精度**: 需要精细时间分辨率时使用sliding_window模型
5. **整体趋势**: 需要宏观视角时使用flexible模型

## 技术说明

- **图表库**: matplotlib
- **输出格式**: PNG (高分辨率)
- **字体设置**: 支持中文显示
- **颜色方案**: 区分不同语义类型和目标词
- **布局优化**: 自动调整以避免标签重叠
