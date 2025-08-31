# Topic Word Analysis Suite

这个目录包含了主题词分析的完整工具套件，用于分析"法治"与"法制"在不同主题维度上的语义相似度。

## 📁 脚本概览

### 📊 数据处理器 (Data Processing)
**文件:** `topic_data_processor.py`

- **功能**: 读取Excel文件中的专家标注数据，转换为结构化JSON
- **支持**: 多种topic类型（法律功能、法律流程等）
- **输出**: 结构化的JSON数据文件

### 🏛️ 法律功能分析器 (Law Function Analysis)
**文件:** `law_function_analysis.py`  
**输出目录:** `output/topic_analysis/law-function/`

- **分析维度**: 发展、秩序、规范、权力限制
- **功能**: 计算"法治"/"法制"与各法律功能维度的相似度
- **特点**: 使用General Union模式确保词包一致性

### ⚖️ 法律流程分析器 (Legal Process Analysis)
**文件:** `legal_process_analysis.py`  
**输出目录:** `output/topic_analysis/legal_process/`

- **分析维度**: 立法、司法、执法、守法
- **功能**: 计算"法治"/"法制"与各法律流程维度的相似度
- **特点**: 支持多种era-keyword组合和归一化模式

### 🔧 通用维度分析器 (Generic Dimension Analysis)
**文件:** `dimension_analysis.py`  
**输出目录:** `output/topic_analysis/`

- **功能**: 通用的topic相似度分析框架
- **特点**: 灵活配置、多种分析模式、可扩展

## 🎯 核心功能

### 📈 分析模式
- **Union模式**: 跨时期词包并集
- **General Union模式**: 跨关键词+跨时期完全并集
- **混合模式**: 支持era内多关键词组合（如era2-[法制+法治]）

### 📊 归一化选项
- **Same Era**: 同一era内各topic相似度归一化
- **Cross Era**: 跨era标准化
- **无归一化**: 原始相似度值

### 📋 Era-Keyword配置
```python
# 基础设置
basic_keywords = {
    'era1': '法制',
    'era2': '法治',
    'era3': '法治'
}

# 混合模式
mixed_keywords = {
    'era1': '法制',
    'era2': ['法制', '法治'],  # 平均两个关键词的相似度
    'era3': '法治'
}
```

### 📊 可视化输出
- **雷达图**: 多维度相似度对比
- **趋势图**: 时间序列变化趋势
- **热力图**: era×topic相似度矩阵（era在x轴）

## 🚀 使用示例

### 数据处理
```bash
python src/topic_word_analysis/topic_data_processor.py
```

### 法律功能分析
```bash
python src/topic_word_analysis/law_function_analysis.py
```

### 法律流程分析
```bash
python src/topic_word_analysis/legal_process_analysis.py
```

### 通用分析（演示）
```bash
python src/topic_word_analysis/dimension_analysis.py
```

## 📂 输出文件结构

```
output/topic_analysis/
├── law-function/                    # 法律功能分析
│   ├── topic_word_sets_law_function.json
│   ├── general_union_wordset_law_function.json
│   └── keywords_[配置]/
│       ├── radar_chart.png
│       ├── trend_chart.png
│       └── heatmap.png
└── legal_process/                   # 法律流程分析
    ├── topic_word_sets_legal_process.json
    ├── general_union_wordset_legal_process.json
    └── keywords_[配置]/
        ├── radar_chart.png
        ├── trend_chart.png
        └── heatmap.png
```

## 🔧 技术特性

### 🎯 智能路径管理
- 自动生成描述性输出目录名称
- 基于分析配置的动态路径构建
- 避免不同设置的结果冲突

### 📊 数据处理
- 自动创建General Union词包
- 支持词向量模型热加载
- 智能错误处理和调试信息

### 🖼️ 可视化优化
- 中文字体自动配置
- 高质量图表输出（300 DPI）
- Era在x轴的热力图布局

## 🔧 依赖要求

- Python 3.7+
- pandas, numpy
- matplotlib, seaborn
- gensim (词向量模型)
- openpyxl (Excel文件读取)

## 📝 说明

所有分析器都继承自相同的核心框架，确保结果的一致性和可比性。使用General Union模式可以确保跨时期分析的词包一致性，提高分析结果的可靠性。