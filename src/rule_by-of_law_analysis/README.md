# Rule by Law vs Rule of Law Analysis Suite

这个目录包含了"法制"与"法治"的全面分析工具套件。

## 📁 脚本概览

### 🔢 频率分析 (Frequency Analysis)
**输出目录:** `output/rule_by-of_law_freq/`

- **`rule_by_of_law_freq_analyzer.py`** - 主要频率分析器
  - 统计"法制"和"法治"两个词的年度词频
  - 计算每百万词中的出现次数
  - 绘制词频趋势图
  - 生成统计报告和数据文件

- **`rule_by_of_law_freq_replot.py`** - 频率趋势重绘工具
  - 重新绘制词频趋势图，支持移动平均线
  - 解决中文显示问题
  - 自定义窗口长度和间隔

### 📊 相似度分析 (Similarity Analysis)
**输出目录:** `output/rule_by-of_law_sim/`

- **`rule_by_of_law_similarity_analyzer.py`** - 相似度分析器
  - 计算"法制"和"法治"在不同时期词向量模型中的余弦相似度
  - 绘制相似度变化趋势图
  - 支持多种词向量模型

### 📈 指数分析 (Index Analysis)
**输出目录:** `output/rule_by-of_law_index/`

- **`rule_by_of_law_index_analyzer.py`** - 指数分析器
  - 读取专家标注的词包数据
  - 计算"法治"和"法制"与特定词语集合的平均余弦相似度
  - 计算gap指数（语义动态性指数）
  - 生成结构化的JSON结果

- **`rule_by_of_law_plotter.py`** - 绘图工具
  - 读取指数分析结果
  - 生成柱状图和趋势图
  - 支持exactly和combined两种设置模式

## 🚀 使用示例

```bash
# 频率分析
python src/rule_by-of_law_analysis/rule_by_of_law_freq_analyzer.py

# 频率趋势重绘（带移动平均线）
python src/rule_by-of_law_analysis/rule_by_of_law_freq_replot.py --moving-average --window-length 10 --window-gap 5

# 相似度分析
python src/rule_by-of_law_analysis/rule_by_of_law_similarity_analyzer.py --model-dir models/fine_tuned_vectors_flexible

# 指数分析
python src/rule_by-of_law_analysis/rule_by_of_law_index_analyzer.py

# 绘制指数图表
python src/rule_by-of_law_analysis/rule_by_of_law_plotter.py
```

## 📂 输出文件结构

```
output/
├── rule_by-of_law_freq/          # 频率分析结果
│   ├── 法制法治词频统计.json
│   ├── 法制法治词频统计.csv
│   ├── 法制法治词频统计报告.txt
│   └── 法制法治词频趋势图*.png
├── rule_by-of_law_sim/           # 相似度分析结果
│   └── 法制_法治_similarity_trend*.png
└── rule_by-of_law_index/         # 指数分析结果
    ├── similarity_analysis_results.json
    ├── word_sets.json
    ├── combined/
    │   ├── similarity_bar_chart_combined.png
    │   └── semantic_dynamism_trend_chart_combined.png
    └── exactly/
        ├── similarity_bar_chart_exactly.png
        └── semantic_dynamism_trend_chart_exactly.png
```

## 🔧 依赖要求

- Python 3.7+
- pandas
- matplotlib
- gensim
- numpy
- openpyxl (用于读取Excel文件)

## 📝 说明

所有脚本都支持中文显示，并自动处理字体问题。输出文件会按功能分类保存到对应的目录中，避免文件混乱。