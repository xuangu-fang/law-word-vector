# law-word-vector

法律语言学词向量分析项目

本项目基于人民日报语料库，通过词向量技术分析法律概念的语义演变，特别关注"法治"与"法制"概念的历史发展轨迹。

## 项目环境

**重要提示：** 请确保激活并使用 conda 环境 `law_word_vector` 运行所有代码。

## 目前主要结果：

1. 法治-法制 的词频变化: [link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/rule_by-of_law_freq/%E6%B3%95%E5%88%B6%E6%B3%95%E6%B2%BB%E8%AF%8D%E9%A2%91%E8%B6%8B%E5%8A%BF%E5%9B%BE_MA_len10_gap5.png)
2. 法治-法制 的相似度变化:

- 10年滑动窗口：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/rule_by-of_law_sim/%E6%B3%95%E5%88%B6_%E6%B3%95%E6%B2%BB_similarity_trend_Year1978-2024_10_5.png)
- 三个时期：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/rule_by-of_law_sim/%E6%B3%95%E5%88%B6_%E6%B3%95%E6%B2%BB_similarity_trend_fine_tuned_vectors_flexible.png)

3. 法治-法制 的 正负指数变化:
   -10年滑动窗口：
   - 发展-约束-柱状图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/rule_by-of_law_index/plots/combined/sliding_window/sliding_window_similarity_bar_chart_combined.png)
   - 语义动态指数（ 发展指数 减去 约束指数 ）趋势图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/rule_by-of_law_index/plots/combined/sliding_window/sliding_window_semantic_dynamism_trend_chart_combined.png)

- 三个时期：
  - 发展-约束-柱状图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/rule_by-of_law_index/combined/similarity_bar_chart_combined.png)
  - 语义动态指数（ 发展指数 减去 约束指数 ）趋势图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/rule_by-of_law_index/combined/semantic_dynamism_trend_chart_combined.png)

4. 内在价值-4维度 分析：

- 维度词表：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/inner_value/general_union_wordset_inner_value.json)
- 热力图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/inner_value/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/heatmap.png)
- 雷达图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/inner_value/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/radar_chart.png)
- 趋势图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/inner_value/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/trend_chart.png)
- 稳健性检验（滑动窗口）：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis_sensitive/inner_value/keywords_era1-%E6%B3%95%E5%88%B6_era2-%E6%B3%95%E5%88%B6_era3-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era4-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era5-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era6-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era7-%E6%B3%95%E6%B2%BB_era8-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/trend_chart.png)

5. 立法过程-4维度 分析：

- 维度词表：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/legal_process/general_union_wordset_legal_process.json)
- 热力图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/legal_process/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/heatmap.png)
- 雷达图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/legal_process/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/radar_chart.png)
- 趋势图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/legal_process/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/trend_chart.png)
- 稳健性检验（滑动窗口）：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis_sensitive/legal_process/keywords_era1-%E6%B3%95%E5%88%B6_era2-%E6%B3%95%E5%88%B6_era3-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era4-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era5-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era6-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era7-%E6%B3%95%E6%B2%BB_era8-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/heatmap.png)

6. 功能价值-4维度 分析：

- 维度词表：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/domain/general_union_wordset_domain.json)
- 热力图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/domain/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/heatmap.png)
- 雷达图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/domain/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/radar_chart.png)
- 趋势图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/domain/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/trend_chart.png)
- 稳健性检验（滑动窗口）：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis_sensitive/domain/keywords_era1-%E6%B3%95%E5%88%B6_era2-%E6%B3%95%E5%88%B6_era3-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era4-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era5-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era6-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era7-%E6%B3%95%E6%B2%BB_era8-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/trend_chart.png)

7. 法治-内在-外在 合计8维度分析

- 维度词表：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/combine_domain/general_union_wordset_combine_domain.json)
- 热力图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/combine_domain/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/heatmap.png)
- 雷达图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/combine_domain/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/radar_chart.png)
- 趋势图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/combine_domain/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/trend_chart.png)
- 稳健性检验（滑动窗口）：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis_sensitive/combine_domain/keywords_era1-%E6%B3%95%E5%88%B6_era2-%E6%B3%95%E5%88%B6_era3-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era4-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era5-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era6-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era7-%E6%B3%95%E6%B2%BB_era8-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/heatmap.png)

8. 最新- 治理与发展-权力制约-权利保障-正义理念 4维度分析

- 维度词表：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/law-function-new/general_union_wordset_law_function.json)
- 热力图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/law-function-new/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/heatmap.png)
- 雷达图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/law-function-new/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/radar_chart.png)
- 趋势图：[link](https://github.com/xuangu-fang/law-word-vector/blob/master/output/topic_analysis/law-function-new/keywords_era1-%E6%B3%95%E5%88%B6_era2-%5B%E6%B3%95%E5%88%B6%2B%E6%B3%95%E6%B2%BB%5D_era3-%E6%B3%95%E6%B2%BB-general_union-normalize_same_era/trend_chart.png)

  注意，,4，5，6, 7,8 的结果，是基于 “法制-era1, 法制+法治-era2， 法治-era3” 的设定计算得到的
