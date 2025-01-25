# law-word-vector

project prosose:

## 目的
  具体来说，我想通过研究 不同时期的社会文献（比如80年代 和 90年代）中，特定词语（比如“法制”）的词嵌入 变化，最紧密关联词的变化等，来研究特定词语的 概念变迁。

## 技术步骤
  1. 获取不同时期的社会文献:
      - 人民日报
      - 政府工作报告
      - 裁判文书


  2. 对社会文献进行预处理:
      - unzip
      - 分词，除停用词、标点符号等
      - list of words of interest (手动列出/自动列出)

  3. 下载静态的中文词向量预训练模型，

  4. 在不同语料集上进行词向量微调
  - 或者采用其他动态词向量的分析方法
  
  5. 对词向量进行分析，比如计算特定词语的词嵌入变化，最紧密关联词的变化等

  6. 对分析结果进行可视化，比如使用matplotlib、seaborn等

  
  ## to-do list:
  - make a requirements.txt, and install all the dependencies
  - process the data
  - define a project structure, include data-class, model-class, analysis-class
  - start from simple case, like a notebook