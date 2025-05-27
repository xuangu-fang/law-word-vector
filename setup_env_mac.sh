#!/bin/bash

# 设置环境名称
ENV_NAME="law_word_vector"

# 检查是否已存在同名环境
if conda info --envs | grep -q $ENV_NAME; then
    echo "环境 $ENV_NAME 已存在。是否要删除它？(y/n)"
    read answer
    if [ "$answer" == "y" ]; then
        conda env remove -n $ENV_NAME
    else
        echo "退出..."
        exit 1
    fi
fi

# 检查是否安装了 Miniconda
if ! command -v conda &> /dev/null; then
    echo "未检测到 Miniconda，正在下载并安装..."
    # 下载 Miniconda for Mac
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    # 安装 Miniconda
    bash Miniconda3-latest-MacOSX-arm64.sh
    # 清理安装文件
    rm Miniconda3-latest-MacOSX-arm64.sh
    # 重新加载 shell 配置
    source ~/.zshrc
fi

# 创建新的conda环境
echo "正在创建新的 conda 环境: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# 激活环境
echo "正在激活环境..."
source activate $ENV_NAME

# 安装 PyTorch (Mac 版本)
echo "正在安装 PyTorch..."
conda install -y pytorch torchvision torchaudio -c pytorch

# 从 requirements.txt 安装其他依赖
echo "正在从 requirements.txt 安装其他依赖..."
pip install -r requirements.txt

# 验证安装
echo "正在验证安装..."
python -c "import numpy; import pandas; import jieba; import gensim; import torch; import transformers; print('所有包导入成功！')"

echo "环境配置完成！您可以使用以下命令激活环境："
echo "conda activate $ENV_NAME" 