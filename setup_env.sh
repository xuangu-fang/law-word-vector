#!/bin/bash

# download miniconda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install miniconda
# bash Miniconda3-latest-Linux-x86_64.sh

# 设置环境名称
ENV_NAME="law_word_vector"

# 检查是否已存在同名环境
if conda info --envs | grep -q $ENV_NAME; then
    echo "Environment $ENV_NAME already exists. Do you want to remove it? (y/n)"
    read answer
    if [ "$answer" == "y" ]; then
        conda env remove -n $ENV_NAME
    else
        echo "Exiting..."
        exit 1
    fi
fi

# 创建新的conda环境
echo "Creating new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# 激活环境
echo "Activating environment..."
source activate $ENV_NAME

# 安装PyTorch (特殊处理是因为需要从pytorch channel安装)
# conda install -y pytorch torchvision torchaudio -c pytorch

# 从requirements.txt安装其他依赖
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# 验证安装
echo "Verifying installation..."
python -c "import numpy; import pandas; import jieba; import gensim; import torch; import transformers; print('All packages imported successfully!')"

echo "Setup completed! You can now activate the environment using:"
echo "conda activate $ENV_NAME" 