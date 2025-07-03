#!/bin/bash

# ========== 配置参数 ==========
GPU_IDS="0"
PORT=8011
MAX_LEN=5120
TP_SIZE=2

# MODEL_PATH_HOST="/home/algo/qwen-14b-gptq-int4"   # 本地模型目录
# MODEL_PATH_HOST="/home/algo/DeepSeek-Chat-7B"   # 本地模型目录
# MODEL_PATH_HOST="/home/zlsd/qwenawq"   # 本地模型目录
# MODEL_PATH_HOST="/home/algo/hrz/Qwen-7B-Chat"   # 本地模型目录
MODEL_PATH_HOST="/home/algo/Qwen3-30B-A3B-GPTQ-Int4"   # 本地模型目录
MODEL_PATH_DOCKER="/model"                        # 容器内路径
CONTAINER_NAME="qwen"


sudo docker run --rm --shm-size 32g --runtime nvidia \
 --gpus=1 \
 -v $MODEL_PATH_HOST:/models \
 -p 9000:8000 \
 -e "NCCL_DEBUG=INFO" \
 vllm/vllm-openai:v0.9.1 \
  --model /models \
  --served_model_name glm \
  --max_model_len 4096 \
  --trust-remote-code \
  --enable-prefix-caching \

