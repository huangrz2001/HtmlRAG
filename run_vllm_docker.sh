#!/bin/bash

# ========== 配置参数 ==========
GPU_IDS="0,1"
PORT=8011
MEM_UTIL=0.7               # 显存使用率调整为 0.6
MAX_LEN=5120
TP_SIZE=2
SWAP_SPACE=1
MAX_SEQS=2

MODEL_PATH_HOST="/home/algo/qwen-14b-gptq-int4"   # 本地模型目录
MODEL_PATH_DOCKER="/model"                        # 容器内路径
CONTAINER_NAME="qwen"

# ========== 启动 Docker 容器 ==========
sudo docker run -it --rm \
  --name $CONTAINER_NAME \
  --runtime=nvidia \
  --gpus all \
  --shm-size 32g \
  -p ${PORT}:${PORT} \
  -v ${MODEL_PATH_HOST}:${MODEL_PATH_DOCKER} \
  -e "VLLM_SLEEP_WHEN_IDLE=1" \
  vllm/vllm-openai:v0.9.1 \
    --model ${MODEL_PATH_DOCKER} \
    --tokenizer ${MODEL_PATH_DOCKER} \
    --served-model-name "glm" \
    --trust-remote-code \
    --tokenizer-mode auto \
    --dtype auto \
    --quantization gptq \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --max-model-len ${MAX_LEN} \
    --max-num-seqs ${MAX_SEQS} \
    --enable-prefix-caching \
    --swap-space ${SWAP_SPACE} \
    --host 0.0.0.0 \
    --port ${PORT}
