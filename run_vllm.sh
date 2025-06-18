#!/bin/bash

# ========== 配置参数 ==========
GPU_IDS="0,1"
PORT=8011
MEM_UTIL=0.7        # 显存使用率
MAX_LEN=4096
TP_SIZE=2
SWAP_SPACE=4
MAX_SEQS=4
BATCHED_TOKENS=8192

# ========== 启动 Qwen-7B-Chat ==========
CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m vllm.entrypoints.openai.api_server \
  --model /home/algo/DeepSeek-Chat-7B \
  --tokenizer /home/algo/DeepSeek-Chat-7B  \
  --served-model-name glm \
  --trust-remote-code \
  --tokenizer-mode auto \
  --dtype float16 \
  --tensor-parallel-size $TP_SIZE \
  --gpu-memory-utilization $MEM_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --max-num-batched-tokens $BATCHED_TOKENS \
  --enable-prefix-caching \
  --swap-space $SWAP_SPACE \
  --host 0.0.0.0 \
  --port $PORT

# # ========== 启动 ChatGLM-4-9B（如需切换请取消注释） ==========
# CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m vllm.entrypoints.openai.api_server \
#   --model THUDM/glm-4-9b-chat \
#   --tokenizer THUDM/glm-4-9b-chat \
#   --served-model-name glm \
#   --trust-remote-code \
#   --tokenizer-mode auto \
#   --dtype float16 \
#   --tensor-parallel-size $TP_SIZE \
#   --gpu-memory-utilization $MEM_UTIL \
#   --max-model-len $MAX_LEN \
#   --max-num-seqs $MAX_SEQS \
#   --max-num-batched-tokens $BATCHED_TOKENS \
#   --enable-prefix-caching \
#   --swap-space $SWAP_SPACE \
#   --host 0.0.0.0 \
#   --port $PORT
