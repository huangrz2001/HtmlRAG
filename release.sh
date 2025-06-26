#!/bin/bash

set -e

# 💡 配置
RELEASE_BRANCH="release/1.1.2"
SOURCE_BRANCH="master"

# Step 1. 确保工作区干净
git stash -u

# Step 2. 切到 master，创建 release 分支
git checkout "$SOURCE_BRANCH"
git pull origin "$SOURCE_BRANCH"  # 可选
git checkout -b "$RELEASE_BRANCH"

# Step 3. 恢复工作区修改
git stash pop

# Step 4. 移除不需要的目录和文件（根据你框出的内容）

echo "🧹 删除不需要的内容..."
rm -rf __pycache__ \
       测试知识库* \
       总知识库* \
       logs \
       archive \
       finetune\\trained_reranker_single\\ \
       finetune\\wandb\\ \
       run_vllm_back.sh \
       test_insert_api.py \
       test_rewrite_api.py \
       test_rewrite_vllm.py \
       test.py \
       nohup.out \
       *test* \
       *.yaml \
       *.log \
       *.json \
       *.jsonl \
       *.txt \
       *.dic \

# Step 5. 查看剩余内容
echo "📂 当前保留文件为："
git status

# Step 6. 重新提交为新的 commit
git add .
git commit -m "Release 1.1.1: 精简版本，仅保留核心代码文件"

# Step 7. 可选：推送到远程
# git push origin "$RELEASE_BRANCH"

echo "✅ Release 分支准备完成: $RELEASE_BRANCH"
