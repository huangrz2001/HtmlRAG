import os
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from utils.db_utils import (
    reset_es, reset_milvus,
    insert_block_to_es, insert_block_to_milvus,
)

# 关闭 tokenizers 并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_all_json_files(json_root):
    """获取所有 block json 文件路径"""
    json_files = []
    for root, _, files in os.walk(json_root):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files


# ======================== 主程序入口 ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_dir", type=str, default="./总知识库_cleaned_block")
    # parser.add_argument("--block_dir", type=str, default="./测试知识库_cleaned_block")
    parser.add_argument("--lang", type=str, default="zh")
    # parser.add_argument("--embed_model", type=str, default="/home/algo/AD_agent/models/bce-embedding-base_v1")
    parser.add_argument("--embed_model", type=str, default="/home/algo/AD_agent/models/qwen3-embedding-0.6B")
    parser.add_argument("--max_node_words_embed", type=int, default=4096)
    parser.add_argument("--min_node_words_embed", type=int, default=48)
    parser.add_argument("--max_context_window_embed", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--env", type=str, default="dev")
    args = parser.parse_args()

    # 初始化模型
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    embedder = HuggingFaceEmbeddings(
        model_name=args.embed_model,
        model_kwargs={"device": device}
    )

    # 重建 ES 和 Milvus 索引
    reset_es(args.env)
    reset_milvus(args.env, dim=len(embedder.embed_query("你好")))

    # 遍历所有 JSON 文件进行构建
    block_dir = args.block_dir
    json_files = get_all_json_files(block_dir)
    print(f"📁 共发现 {len(json_files)} 个 JSON 文件待构建索引")

    cnt4Milvus = 0
    cnt4ES = 0
    for json_path in json_files:
        print(f"\n📄 文档块文件: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            doc_meta_list = json.load(f)
        if not len(doc_meta_list):
            print(f"⚠️ {json_path} 中没有有效的文档块，跳过")
            continue

        # print(doc_meta_list)
        cnt4Milvus += insert_block_to_milvus(doc_meta_list, embedder, args.env)
        cnt4ES += insert_block_to_es(doc_meta_list, args.env)



    print(f"\n✅ 所有文档块构建完成，Milvus总计插入 {cnt4Milvus} 条文档块，ES总计插入 {cnt4ES} 条文档块")

