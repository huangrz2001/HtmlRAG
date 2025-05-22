import os
import argparse
import torch
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


from utils.html_utils import build_block_tree
from utils.db_utils import insert_block_documents, query_milvus_blocks, query_milvus_blocks, Reranker
# 关闭 tokenizers 并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ======================== 主入口 ========================
# ======================== 主程序入口 ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--html_dir", type=str, default="./测试知识库_cleaned")
    parser.add_argument("--question", type=str, default="如何运营巨量千川平台")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--index_name", type=str, default="test_env")
    parser.add_argument("--embed_model", type=str, default="/data/huangruizhi/htmlRAG/bce-embedding-base_v1")
    parser.add_argument("--rerank_model", type=str, default="maidalun1020/bce-reranker-base_v1")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    # 设置设备
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"




    print("📦 加载 Embedder 模型...")
    embedder = HuggingFaceEmbeddings(model_name=args.embed_model,model_kwargs={"device": device})
    # 加载 reranker 模型（作为对象传入，不负责推理逻辑）
    print("📦 加载 Reranker 模型...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.rerank_model)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(args.rerank_model).to(device)
    reranker_model.eval()
    reranker = Reranker(reranker_model, reranker_tokenizer, device)

    

    # 无限循环，获取用户输入的问题进行检索
    while True:
        question = input("\n请输入查询问题（输入 exit 或 quit 退出）：\n>>> ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 已退出查询模式")
            break

        # query_milvus_blocks(
        #     question,
        #     embedder,
        #     milvus_collection_name=args.index_name,
        #     top_k=args.top_k,
        #     include_content=True,
        # )

        query_milvus_blocks(
            "192.168.7.247",
            question,
            embedder,
            reranker=reranker,
            milvus_collection_name=args.index_name,
            top_k=100,
            rerank_top_k=5
        )