import os
import argparse
import torch
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel

from utils.html_utils import build_block_tree
from utils.db_utils import insert_block_documents, query_block_rankings, reset_es, reset_milvus

# 关闭 tokenizers 并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================== HTML 文件处理函数 ========================


def process_html_file(html_path,
                      args,
                      embedder,
                      summary_model,
                      summary_tokenizer,
                      index_name="jvliangqianchuan",
                      insert_num=0):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    print(f"\n=== Processing: {html_path} ===")

    # 提取起始 <time> 标签（如存在）
    time_pattern = r"^\s*<time[^>]*?>(.*?)</time>"
    time_match = re.match(time_pattern, html, flags=re.IGNORECASE | re.DOTALL)

    time_value = ""
    if time_match:
        time_value = time_match.group(1).strip()
        html = html[time_match.end():].lstrip()

    # 清洗并构建结构树
    prune_zh = args.lang == "zh"
    simplified_html = html
    block_tree, simplified_html = build_block_tree(
        simplified_html,
        max_node_words=args.max_node_words_embed,
        min_node_words=args.min_node_words_embed,
        zh_char=prune_zh,
    )

    return insert_block_documents(
        block_tree,
        embedder,
        collection_name=args.index_name,
        page_url=os.path.relpath(html_path),
        insert_num=insert_num,
        summary_model=summary_model,
        summary_tokenizer=summary_tokenizer,
        time_value=time_value,  # 👈 传入提取的 time
    )



def get_all_html_files(html_dir):
    html_files = []
    for root, _, files in os.walk(html_dir):
        for file in files:
            if file.lower().endswith(".html"):
                html_files.append(os.path.join(root, file))
    return html_files


# ======================== 主入口 ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--html_dir", type=str, default="./巨量千川知识库_cleaned/粤理知识库")
    parser.add_argument("--html_dir", type=str, default="./测试")
    # parser.add_argument("--html_dir", type=str, default="./巨量千川知识库_cleaned")
    parser.add_argument("--question", type=str, default="如何运营巨量千川平台")
    parser.add_argument("--lang", type=str, default="zh")
    # parser.add_argument("--index_name", type=str, default="jvliangqianchuan")
    parser.add_argument("--index_name", type=str, default="test_env")
    # parser.add_argument("--embed_model", type=str, default="./bge-m3-local")
    parser.add_argument("--embed_model",type=str,default="../htmlRAG/bce-embedding-base_v1")
    parser.add_argument("--summary_tokenizer", type=str, default="../htmlRAG/chatglm3-6b")
    parser.add_argument("--summary_model", type=str, default="../htmlRAG/chatglm3-6b")
    parser.add_argument("--max_node_words_embed", type=int, default=1024)
    parser.add_argument("--min_node_words_embed", type=int, default=48)
    parser.add_argument("--max_context_window_embed", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedder = HuggingFaceEmbeddings(model_name=args.embed_model,model_kwargs={"device": device})
    summary_tokenizer = AutoTokenizer.from_pretrained(args.summary_tokenizer, trust_remote_code=True)
    summary_model = AutoModel.from_pretrained(args.summary_model, trust_remote_code=True).half().cuda()

    summary_model.eval()

    # 重置 ES 和 Milvus 索引
    reset_es(args)
    reset_milvus(args.index_name, dim=len(embedder.embed_query("0")))

    insert_num = 0
    # 处理目录中所有 HTML 文件
    html_files = get_all_html_files(args.html_dir)
    print(f"📄 共发现 HTML 文件数: {len(html_files)}")
    for html_file in html_files:
        insert_num += process_html_file(html_file, args, embedder,summary_model,summary_tokenizer, args.index_name, insert_num)
    print(f"✅ 成功插入文档块总数: {insert_num}")

    # 无限循环，获取用户输入的问题进行检索
    while True:
        question = input("\n请输入查询问题（输入 exit 或 quit 退出）：\n>>> ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 已退出查询模式")
            break

        query_block_rankings(
            question,
            embedder,
            es_index_name=args.index_name,
            milvus_collection_name=args.index_name,
            top_k=args.top_k,
            include_content=True,
        )
