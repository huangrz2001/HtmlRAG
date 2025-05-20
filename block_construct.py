import os
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings

from utils.html_utils import build_block_tree
from utils.text_process_utils import (
    generate_block_documents,
    save_doc_meta_to_block_dir,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================== HTML 文件处理函数 ========================

def process_html_file(html_path, args, summary_model, summary_tokenizer):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    print(f"\n=== Processing: {html_path} ===")

    # 提取起始 <time> 标签
    time_pattern = r"^\s*<time[^>]*?>(.*?)</time>"
    time_match = re.match(time_pattern, html, flags=re.IGNORECASE | re.DOTALL)

    time_value = ""
    if time_match:
        time_value = time_match.group(1).strip()
        html = html[time_match.end():].lstrip()

    # 清洗并构建结构树
    prune_zh = args.lang == "zh"
    block_tree, _ = build_block_tree(
        html,
        max_node_words=args.max_node_words_embed,
        min_node_words=args.min_node_words_embed,
        zh_char=prune_zh,
    )

    # 构建文档块元数据（摘要可选）
    doc_meta = generate_block_documents(
        block_tree,
        page_url=os.path.relpath(html_path),
        summary_model=summary_model,  # 可切换为 summary_model
        summary_tokenizer=summary_tokenizer,
        time_value=time_value,
    )

    # 保存为 JSON 文件
    save_doc_meta_to_block_dir(
        doc_meta,
        html_path,
        html_root_dir=args.html_dir,
        block_root_dir=args.html_dir + "_block",
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
    parser.add_argument("--html_dir", type=str, default="./总知识库_cleaned")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--embed_model", type=str, default="/data/huangruizhi/htmlRAG/bce-embedding-base_v1")
    # parser.add_argument("--summary_tokenizer", type=str, default="/data/huangruizhi/htmlRAG/chatglm3-6b")
    # parser.add_argument("--summary_model", type=str, default="/data/huangruizhi/htmlRAG/chatglm3-6b")
    parser.add_argument("--summary_tokenizer", type=str, default="THUDM/glm-4-9b-chat")
    parser.add_argument("--summary_model", type=str, default="THUDM/glm-4-9b-chat")
    parser.add_argument("--max_node_words_embed", type=int, default=4096)
    parser.add_argument("--min_node_words_embed", type=int, default=48)
    args = parser.parse_args()


    # 加载模型（可注释掉摘要部分以加速）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary_tokenizer = AutoTokenizer.from_pretrained(args.summary_tokenizer, trust_remote_code=True)
    summary_model = AutoModel.from_pretrained(args.summary_model, trust_remote_code=True).half().to(device)
    summary_model.eval()

    # 遍历并处理 HTML 文件
    html_files = get_all_html_files(args.html_dir)
    print(f"📄 共发现 HTML 文件数: {len(html_files)}")

    for html_file in html_files:
        process_html_file(html_file, args, summary_model, summary_tokenizer)

    print("✅ 全部文档块 JSON 已生成完毕。")
