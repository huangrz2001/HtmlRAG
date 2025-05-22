# -*- coding: utf-8 -*-
"""
函数式接口服务：统一封装三步处理流程（clean → block → insert）
1. 接收 HTML 文件路径
2. 清洗 + 分块 + 构造 JSON 元数据
3. 插入 Milvus / Elasticsearch
"""

import os
import re
import json
import shutil
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings

from utils.html_utils import clean_html, build_block_tree
from utils.text_process_utils import generate_block_documents
from utils.db_utils import (
    insert_block_to_es, insert_block_to_milvus,
    delete_blocks_from_es, delete_blocks_from_milvus,
    get_max_global_idx_es, get_max_global_idx_milvus
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================== 全局配置与模型加载 ========================
class Config:
    lang = "zh"
    embed_model = "/data/huangruizhi/htmlRAG/bce-embedding-base_v1"
    # summary_model = "/data/huangruizhi/htmlRAG/chatglm3-6b"
    # summary_tokenizer = "/data/huangruizhi/htmlRAG/chatglm3-6b"
    summary_model = "THUDM/glm-4-9b-chat"
    summary_tokenizer = "THUDM/glm-4-9b-chat"
    index_name = "curd_env"
    Milvus_host = "192.168.7.247"
    ES_host = "192.168.7.247"
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    max_node_words_embed = 4096
    min_node_words_embed = 48

summary_tokenizer = AutoTokenizer.from_pretrained(Config.summary_tokenizer, trust_remote_code=True)
summary_model = AutoModel.from_pretrained(Config.summary_model, trust_remote_code=True).half().to(Config.device)
summary_model.eval()
embedder = HuggingFaceEmbeddings(
    model_name=Config.embed_model,
    model_kwargs={"device": Config.device}
)
cnt4Milvus = get_max_global_idx_milvus(Config.Milvus_host, Config.index_name)
cnt4ES = get_max_global_idx_es(Config.ES_host, Config.index_name)
try: 
    assert cnt4Milvus == cnt4ES # 确保索引一致性
except AssertionError:
    print("❌ 初始索引不一致，请检查 Milvus 和 Elasticsearch 的索引状态！")
    print(f"Milvus: {cnt4Milvus}, ES: {cnt4ES}")
    exit(1)


# ======================== 工具函数 ========================
def parse_time_tag(html: str):
    time_pattern = r"^\s*<time[^>]*?>(.*?)</time>"
    time_match = re.match(time_pattern, html, flags=re.IGNORECASE | re.DOTALL)
    time_value = ""
    if time_match:
        time_value = time_match.group(1).strip()
        html = html[time_match.end():].lstrip()
    return time_value, html

# ======================== 插入函数接口 ========================
def insert_html(html_path: str):
    if not os.path.exists(html_path):
        return {"status": "fail", "msg": f"HTML 不存在: {html_path}"}

    page_url = os.path.relpath(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_raw = f.read()

    # Step 1: 提取 <time> 标签
    time_pattern = r"^\s*<time[^>]*?>(.*?)</time>"
    time_match = re.match(time_pattern, html_raw, flags=re.IGNORECASE | re.DOTALL)
    time_value = ""
    if time_match:
        time_value = time_match.group(1).strip()
        html_raw = html_raw[time_match.end():].lstrip()

    # Step 2: 清洗 HTML 内容
    from utils.html_utils import clean_html, build_block_tree
    cleaned_html = clean_html(html_raw)

    clean_path = html_path.replace(".html", ".clean.html")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned_html)
    print(f"✅ Clean HTML 写入：{clean_path}")

    # Step 3: 构建 block tree
    block_tree, _ = build_block_tree(
        cleaned_html,
        max_node_words=Config.max_node_words_embed,
        min_node_words=Config.min_node_words_embed,
        zh_char=(Config.lang == "zh")
    )

    # Step 4: 文本块抽取 + 摘要 + 问题生成
    from utils.text_process_utils import generate_block_documents, save_doc_meta_to_block_dir

    doc_meta = generate_block_documents(
        block_tree,
        max_node_words=Config.max_node_words_embed,
        page_url=page_url,
        summary_model=summary_model,
        summary_tokenizer=summary_tokenizer,
        time_value=time_value,
    )

    block_json_path = html_path.replace(".html", ".block.json")
    with open(block_json_path, "w", encoding="utf-8") as f:
        json.dump(doc_meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Block JSON 写入：{block_json_path}")

    # Step 5: 插入 Milvus / ES
    from utils.db_utils import (
        insert_block_to_milvus,
        insert_block_to_es,
        get_max_global_idx_milvus,
        get_max_global_idx_es,
    )

    cnt4Milvus = get_max_global_idx_milvus(Config.Milvus_host, Config.index_name)
    cnt4ES = get_max_global_idx_es(Config.ES_host, Config.index_name)
    assert cnt4Milvus == cnt4ES, "Milvus 和 ES 索引不一致，请检查数据完整性"

    cnt4Milvus = insert_block_to_milvus(doc_meta, embedder, Config.Milvus_host, Config.index_name, cnt4Milvus)
    cnt4ES = insert_block_to_es(doc_meta, Config.ES_host, Config.index_name, cnt4ES)

    return {
        "status": "ok",
        "inserted": len(doc_meta),
        "clean_path": clean_path,
        "block_path": block_json_path
    }

# ======================== 删除函数接口 ========================
def delete_html(html_path: str):
    page_url = os.path.relpath(html_path)

    if not page_url or not isinstance(page_url, str):
        return {"status": "fail", "error": "参数 page_url 无效"}

    results = {"status": "ok", "milvus_deleted": False, "es_deleted": False}

    try:
        delete_blocks_from_milvus(Config.Milvus_host, Config.index_name, page_url)
        results["milvus_deleted"] = True
    except Exception as e:
        results["status"] = "partial_fail"
        results["milvus_error"] = str(e)

    try:
        delete_blocks_from_es(Config.ES_host, Config.index_name, page_url)
        results["es_deleted"] = True
    except Exception as e:
        results["status"] = "partial_fail"
        results["es_error"] = str(e)

    if results["milvus_deleted"] and results["es_deleted"]:
        results["status"] = "ok"

    return results

# ======================== 主函数入口 ========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--html_path", type=str, default="./测试知识库/_ 巨量千川投放_树__V2.0版_.html", help="输入 HTML 文件路径")
    parser.add_argument("--mode", type=str, choices=["insert", "delete"], default="delete", help="操作类型：insert 或 delete")
    args = parser.parse_args()

    if args.mode == "insert":
        result = insert_html(args.html_path)
    else:
        result = delete_html(args.html_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))