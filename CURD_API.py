# -*- coding: utf-8 -*-
"""
API服务模块：统一封装 HTML 的清洗、分块与插入/删除操作
支持多环境配置 + 前端传入 document_index 参数
"""

import os
import json
import re
import torch
from time import sleep
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from utils.html_utils import clean_html, build_block_tree, parse_time_tag
from utils.text_process_utils import generate_block_documents
from utils.db_utils import (
    insert_block_to_es, insert_block_to_milvus,
    delete_blocks_from_es, delete_blocks_from_milvus
)



# ======================== 加载配置 ========================
def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

CONFIG = load_config()
DEVICE = CONFIG.get("device", f"cuda:0" if torch.cuda.is_available() else "cpu")

# ======================== 嵌入模型加载 ========================
embedder = HuggingFaceEmbeddings(
    model_name=CONFIG["embed_model"],
    model_kwargs={"device": DEVICE}
)

# ======================== 插入接口 ========================
def insert_html_api(html_path: str, document_index: int, env: str = "dev"):
    if not os.path.exists(html_path):
        return {"status": "fail", "msg": f"HTML 文件不存在: {html_path}"}

    try:
        page_url = os.path.relpath(html_path)

        with open(html_path, "r", encoding="utf-8") as f:
            html_raw = f.read()

        # Step 1: 提取 <time> 标签
        time_value, html_clean_start = parse_time_tag(html_raw)

        # Step 2: 清洗 HTML
        cleaned_html = clean_html(html_clean_start)

        # Step 3: 构建 block tree
        block_tree, _ = build_block_tree(
            cleaned_html,
            max_node_words=CONFIG["max_node_words_embed"],
            min_node_words=CONFIG["min_node_words_embed"],
            zh_char=(CONFIG["lang"] == "zh")
        )

        # Step 4: 抽取元信息
        doc_meta = generate_block_documents(
            block_tree,
            max_node_words=CONFIG["max_node_words_embed"],
            page_url=page_url,
            time_value=time_value,
            use_vllm=True
        )

        # Step 5: 添加字段
        for i, doc in enumerate(doc_meta):
            doc["document_index"] = document_index
            doc["chunk_idx"] = i

        # Step 6: 插入 Milvus / ES
        inserted_milvus = insert_block_to_milvus(doc_meta, embedder, env=env)
        inserted_es = insert_block_to_es(doc_meta, env=env)

        return {
            "status": "ok",
            "inserted": len(doc_meta),
            "inserted_chunks_milvus": inserted_milvus,
            "inserted_chunks_es": inserted_es,
            "document_index": document_index
        }

    except Exception as e:
        return {
            "status": "fail",
            "error": f"[{env}] 插入失败：{str(e)}"
        }

# ======================== 删除接口 ========================
def delete_html_api(document_index: int, html_path: str = None, env: str = "dev"):
    try:
        # 删除中间缓存文件
        if html_path:
            for suffix in ["_clean.html", "_block.json"]:
                fpath = html_path.replace(".html", suffix)
                if os.path.exists(fpath):
                    os.remove(fpath)

        deleted_milvus = delete_blocks_from_milvus(document_index, env=env)
        deleted_es = delete_blocks_from_es(document_index, env=env)

        return {
            "status": "ok",
            "deleted_chunks_milvus": deleted_milvus,
            "deleted_chunks_es": deleted_es,
            "document_index": document_index
        }

    except Exception as e:
        return {
            "status": "fail",
            "error": f"[{env}] 删除失败：{str(e)}"
        }
