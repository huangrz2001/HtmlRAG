import os
import json
import base64
import requests
import re
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from utils.html_utils import clean_html, build_block_tree
from utils.text_process_utils import generate_block_documents
from utils.db_utils import (
    insert_block_to_es, insert_block_to_milvus,
    delete_blocks_from_es, delete_blocks_from_milvus
)
from utils.llm_api import rewrite_query_vllm
from utils.config import CONFIG

# ======================== 嵌入模型加载 ========================
DEVICE = CONFIG.get("device", f"cuda:0" if torch.cuda.is_available() else "cpu")

embedder = HuggingFaceEmbeddings(
    model_name=CONFIG["embed_model"],
    model_kwargs={"device": DEVICE}
)

# ======================== 公共函数 ========================
def parse_time_tag(html: str):
    pattern = r"^\s*<time[^>]*?>(.*?)</time>"
    match = re.match(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    return (match.group(1).strip(), html[match.end():].lstrip()) if match else ("", html)

def get_local_html_path(page_url: str) -> str:
    return os.path.join("tmp", page_url)

def download_html(document_index: int, page_url: str) -> str:
    """从远程服务器下载 base64 HTML 内容并保存为本地文件"""
    url = "http://ip:port/resource/withoutLogin/downloadWithBase64"  # TODO: 替换为实际地址
    try:
        resp = requests.post(url, json={"resourceId": document_index}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        content_bytes = base64.b64decode(data["content"])
        local_path = get_local_html_path(page_url)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content_bytes)
        return local_path
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

# ======================== 插入接口实现 ========================
def insert_html_api(document_index: int, page_url: str):
    # html_path = download_html(document_index, page_url)
    html_path = '/home/algo/hrz/db_construct/测试知识库/_全球购_发布违禁商品_信息_细则.html'
    if not html_path or not os.path.exists(html_path):
        return {"result": "fail", "error": "HTML 文件下载失败"}

    try:
        html_raw = open(html_path, "r", encoding="utf-8").read()
        time_value, cleaned_part = parse_time_tag(html_raw)
        cleaned_html = clean_html(cleaned_part)
        block_tree, _ = build_block_tree(
            cleaned_html,
            max_node_words=CONFIG["max_node_words_embed"],
            min_node_words=CONFIG["min_node_words_embed"],
            zh_char=(CONFIG["lang"] == "zh")
        )
        doc_meta = generate_block_documents(
            block_tree,
            max_node_words=CONFIG["max_node_words_embed"],
            page_url=page_url,
            time_value=time_value,
            use_vllm=True
        )
        for i, doc in enumerate(doc_meta):
            doc["file_idx"] = document_index
            doc["chunk_idx"] = i

        milvus_cnt = insert_block_to_milvus(doc_meta, embedder, CONFIG["index_name"])
        es_cnt = insert_block_to_es(doc_meta, CONFIG["index_name"])

        return {
            "result": "ok",
            "file_idx": document_index,
            "inserted_chunks_milvus": milvus_cnt,
            "inserted_chunks_es": es_cnt
        }

    except Exception as e:
        return {"result": "fail", "error": f"插入失败: {e}"}

# ======================== 删除接口实现 ========================
def delete_html_api(document_index: int, page_url: str):
    try:
        milvus_cnt = delete_blocks_from_milvus(CONFIG["index_name"], document_index)
        es_cnt = delete_blocks_from_es(CONFIG["index_name"], document_index)

        local_path = get_local_html_path(page_url)
        if os.path.exists(local_path):
            os.remove(local_path)

        return {
            "result": "ok",
            "file_idx": document_index,
            "deleted_chunks_milvus": milvus_cnt,
            "deleted_chunks_es": es_cnt
        }

    except Exception as e:
        return {"result": "fail", "error": f"删除失败: {e}"}

# ======================== FastAPI 定义 ========================
app = FastAPI(title="RAG 文档接口", description="支持文档插入、删除、query 重写")

class InsertRequest(BaseModel):
    document_index: int
    page_url: str

@app.post("/chat/python/document/add", summary="新增文档")
async def add_doc(req: InsertRequest):
    print(f"📥 插入请求: {req.document_index}, {req.page_url}")
    result = insert_html_api(req.document_index, req.page_url)
    return JSONResponse(content=result)

class DeleteRequest(BaseModel):
    document_index: int
    page_url: str

@app.post("/chat/python/document/delete", summary="删除文档")
async def delete_doc(req: DeleteRequest):
    print(f"📤 删除请求: {req.document_index}, {req.page_url}")
    result = delete_html_api(req.document_index, req.page_url)
    return JSONResponse(content=result)

class DialogueTurn(BaseModel):
    speaker: str
    text: str

class RewriteRequest(BaseModel):
    dialogue: List[DialogueTurn]
    final_query: str

@app.post("/chat/python/query/rewrite", summary="重写 Query")
async def rewrite_query(req: RewriteRequest):
    print(f"🔄 重写请求: {req}")
    try:
        dialogue = [{"speaker": d.speaker, "text": d.text} for d in req.dialogue]
        rewritten = rewrite_query_vllm(dialogue, req.final_query)
        return {"status": "ok", "rewritten_query": rewritten}
    except Exception as e:
        return {"status": "fail", "error": f"重写失败: {str(e)}"}

@app.get("/ping", summary="健康检查")
def ping():
    return {"status": "ok"}
