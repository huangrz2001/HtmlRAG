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
from utils.text_process_utils import generate_block_documents, generate_block_documents_async
from utils.db_utils import (
    insert_block_to_es, insert_block_to_milvus,
    delete_blocks_from_es, delete_blocks_from_milvus
)
from utils.llm_api import rewrite_query_vllm, rewrite_query_vllm_async
from utils.config import CONFIG
import aiofiles 


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

# ========== 辅助函数 ==========
def get_local_html_path(page_url: str) -> str:
    return os.path.join("tmp", page_url)

def download_html(resource_id: int, blackhole_url: str, page_url: str) -> str:
    """从远程服务器下载 base64 HTML 内容并保存为本地文件"""
    url = f"http://{blackhole_url}/resource/withoutLogin/downloadWithBase64"
    try:
        resp = requests.post(url, json={"resourceId": resource_id}, timeout=10)
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


# ========== 插入异步接口 ==========
async def insert_html_api_async(document_index: int, page_url: str, resource_id: int, blackhole_url: str):
    # ✅ 下载 HTML 文件（或使用本地缓存路径）
    # html_path = download_html(resource_id, blackhole_url, page_url)
    # html_path = "/home/algo/hrz/db_construct/测试知识库/_全球购_商家违规管理规则.html"
    html_path = page_url
    if not html_path or not os.path.exists(html_path):
        return {"result": "fail", "error": "HTML 文件下载失败"}

    try:
        # ✅ 异步读取 HTML 内容
        async with aiofiles.open(html_path, "r", encoding="utf-8") as f:
            html_raw = await f.read()

        # HTML 清洗 + 结构解析
        time_value, cleaned_part = parse_time_tag(html_raw)
        cleaned_html = clean_html(cleaned_part)
        block_tree, _ = build_block_tree(
            cleaned_html,
            max_node_words=CONFIG["max_node_words_embed"],
            min_node_words=CONFIG["min_node_words_embed"],
            zh_char=(CONFIG["lang"] == "zh")
        )

        # ✅ 异步生成摘要块
        doc_meta = await generate_block_documents_async(
            block_tree=block_tree,
            max_node_words=CONFIG["max_node_words_embed"],
            page_url=page_url,
            time_value=time_value,
            use_vllm=True,
            batch_size=CONFIG.get("vllm_batch_size", 32)
        )

        for i, doc in enumerate(doc_meta):
            doc["file_idx"] = document_index
            doc["chunk_idx"] = i

        # ✅ 插入 Milvus 和 ES（同步）
        milvus_cnt = insert_block_to_milvus(doc_meta, embedder, CONFIG["index_name"])
        es_cnt = insert_block_to_es(doc_meta, CONFIG["index_name"])

        return {
            "result": "ok",
            "file_idx": document_index,
            "inserted_chunks_milvus": milvus_cnt,
            "inserted_chunks_es": es_cnt
        }

    except Exception as e:
        return {"result": "fail", "error": f"插入失败: {str(e)}"}


# ======================== 删除接口实现 ========================
async def delete_html_api_async(document_index: int, page_url: str):
    try:
        # ✅ DB 删除操作仍为同步（推荐后续改造为异步驱动库）
        milvus_cnt = delete_blocks_from_milvus(CONFIG["index_name"], document_index)
        es_cnt = delete_blocks_from_es(CONFIG["index_name"], document_index)

        # ✅ 异步删除本地文件
        local_path = get_local_html_path(page_url)
        if os.path.exists(local_path):
            try:
                await aiofiles.os.remove(local_path)
            except Exception as e:
                print(f"⚠️ 异步删除文件失败: {e}")

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
    resource_id: int  # long结构，文档资源id
    blackhole_url: str  # ✅ 应为 str 类型
    page_url: str

@app.post("/chat/python/document/add", summary="新增文档")
async def add_doc(req: InsertRequest):
    print(f"📥 插入请求: {req.document_index}, {req.page_url}")
    result = await insert_html_api_async(req.document_index, req.page_url, req.resource_id, req.blackhole_url)
    return JSONResponse(content=result)


class DeleteRequest(BaseModel):
    document_index: int
    page_url: str

@app.post("/chat/python/document/delete", summary="删除文档")
async def delete_doc_async(req: DeleteRequest):
    print(f"📤 删除请求: {req.document_index}, {req.page_url}")
    result = await delete_html_api_async(req.document_index, req.page_url)
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
        # 不阻塞当前线程，而是等待 rewrite_query_vllm_async 执行完毕，再继续
        rewritten = await rewrite_query_vllm_async(dialogue, req.final_query)
        return {"status": "ok", "rewritten_query": rewritten}
    except Exception as e:
        return {"status": "fail", "error": f"重写失败: {str(e)}"}


@app.get("/ping", summary="健康检查")
def ping():
    return {"status": "ok"}
