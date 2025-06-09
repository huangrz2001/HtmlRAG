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
from utils.config import CONFIG, logger
import asyncio
import aiofiles
from functools import partial
import time


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
    """
    在当前项目目录下构造缓存文件的路径，目录为 ./archive/
    """
    return os.path.join(os.path.dirname(__file__), "archive", page_url)

async def async_remove(path: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, partial(os.remove, path))


def download_html(resource_id: int, blackhole_url: str, save_path: str) -> str:
    """
    从黑洞服务器下载 Base64 文件内容并保存到本地指定路径，并记录耗时
    """
    url = f"http://{blackhole_url}/resource/withoutLogin/downloadWithBase64"
    start_time = time.perf_counter()

    try:
        resp = requests.post(url, json={"resourceId": resource_id}, timeout=10)
        resp.raise_for_status()
        resp_data = resp.json()

        if resp_data.get("code") != 200000 or "data" not in resp_data:
            raise ValueError(f"请求失败或响应无效: {resp_data}")

        content_base64 = resp_data["data"].get("content")
        if not content_base64:
            raise ValueError(f"响应中缺少 content 字段: {resp_data}")

        content_bytes = base64.b64decode(content_base64)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(content_bytes)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"✅ 黑洞下载成功: {resource_id} -> {save_path}，耗时 {elapsed_ms:.2f} ms")
        return save_path

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"❌ 黑洞下载失败: {resource_id} -> {save_path}，耗时 {elapsed_ms:.2f} ms，错误: {e}")
        return None



# ========== 插入异步接口 ==========
async def insert_html_api_async(document_index: int, page_url: str, resource_id: int, blackhole_url: str):
    logger.info(f"📥 插入请求: document_index={document_index}, page_url={page_url}, resource_id={resource_id}")

    html_path = get_local_html_path(page_url)
    if not os.path.exists(html_path):
        logger.info(f"{html_path} 文件不存在，本地路径: %s，尝试从黑洞下载...", html_path)
        downloaded_path = download_html(resource_id, blackhole_url, html_path)
        if not downloaded_path or not os.path.exists(downloaded_path):
            logger.error("❌ HTML 文件下载失败: %s", html_path)
            return {"result": "fail", "error": "HTML 文件下载失败"}

    try:
        async with aiofiles.open(html_path, "r", encoding="utf-8") as f:
            html_raw = await f.read()
        logger.info("%s HTML 文件读取完成", page_url)

        time_value, cleaned_part = parse_time_tag(html_raw)
        cleaned_html = clean_html(cleaned_part)
        logger.info("%s HTML 清洗与时间标签提取完成", page_url)

        block_tree, _ = build_block_tree(
            cleaned_html,
            max_node_words=CONFIG["max_node_words_embed"],
            min_node_words=CONFIG["min_node_words_embed"],
            zh_char=(CONFIG["lang"] == "zh")
        )
        logger.info("%s Block Tree 构建完成", page_url)

        doc_meta = await generate_block_documents_async(
            block_tree=block_tree,
            max_node_words=CONFIG["max_node_words_embed"],
            page_url=page_url,
            time_value=time_value,
            use_vllm=True,
            batch_size=CONFIG.get("vllm_batch_size", 32)
        )
        logger.info("%s 文档块生成完成，chunk 数: %d", page_url, len(doc_meta))
        if len(doc_meta) == 0:
            logger.warning("%s 生成的文档块为空，请检查 HTML 内容", page_url)
            return {"result": "fail", "error": "原文档为空，无法插入"}

        for i, doc in enumerate(doc_meta):
            doc["document_index"] = document_index
            doc["chunk_idx"] = i
        logger.info("%s 开始插入 Milvus...", page_url)
        milvus_cnt = insert_block_to_milvus(doc_meta, embedder, CONFIG["index_name"])
        logger.info("%s 开始插入 ES...", page_url)
        es_cnt = insert_block_to_es(doc_meta, CONFIG["index_name"])
        logger.info("%s 向量插入完成: Milvus=%d, ES=%d", page_url, milvus_cnt, es_cnt)

        return {
            "result": "ok",
            "document_index": document_index,
            "inserted_chunks_milvus": milvus_cnt,
            "inserted_chunks_es": es_cnt
        }

    except Exception as e:
        logger.exception("%s 插入失败:", page_url)
        return {"result": "fail", "error": f"插入失败: {str(e)}"}

# ======================== 删除接口实现 ========================
async def delete_html_api_async(document_index: int, page_url: str):
    logger.info(f"📤 删除请求: document_index={document_index}, page_url={page_url}")
    try:
        milvus_cnt = delete_blocks_from_milvus(CONFIG["index_name"], document_index)
        es_cnt = delete_blocks_from_es(CONFIG["index_name"], document_index)
        logger.info(f"{page_url} 向量删除完成: Milvus=%d, ES=%d", milvus_cnt, es_cnt)

        local_path = get_local_html_path(page_url)
        if os.path.exists(local_path):
            logger.info(f"{page_url} 检测到本地文件，准备删除: {local_path}")
            try:
                await async_remove(local_path)
                logger.info(f"{page_url} 本地文件删除成功: {local_path}")
            except Exception as e:
                logger.warning(f"{page_url} 本地文件异步删除失败: {e}")

        return {
            "result": "ok",
            "document_index": document_index,
            "deleted_chunks_milvus": milvus_cnt,
            "deleted_chunks_es": es_cnt
        }

    except Exception as e:
        logger.exception("❌ 删除失败:")
        return {"result": "fail", "error": f"删除失败: {str(e)}"}


# ======================== FastAPI 定义 ========================
app = FastAPI(title="RAG 文档接口", description="支持文档插入、删除、query 重写", root_path="/document")

class InsertRequest(BaseModel):
    document_index: int
    resource_id: int  # long结构，文档资源id
    page_url: str

@app.post("/add", summary="新增文档")
async def add_doc(req: InsertRequest):
    print(f"📥 插入请求: {req.document_index}, {req.page_url}")
    # 4008893141271707648
    result = await insert_html_api_async(req.document_index, req.page_url, req.resource_id, CONFIG.get("blackhole_url", "172.16.4.51:8082"))
    return JSONResponse(content=result)


class DeleteRequest(BaseModel):
    document_index: int
    page_url: str

@app.post("/delete", summary="删除文档")
async def delete_doc_async(req: DeleteRequest):
    result = await delete_html_api_async(req.document_index, req.page_url)
    return JSONResponse(content=result)

class DialogueTurn(BaseModel):
    speaker: str
    text: str

class RewriteRequest(BaseModel):
    dialogue: List[DialogueTurn]
    final_query: str

@app.post("/query_rewrite", summary="重写 Query")
async def rewrite_query(req: RewriteRequest):
    logger.info(f"重写请求: 原始query={req.final_query}")
    try:
        dialogue = [{"speaker": d.speaker, "text": d.text} for d in req.dialogue]
        rewritten = await rewrite_query_vllm_async(dialogue, req.final_query)
        logger.info(f"{req.final_query} 重写完成: {rewritten}")
        return {"status": "ok", "rewritten_query": rewritten}
    except Exception as e:
        logger.exception("{req.final_query} 重写失败:")
        return {"status": "fail", "error": f"重写失败: {str(e)}"}


@app.get("/ping", summary="健康检查")
def ping():
    return {"status": "ok"}
