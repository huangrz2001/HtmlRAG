# -*- coding: utf-8 -*-
"""
HTML 文档索引与多模态检索核心模块

本模块提供面向知识型问答系统（如 RAG）的一体化向量索引管理与查询能力，支持基于 Milvus 和 Elasticsearch 的双模检索方案，
涵盖向量索引构建、文档块插入与删除、索引重建、文本摘要与语义去重、多源融合检索与 reranker 精排等功能。

模块功能概览：
------------------------------------------------
1. 向量库（Milvus）管理：
   - `reset_milvus`: 重建 Milvus collection，支持主键自增与多字段结构。
   - `insert_block_to_milvus`: 向量块插入，支持批量写入与嵌入生成。
   - `delete_blocks_from_milvus`: 按 document_index 删除 Milvus 向量。
   - `query_milvus_blocks`: 基于语义向量进行 ANN 检索，支持 reranker 精排。

2. 关键词库（Elasticsearch）管理：
   - `reset_es`: 重建 Elasticsearch 索引结构，使用 IK 分词器进行中文优化。
   - `insert_block_to_es`: 批量插入文档块文本至 Elasticsearch。
   - `delete_blocks_from_es`: 按 document_index 删除 ES 文档块。
   - `query_es_blocks`: 基于关键词抽取构建查询语句，执行倒排检索。

3. 多源融合检索与去重：
   - `query_blocks`: 同时从 Milvus 与 ES 检索文档块，支持时间优先的去重策略与 reranker 精排逻辑。

4. reranker 模型精排：
   - `Reranker`: 使用 transformer 模型对候选块进行 query-passage 精排。
   - `rerank_results`: 对候选文档块按照 relevance 分数重新排序。

依赖环境与配置说明：
------------------------------------------------
- Milvus >= 2.x（需预启动服务，监听 19530 端口）
- Elasticsearch >= 7.x（需启用 IK 分词器）
- LangChain、PyMilvus、transformers、torch 等
- 外部依赖配置通过 utils/config.py 注入：如 milvus_host, es_host, index_name

"""

import os
import jieba
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import (
    connections,
    utility,
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
)
import torch
from utils.text_process_utils import build_optimal_jieba_query, deduplicate_ranked_blocks_pal
from time import sleep
from langchain_community.vectorstores import Milvus
from langchain.schema import Document
from time import sleep
from elasticsearch import Elasticsearch, helpers
import jieba.analyse
from utils.config import CONFIG, logger
from utils.llm_api import get_embeddings_from_vllm_async, get_embedding_from_vllm
import numpy as np
from typing import List


# 加载自定义词典（适用于电商/运营场景）
jieba.load_userdict("./user_dict.txt")

# 全局连接池
_es_clients = {}
_milvus_alias_map = {}  # 存储已连接的别名
_milvus_collection_cache = {}  # 存储已创建并加载的 Collection 对象


def get_env_config(env=None):
    """获取指定环境下的完整配置项"""
    env = env or os.getenv("RAG_ENV") or CONFIG.get("env_default", "dev")
    env_cfg = CONFIG.get("env_config", {}).get(env)
    if not env_cfg:
        raise ValueError(f"❌ 未找到环境配置: {env}")
    return env, env_cfg

def get_es(env=None):
    env, env_cfg = get_env_config(env)
    
    if env not in _es_clients:
        logger.debug(f"🔌 初始化 ES 连接 [{env}]：{env_cfg['es_host']}")
    if "es_user" in env_cfg and "es_password" in env_cfg:
        es = Elasticsearch(
            hosts=[f"http://{env_cfg['es_host']}:9200"],
            basic_auth=(env_cfg["es_user"], env_cfg["es_password"]),
            request_timeout=30
        )
        _es_clients[env] = es
    else:
        _es_clients[env] = Elasticsearch(f"http://{env_cfg['es_host']}:9200")


    return _es_clients[env]


# def get_milvus_collection(env=None):
#     """
#     获取指定环境下的 Milvus collection，使用缓存机制避免重复创建和加载。
#     """
#     env, env_cfg = get_env_config(env)
#     alias = env  # 使用环境名作为连接别名
#     collection_name = env_cfg["collection_name"]
#     print(collection_name)
    
#     # 构建缓存键
#     cache_key = f"{alias}_{collection_name}"
    
#     # 检查连接是否已存在，不存在则创建
#     if alias not in _milvus_alias_map:
#         logger.debug(f"🔌 初始化 Milvus 连接 [{env}]：{env_cfg['milvus_host']}")
#         connections.connect(alias=alias, host=env_cfg["milvus_host"], port="19530")
#         _milvus_alias_map[alias] = True
    
#     # 检查 Collection 是否已缓存，不存在则创建并加载
#     if cache_key not in _milvus_collection_cache:
#         logger.debug(f"📚 加载 Milvus Collection: {collection_name}")
#         col = Collection(name=collection_name, using=alias)
#         col.load()
#         _milvus_collection_cache[cache_key] = col
    
#     return _milvus_collection_cache[cache_key]




def get_milvus_collection(env=None):
    env, env_cfg = get_env_config(env)
    alias = env
    collection_name = env_cfg["collection_name"]
    print(f"📝 正在加载 Collection: {collection_name}")
    cache_key = f"{alias}_{collection_name}"
    
    if alias not in _milvus_alias_map:
        logger.debug(f"🔌 初始化 Milvus 连接 [{env}]：{env_cfg['milvus_host']}")
        connect_params = {
            "alias": alias,
            "host": env_cfg["milvus_host"],
            "port": "19530",
            "secure": False
        }
        if "milvus_user" in env_cfg and "milvus_password" in env_cfg:
            connect_params["user"] = env_cfg["milvus_user"]
            connect_params["password"] = env_cfg["milvus_password"]

        connections.connect(**connect_params)
        _milvus_alias_map[alias] = True

    if cache_key not in _milvus_collection_cache:
        logger.debug(f"📚 加载 Milvus Collection: {collection_name}")
        col = Collection(name=collection_name, using=alias)
        col.load()
        _milvus_collection_cache[cache_key] = col
    
    return _milvus_collection_cache[cache_key]



def get_index_name(env=None):
    """获取当前环境的 Elasticsearch 索引名"""
    _, env_cfg = get_env_config(env)
    return env_cfg["index_name"]

# ======================== 获取最大全局索引(milvus) ========================
def get_max_global_idx_milvus(env="dev"):
    """获取当前环境下 Milvus 中 global_chunk_idx 的最大值"""
    try:
        _, cfg = get_env_config(env)
        alias = env
        host = cfg["milvus_host"]
        collection_name = cfg["collection_name"]

        # 初始化连接（如果未连接）
        if not connections.has_connection(alias):
            connections.connect(alias=alias, host=host, port="19530")

        collection = Collection(name=collection_name, using=alias)
        collection.load()

        results = collection.query(
            expr="global_chunk_idx >= 0",
            output_fields=["global_chunk_idx"],
        )
        if not results:
            return 0
        max_idx = max(item["global_chunk_idx"] for item in results)
        return max_idx + 1
    except Exception as e:
        print(f"⚠️ 查询失败: {e}")
        return 0


# ======================== ES 索引重建 ========================
def reset_es(env="dev"):
    """重建 Elasticsearch 索引（含 ik 分词与字段映射）"""
    _, cfg = get_env_config(env)
    index_name = cfg["index_name"]
    es = get_es(env)

    logger.info("Connected to Elasticsearch!" if es.ping() else "Connection failed.")

    if es.indices.exists(index=index_name):
        logger.info(f"⚠️ {env} 环境索引 {index_name} 已存在，删除中...")
        es.indices.delete(index=index_name, ignore_unavailable=True, request_timeout=20)

    # es.indices.create(
    #     index=index_name,
    #     body={
    #         "settings": {
    #             "analysis": {
    #                 "analyzer": {
    #                     "ik_max_word": {
    #                         "type": "custom",
    #                         "tokenizer": "ik_max_word"
    #                     }
    #                 }
    #             }
    #         },
    #         "mappings": {
    #             "properties": {
    #                 "document_index": { "type": "long" },
    #                 "chunk_idx": { "type": "integer" },
    #                 "text": {
    #                     "type": "text",
    #                     "analyzer": "ik_max_word",
    #                     "fields": { "keyword": { "type": "keyword" } }
    #                 },
    #                 "page_url": {
    #                     "type": "text",
    #                     "fields": { "keyword": { "type": "keyword" } }
    #                 },
    #                 "page_name": {
    #                     "type": "text",
    #                     "analyzer": "ik_max_word",
    #                     "fields": { "keyword": { "type": "keyword" } }
    #                 },
    #                 "title": {
    #                     "type": "text",
    #                     "analyzer": "ik_max_word",
    #                     "fields": { "keyword": { "type": "keyword" } }
    #                 },
    #                 "summary": {
    #                     "type": "text",
    #                     "analyzer": "ik_max_word",
    #                     "fields": { "keyword": { "type": "keyword" } }
    #                 },
    #                 "time": {
    #                     "type": "text",
    #                     "fields": { "keyword": { "type": "keyword" } }
    #                 },
    #                 "question": {
    #                     "type": "text",
    #                     "analyzer": "ik_max_word",
    #                     "fields": { "keyword": { "type": "keyword" } }
    #                 }
    #             }
    #         }
    #     },
    # )
    es.indices.create(
    index=index_name,
    body={
        "mappings": {
            "properties": {
                "document_index": { "type": "long" },
                "chunk_idx": { "type": "integer" },
                "text": {
                    "type": "text",
                    "fields": { "keyword": { "type": "keyword" } }
                },
                "page_url": {
                    "type": "text",
                    "fields": { "keyword": { "type": "keyword" } }
                },
                "page_name": {
                    "type": "text",
                    "fields": { "keyword": { "type": "keyword" } }
                },
                "title": {
                    "type": "text",
                    "fields": { "keyword": { "type": "keyword" } }
                },
                "summary": {
                    "type": "text",
                    "fields": { "keyword": { "type": "keyword" } }
                },
                "time": {
                    "type": "text",
                    "fields": { "keyword": { "type": "keyword" } }
                },
                "question": {
                    "type": "text",
                    "fields": { "keyword": { "type": "keyword" } }
                }
                }
            }
        },
    )

    logger.info(f"✅ {env} 环境 ES 索引 '{index_name}' 已成功创建")


# ======================== Milvus 向量库重建 ========================
def reset_milvus(env="dev", dim=768):
    """重建 Milvus 向量集合（自动获取 collection_name）"""
    _, cfg = get_env_config(env)
    collection_name = cfg["collection_name"]
    host = cfg["milvus_host"]
    alias = env

    if not connections.has_connection(alias):
        connections.connect(alias=alias, host=host, port="19530")

    if utility.has_collection(collection_name, using=alias):
        logger.info(f"⚠️ {env} 环境 Milvus 集合 '{collection_name}' 已存在，正在删除...")
        utility.drop_collection(collection_name, using=alias)

    logger.info(f"🚀 {env} 环境正在创建 Milvus collection: {collection_name}")
    fields = [
        FieldSchema(name="global_chunk_idx", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_index", dtype=DataType.INT64),
        FieldSchema(name="chunk_idx", dtype=DataType.INT64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="page_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="time", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields=fields, description="HTML块向量索引")
    Collection(name=collection_name, schema=schema, using=alias)
    logger.info(f"✅ {env} 环境 Milvus collection '{collection_name}' 已创建")

# ======================== 插入 Milvus ========================
def insert_block_to_milvus(doc_meta_list, embedder, env="dev", batch_size=2) -> int:
    """
    向指定环境的 Milvus 插入文档块，自动获取 collection_name 和 host
    """
    _, cfg = get_env_config(env)
    collection_name = cfg["collection_name"]
    host = cfg["milvus_host"]

    all_docs = []
    for doc in doc_meta_list:
        doc.setdefault("document_index", -1)
        node = Document(
            page_content=doc["text"],
            metadata={k: v for k, v in doc.items() if k != "text"}
        )
        all_docs.append(node)

    logger.debug(f"🧠 正在插入向量到 Milvus（{env}）collection: {collection_name} ...")

    # 初始化 Milvus 向量库对象（from_documents 会自动建表，但我们建议先 reset）
    milvus = Milvus.from_documents(
        [all_docs[0]],  # 用第一条初始化 collection
        embedder,
        collection_name=collection_name,
        connection_args={
                "host": host,
                "port": "19530",
                "user": cfg.get("milvus_user", ""),
                "password": cfg.get("milvus_password", ""),
                "secure": False  # 如果你没启用TLS，一定设为 False！
            },
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 64}
        },
    )
    


    inserted = 1
    for i in range(1, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        try:
            milvus.add_documents(batch)
            inserted += len(batch)
            logger.debug(f"✅ 插入 batch {i // batch_size + 1}: {len(batch)} 条")
        except Exception as e:
            logger.error(f"❌ 插入 batch 失败: {e}")

    logger.debug(f"✅ 已插入 Milvus（{env}）：{inserted} 条文档块")
    return inserted


async def insert_blocks_to_milvus_vllm_async(
    doc_meta_list: List[dict],
    url: str,
    env: str = "dev",
    batch_size: int = 2,
):
    _, cfg = get_env_config(env)
    collection = get_milvus_collection(env)
    logger.debug(f"🧠 准备插入 {len(doc_meta_list)} 条文档块 到 Milvus[{env}]：{cfg['collection_name']}")

    total_inserted = 0
    N = len(doc_meta_list)

    for start in range(0, N, batch_size):
        batch = doc_meta_list[start : start + batch_size]

        try:
            # 先收集一批用于 VLLM 获取 embedding 的文本（比如 title 或 question）
            texts_for_embedding = [doc.get("text", "") for doc in batch]
            embeddings = await get_embeddings_from_vllm_async(texts_for_embedding, url)

            # 构造每列
            document_index =   [doc.get("document_index", -1) for doc in batch]
            chunk_idx =        [doc.get("chunk_idx", -1) for doc in batch]
            vector =           embeddings
            text =             [doc.get("text", "") for doc in batch]
            page_url =         [doc.get("page_url", "") for doc in batch]
            page_name =        [doc.get("page_name", "") for doc in batch]
            title =            [doc.get("title", "") for doc in batch]
            summary =          [doc.get("summary", "") for doc in batch]
            time =             [doc.get("time", "") for doc in batch]
            question =         [doc.get("question", "") for doc in batch]

            milvus_data = [
                document_index,
                chunk_idx,
                vector,
                text,
                page_url,
                page_name,
                title,
                summary,
                time,
                question,
            ]

            collection.insert(milvus_data)
            logger.debug(f"✅ Batch {start//batch_size + 1} 插入成功：{len(batch)} 条")
            total_inserted += len(batch)

        except Exception as e:
            logger.error(f"❌ Batch {start//batch_size + 1} 插入失败：{e}")

    logger.info(f"🚀 插入完成，共成功插入 {total_inserted} 条")
    return total_inserted



def insert_blocks_to_milvus_vllm(
    doc_meta_list: List[dict],
    url: str,
    env: str = "dev",
):
    """将文档块逐条插入 Milvus"""
    _, cfg = get_env_config(env)
    collection = get_milvus_collection(env)
    logger.debug(f"🧠 准备插入 {len(doc_meta_list)} 条文档块 到 Milvus[{env}]：{cfg['collection_name']}")

    success_count = 0
    failed_count = 0
    failed_indices = []

    for idx, doc in enumerate(doc_meta_list):
        try:
            # 获取文本嵌入向量
            text = doc.get("text", "")
            embeddings = get_embedding_from_vllm(text, url)  # 同步版本的嵌入函数
            
            # 构造插入数据
            milvus_data = [
                [doc.get("document_index", -1)],
                [doc.get("chunk_idx", -1)],
                [embeddings[0]],  # 取第一个嵌入向量
                [text],
                [doc.get("page_url", "")],
                [doc.get("page_name", "")],
                [doc.get("title", "")],
                [doc.get("summary", "")],
                [doc.get("time", "")],
                [doc.get("question", "")],
            ]
            
            # 单条插入
            collection.insert(milvus_data)
            success_count += 1
            logger.debug(f"✅ 插入成功：文档索引 {doc.get('document_index', -1)}，块索引 {doc.get('chunk_idx', -1)}")
            
        except Exception as e:
            failed_count += 1
            failed_indices.append(idx)
            logger.error(f"❌ 插入失败：文档索引 {doc.get('document_index', -1)}，块索引 {doc.get('chunk_idx', -1)}，错误：{e}")
    
    logger.info(f"🚀 插入完成，成功 {success_count} 条，失败 {failed_count} 条")
    
    if failed_count > 0:
        logger.warning(f"⚠️ 以下索引的文档插入失败：{failed_indices}")
    
    return success_count




# ======================== 插入 ES ========================
def insert_block_to_es(doc_meta_list, env="dev") -> int:
    """
    向指定环境的 Elasticsearch 索引插入文档块，自动获取 index_name
    """
    _, cfg = get_env_config(env)
    index_name = cfg["index_name"]
    es = get_es(env)

    logger.debug(f"📥 正在插入文档到 Elasticsearch 索引（{env}）: {index_name} ...")

    actions = []
    for doc in doc_meta_list:
        if "document_index" not in doc:
            doc["document_index"] = -1
        actions.append({
            "_index": index_name,
            "_source": {
                "document_index": doc["document_index"],
                "chunk_idx": doc["chunk_idx"],
                "title": doc["title"],
                "summary": doc.get("summary", ""),
                "text": doc["text"],
                "page_url": doc["page_url"],
                "page_name": doc["page_name"],
                "time": doc.get("time", ""),
                "question": doc.get("question", "")
            }
        })

    try:
        resp = helpers.bulk(es, actions)
        logger.debug(f"✅ 已插入 ES（{env}）：{resp[0]} 条文档块")
        return resp[0]
    except Exception as e:
        logger.error(f"❌ ES 插入失败（{env}）: {e}")
        return 0




# ======================== 删除 Milvus 中的文档块 ========================
def delete_blocks_from_milvus(document_index: int, env: str = "dev") -> int:
    """从指定环境下的 Milvus collection 中删除某个 document_index"""
    try:
        col = get_milvus_collection(env=env)
        expr = f"document_index == {document_index}"
        result = col.delete(expr)
        count = result.delete_count if hasattr(result, "delete_count") else 0
        logger.debug(f"🗑️ Milvus（{env}）: 已删除 document_index = {document_index} 的文档块，共 {count} 条")
        return count
    except Exception as e:
        logger.error(f"❌ Milvus 删除失败（{env}）: {e}")
        return 0


# ======================== 删除 ES 中的文档块 ========================
def delete_blocks_from_es(document_index: int, env: str = "dev") -> int:
    """从指定环境下的 Elasticsearch 索引中删除某个 document_index"""
    try:
        _, cfg = get_env_config(env)
        index_name = cfg["index_name"]
        es = get_es(env)

        query = {
            "query": {
                "term": {
                    "document_index": document_index
                }
            }
        }

        hits = es.search(index=index_name, body=query, size=10000)["hits"]["hits"]
        if not hits:
            logger.debug(f"📭 ES（{env}）: 未找到 document_index = {document_index} 的文档")
            return 0

        resp = helpers.bulk(
            client=es,
            actions=[
                {"_op_type": "delete", "_index": index_name, "_id": hit["_id"]}
                for hit in hits
            ]
        )
        logger.debug(f"🗑️ ES（{env}）: 已删除 document_index = {document_index} 的文档块，共 {resp[0]} 条")
        return resp[0]
    except Exception as e:
        logger.error(f"❌ ES 删除失败（{env}）: {e}")
        return 0



# ======================== Milvus 查询函数 ========================
def query_milvus_blocks(
    question,
    embedder,
    env="dev",
    top_k=10,
    rerank_top_k=5,
    reranker=None,
):
    _, cfg = get_env_config(env)
    collection_name = cfg["collection_name"]

    collection = get_milvus_collection(env=env)
    if not collection.has_index():
        print(f"⚙️ Creating index on vector ...")
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 64},
            },
            index_name="default",
        )
        print("✅ Index created.")
    collection.load()

    query_vec = embedder.embed_query(question)

    results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 100}},
        limit=top_k,
        output_fields=[
            "text", "page_url", "chunk_idx", "page_name",
            "title", "summary", "time", "question", "document_index"
        ],
    )

    milvus_rank = []
    for hits in results:
        for hit in hits:
            milvus_rank.append({
                "chunk_idx": hit.entity.get("chunk_idx", -1),
                "page_url": hit.entity.get("page_url", "unknown"),
                "page_name": hit.entity.get("page_name", "none"),
                "title": hit.entity.get("title", "none"),
                "summary": hit.entity.get("summary", ""),
                "time": hit.entity.get("time", ""),
                "text": hit.entity.get("text", ""),
                "question": hit.entity.get("question", ""),
                "document_index": hit.entity.get("document_index", -1),
            })

    return milvus_rank



# ======================== ES 查询函数 ========================
def query_es_blocks(
    question,
    env="dev",
    top_k=10
):
    _, cfg = get_env_config(env)
    es = get_es(env)
    index_name = cfg["index_name"]

    keywords = jieba.analyse.extract_tags(question, topK=8, withWeight=True)
    keywords = [_[0] for _ in keywords if _[1] > 0.0]

    fields_config = {"text": {"boost": 1}, "title": {"boost": 2}}
    query = build_optimal_jieba_query(keywords, fields_config)
    es_response = es.search(index=index_name, query=query.get("query", {}), size=top_k)

    es_rank = [
        {
            "document_index": hit["_source"].get("document_index", -1),
            "chunk_idx": hit["_source"].get("chunk_idx", -1),
            "page_url": hit["_source"].get("page_url", "unknown"),
            "page_name": hit["_source"].get("page_name", "none"),
            "title": hit["_source"].get("title", "none"),
            "summary": hit["_source"].get("summary", ""),
            "time": hit["_source"].get("time", ""),
            "question": hit["_source"].get("question", ""),
            "text": hit["_source"].get("text", ""),
        } for hit in es_response["hits"]["hits"]
    ]

    return es_rank





# ======================== 多源查询函数 ========================
def query_blocks(
    question,
    embedder,
    env="dev",
    top_k=10,
    reranker=None,
    rerank_top_k=5
):
    # 1. Milvus 查询
    milvus_raw = query_milvus_blocks(
        question=question,
        embedder=embedder,
        env=env,
        top_k=top_k,
        reranker=None,
        rerank_top_k=rerank_top_k
    )

    # 2. ES 查询
    es_raw = query_es_blocks(
        question=question,
        env=env,
        top_k=top_k
    )

    print(f"📥 合并前: Milvus={len(milvus_raw)}, ES={len(es_raw)}")

    # 3. 合并去重（目前仅拼接，如需可替换为 deduplicate_ranked_blocks_pal）
    final_blocks = milvus_raw + es_raw

    # 4. 可选 rerank
    if reranker is not None:
        final_blocks = rerank_results(final_blocks, question, reranker, top_k=rerank_top_k)

    print(f"✅ 最终返回文档块数: {len(final_blocks)}")
    print("📦 返回的文档块示例:")
    for i, doc in enumerate(final_blocks[:5]):
        print(f"  [#{i+1}] document_index={doc.get('document_index',-1)}  page_url={doc['page_url']:<30} chunk_idx={doc['chunk_idx']:<4} title={doc['title'][:30]:<30}")
    return final_blocks


# ======================== Reranker 函数 ========================
class Reranker:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute_score_pairs(self, pairs):
        inputs = self.tokenizer(
            [q for q, a in pairs], [a for q, a in pairs],
            padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
        return scores.cpu().tolist()

def rerank_results(docs, query, reranker, top_k):
    """
    使用 reranker 对检索结果进行精排
    """
    print("🔁 Running Reranker...")
    texts = [d["text"] for d in docs]
    pairs = [(query, t) for t in texts]
    scores = reranker.compute_score_pairs(pairs)

    print("📊 原始顺序及分数:")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"  [#{i+1}] chunk_idx={doc['chunk_idx']:<4} summary={doc['summary'][:30]:<30} score={score:.4f}")

    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [x[0] for x in reranked[:top_k]]

    print("\n🏆 Reranked Top Results:")
    for i, (doc, score) in enumerate(reranked[:top_k]):
        print(f"  [#{i+1}] chunk_idx={doc['chunk_idx']:<4} summary={doc['summary'][:30]:<30} score={score:.4f}")

    print(f"✅ 精排完成，选取前 {top_k} 条")
    return reranked_docs



