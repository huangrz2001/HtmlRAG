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
   - `delete_blocks_from_milvus`: 按 file_idx 删除 Milvus 向量。
   - `query_milvus_blocks`: 基于语义向量进行 ANN 检索，支持 reranker 精排。

2. 关键词库（Elasticsearch）管理：
   - `reset_es`: 重建 Elasticsearch 索引结构，使用 IK 分词器进行中文优化。
   - `insert_block_to_es`: 批量插入文档块文本至 Elasticsearch。
   - `delete_blocks_from_es`: 按 file_idx 删除 ES 文档块。
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
from utils.config import CONFIG

# 加载自定义词典（适用于电商/运营场景）
jieba.load_userdict("./user_dict.txt")


index_name = CONFIG.get("index_name", "curd_env")
Milvus_host = CONFIG.get('milvus_host', "192.168.7.247")
ES_host = CONFIG.get('es_host', "192.168.7.247")


# 全局 Elasticsearch 和 Milvus 客户端（内建连接池）
es_client = Elasticsearch(f"http://{ES_host}:9200")
connections.connect(alias="default", host=Milvus_host, port="19530")


def get_es():
    """返回全局 ES 客户端"""
    return es_client

def get_milvus_collection(collection_name):
    """返回已连接的 Milvus collection 实例"""
    col = Collection(name=collection_name)
    col.load()
    return col


# ======================== 获取最大全局索引(milvus) ========================
def get_max_global_idx_milvus(host, collection_name):
    try:
        connections.connect(alias="default", host=host, port="19530")
        collection = Collection(name=collection_name)
        collection.load()
        results = collection.query(
            expr="global_chunk_idx >= 0",
            output_fields=["global_chunk_idx"],
        )
        if not results:
            return 0
        # 提取所有 idx，取最大值
        max_idx = max(item["global_chunk_idx"] for item in results)
        return max_idx + 1
    except Exception as e:
        print(f"⚠️ 查询失败: {e}")
        return 0


# ======================== ES 索引重建 ========================
def reset_es(index_name=index_name):
    """重建 Elasticsearch 索引，使用 IK 分词器，去除 global_chunk_idx，增加 file_idx"""
    es = get_es()
    print("Connected to ElasticSearch!" if es.ping() else "Connection failed.")

    if es.indices.exists(index=index_name):
        print(f"⚠️ 索引 {index_name} 已存在，删除中...")
        es.indices.delete(index=index_name, ignore_unavailable=True, request_timeout=20)

    es.indices.create(
        index=index_name,
        body={
            "settings": {
                "analysis": {
                    "analyzer": {
                        "ik_max_word": {
                            "type": "custom",
                            "tokenizer": "ik_max_word"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "file_idx": { "type": "long" },
                    "chunk_idx": { "type": "integer" },
                    "text": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "fields": { "keyword": { "type": "keyword" } }
                    },
                    "page_url": {
                        "type": "text",
                        "fields": { "keyword": { "type": "keyword" } }
                    },
                    "page_name": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "fields": { "keyword": { "type": "keyword" } }
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "fields": { "keyword": { "type": "keyword" } }
                    },
                    "summary": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "fields": { "keyword": { "type": "keyword" } }
                    },
                    "time": {
                        "type": "text",
                        "fields": { "keyword": { "type": "keyword" } }
                    },
                    "question": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "fields": { "keyword": { "type": "keyword" } }
                    }
                }
            }
        },
    )
    print(f"✅ ES 索引 '{index_name}' 已成功创建（含 file_idx）")

# ======================== Milvus 向量库重建 ========================
def reset_milvus(collection_name=index_name, dim=768):
    """重建 Milvus 向量集合，含主键自增和 file_idx 字段"""
    # ⚠️ 注意：连接初始化已在 connections.py 执行
    if utility.has_collection(collection_name):
        print(f"⚠️ Milvus 集合 '{collection_name}' 已存在，正在删除...")
        utility.drop_collection(collection_name)

    print(f"🚀 正在创建 Milvus collection: {collection_name}")
    fields = [
        FieldSchema(name="global_chunk_idx", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_idx", dtype=DataType.INT64),
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
    Collection(name=collection_name, schema=schema)
    print(f"✅ Milvus collection '{collection_name}' 已创建（含主键自增 + file_idx）")

# ======================== 插入 Milvus ========================
def insert_block_to_milvus(doc_meta_list, embedder, collection_name, batch_size=100) -> int:
    print(f"🧠 正在插入向量到 Milvus collection: {collection_name} ...")
    all_docs = []
    for doc in doc_meta_list:
        # doc.setdefault("file_idx", -1)
        node = Document(
            page_content=doc["text"],
            metadata={k: v for k, v in doc.items() if k != "text"}
        )
        all_docs.append(node)

    milvus = Milvus.from_documents(
        [all_docs[0]],
        embedder,
        collection_name=collection_name,
        connection_args={"host": CONFIG["milvus_host"], "port": "19530"},
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
            print(f"✅ 插入 batch {i // batch_size + 1}: {len(batch)} 条")
        except Exception as e:
            print(f"❌ 插入 batch 失败: {e}")
    return inserted

# ======================== 插入 ES ========================
def insert_block_to_es(doc_meta_list, es_index_name) -> int:
    es = get_es()
    print(f"📥 正在插入文档到 Elasticsearch 索引: {es_index_name} ...")

    actions = []
    for doc in doc_meta_list:
        doc.setdefault("file_idx", -1)
        actions.append({
            "_index": es_index_name,
            "_source": {
                "file_idx": doc["file_idx"],
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
        print(f"✅ 已插入 ES：{resp[0]} 条文档块")
        return resp[0]  # 成功数
    except Exception as e:
        print(f"❌ ES 插入失败: {e}")
        return 0

# ======================== 删除 Milvus 中的文档块 ========================
def delete_blocks_from_milvus(collection_name, file_idx) -> int:
    try:
        col = get_milvus_collection(collection_name)
        expr = f"file_idx == {file_idx}"
        count = col.num_entities  # 删除前实体总数（可能非严格对应）
        result = col.delete(expr)
        print(f"🗑️ Milvus: 已删除 file_idx = {file_idx} 的文档块")
        return result.delete_count if hasattr(result, "delete_count") else 0
    except Exception as e:
        print(f"❌ Milvus 删除失败: {e}")
        return 0

# ======================== 删除 ES 中的文档块 ========================
def delete_blocks_from_es(index_name, file_idx) -> int:
    try:
        es = get_es()
        query = {
            "query": {
                "term": {
                    "file_idx": file_idx
                }
            }
        }

        hits = es.search(index=index_name, body=query, size=10000)["hits"]["hits"]
        if not hits:
            print(f"📭 ES: 未找到 file_idx = {file_idx} 的文档")
            return 0

        resp = helpers.bulk(
            client=es,
            actions=[
                {"_op_type": "delete", "_index": index_name, "_id": hit["_id"]}
                for hit in hits
            ]
        )
        print(f"🗑️ ES: 已删除 file_idx = {file_idx} 的文档块，共 {resp[0]} 条")
        return resp[0]
    except Exception as e:
        print(f"❌ ES 删除失败: {e}")
        return 0

# ======================== Milvus 查询函数 ========================
def query_milvus_blocks(
    host,
    question,
    embedder,
    reranker=None,
    milvus_collection_name="jvliangqianchuan",
    top_k=10,
    rerank_top_k=5
):

    # print("📦 Connecting to Milvus ...")
    connections.connect(alias="default", host=host, port="19530")
    collection = Collection(name=milvus_collection_name)
    if not collection.has_index():
        print(f"⚙️ Creating index on 'vector' ...")
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
    # print(f"✅ 查询向量维度: {len(query_vec)}, 范数: {np.linalg.norm(query_vec):.4f}")

    # print("🔎 Searching Milvus ...")
    results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 100}},
        limit=top_k,
        output_fields=["text", "page_url", "chunk_idx", "page_name", "title", "summary", "time", "question", 'file_idx'],
    )
    # print(results)

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
            })

    # print(f"🔎 Milvus 初始返回数量: {len(milvus_rank)}")
    # milvus_rank = deduplicate_ranked_blocks(milvus_rank)
    # print(f"✅ 去重后保留数量: {len(milvus_rank)}")

    # if reranker is not None:
    #     milvus_rank = rerank_results(milvus_rank, question, reranker, rerank_top_k)

    # print("=" * 60)
    # print("[Milvus] Top Results:")
    # for i, doc in enumerate(milvus_rank):
    #     print(f"#{i+1} 🔸 Chunk ID: {doc['chunk_idx']} | summary: {doc['summary']}")
    #     print("-" * 100)
    # print("=" * 60)

    return milvus_rank


# ======================== ES 查询函数 ========================
def query_es_blocks(
    host,
    question,
    es_index_name="jvliangqianchuan",
    top_k=10
):
    es = Elasticsearch(f"http://{host}:9200")
    
    keywords = jieba.analyse.extract_tags(question, topK=8, withWeight=True)
    keywords = [_[0] for _ in keywords if _[1] > 0.0]
    # print("Keywords:", keywords)

    fields_config = {"text": {"boost": 1}, "title": {"boost": 2}}
    query = build_optimal_jieba_query(keywords, fields_config)
    es_response = es.search(index=es_index_name, query=query.get("query", {}), size=top_k)

    es_rank = [
        {
            "file_idx": hit["_source"].get("file_idx", -1),
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
    # print(f"🔎 ES 初始返回数量: {len(es_rank)}")
    # milvus_rank = deduplicate_ranked_blocks(es_rank)
    # print(f"✅ 去重后保留数量: {len(es_rank)}")
    return es_rank



# ======================== 多源查询函数 ========================
def query_blocks(
    question,
    embedder,
    host="localhost",
    milvus_collection_name="jvliangqianchuan",
    es_index_name="jvliangqianchuan",
    top_k=10,
    reranker=None,
    rerank_top_k=5
):
    # 1. 查询 Milvus
    milvus_raw = query_milvus_blocks(
        host=host,
        question=question,
        embedder=embedder,
        reranker=None,
        milvus_collection_name=milvus_collection_name,
        top_k=top_k,
        rerank_top_k=rerank_top_k
    )

    # 2. 查询 ES
    es_raw = query_es_blocks(
        host=host,
        question=question,
        es_index_name=es_index_name,
        top_k=top_k
    )

    print(f"📥 合并前: Milvus={len(milvus_raw)}, ES={len(es_raw)}")

    # 3. 时间优先去重：先内部再交叉（Milvus against ES）
    # final_blocks = deduplicate_ranked_blocks_pal(milvus_raw + es_raw)
    final_blocks = milvus_raw + es_raw
    # 4. 可选重排序
    if reranker is not None:
        final_blocks = rerank_results(final_blocks, question, reranker, top_k=rerank_top_k)

    print(f"✅ 最终返回文档块数: {len(final_blocks)}")
    print("📦 返回的文档块示例:")
    for i, doc in enumerate(final_blocks[:5]):
        # print(doc)
        print(f"  [#{i+1}] file_idx={doc.get('file_idx',-1)}  page_url={doc['page_url']:<30} chunk_idx={doc['chunk_idx']:<4} title={doc['title'][:30]:<30}")
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



