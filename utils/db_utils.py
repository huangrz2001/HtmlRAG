# -*- coding: utf-8 -*-
"""
文档索引与检索核心模块

本模块包含：
- Elasticsearch 索引重建逻辑
- Milvus 向量库结构定义与重建
- HTML 文档块的摘要生成与向量入库
- 多源检索与去重处理逻辑（Milvus + ES）
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
from utils.text_process_utils import extract_title_from_block, deduplicate_ranked_blocks, clean_invisible, generate_summary_ChatGLM


# 加载自定义词典（适用于电商/运营场景）
jieba.load_userdict("./user_dict.txt")

# ======================== ES 索引重建 ========================
def reset_es(host="192.168.7.247", index_name="test_env"):
    """重建 Elasticsearch 索引，含中文分词配置"""
    es = Elasticsearch(f"http://{host}:9200", request_timeout=10)
    print("Connected to ElasticSearch!" if es.ping() else "Connection failed.")

    if es.indices.exists(index=index_name):
        print(f"⚠️ 索引 {index_name} 已存在，删除中...")
        es.indices.delete(index="test_env", ignore_unavailable=True, request_timeout=20)

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
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "chunk_idx": {"type": "integer"},
                    "page_url": {"type": "keyword"},
                    "page_name": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                }
            },
        },
    )


# ======================== Milvus 向量库重建 ========================
def reset_milvus(host="localhost", collection_name="test_env", dim=768):
    """重建 Milvus 向量索引集合，字段与 ES 对齐"""
    connections.connect(alias="default", host=host, port="19530")
    if utility.has_collection(collection_name):
        print(f"⚠️ Milvus 集合 '{collection_name}' 已存在，正在删除...")
        utility.drop_collection(collection_name)

    print(f"🚀 正在创建 Milvus collection: {collection_name}")
    fields = [
        FieldSchema(name="global_chunk_idx", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="chunk_idx", dtype=DataType.INT64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="page_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="time", dtype=DataType.VARCHAR, max_length=128),
    ]

    schema = CollectionSchema(fields=fields, description="HTML块向量索引")
    Collection(name=collection_name, schema=schema)
    print(f"✅ Milvus collection '{collection_name}' 已创建")


# ======================== 文档块插入函数，同时插入Milvus和ES，对应clean步骤的文件 ========================
def insert_block_documents(
    block_tree,
    embedder,
    collection_name="jvliangqianchuan",
    page_url="unknown.html",
    insert_num=0,
    summary_model=None,
    summary_tokenizer=None,
    time_value = ""
):
    """将 HTML block 文档插入 Milvus 与 Elasticsearch"""
    es = Elasticsearch("http://localhost:9200")
    path_tags = [b[0] for b in block_tree]
    paths = [b[1] for b in block_tree]
    doc_meta = []

    for pidx, tag in enumerate(path_tags):
        text = tag.get_text().strip().replace("\x00", "")
        text = clean_invisible(text)
        if not text:
            continue

        title = extract_title_from_block(tag)
        page_name = os.path.splitext(os.path.basename(page_url))[0]
        summary = ""

        if summary_model and summary_tokenizer:
            summary = generate_summary_ChatGLM(text, summary_model, summary_tokenizer)

        print("\n" + "=" * 80)
        print("📄 原文内容：")
        print(text[:500] + ("..." if len(text) > 500 else ""))
        print("-" * 80)
        print("📝 生成摘要：")
        print(summary)
        print("=" * 80 + "\n")

        doc_meta.append({
            "chunk_idx": pidx + insert_num,
            "page_name": page_name,
            "title": title[:64],
            "page_url": page_url,
            "summary": summary,
            "text": text,
            "time": time_value,
        })

    # Elasticsearch 插入
    actions = [
        {
            "_index": collection_name,
            "_id": f"{doc['page_url']}#{doc['chunk_idx']}",
            "_source": {
                "chunk_idx": doc["chunk_idx"],
                "title": doc["title"],
                "summary": doc.get("summary", ""),
                "text": doc["text"],
                "page_url": doc["page_url"],
                "page_name": doc["page_name"],
                "time": doc.get("time", ""),
            }
        } for doc in doc_meta
    ]
    bulk(es, actions)
    print(f"✅ 已插入 ES：{len(doc_meta)} 条文档块")

    # Milvus 插入
    connections.connect(alias="default", host="localhost", port="19530")
    print(f"🧠 正在插入向量到 Milvus collection: {collection_name} ...")

    node_docs = [Document(page_content=doc["text"], metadata=doc) for doc in doc_meta]
    Milvus.from_documents(
        node_docs,
        embedder,
        collection_name=collection_name,
        connection_args={
            "host": "localhost",
            "port": "19530",
            "field_map": {
                "chunk_idx": "chunk_idx",
                "title": "title",
                "text": "text",
                "page_name": "page_name",
                "page_url": "page_url",
                "summary": "summary",
                "time": "time",
            },
        },
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 64},
        },
    )
    print(f"✅ 已插入 Milvus：{len(doc_meta)} 条向量")
    return len(doc_meta)



# ======================== 插入 Milvus ========================
def insert_block_to_milvus(doc_meta_list, embedder, host, collection_name, cnt):
    connections.connect(alias="default", host=host, port="19530")
    print(f"🧠 正在插入向量到 Milvus collection: {collection_name} ...")

    new_docs = []
    for doc in doc_meta_list:
        local_idx = doc["chunk_idx"]
        global_idx = cnt + local_idx
        doc["global_chunk_idx"] = global_idx
        node = Document(page_content=doc["text"][:65535], metadata=doc)
        new_docs.append(node)
    Milvus.from_documents(
        new_docs,
        embedder,
        collection_name=collection_name,
        connection_args={
            "host": "localhost",
            "port": "19530",
            "field_map": {
                "chunk_idx": "chunk_idx",
                "global_chunk_idx": "global_chunk_idx",
                "title": "title",
                "text": "text",
                "page_name": "page_name",
                "page_url": "page_url",
                "summary": "summary",
                "time": "time",
            },
        },
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 64},
        },
    )
    print(f"✅ 已插入 Milvus：{len(doc_meta_list)} 条向量")
    return cnt + len(doc_meta_list)


# ======================== 插入 ES ========================
def insert_block_to_es(doc_meta_list, host, es_index_name, cnt):
    es = Elasticsearch(f"http://{host}:9200")
    print(f"📥 正在插入文档到 Elasticsearch 索引: {es_index_name} ...")

    actions = []
    for doc in doc_meta_list:
        local_idx = doc["chunk_idx"]
        global_idx = cnt + local_idx
        doc["global_chunk_idx"] = global_idx

        actions.append({
            "_index": es_index_name,
            "_id": f"{doc['page_url']}#{global_idx}",
            "_source": {
                "chunk_idx": local_idx,
                "global_chunk_idx": global_idx,
                "title": doc["title"],
                "summary": doc.get("summary", ""),
                "text": doc["text"],
                "page_url": doc["page_url"],
                "page_name": doc["page_name"],
                "time": doc.get("time", ""),
            }
        })

    bulk(es, actions)
    print(f"✅ 已插入 ES：{len(doc_meta_list)} 条文档块")
    return cnt + len(doc_meta_list)


# ======================== Milvus 查询函数 ========================
def query_milvus_blocks(
    question,
    embedder,
    reranker=None,
    milvus_collection_name="jvliangqianchuan",
    top_k=10,
    rerank_top_k=5,
    include_content=True,
):
    print(f"\n🔍 Query (Milvus): {question}")

    print("📦 Connecting to Milvus ...")
    connections.connect(alias="default", host="localhost", port="19530")

    collection = Collection(name=milvus_collection_name)
    collection.load()
    collection.flush()

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

    query_vec = embedder.embed_query(question)
    print(f"✅ 查询向量维度: {len(query_vec)}, 范数: {np.linalg.norm(query_vec):.4f}")

    print("🔎 Searching Milvus ...")
    results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 100}},
        limit=top_k,
        output_fields=["text", "page_url", "chunk_idx", "page_name", "title", "summary", "time"],
    )
    print(results)

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
                "text": hit.entity.get("text", "") if include_content else "",
            })

    print(f"🔎 Milvus 初始返回数量: {len(milvus_rank)}")
    milvus_rank = deduplicate_ranked_blocks(milvus_rank)
    print(f"✅ 去重后保留数量: {len(milvus_rank)}")

    if reranker is not None:
        milvus_rank = rerank_results(milvus_rank, question, reranker, rerank_top_k)

    print("=" * 60)
    print("[Milvus] Top Results:")
    for i, doc in enumerate(milvus_rank):
        print(f"#{i+1} 🔸 Chunk ID: {doc['chunk_idx']} | summary: {doc['summary']}")
        print("-" * 100)
    print("=" * 60)

    return milvus_rank


# ======================== ES 查询函数 ========================
def query_es_blocks(
    question,
    es_index_name="jvliangqianchuan",
    top_k=10,
    include_content=True,
):
    print(f"\n🔍 Query (Elasticsearch): {question}")
    from elasticsearch import Elasticsearch
    import jieba
    from utils.text_process_utils import deduplicate_ranked_blocks

    es = Elasticsearch("http://localhost:9200")

    def jieba_tokenize(text):
        return [t for t in jieba.cut(text) if t.strip()]

    def build_optimal_jieba_query(tokens, field_cfg):
        should_clauses = []
        for token in tokens:
            for field, cfg in field_cfg.items():
                clause = {
                    "match": {
                        field: {
                            "query": token,
                            "boost": cfg["boost"]
                        }
                    }
                }
                if cfg.get("fuzzy"):
                    clause["match"][field]["fuzziness"] = "AUTO"
                should_clauses.append(clause)
        return {"query": {"bool": {"should": should_clauses}}}

    keywords = jieba_tokenize(question)
    es_query = build_optimal_jieba_query(keywords, {"content": {"boost": 1.0, "fuzzy": True}})
    es_response = es.search(index=es_index_name, body=es_query, size=top_k)

    es_rank = [
        {
            "chunk_idx": hit["_source"].get("chunk_idx", -1),
            "page_url": hit["_source"].get("page_url", "unknown"),
            "page_name": hit["_source"].get("page_name", "none"),
            "title": hit["_source"].get("title", "none"),
            "summary": hit["_source"].get("summary", ""),
            "time": hit["_source"].get("time", ""),
            "content": hit["_source"].get("content", "") if include_content else "",
        } for hit in es_response["hits"]["hits"]
    ]

    es_rank = deduplicate_ranked_blocks(es_rank)
    print(f"[ES] Top {top_k}:")
    for doc in es_rank:
        print("-" * 100)
        print(doc)
    print("-" * 100)
    print("=" * 60)

    return es_rank



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
