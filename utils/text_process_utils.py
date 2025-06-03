
# -*- coding: utf-8 -*-
"""
文本处理与文档块生成模块（用于 HTML-RAG 知识库构建）

本模块负责从 HTML 清洗结果中提取结构化文档块，生成摘要与问句，并提供查询构建、文本清洗与去重能力，
是智能客服/文档问答等 RAG 系统中“切块-摘要-向量化”流程的核心组件。

核心功能结构：
------------------------------------------------
1. 文本清洗与分词：
   - `clean_text`: 移除所有非中英文与数字字符。
   - `jieba_cut_clean`: 清洗后进行自定义分词（支持外部 user_dict）。
   - `clean_invisible`: 清除零宽与控制类不可见字符。

2. 语义块标题提取：
   - `extract_title_from_block`: 提取 HTML 块中的第一个标题或非空文本作为 chunk title。

3. 查询构建（用于 ES 倒排检索）：
   - `build_optimal_jieba_query`: 综合精确匹配、模糊查询、短语匹配与同义词扩展构建结构化 bool 查询。

4. 相似内容去重：
   - `deduplicate_ranked_blocks_pal`: 基于 TF-IDF 和 cosine 相似度计算文本和页面名的相似性，按时间优先保留最优版本。

5. 文档块生成：
   - `generate_block_documents`: 将结构化 HTML 节点生成带 metadata 的 chunk 列表，支持表格行切分、摘要生成、问句构造。
     - 支持摘要生成方式：
       - `generate_summary_ChatGLM`（调用 ChatGLM 接口）
       - `generate_summary_vllm`（使用 vLLM HTTP 接口）
     - 可选生成问句 `generate_question_ChatGLM`

6. 块数据持久化：
   - `save_doc_meta_to_block_dir`: 将 HTML 块的结构化信息以 JSON 格式写入指定路径，路径结构与原始 HTML 保持一致。

   """

import os
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import torch
from difflib import SequenceMatcher
from utils.llm_api import generate_summary_ChatGLM, generate_question_ChatGLM, generate_summary_vllm, generate_summary_vllm_async
import numpy as np
from collections import defaultdict
import aiohttp
import asyncio
import time


# 关闭并行化警告，避免控制台冗余信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化自定义分词器，不使用全局影响的 jieba 加载词典
pure_tokenizer = jieba.Tokenizer(dictionary=jieba.DEFAULT_DICT)
jieba.load_userdict("./user_dict.txt")


# ======================== 文本处理工具函数 ========================

def clean_text(text: str) -> str:
    """移除文本中的特殊字符，仅保留中英文与数字"""
    return "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text))


def jieba_cut_clean(text: str) -> list:
    """结合 clean_text 与自定义分词器进行分词处理"""
    text = clean_text(text)
    return list(pure_tokenizer.cut(text, HMM=False))


def clean_invisible(text):
    # 去除所有 Unicode 控制字符
    return re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f]', '', text)

# ======================== 文档块标题提取函数 ========================

def extract_title_from_block(tag) -> str:
    """
    从 HTML tag 中提取第一个 heading 标签（h1~h6）作为标题
    若不存在，则回退为第一个非空文本
    """
    from bs4 import Tag

    for descendant in tag.descendants:
        if isinstance(descendant, Tag) and descendant.name and descendant.name.lower().startswith("h"):
            return descendant.get_text(separator="", strip=True)[:48]

    for t in tag.stripped_strings:
        if t.strip():
            return t.strip()[:48]
    return ""



# ======================== 优化的 Jieba 查询构建函数 ========================
def build_optimal_jieba_query(
    jieba_keywords, fields_config, synonym_map=None, use_phrase=True, use_fuzzy=True
):
    """
    综合多种技术的优化查询，增强同义词的使用

    :param jieba_keywords: jieba库所提取的关键词
    :param fields_config: {'title': {'boost':5, 'fuzzy':False}, ...}
    :param synonym_map: 同义词词典，格式: {'关键词': ['同义词1', '同义词2']}
    """
    should_clauses = []

    for word in jieba_keywords:
        # 获取关键词及其同义词
        synonyms = synonym_map.get(word, [word]) if synonym_map else [word]

        for field, config in fields_config.items():
            boost = config.get("boost", 1)

            # 1. 为每个关键词及其同义词构建OR查询
            synonym_queries = []

            # 精确匹配（使用terms查询替代多个term查询）
            if len(synonyms) > 0:
                synonym_queries.append(
                    {"terms": {f"{field}.keyword": synonyms, "boost": boost * 1.2}}
                )

            # 模糊匹配
            if use_fuzzy and config.get("fuzzy", True):
                for syn in synonyms:
                    synonym_queries.append(
                        {
                            "match": {
                                field: {
                                    "query": syn,
                                    "fuzziness": "AUTO",
                                    "boost": boost * 0.5,
                                }
                            }
                        }
                    )

            # 短语匹配
            if use_phrase and len(word) > 1:
                for syn in synonyms:
                    synonym_queries.append(
                        {
                            "match_phrase": {
                                field: {"query": syn, "slop": 2,
                                        "boost": boost * 0.8}
                            }
                        }
                    )

            # 将所有同义词相关的查询组合到一个bool查询中
            if synonym_queries:
                should_clauses.append(
                    {"bool": {"should": synonym_queries, "minimum_should_match": 1}}
                )

    return {
        "query": {"bool": {"should": should_clauses, "minimum_should_match": "30%"}},
        "highlight": {
            "fields": {
                "*": {
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                }  # 添加简单的高亮标签
            }
        },
    }


# ======================== 检索结果去重函数（适用于 Milvus/ES） ========================
def parse_time(t: str) -> datetime:
    try:
        return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.min

def str_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def clean_text(text: str) -> str:
    """移除文本中的特殊字符，仅保留中英文与数字"""
    return "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text))


def deduplicate_ranked_blocks_pal(docs, threshold_content=0.9, threshold_page_name=0.6):
    n = len(docs)
    if n <= 1:
        return docs

    texts = [clean_text(doc.get("text", "")) for doc in docs]
    names = [clean_text(doc.get("page_name", "")) for doc in docs]
    times = np.array([parse_time(doc.get("time", "")) for doc in docs])

    tfidf = TfidfVectorizer().fit(texts + names)
    text_vecs = tfidf.transform(texts)
    name_vecs = tfidf.transform(names)

    sim_text = cosine_similarity(text_vecs)
    sim_name = cosine_similarity(name_vecs)

    # 上三角重复对
    triu_idx = np.triu_indices(n, k=1)
    sim_mask = (sim_text[triu_idx] >= threshold_content) & (sim_name[triu_idx] >= threshold_page_name)
    dup_pairs = list(zip(triu_idx[0][sim_mask], triu_idx[1][sim_mask]))

    # 构建重复簇：用图表示
    graph = defaultdict(set)
    for i, j in dup_pairs:
        graph[i].add(j)
        graph[j].add(i)

    visited = set()
    keep = set()

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for i in range(n):
        if i not in visited:
            group = []
            dfs(i, group)
            if len(group) == 1:
                keep.add(group[0])
            else:
                latest = max(group, key=lambda x: times[x])
                keep.add(latest)

    kept = sorted(list(keep))
    print(f"✅ 原始 {n} 个块，重复对 {len(dup_pairs)}，去重后保留 {len(kept)}")
    return [docs[i] for i in kept]

def save_doc_meta_to_block_dir(doc_meta, html_path, html_root_dir, block_root_dir):
    """
    保存 JSON 文件，路径映射：
    html_path = a/b/c.html → 保存为 a_blocks/b/c.json
    """
    # 相对路径：b/c.html
    rel_path = os.path.relpath(html_path, html_root_dir)

    # 输出路径：a_blocks/b/c.json
    rel_json_path = os.path.splitext(rel_path)[0] + ".json"
    json_full_path = os.path.join(block_root_dir, rel_json_path)

    # 创建目标目录
    os.makedirs(os.path.dirname(json_full_path), exist_ok=True)

    # 写入 JSON 文件
    with open(json_full_path, "w", encoding="utf-8") as f:
        json.dump(doc_meta, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON 已保存：{json_full_path}")
    return json_full_path




# ======================== 文档块生成函数 ========================
def generate_block_documents(
    block_tree,
    max_node_words,
    page_url="unknown.html",
    summary_model=None,
    summary_tokenizer=None,
    time_value="",
    gen_question=False,
    use_vllm=True,
):
    """
    生成结构化文档块，支持表格自动切分，统一生成 summary 和 question。
    """

    path_tags = [b[0] for b in block_tree]
    doc_meta = []
    chunk_idx = 0

    print(f"📦 共提取块数：{len(path_tags)}")

    for pidx, tag in enumerate(path_tags):
        print(f"\n🧩 正在处理第 {pidx+1}/{len(path_tags)} 个 block")

        page_name = os.path.splitext(os.path.basename(page_url))[0]
        title = extract_title_from_block(tag)
        print(f"🏷️ 提取标题：{title[:128]}")

        is_table_block = (tag.name == "table") or tag.find("table") is not None

        if is_table_block:
            print("📊 表格类型，执行按行拼接切分")
            table = tag.find("table") if tag.name != "table" else tag
            rows = table.find_all("tr")
            print(f"📊 表格行数：{len(rows)}")
            if not rows:
                continue

            def row_to_text(row):
                return " ".join(cell.strip() for cell in row.stripped_strings) + "\n"

            header_text = row_to_text(rows[0])
            current_text = header_text
            current_words = len(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", header_text))
            start_row = 1  # header 是第1行
            row_range_start = 1

            for idx, row in enumerate(rows[1:], start=2):
                row_text = row_to_text(row)
                row_words = len(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", row_text))

                if current_words + row_words > max_node_words:
                    # ✅ 提交当前块
                    text = clean_invisible(current_text.strip())
                    if text:
                        summary, question = "", ""
                        if use_vllm:
                            generate_summary_vllm(text, page_url)
                        elif summary_model and summary_tokenizer:
                            summary = generate_summary_ChatGLM(text, page_url, summary_model, summary_tokenizer)
                            if gen_question:
                                question = generate_question_ChatGLM(text, page_url, summary_model, summary_tokenizer)

                        title_with_range = f"{title[:96]} 表格行{row_range_start}-{idx-1}"
                        doc_meta.append({
                            "chunk_idx": chunk_idx,
                            "page_name": page_name,
                            "title": title_with_range,
                            "page_url": page_url,
                            "summary": summary,
                            "question": question,
                            "text": text,
                            "time": time_value,
                        })
                        chunk_idx += 1

                    # ✅ 重置
                    current_text = header_text + row_text
                    current_words = len(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", current_text))
                    row_range_start = idx
                else:
                    current_text += row_text
                    current_words += row_words

            # ✅ 提交最后一个块
            text = clean_invisible(current_text.strip())
            if text:
                summary, question = "", ""
                if use_vllm:
                    generate_summary_vllm(text, page_url)
                elif summary_model and summary_tokenizer:
                    summary = generate_summary_ChatGLM(text, page_url, summary_model, summary_tokenizer)
                    if gen_question:
                        question = generate_question_ChatGLM(text, page_url, summary_model, summary_tokenizer)
                title_with_range = f"{title[:96]} 表格行{row_range_start}-{len(rows)}"
                doc_meta.append({
                    "chunk_idx": chunk_idx,
                    "page_name": page_name,
                    "title": title_with_range,
                    "page_url": page_url,
                    "summary": summary,
                    "question": question,
                    "text": text,
                    "time": time_value,
                })
                chunk_idx += 1

        else:
            # text = tag.get_text().strip().replace("\x00", "")
            text = tag.get_text().replace("\x00", "")  # 不 strip()，保留换行

            text = clean_invisible(text)
            if not text:
                print("⚠️ 空内容，跳过")
                continue

            preview = text[:80].replace('\n', ' ') + ("..." if len(text) > 80 else "")
            print(f"📄 文本预览：{preview}")

            summary, question = "", ""
            if use_vllm:
                generate_summary_vllm(text, page_url)
            elif summary_model and summary_tokenizer:
                summary = generate_summary_ChatGLM(text, page_url, summary_model, summary_tokenizer)
                if gen_question:
                    question = generate_question_ChatGLM(text, page_url, summary_model, summary_tokenizer)
            doc_meta.append({
                "chunk_idx": chunk_idx,
                "page_name": page_name,
                "title": title[:128],
                "page_url": page_url,
                "summary": summary,
                "question": question,
                "text": text,
                "time": time_value,
            })
            chunk_idx += 1

    print(f"\n✅ 所有块处理完毕，共生成 {len(doc_meta)} 条有效文档块")
    return doc_meta



async def generate_block_documents_async(
    block_tree,
    max_node_words,
    page_url="unknown.html",
    summary_model=None,
    summary_tokenizer=None,
    time_value="",
    gen_question=False,
    use_vllm=True,
    batch_size=32
):
    path_tags = [b[0] for b in block_tree]
    doc_meta, chunk_idx, tasks = [], 0, []
    page_name = os.path.splitext(os.path.basename(page_url))[0]

    def row_to_text(row):
        return " ".join(cell.strip() for cell in row.stripped_strings) + "\n"

    for tag in path_tags:
        title = extract_title_from_block(tag)
        is_table_block = (tag.name == "table") or tag.find("table") is not None

        if is_table_block:
            table = tag.find("table") if tag.name != "table" else tag
            rows = table.find_all("tr")
            if not rows:
                continue

            header_text = row_to_text(rows[0])
            current_text = header_text
            current_words = len(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", header_text))
            row_range_start = 1

            for idx, row in enumerate(rows[1:], start=2):
                row_text = row_to_text(row)
                row_words = len(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", row_text))

                if current_words + row_words > max_node_words:
                    text = clean_invisible(current_text.strip())
                    if text:
                        doc_meta.append({
                            "chunk_idx": chunk_idx,
                            "page_name": page_name,
                            "title": f"{title[:96]} 表格行{row_range_start}-{idx-1}",
                            "page_url": page_url,
                            "summary": "",
                            "question": "",
                            "text": text,
                            "time": time_value,
                        })
                        tasks.append((chunk_idx, text, page_url))
                        chunk_idx += 1
                    current_text = header_text + row_text
                    current_words = len(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", current_text))
                    row_range_start = idx
                else:
                    current_text += row_text
                    current_words += row_words

            text = clean_invisible(current_text.strip())
            if text:
                doc_meta.append({
                    "chunk_idx": chunk_idx,
                    "page_name": page_name,
                    "title": f"{title[:96]} 表格行{row_range_start}-{len(rows)}",
                    "page_url": page_url,
                    "summary": "",
                    "question": "",
                    "text": text,
                    "time": time_value,
                })
                tasks.append((chunk_idx, text, page_url))
                chunk_idx += 1

        else:
            text = clean_invisible(tag.get_text().replace("\x00", ""))
            if not text:
                continue
            doc_meta.append({
                "chunk_idx": chunk_idx,
                "page_name": page_name,
                "title": title[:128],
                "page_url": page_url,
                "summary": "",
                "question": "",
                "text": text,
                "time": time_value,
            })
            tasks.append((chunk_idx, text, page_url))
            chunk_idx += 1

    print(f"\n🚀 开始分批并发生成 {len(tasks)} 个摘要 ...")
    start = time.time()

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        summaries = await asyncio.gather(*[
            generate_summary_vllm_async(text, url) for _, text, url in batch
        ])
        for j, (chunk_idx_i, _, _) in enumerate(batch):
            doc_meta[chunk_idx_i]["summary"] = summaries[j]

    print(f"✅ 摘要生成完成（耗时 {time.time() - start:.2f}s）")
    return doc_meta

