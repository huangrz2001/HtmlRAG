import os
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import torch
import jieba.analyse

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


# ======================== Elasticsearch 去重 ========================

def is_duplicate_in_es(
    es,
    index_name,
    text,
    page_name,
    threshold_content=0.9,
    threshold_page_name=0.6,
    top_k=5,
) -> bool:
    """
    基于 ES 查询相似内容块，判断是否重复：
    - 内容相似度 >= 阈值 且
    - 页面名称相似度 >= 阈值
    """
    query = {"query": {"match": {"content": text}}}

    try:
        resp = es.search(index=index_name, body=query, size=top_k)
    except Exception as e:
        print(f"⚠️ Elasticsearch 查询失败: {e}")
        return False

    text_cleaned = clean_text(text)
    page_name_cleaned = clean_text(page_name)

    for hit in resp["hits"]["hits"]:
        content_existing = hit["_source"].get("content", "")
        page_name_existing = hit["_source"].get("page_name", "")

        try:
            vectorizer = TfidfVectorizer(tokenizer=jieba_cut_clean)
            content_sim = cosine_similarity(
                vectorizer.fit_transform([text_cleaned, clean_text(content_existing)])
            )[0, 1]
            title_sim = cosine_similarity(
                vectorizer.fit_transform([page_name_cleaned, clean_text(page_name_existing)])
            )[0, 1]
        except Exception as e:
            print(f"⚠️ 相似度计算失败: {e}")
            continue

        if content_sim >= threshold_content and title_sim >= threshold_page_name:
            print(f"\n⛔️ 内容重复度 {content_sim:.3f}，标题重复度 {title_sim:.3f}，判为重复")
            print("👉 当前文本：", text_cleaned[:300] + ("..." if len(text_cleaned) > 300 else ""))
            print("👉 相似 ES 文本：", clean_text(content_existing)[:300] + ("..." if len(content_existing) > 300 else ""))
            print("👉 当前标题：", page_name_cleaned)
            print("👉 ES 中标题：", clean_text(page_name_existing))
            print("=" * 80)
            return True

    return False


# ======================== 文档内重复块过滤 ========================

def filter_duplicate_blocks(texts: list, threshold=0.9) -> list:
    """
    基于 TF-IDF 向量与 cosine 相似度，过滤重复文本块
    返回保留的索引列表
    """
    if len(texts) <= 1:
        return list(range(len(texts)))

    cleaned_texts = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer(tokenizer=jieba_cut_clean)
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    keep_indices = []
    seen = set()
    for i in range(len(cleaned_texts)):
        if i in seen:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(cleaned_texts)):
            if similarity_matrix[i][j] >= threshold:
                seen.add(j)

    return keep_indices





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
from datetime import datetime
from difflib import SequenceMatcher

from datetime import datetime
from difflib import SequenceMatcher

from datetime import datetime
from difflib import SequenceMatcher

def deduplicate_ranked_blocks(docs: list,
                              threshold_content=0.9,
                              threshold_page_name=0.6,
                              window: int = 3) -> list:
    """
    多窗口滑动去重逻辑（带详细打印）：
    - 若后续 window 个块中存在重复，则用时间更新最新项，继续滑动比较
    - 直到无重复，保留该块并继续下一个
    """
    def parse_time(t: str) -> datetime:
        try:
            return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min

    def str_sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    seen = set()
    keep = []
    i = 0

    while i < len(docs):
        if i in seen:
            i += 1
            continue

        base = docs[i]
        base_text = clean_text(base.get("text", ""))
        base_name = clean_text(base.get("page_name", ""))
        base_time = parse_time(base.get("time", ""))
        best_doc = base
        best_time = base_time

        # print(f"\n🟩 当前基准块 i={i}：")
        # print(f"🔹标题: {base.get('page_name', '')}")
        # print(f"🔹时间: {base.get('time', '')}")
        # print(f"🔹内容前50字: {base.get('text', '')[:50]}")

        for j in range(i + 1, min(i + 1 + window, len(docs))):
            if j in seen:
                continue

            comp = docs[j]
            sim_text = str_sim(base_text, clean_text(comp.get("text", "")))
            sim_name = str_sim(base_name, clean_text(comp.get("page_name", "")))

            if sim_text >= threshold_content and sim_name >= threshold_page_name:
                comp_time = parse_time(comp.get("time", ""))
                seen.add(j)

                # print(f"\n⚠️ 发现重复块 j={j}：")
                # print(f"   - 标题相似度: {sim_name:.3f}，内容相似度: {sim_text:.3f}")
                # print(f"   - 标题: {comp.get('page_name', '')}")
                # print(f"   - 时间: {comp.get('time', '')}")
                # print(f"   - 内容前50字: {comp.get('text', '')[:50]}")

                if comp_time > best_time:
                    seen.add(i)
                    best_doc = comp
                    best_time = comp_time
                    # print("✅ 当前块被替换为较新的重复块")

        keep.append(best_doc)
        i += 1

    print(f"\n✅ 去重完成，原始 {len(docs)} 个块，保留 {len(keep)} 个块\n")
    return keep


# ======================== 文档块分类函数 ========================
def infer_chunk_category(page_url):
    if any(k in page_url for k in ["规则", "制度", "法律", "审核"]):
        return "规则类"
    elif any(k in page_url for k in ["使用", "指南", "帮助", "操作", "功能"]):
        return "操作类"
    elif any(k in page_url for k in ["生态", "角色", "策略", "推广", "平台信息"]):
        return "信息类"
    else:
        return "泛用类"


# ======================== ChatGLM 摘要生成函数 ========================
def generate_summary_ChatGLM(
    text,
    page_url,
    model,
    tokenizer,
    max_new_tokens=150,
):
    if len(text) < max_new_tokens * 2:
        print("⚠️ 文本长度不足，使用原文本")
        return text[:max_new_tokens]

    category = infer_chunk_category(page_url)
    text = text.strip().replace("\x00", "")

    prompt = (
            f"你正在处理一篇电商平台的知识内容，属于“{category}”类。\n"
            f"请你根据下方内容提炼其主要信息，要求如下：\n"
            f"1. 概括要点，不要重复原文原句；\n"
            f"2. 总长度不超过{max_new_tokens}字，使用简体中文；\n"
            f"3. 输出格式为完整一句话。\n"
            f"📂 来源路径：{page_url}\n"
            f"📄 内容：\n{text}"
        )

    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.8,
                temperature=0.4
            )
        # 裁剪掉 prompt 部分
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return response if response else text[:max_new_tokens]
    except Exception as e:
        print(f"⚠️ ChatGLM 摘要生成失败: {e}，使用 fallback")
        return text[:max_new_tokens]



# ======================== ChatGLM 问题生成函数 ========================
def generate_question_ChatGLM(
    text,
    page_url,
    model,
    tokenizer,
    max_new_tokens=64,
    fallback_question="该内容可构造相关业务问题"
):

    category = infer_chunk_category(page_url)
    text = text.strip().replace("\x00", "")

    if category == "规则类":
        hint = "平台是否允许、规则约束、违规处理"
    elif category == "操作类":
        hint = "如何操作、是否可用、使用方法"
    elif category == "信息类":
        hint = "平台背景、产品定位、策略设计"
    else:
        hint = "用户实际可能会问的问题"

    prompt = (
        f"你是一个电商平台知识问答构建助手，请根据以下内容生成一个有实际价值的用户问题。\n"
        f"要求：\n"
        f"- 问题应体现“{hint}”；\n"
        f"- 禁止复述原文，应提炼操作、判断或咨询点；\n"
        f"- 只输出一个简体中文问题句，不加说明。\n"
        f"📂 来源路径：{page_url}\n"
        f"📄 内容：\n{text}"
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return response if response else fallback_question
    except Exception as e:
        print(f"⚠️ ChatGLM 问题生成失败: {e}，使用 fallback")
        return fallback_question




# ======================== 文档块生成函数 ========================
def _generate_block_documents(
    block_tree,
    max_node_words,
    page_url="unknown.html",
    summary_model=None,
    summary_tokenizer=None,
    time_value=""
):
    """
    生成文档块的元信息列表（doc_meta），用于保存为 JSON 或入库。
    打印处理进度和内容摘要信息。
    """
    from utils.text_process_utils import extract_title_from_block, clean_invisible

    path_tags = [b[0] for b in block_tree]
    doc_meta = []

    print(f"📦 共提取块数：{len(path_tags)}")

    for pidx, tag in enumerate(path_tags):
        print(f"\n🧩 正在处理第 {pidx+1}/{len(path_tags)} 个 block")
        text = tag.get_text().strip().replace("\x00", "")
        text = clean_invisible(text)
        if not text:
            print("⚠️ 空内容，跳过")
            continue

        preview = text[:80].replace('\n', ' ') + ("..." if len(text) > 80 else "")
        print(f"📄 文本预览：{preview}")

        title = extract_title_from_block(tag)
        print(f"🏷️ 提取标题：{title[:128]}")

        page_name = os.path.splitext(os.path.basename(page_url))[0]
        summary = ""

        if summary_model and summary_tokenizer:
            summary = generate_summary_ChatGLM(text, page_url, summary_model, summary_tokenizer)
            print(f"✅ 摘要生成成功：{summary}")
            question = generate_question_ChatGLM(text, page_url, summary_model, summary_tokenizer)
            print(f"✅ 问题生成成功：{question}")

        doc_meta.append({
            "chunk_idx": pidx,
            "page_name": page_name,
            "title": title[:128],
            "page_url": page_url,
            "summary": summary,
            "question": question,
            "text": text,
            "time": time_value,
        })

    print(f"\n✅ 所有块处理完毕，共生成 {len(doc_meta)} 条有效文档块")
    return doc_meta


def generate_block_documents(
    block_tree,
    max_node_words,
    page_url="unknown.html",
    summary_model=None,
    summary_tokenizer=None,
    time_value=""
):
    """
    生成结构化文档块，支持表格自动切分，统一生成 summary 和 question。
    """
    from utils.text_process_utils import extract_title_from_block, clean_invisible
    import os

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
                        if summary_model and summary_tokenizer:
                            summary = generate_summary_ChatGLM(text, page_url, summary_model, summary_tokenizer)
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
                if summary_model and summary_tokenizer:
                    summary = generate_summary_ChatGLM(text, page_url, summary_model, summary_tokenizer)
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
            if summary_model and summary_tokenizer:
                summary = generate_summary_ChatGLM(text, page_url, summary_model, summary_tokenizer)
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


