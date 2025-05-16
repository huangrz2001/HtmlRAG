import os
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

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


# ======================== 检索结果去重函数（适用于 Milvus/ES） ========================

def deduplicate_ranked_blocks(docs: list,
                              threshold_content=0.9,
                              threshold_page_name=0.6) -> list:
    """
    去重检索结果列表，判断依据：
    - 内容相似度
    - 页面名相似度
    """
    if len(docs) <= 1:
        return docs

    keep, seen = [], set()

    for i, base in enumerate(docs):
        if i in seen:
            continue

        base_text = clean_text(base.get("text", ""))
        base_name = clean_text(base.get("page_name", ""))
        keep.append(base)

        for j in range(i + 1, len(docs)):
            if j in seen:
                continue

            comp = docs[j]
            comp_text = clean_text(comp.get("text", ""))
            comp_name = clean_text(comp.get("page_name", ""))

            try:
                vectorizer = TfidfVectorizer(tokenizer=jieba_cut_clean)
                sim_text = cosine_similarity(vectorizer.fit_transform([base_text, comp_text]))[0, 1]
                sim_name = cosine_similarity(vectorizer.fit_transform([base_name, comp_name]))[0, 1]
            except Exception as e:
                print(f"⚠️ 相似度计算失败: {e}")
                continue
            if sim_text >= threshold_content and sim_name >= threshold_page_name:
                print(f"\n🔍 比较块 i={i} vs j={j}")
                print(f"📎 内容相似度: {sim_text:.3f}，标题相似度: {sim_name:.3f}")
                print("⛔️ 判为重复，跳过块 j\n" + "=" * 80)
                seen.add(j)

    return keep



def generate_summary_ChatGLM(
        text, model, tokenizer, 
        max_new_tokens=200, 
        min_trigger_length=200, 
        fallback_length=100
    ):
        """
        用 ChatGLM 生成摘要：
        - 如果正文长度小于 min_trigger_length，则直接返回全文；
        - 如果摘要失败，则兜底返回正文前 fallback_length 个字符。
        """
        text = text.strip().replace("\x00", "")
                
        if len(text) < min_trigger_length * 2:
            return text[:min_trigger_length]  # 正文太短，直接返回
        
        prompt = (
            "请你阅读以下内容，并用简洁的语言总结出其主要信息和核心要点，"
            "突出运营策略或平台规则，限制在100字以内：\n\n"
            f"【文档内容】\n{text[:6000]}"
        )
        
        try:
            response, _ = model.chat(
                tokenizer=tokenizer,
                query=prompt,
                history=[],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.8,
            )
            response = response.strip()
            if response:
                return response
            else:
                print("⚠️ ChatGLM 返回空摘要，启用兜底文本。")
                return text[:fallback_length]
        except Exception as e:
            print(f"⚠️ ChatGLM 摘要生成失败: {e}，启用兜底文本。")
            return text[:fallback_length]

def generate_block_documents(
    block_tree,
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
            summary = generate_summary_ChatGLM(text, summary_model, summary_tokenizer)
            print(f"✅ 摘要生成成功：{summary}")

        doc_meta.append({
            "chunk_idx": pidx,
            "page_name": page_name,
            "title": title[:64],
            "page_url": page_url,
            "summary": summary,
            "text": text,
            "time": time_value,
        })

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


