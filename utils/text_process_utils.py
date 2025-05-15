import os
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

        base_text = clean_text(base.get("content", ""))
        base_name = clean_text(base.get("page_name", ""))
        keep.append(base)

        for j in range(i + 1, len(docs)):
            if j in seen:
                continue

            comp = docs[j]
            comp_text = clean_text(comp.get("content", ""))
            comp_name = clean_text(comp.get("page_name", ""))

            try:
                vectorizer = TfidfVectorizer(tokenizer=jieba_cut_clean)
                sim_text = cosine_similarity(vectorizer.fit_transform([base_text, comp_text]))[0, 1]
                sim_name = cosine_similarity(vectorizer.fit_transform([base_name, comp_name]))[0, 1]
            except Exception as e:
                print(f"⚠️ 相似度计算失败: {e}")
                continue

            print(f"\n🔍 比较块 i={i} vs j={j}")
            print(f"📎 内容相似度: {sim_text:.3f}，标题相似度: {sim_name:.3f}")
            print(f"👉 块{i} 内容: {base_text[:80]}...")
            print(f"👉 块{j} 内容: {comp_text[:80]}...")
            print(f"👉 块{i} 页面名: {base_name}")
            print(f"👉 块{j} 页面名: {comp_name}")

            if sim_text >= threshold_content and sim_name >= threshold_page_name:
                print("⛔️ 判为重复，跳过块 j\n" + "=" * 80)
                seen.add(j)
            else:
                print("✅ 保留块 j\n" + "-" * 80)

    return keep
