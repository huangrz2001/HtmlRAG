from typing import List, Dict
import random
import json
from openai import OpenAI
import os
from pymilvus import connections, Collection
import random
from utils.db_utils import query_milvus_blocks, query_es_blocks, deduplicate_ranked_blocks
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os
import json
import random
from typing import List, Dict
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


os.environ["OPENAI_API_KEY"] = "sk-d14cb74d17974a1fb4db738eaf5e15e6"

# ✅ 初始化 DeepSeek 模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("sk-d14cb74d17974a1fb4db738eaf5e15e6"),
    base_url="https://api.deepseek.com",
)


def format_chunk_for_reranker(chunk: Dict) -> str:
    """
    将 chunk 中的结构化字段拼接为适用于 reranker 的 doc 字段。
    """
    page_name = chunk.get("page_name", "").strip()
    title = chunk.get("title", "").strip()
    page_url = chunk.get("page_url", "").strip()
    summary = chunk.get("summary", "").strip()
    text = chunk.get("text", "").strip()

    return (
        f"【页面名称】：{page_name}\n"
        f"【段落标题】：{title}\n"
        f"【来源路径】：{page_url}\n"
        f"【摘要】：{summary}\n"
        f"【正文】：{text}"
    )


def build_contrastive_prompt_with_selection(chunks: List[Dict]) -> str:
    prompt = (
        "你是一个电商平台智能问答系统训练数据构建助手。\n"
        "以下是几段用户检索可能命中的平台知识内容，每段包含：页面名称、段落标题、来源路径、摘要、正文片段。\n\n"
        "🎯 你的任务：\n"
        "1. 阅读所有段落，构造一个自然口语化的问题；\n"
        "2. 该问题必须只能由其中一段准确回答，其他段容易混淆；\n"
        "3. 问题必须体现出区分性的特征，可以基于路径、页面名称、标题或摘要，而不仅仅是正文内容；\n"
        "4. 然后告诉我问题最匹配的是哪一段（用编号1~6表示）。\n\n"
        "📌 输出格式：\n"
        "问题：<一句自然中文问题>\n"
        "对应段编号：<1~>\n"
    )

    for i, chunk in enumerate(chunks):
        prompt += (
            f"\n【候选内容编号{i+1}】\n"
            f"页面名称：{chunk.get('page_name', '')}\n"
            f"段落标题：{chunk.get('title', '')}\n"
            f"来源路径：{chunk.get('page_url', '')}\n"
            f"摘要：{chunk.get('summary', '')}\n"
            f"正文：{chunk.get('text', '')}\n"
        )

    return prompt


def call_deepseek_chat(prompt: str) -> str:
    return llm.invoke(prompt).content.strip()


def parse_and_save_qa_pairs(response: str, chunks: List[Dict], output_path: str):
    try:
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        question_line = next((line for line in lines if "问题：" in line), None)
        index_line = next((line for line in lines if "编号" in line or "对应段编号" in line), None)

        query = question_line.split("问题：", 1)[-1].strip()
        idx_str = index_line.split("编号", 1)[-1].strip("：:.)） ")
        local_idx = int(idx_str) - 1
    except Exception as e:
        print(f"⚠️ 解析失败：{e}，使用 fallback")
        query = response.strip()
        local_idx = random.randint(0, 5)

    data = []
    for i, chunk in enumerate(chunks):
        label = 1 if i == local_idx else 0
        formatted_doc = format_chunk_for_reranker(chunk)
        data.append({
            "query": query,
            "doc": formatted_doc,
            "label": label
        })

    with open(output_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ✅ 主流程：对一个 sample 执行构建 QA 训练样本

def process_one_sample(sample: Dict, query_fn, output_path: str):
    chunklist = query_fn(sample)  # 你负责实现此函数，返回 List[Dict]，共6条 chunk
    # print((len(chunklist)))
    # print(chunklist)
    # exit()
    if len(chunklist) < 2:
        print("❌ 检索结果不足2条，跳过该样本")
        return

    prompt = build_contrastive_prompt_with_selection(chunklist)
    response = call_deepseek_chat(prompt)
    parse_and_save_qa_pairs(response, chunklist, output_path)




if __name__ == "__main__":

    # ✅ 参数设置
    embedder_model = "/data/huangruizhi/htmlRAG/bce-embedding-base_v1"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    Milvus_host = "192.168.7.247"
    ES_host = "192.168.7.247"
    COLLECTION_NAME = "test_env"
    SAMPLE_SIZE = 2500
    top_k = 2
    OUTPUT_PATH = "reranker_qa_dataset.jsonl"
    
    print("📦 加载 Embedder 模型...")
    embedder = HuggingFaceEmbeddings(model_name=embedder_model,model_kwargs={"device": device})


    # ✅ 连接 Milvus 并加载集合
    connections.connect(alias="default", host=Milvus_host, port="19530")
    collection = Collection(name=COLLECTION_NAME)
    all_ids = collection.query(
        expr="",
        output_fields=["global_chunk_idx"],
        limit=10000
    )
    valid_ids = [item["global_chunk_idx"] for item in all_ids]
    sample_ids = random.sample(valid_ids, SAMPLE_SIZE)

    def query_fn(sample: Dict) -> List[Dict]:
        global_chunk_idx = sample["global_chunk_idx"]
        # ✅ 查询主键对应问题
        res = collection.query(
            expr=f"global_chunk_idx == {global_chunk_idx}",
            output_fields=["question"]
        )
        print(f"🔍 查询主键：{global_chunk_idx}")
        if not res or not res[0].get("question"):
            return []
        question = res[0]["question"]
        print(f"🔍 查询问题：{question}")

        # 你需要实现这两个函数（语义+关键词）
        milvus_results = query_milvus_blocks(Milvus_host, question, embedder, top_k=top_k, milvus_collection_name=COLLECTION_NAME)
        es_results = query_es_blocks(ES_host, question, top_k=top_k, es_index_name=COLLECTION_NAME)
        # print(f"🔍 Milvus 检索结果：{len(milvus_results)}")
        # print(f"🔍 ES 检索结果：{len(es_results)}")
        results = deduplicate_ranked_blocks(milvus_results + es_results)
        return results

    # ✅ 主流程：采样并构建训练样本
    for idx in sample_ids:
        sample = {"global_chunk_idx": idx}
        process_one_sample(sample, query_fn=query_fn, output_path=OUTPUT_PATH)
