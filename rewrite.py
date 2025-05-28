import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.llm_api import rewrite_query_ChatGLM
from transformers import AutoTokenizer, AutoModel
import argparse
import time

# ======================== 主入口 ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="zh")
    # parser.add_argument("--summary_tokenizer", type=str, default="/data/huangruizhi/htmlRAG/chatglm3-6b")
    # parser.add_argument("--summary_model", type=str, default="/data/huangruizhi/htmlRAG/chatglm3-6b")
    parser.add_argument("--summary_tokenizer", type=str, default="THUDM/glm-4-9b-chat")
    parser.add_argument("--summary_model", type=str, default="THUDM/glm-4-9b-chat")
    parser.add_argument("--max_node_words_embed", type=int, default=4096)
    parser.add_argument("--min_node_words_embed", type=int, default=48)
    args = parser.parse_args()


    # 加载模型（可注释掉摘要部分以加速）
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.summary_tokenizer, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.summary_model, trust_remote_code=True).half().to(device)
    model.eval()

    start_time = time.time()  # 记录开始时间
    # ======== 遍历重写 demo ========
    with open("rewriting_test_set.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            dialogue = item["dialogue"]
            final_query = item["final_query"]
            predicted = rewrite_query_ChatGLM(dialogue, final_query, model, tokenizer)
            print(f"🔹 原问题: {final_query}")
            print(f"✅ 重写后: {predicted}")
            print(f"🎯 标准答案: {item['rewriting_target']}")
            print("------\n")
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time
    print(f"十条重写总耗时: {total_time} 秒")

    while True:
        pass