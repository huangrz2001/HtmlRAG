import json
import argparse
import requests
import time
import asyncio
import aiohttp
import json
import time


def build_prompt(dialogue, final_query):
    prompt = (
        "你是一个电商平台智能客服的对话清晰化助手。\n"
        "用户提出的问题可能存在复杂指代、上下文依赖或表达模糊等问题。\n"
        "你需要根据多轮历史对话，丰富润色用户的当前问题，使其成为一个清晰、完整的独立问题。\n\n"
        "下面是要求：\n"
        "- 准确解析用户真实意图，使得这个独立问题尽量完整，尽可能包含所有信息；\n"
        "- 问题越丰富越好，特别是要捕捉到关键的指代，场景，特别针对的问题和例子等等；\n"
        "- 不可以捏造不存在的信息；\n"
        "- 不添加解释、注释、引导语等，只输出润色后的问题句。\n\n"
        "下面是历史对话：\n"
    )
    for turn in dialogue:
        role = "用户" if turn["speaker"] == "user" else "系统"
        content = turn["text"].replace("\n", " ").strip()
        prompt += f"{role}：{content}\n"

    prompt += f"用户当前问题是：{final_query.strip()}\n请你遵循要求润色为一个清晰、完整的独立问题："
    return prompt


def rewrite_query_vllm_requests(dialogue, final_query, api_url="http://localhost:8000/v1/chat/completions", max_tokens=128):
    prompt = build_prompt(dialogue, final_query)
    payload = {
        "model": "glm",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4,
        "top_p": 0.8,
    }

    try:
        start = time.time()
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        duration = time.time() - start
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        return result or final_query, duration
    except Exception as e:
        print(f"⚠️ vLLM 请求失败: {e}")
        return final_query, -1


async def query_once(session, url, payload):
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()

async def main(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    prompts = []
    for item in dataset:
        prompt = build_prompt(item["dialogue"], item["final_query"])
        prompts.append({
            "model": "glm",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.4,
            "top_p": 0.8
        })

    start_time = time.time()
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [query_once(session, url, p) for p in prompts]
        responses = await asyncio.gather(*tasks)

    # print(responses)
    end_time = time.time()
    print(f"✅ 并发完成 {len(responses)} 条请求，耗时: {end_time - start_time:.2f}s")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="rewriting_test_set.jsonl", help="输入的 JSONL 文件路径")
    args = parser.parse_args()

    # 使用异步方式批量查询
    asyncio.run(main(args.input_file))

    # 测试单条查询
    # dialogue = [{"speaker": "user", "text": "如何运营巨量千川平台？"}]
    # final_query = "如何运营巨量千川平台？"
    # result, duration = rewrite_query_vllm_requests(dialogue, final_query)
    # print(f"🔹 原问题: {final_query}")
    # print(f"✅ 重写后: {result} (耗时: {duration:.2f} 秒)")