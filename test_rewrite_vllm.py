import json
import argparse
import time
import asyncio
import aiohttp
import re
import requests


def build_payload(dialogue, final_query, model="glm", max_new_tokens=2048):
    system_prompt = (
        "你是一个专业的问题重写模块，专门用于多轮对话场景下的指代补全与语义还原任务。\n"
        "请严格按照以下规则执行：\n\n"
        "【任务目标】\n"
        "1. 你的目标是准确解析用户真实意图，从历史中发掘指代和意图，使得这个独立问题尽量完整，尽可能包含所有信息关键词,特别是要补充关键的动机，指代和场景（润色后的问题至少涉及：用户面对的前置情况，什么限制，什么疑问等）。\n"
        "2. 仅当历史信息能够提供明确上下文时才进行补全，否则保持当前问题不变。\n"
        "3. 对于明显不需要补全的问题如：语气词：\"呵呵\"，命令：\"关机\"，无关内容：\"人生的意义\"等，返回原句。\n"
        "5. 不编造任何不存在的假设背景或新信息，不进行任何无关发挥、扩写或解释。\n"
        "6. 仅输出最终重写结果文本，不包含任何前缀、提示词、说明性文字或换行符。"
        "7. 当无需重写时，直接输出原问题。\n\n"
        "【重写示例】\n"
        "1.\n用户：小红鞋什么时候有货？\n系统：请问您指的是哪款小红鞋？\n用户：就是上次缺货那款\n当前问题：就是上次缺货那款\n重写结果：我想知道上次缺货的那款小红鞋什么时候有货？\n\n"
        "2.\n用户：我们前几天账号被限流了，不知道什么原因\n系统：限流可能是因为素材违规或账户表现不佳。\n用户：那我们还有其他计划，能投放吗？\n当前问题：那我们还有其他计划，能投放吗？\n重写结果：我想知道当账号处于限流状态时，是否可以继续开启新的广告投放计划？\n\n"
        "3.\n用户：你是谁？\n当前问题：你是谁？\n重写结果：你是谁？\n\n"
        "4.\n用户：开心\n当前问题：开心\n重写结果：开心\n\n"
        "5.\n用户：我要买它\n系统：请问您指的是哪款商品？\n用户：之前推荐的那双\n当前问题：之前推荐的那双\n重写结果：我想要买你之前推荐的那双鞋。\n\n"
    )

    history_content = "下面是历史对话和你要重写的问题，你只需要输出改写后的问题，不需要任何解释性语句：\n"
    for turn in dialogue:
        role = "用户" if turn.get("speaker") == "user" else "系统"
        content = turn.get("text", "").replace("\n", " ").strip()
        history_content += f"{role}：{content}\n"

    current_question = f'\n当前问题："{final_query.strip()}"'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{history_content}\n{current_question}\n请你严格按照输出重写结果，禁止任何额外输出"}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.5,
        "top_k": -1,
    }

    return payload


async def query_once(session, url, payload, sem):
    async with sem:  # 控制并发数量
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            rewritten = data["choices"][0]["message"]["content"].strip()
            # rewritten = rewritten.split("</think>")[-1].strip() if "</think>" in rewritten else rewritten.strip()
            rewritten = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.IGNORECASE | re.DOTALL).strip()
            return rewritten


async def main(input_file, concurrency = 16):
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    sem = asyncio.Semaphore(concurrency)
    url = "http://192.168.7.179:9000/v1/chat/completions"  # 修改为 Ollama 默认接口
    # url = "http://192.168.7.247:8011/v1/chat/completions"  # 修改为 Ollama 默认接口
    headers = {"Content-Type": "application/json"}

    payloads = []
    for item in dataset:
        payload = build_payload(item["dialogue"], item["final_query"])
        # 可选跳过长问题（maxtoken限制会报错）
        if len(payload['messages'][-1]['content'].strip()) > 5000:
            print(f"❗️ 跳过过长问题: {item['final_query']}")
            continue

        payloads.append(build_payload(item["dialogue"], item["final_query"]))

    start_time = time.time()
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [query_once(session, url, p, sem) for p in payloads]
        responses = await asyncio.gather(*tasks)

    for i, res in enumerate(responses):
        print(f"[{i+1}] ✏️ {res}")

    test_acc([item["final_query"] for item in dataset], responses)
    end_time = time.time()
    print(f"\n✅ 并发完成 {len(responses)} 条请求，耗时: {end_time - start_time:.2f}s")


def test_acc(queries, responses):
    print("\n====================================== 测试结果 ======================================")
    correct = 0
    total = len(responses)
    for i, (query, res) in enumerate(zip(queries, responses)):
        match = query == res
        if match:
            correct += 1
        print(f"[{i+1}] {'✅' if match else '❌'} GT: {query} | ✏️ Rewrite: {res}")

    acc = correct / total * 100
    print(f"\n🎯 准确率：{correct}/{total} = {acc:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, default="rewriting_easy.jsonl", help="输入的 JSONL 文件路径")
    parser.add_argument("--input_file", type=str, default="rewriting_hard.jsonl", help="输入的 JSONL 文件路径")
    args = parser.parse_args()

    asyncio.run(main(args.input_file))
    # main_serial(args.input_file)
