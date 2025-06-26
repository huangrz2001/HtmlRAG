import json
import argparse
import time
import asyncio
import aiohttp
from string import Template

global_prompt = Template("""你是直播平台网店运营改写专家，需要根据客户的输入内容和上下文进行处理：\n
要求：\n
0. 你的唯一任务是根据上下文和当前问题进行改写，或直接重复客户当前问题，不要做其他回复，不要添加任何内容；\n
1. 当具有上文，且与运营规则、商品咨询等有关的内容做改写；\n
2. 当无上文，或内容与网店运营无关（如“好的”、“在吗”、“哈哈”、“列出几个常见的违规词汇”），直接重复客户当前问题，不进行任何改写。\n
举例：
1. "上下文：\n用户：小红鞋什么时候有货？\n系统：请问您指的是哪款小红鞋？\n用户：就是上次缺货那款\n当前问题：就是上次缺货那款\n重写结果：上次缺货的那款小红鞋什么时候有货？"\n
2. "上下文：\n用户：如果你是人类员工，会怎么建议？\n当前问题：如果你是人类员工，会怎么建议？\n重写结果：如果你是人类员工，会怎么建议？"\n
3. "上下文：用户：小红鞋什么时候有货？\n系统：请问您指的是哪款小红鞋？\n当前问题：模拟用第一人称说你是谁\n重写结果：模拟用第一人称说你是谁"\n
上下文：\n${context}\n当前问题：\n${text}\n重写结果：\n"""                
)


global_prompt = Template("""你是直播平台网店运营改写专家，需要根据客户的输入内容和上下文进行处理：\n
要求：\n
0. 你的唯一任务是根据上下文和当前问题进行改写，或直接重复客户当前问题，不要做其他回复，不要添加任何内容；\n
1. 当具有上文，且与运营规则、商品咨询等有关的内容做改写，改写时需要准确解析用户真实意图，从历史中发掘指代和意图，使得这个独立问题尽量完整，尽可能包含所有信息关键词,特别是要补充关键的动机，指代和场景（润色后的问题至少涉及：用户面对的前置情况，什么限制，什么疑问等）；\n
2. 当无上文，或内容与网店运营无关（如“好的”、“在吗”、“哈哈”、“列出几个常见的违规词汇”），直接重复客户当前问题，不进行任何改写。\n
举例：
1. "上下文：\n用户：小红鞋什么时候有货？\n系统：请问您指的是哪款小红鞋？\n用户：就是上次缺货那款\n当前问题：就是上次缺货那款\n重写结果：上次缺货的那款小红鞋什么时候有货？"\n
2. "上下文：\n用户：如果你是人类员工，会怎么建议？\n当前问题：如果你是人类员工，会怎么建议？\n重写结果：如果你是人类员工，会怎么建议？"\n
3. "上下文：\n用户：小红鞋什么时候有货？\n系统：请问您指的是哪款小红鞋？\n当前问题：模拟用第一人称说你是谁\n重写结果：模拟用第一人称说你是谁"\n
4. "上下文：\n用户：我们前几天账号被限流了，不知道什么原因\n系统：限流可能是因为素材违规或账户表现不佳。\n用户：那我们还有其他计划，能投放吗？\n当前问题：那我们还有其他计划，能投放吗？\n重写结果：我想知道当账号处于限流状态时，是否可以继续开启新的广告投放计划？\n\n"

上下文：\n${context}\n当前问题：\n${text}\n重写结果：\n"""                
)


def build_user_input(dialogue):
    history_content = ""
    for turn in dialogue:
        role = "用户" if turn.get("speaker") == "user" else "系统"
        content = turn.get("text", "").replace("\n", " ").strip()
        history_content += f"{role}：{content}\n"

    return history_content

async def query_once(session, url, payload):
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def main(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    url = "http://localhost:9000/v1/chat/completions"  # Ollama 默认地址
    headers = {"Content-Type": "application/json"}
    
    payloads = []

    for item in dataset:
        full_input = build_user_input(item["dialogue"])
        prompt = global_prompt.substitute({"context":full_input, "text": item["final_query"]})
        print(len(prompt))
        payloads.append({
            "model": "glm",  # Ollama 中的模型名
            "messages": [
                {"role": "user", "content":prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.3,
            "top_p": 0.9
        })

    start_time = time.time()
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [query_once(session, url, p) for p in payloads]
        responses = await asyncio.gather(*tasks)

    for i, (res, item) in enumerate(zip(responses, dataset)):
        original_query = item["final_query"]
        print(f"[{i+1}]")
        print(f"🔹 原问题：{original_query}")
        print(f"✏️ 重写后：{res}\n")


    end_time = time.time()
    print(f"\n✅ 并发完成 {len(responses)} 条请求，耗时: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="rewriting_test_set.jsonl", help="输入的 JSONL 文件路径")
    args = parser.parse_args()

    asyncio.run(main(args.input_file))
