import json
import argparse
import asyncio
import aiohttp
import time


def build_payload(dialogue, final_query):
    return {
        "dialogue": dialogue,
        "final_query": final_query
    }


async def query_once(session, url, payload):
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        return data.get("rewritten_query", "") if data.get("status") == "ok" else f"❌ error: {data.get('error')}"


async def main(input_file, concurrent_url):
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    url = concurrent_url
    headers = {"Content-Type": "application/json"}

    payloads = []
    for item in dataset:
        payload = build_payload(item["dialogue"], item["final_query"])
        payloads.append(payload)

    start_time = time.time()
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [query_once(session, url, p) for p in payloads]
        responses = await asyncio.gather(*tasks)

    end_time = time.time()
    for i, r in enumerate(responses):
        print(f"[{i+1}] 🔁 {r}")
    print(f"\n✅ 并发完成 {len(responses)} 条请求，耗时: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="rewriting_test_set_robust.jsonl", help="输入的 JSONL 文件路径")
    parser.add_argument("--url", type=str, default="http://192.168.7.179:80/document/query_rewrite", help="FastAPI 重写接口地址")
    args = parser.parse_args()

    asyncio.run(main(args.input_file, args.url))
