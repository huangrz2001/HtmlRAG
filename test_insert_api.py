import os
import argparse
import asyncio
import aiohttp
import time


def build_insert_payload(document_index, resource_id, blackhole_url, page_url):
    return {
        "document_index": document_index,
        "resource_id": resource_id,
        "blackhole_url": blackhole_url,
        "page_url": page_url
    }


async def insert_once(session, url, payload):
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        if data.get("result") == "ok":
            return f"✅ 插入成功: {payload['page_url']}, 块数: {data.get('inserted_chunks_milvus')}"
        else:
            return f"❌ 插入失败: {payload['page_url']}, 错误: {data.get('error')}"


async def main(input_dir, insert_url):
    # 遍历 HTML 文件路径
    html_paths = []
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath) and fname.endswith(".html"):
            html_paths.append(fpath)

    # 构造请求 payloads
    payloads = [
        build_insert_payload(
            document_index=101,
            resource_id=0,
            blackhole_url="http://placeholder:8081",
            page_url=path
        )
        for path in html_paths
    ]

    headers = {"Content-Type": "application/json"}
    start_time = time.time()
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [insert_once(session, insert_url, p) for p in payloads]
        results = await asyncio.gather(*tasks)

    end_time = time.time()
    for i, r in enumerate(results):
        print(f"[{i + 1}] 📄 {r}")
    print(f"\n✅ 批量插入完成 {len(results)} 条请求，耗时: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/home/algo/hrz/db_construct/总知识库/粤理知识库", help="HTML 文件所在目录")
    parser.add_argument("--url", type=str, default="http://localhost:8080/chat/python/document/add", help="插入接口地址")
    args = parser.parse_args()

    asyncio.run(main(args.dir, args.url))
