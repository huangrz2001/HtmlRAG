# -*- coding: utf-8 -*-
"""
大模型摘要生成与多轮问答重写模块（ChatGLM / vLLM 支持）

本模块封装了使用 ChatGLM 与 vLLM 接口完成以下任务的能力：
1. 文档块摘要生成（支持并发控制和类型自适应）
2. 问题生成（根据文档块自动生成潜在用户问题）
3. 多轮对话重写（将用户模糊提问重写为独立清晰问题）
4. 向量生成（基于 vLLM Embedding 接口）

模块亮点：
------------------------------------------------
- 支持 ChatGLM 本地模型与 vLLM 部署服务两种调用方式
- 摘要/问句生成根据 URL 路径智能分类（规则类 / 操作类 / 信息类 / 泛用类）
- 多轮 query 重写任务支持上下文融合，提示词经过 prompt tuning 优化
- 提供 vLLM 并发控制机制（通过 asyncio.Semaphore 实现请求速率调控）
- 支持 vLLM 嵌入接口，可用于后续向量化搜索

主要函数说明：
------------------------------------------------
1. 摘要生成：
   - `generate_summary_vllm`: 使用 vLLM 接口生成摘要，支持超时与并发限制
   - `generate_summary_ChatGLM`: 使用本地 ChatGLM 模型生成摘要

2. 问题生成：
   - `generate_question_ChatGLM`: 根据文档内容和类别生成一个代表性问题

3. 多轮对话重写：
   - `rewrite_query_ChatGLM`: 使用 ChatGLM 对话模板改写用户模糊问题
   - `rewrite_query_vllm`: 使用 vLLM Chat API 改写用户模糊问题

4. 其他辅助：
   - `infer_chunk_category`: 根据 page_url 分类文档内容（规则/操作/信息/泛用）
   - `get_embedding_from_vllm`: 调用 vLLM embedding 接口获取文本向量（适配 BCE 模型）

配置依赖项：
------------------------------------------------
- `CONFIG` 中可配置：
  - `"vllm_api_url"`：vLLM 推理服务地址
  - `"vllm_max_concurrent_requests"`：最大并发数
  - `"vllm_timeout"`：请求超时时间（秒）

"""


import httpx
import torch
import requests
import time
import asyncio
from utils.config import CONFIG, get_aiohttp_session, close_aiohttp_session, sem, logger
import aiohttp
from typing import List, Dict, Optional, Set
import random



timeout = CONFIG.get("vllm_timeout", 60)
# api_url = CONFIG.get("vllm_api_url", "http://localhost:8011/v1/chat/completions")
VLLM_SERVERS = CONFIG.get("vllm_api_servers", [])
VLLM_TIMEOUT = CONFIG.get("vllm_timeout", 60)



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




def get_embedding_from_vllm(text: str) -> list[float]:
    url = "http://0.0.0.0:8010/v1/embeddings"
    payload = {
        "model": "/home/algo/AD_agent/models/bce-embedding-base_v1",
        "input": [text],    # 一定要是列表
    }
    resp = requests.post(url, json=payload, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]
    


# ======================== vLLM 摘要生成函数 ========================
def generate_summary_vllm(text, page_url, max_new_tokens=150, model="glm") -> str:
    """使用 HTTP 调用 vLLM 并发受控摘要生成"""
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

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
        "top_p": 0.8
    }

    api_url = CONFIG.get("vllm_api_url", "http://localhost:8000/v1/chat/completions")

    try:
        start = time.time()
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        duration = time.time() - start
        print(f"✅ vLLM摘要成功 (耗时 {duration:.2f}s)")
        return result or text[:max_new_tokens]
    except Exception as e:
        print(f"⚠️ vLLM 摘要生成失败: {e}，fallback 到截断文本")
        return text[:max_new_tokens]



async def generate_summary_vllm_async(text, page_url, model="glm", max_new_tokens=150):
    """
    真正异步并发调用 vLLM 接口生成摘要
    """
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

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
        "top_p": 0.8,
    }

    url = CONFIG.get("vllm_api_url", "http://localhost:8000/v1/chat/completions")
    headers = {"Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=payload, timeout=CONFIG.get("vllm_timeout", 60)) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ vLLM 异步摘要失败: {e}，返回截断文本")
        return text[:max_new_tokens]


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



# ======================== ChatGLM 多轮 Query Rewriting 函数 ========================
def rewrite_query_ChatGLM(
    dialogue: list,  # 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    final_query: str,
    model,
    tokenizer,
    max_new_tokens=128,
):
    """
    基于对话历史和当前模糊问题，使用 ChatGLM 重写为独立清晰问题
    """
    fallback_rewrite = final_query

    # 构造提示词
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
    # print(f"🔍 重写提示词：\n{prompt[:256]}...")  # 仅打印前256字符

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

        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return response if response else fallback_rewrite
    except Exception as e:
        print(f"⚠️ ChatGLM 重写失败: {e}，返回原问题")
        return fallback_rewrite


def rewrite_query_vllm(
    dialogue: list,
    final_query: str,
    model: str = "glm",
    max_new_tokens: int = 128
) -> str:
    """
    基于对话历史和当前模糊问题，使用 vLLM Chat 接口重写为独立清晰问题（带并发控制）
    """
    fallback_rewrite = final_query
    # 构造 prompt
    prompt = (
        "你是一个问题重写API，只会重写优化或复读用户的问题。\n"
        "用户提出的问题可能存在复杂指代、上下文依赖或表达模糊等问题。\n"
        "你需要根据多轮历史对话，丰富润色用户的当前问题，使其成为一个清晰、完整的独立问题。\n\n"
        "下面是要求：\n"
        "- 准确解析用户真实意图，从历史中发掘指代和意图，使得这个独立问题尽量完整，尽可能包含所有信息关键词,特别是要补充关键的动机，指代和场景（润色后的问题至少涉及：用户面对什么前置情况，强调了什么限制，有什么疑问等），但不可以捏造不存在的信息；\n"
        "- 如果当前问题本身是 1.非问题（如“你好”等寒暄，“退出系统”等指令，“呵呵”等无意义灌水），2. 非技术性问题（“你是谁”，“你是AI吗”等身份询问，“人生的意义是什么”等无关性问题），不进行润色直接返回原句子；\n"
        "- 一定一定不可以回答用户问题，你只关注问题本身的润色；\n"
        "- 不添加解释、注释、引导语等，只输出润色后的问题句。\n\n"
        "下面是历史对话：\n"
    )

    for turn in dialogue:
        role = "用户" if turn.get("speaker") == "user" else "系统"
        content = turn.get("text", "").replace("\n", " ").strip()
        prompt += f"{role}：{content}\n"

    prompt += f"用户当前问题是：{final_query.strip()}\n请你遵循要求润色为一个清晰、完整的独立问题："

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
        "top_p": 0.8
    }
    try:
        start = time.time()
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        duration = time.time() - start
        print(f"✅ 重写成功（耗时 {duration:.2f}s）")
        return result if result else fallback_rewrite
    except Exception as e:
        print(f"⚠️ vLLM 重写失败: {e}")
        return fallback_rewrite


# 均衡负载handler
def weighted_sample_without_replacement(servers: List[Dict[str, any]], tried: Set[str]) -> Optional[str]:
    """在未尝试服务器中按权重随机采样一个"""
    candidates = [(s["url"], s.get("weight", 1)) for s in servers if s["url"] not in tried]
    if not candidates:
        return None
    urls, weights = zip(*candidates)
    return random.choices(urls, weights=weights, k=1)[0]


async def call_vllm_with_retry_weighted(payload: dict, timeout: int = 15, max_retries: Optional[int] = None) -> dict:
    """
    带权重的vLLM异步调用，失败自动重试，按权重采样但不重复。
    """
    tried_urls = set()
    retries = max_retries or len(VLLM_SERVERS)

    errors = []

    for _ in range(retries):
        api_url = weighted_sample_without_replacement(VLLM_SERVERS, tried_urls)
        if api_url is None:
            break
        tried_urls.add(api_url)

        try:
            logger.debug(f"🚀 请求 vLLM: {api_url}")
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=timeout) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    return result  # ✅ 成功
        except Exception as e:
            logger.warning(f"⚠️ vLLM 请求失败: {api_url} → {e}")
            errors.append((api_url, str(e)))
            continue

    # 所有尝试都失败
    error_msg = "所有 vLLM 实例请求失败: " + "; ".join([f"{url}: {err}" for url, err in errors])
    raise RuntimeError(error_msg)



# max_new_tokens=1024主要防止带思维链输出溢出
async def rewrite_query_vllm_async(dialogue, final_query, model="glm", max_new_tokens=1024):
    fallback_rewrite = final_query
    banned_phrases = [
        "你好", "您好", "hi", "hello", "哈喽", "在吗", "喂", "请问在吗", "有人吗", "hello？",
        "你是谁", "你是人吗", "你是机器人吗", "你是AI吗", "你叫什么", "你是客服吗", "你是智能助手吗", "你是人工的吗", "你能听懂我说话吗",
        "呵呵", "哈哈", "嗯", "哼", "额", "好吧", "无语", "。。。", "...", "===", "你猜", "随便", "看你咋说",
        "测试", "test", "just testing", "随便问问", "这是个测试", "debug", "看看你怎么回答",
        "今天是几号", "时间", "天气", "北京天气", "今天天气", "讲个笑话", "背首诗", "给我唱首歌", "来段rap",
        "重启一下", "清除缓存", "退出系统", "保存文件", "打开浏览器", "运行代码", "执行脚本", "回答问题",
        "你扮演谁", "假设你是", "你是人类", "如果你是我", "从你的角度看", "你作为一个AI",
        "存在的意义是什么", "人生的意义", "什么是真实", "你怎么看这个世界", "你觉得我是谁",
        "商品", "服务", "平台", "抖音", "小红书", "视频", "规则", "政策", "报表"
    ]

    if any(phrase == final_query for phrase in banned_phrases):
        logger.debug(f"🔍 命中过滤词 Query 重写跳过：{final_query}")
        return fallback_rewrite

    if len(dialogue) < 2:
        logger.debug(f"🔍 对话历史过短，跳过 Query 重写：{final_query}")
        return fallback_rewrite

    system_prompt = (
        "你是一个专业的问题重写模块，专门用于多轮对话场景下的指代补全与语义还原任务。\n"
        "请严格按照以下规则执行：\n\n"
        "【任务目标】\n"
        "1. 你的目标是准确解析用户真实意图，从历史中发掘指代和意图，使得这个独立问题尽量完整，尽可能包含所有信息关键词,特别是要补充关键的动机，指代和场景（润色后的问题至少涉及：用户面对的前置情况，什么限制，什么疑问等）。\n"
        "2. 仅当历史信息能够提供明确上下文时才进行补全，否则保持当前问题不变。\n"
        "3. 对于明显不需要补全的问题如：语气词：\"呵呵\"，命令：\"关机\"，无关内容：\"人生的意义\"等，返回原句\n"
        "4. 不进行任何无关发挥、扩写、润色、修辞性描述、解释、总结、感情色彩。\n"
        "5. 不编造任何不存在的假设背景或新信息。\n"
        "6. 输出格式必须严格遵循：仅输出最终重写结果文本，不包含任何前缀、提示词、说明性文字或换行符。重写的疑问句以：\"我想知道\"开头，非疑问句你自行适配\n"
        "7. 当无需重写时，直接输出原问题。\n\n"
        "【重写示例】\n"
        "历史对话：\n用户：小红鞋什么时候有货？\n系统：请问您指的是哪款小红鞋？\n用户：就是上次缺货那款\n当前问题：就是上次缺货那款\n重写结果：我想知道上次缺货的那款小红鞋什么时候有货？\n\n"
        "历史对话：\n用户：我们前几天账号被限流了，不知道什么原因\n系统：限流可能是因为素材违规或账户表现不佳。\n用户：那我们还有其他计划，能投放吗？\n当前问题：那我们还有其他计划，能投放吗？\n重写结果：我想知道当账号处于限流状态时，是否可以继续开启新的广告投放计划？\n\n"
        "历史对话：\n用户：你是谁？\n当前问题：你是谁？\n重写结果：你是谁？\n\n"
        "历史对话：\n用户：开心\n当前问题：开心\n重写结果：开心\n\n"
        "历史对话：\n用户：我要买它\n系统：请问您指的是哪款商品？\n用户：之前推荐的那双\n当前问题：之前推荐的那双\n重写结果：我想要买你之前推荐的那双鞋。\n\n"
        "注意：严格按照以上风格工作，禁止任何额外输出。"
    )

    history_content = ""
    for turn in dialogue:
        role = "用户" if turn.get("speaker") == "user" else "系统"
        content = turn.get("text", "").replace("\n", " ").strip()
        history_content += f"{role}：{content}\n"

    current_question = f"当前问题：{final_query.strip()}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{history_content}\n{current_question}\n请你根据任务规则输出重写结果。"}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.5,
        "top_k": -1,
    }

    try:
        async with sem:
            start = time.time()
            result = await call_vllm_with_retry_weighted(payload, timeout=CONFIG.get("vllm_timeout", 60))
            rewritten = result["choices"][0]["message"]["content"].strip()
            # 如果有思维链
            rewritten = rewritten.split("</think>")[-1].strip() if "</think>" in rewritten else rewritten.strip()
            logger.debug(f"{final_query} 重写成功，用时 {time.time() - start:.2f}s，结果：{rewritten}")
            return rewritten or fallback_rewrite
    except Exception as e:
        logger.error(f"⚠️ 重写请求失败，返回原问题：{e}")
        return fallback_rewrite



async def generate_summary_vllm_async(text, page_url, model="glm", max_new_tokens=1024):
    """
    使用 vLLM 异步接口生成摘要，支持多机路由与失败重试。
    """
    category = infer_chunk_category(page_url)
    text = text.strip().replace("\x00", "")
    fallback_summary = text[:max_new_tokens]

    # 构建 system + user 格式的 prompt
    system_prompt = (
        "你是一个智能的内容摘要助手，专门用于提炼电商平台文档的主要信息。\n"
        "你需要根据给出的文档内容，在保留原意的前提下压缩成一句话摘要。\n\n"
        "【任务要求】\n"
        "1. 摘要需准确覆盖原文核心信息，不得添加、编造、扩写。\n"
        "2. 不允许重复粘贴原文原句或冗余内容。\n"
        "3. 语言风格应简洁、清晰、无修辞和主观色彩，使用简体中文。\n"
        "4. 输出格式为一整句话，不包含前缀、项目符号或换行。\n"
        "5. 字数控制在不超过指定上限，能短则短，尽量精准。\n"
        "6. 当文本无明显中心内容或格式异常时，请从中提炼关键词或主旨进行简要总结。"
    )

    user_prompt = (
        f"文档路径：{page_url}\n"
        f"所属类目：{category}\n"
        f"文本内容：{text}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
        "top_p": 0.8,
    }

    try:
        async with sem:
            start = time.time()
            logger.debug(f"📩 vLLM 异步摘要请求: {page_url + ' ' + text[:64]}")
            result = await call_vllm_with_retry_weighted(payload, timeout=CONFIG.get("vllm_timeout", 60))
            summary = result["choices"][0]["message"]["content"].strip()
            duration = time.time() - start
            logger.debug(f"✅ vLLM 异步摘要成功 (耗时 {duration:.2f}s)，{page_url + '：' + text[:64]}。摘要内容: {summary[:64]}...")
            return summary or fallback_summary
    except Exception as e:
        logger.error(f"⚠️ vLLM 异步摘要失败: {e}，返回截断文本")
        return fallback_summary




async def get_embeddings_from_vllm_async(
    texts: list[str],
    url: str,
    timeout: int = 10,
    max_concurrent_tasks: int = 16
) -> list[list[float]]:
    async def _fetch(text: str) -> list[float]:
        payload = {"input": text}
        async with sem:
            session = await get_aiohttp_session()
            try:
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    # 假设接口返回 {"data":[{"embedding": [...]}, ...]}
                    return data["data"][0]["embedding"]
            except Exception as e:
                logger.error(f"❌ 文本推理失败: {e} | text={text!r}")
                raise

    # 创建并发任务
    tasks = [asyncio.create_task(_fetch(txt)) for txt in texts]
    # 并发执行并收集结果
    embeddings = await asyncio.gather(*tasks)
    return embeddings



def get_embedding_from_vllm(text: str, url) -> list[float]:
        "" "从vLLM服务获取文本嵌入向量 """
        payload = {"input": [text]}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except requests.exceptions.Timeout:
            logger.error(f"vLLM请求超时 | 超时时间: 5s")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"vLLM HTTP错误 | 状态码: {resp.status_code} | 响应: {resp.text}")
            raise
        except Exception as e:
            logger.error(f"vLLM嵌入失败 | 错误: {str(e)}")
            raise
