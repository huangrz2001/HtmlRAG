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


timeout = CONFIG.get("vllm_timeout", 60)
api_url = CONFIG.get("vllm_api_url", "http://localhost:8011/v1/chat/completions")


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
        "你是一个电商平台智能客服的对话清晰化助手。\n"
        "用户提出的问题可能存在复杂指代、上下文依赖或表达模糊等问题。\n"
        "你需要根据多轮历史对话，丰富润色用户的当前问题，使其成为一个清晰、完整的独立问题。\n\n"
        "下面是要求：\n"
        "- 准确解析用户真实意图，使得这个独立问题尽量完整，尽可能包含所有信息；\n"
        "- 问题越具体越好，特别是要捕捉到关键的指代，场景，特别针对的问题和例子等等；\n"
        "- 不可以捏造不存在的信息，同时过滤掉与真实意图无关的信息；\n"
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


async def rewrite_query_vllm_async(dialogue, final_query, model="glm", max_new_tokens=128):
    """
    使用 vLLM 异步接口重写 query，带全局 session 和并发控制
    """
    fallback_rewrite = final_query
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
        role = "用户" if turn.get("speaker") == "user" else "系统"
        content = turn.get("text", "").replace("\n", " ").strip()
        prompt += f"{role}：{content}\n"

    prompt += f"用户当前问题是：{final_query.strip()}\n请你遵循要求润色为一个清晰、完整的独立问题："

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
        "top_p": 0.8,
    }
    try:
        async with sem:
            session = await get_aiohttp_session()
            start = time.time()
            logger.debug(f"vLLM 异步重写请求: {final_query}")
            async with session.post(api_url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                rewritten = result["choices"][0]["message"]["content"].strip()
                logger.debug(f"{final_query} Query 重写成功，耗时 {time.time() - start:.2f}s, 重写结果: {rewritten }")
                return rewritten or fallback_rewrite
    except Exception as e:
        logger.error(f"⚠️ vLLM 异步重写失败: {e}, 返回原问题")
        return fallback_rewrite


async def generate_summary_vllm_async(text, page_url, model="glm", max_new_tokens=196):
    """
    使用 vLLM 异步接口生成摘要，带全局 session 和并发控制
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

    try:
        async with sem:
            session = await get_aiohttp_session()
            start = time.time()
            logger.debug(f" vLLM 异步摘要请求: {page_url + ' ' + text[:64]}")
            async with session.post(api_url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                summary = result["choices"][0]["message"]["content"].strip()
                duration = time.time() - start
                logger.debug(f"✅ vLLM 异步摘要成功 (耗时 {duration:.2f}s), {page_url + ' ' + text[:64]}, 摘要内容: {summary[:64]}...")
                return summary
    except Exception as e:
        logger.error(f"⚠️ vLLM 异步摘要失败: {e}，返回截断文本")
        return text[:max_new_tokens]
