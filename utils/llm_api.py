import os
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import torch
from difflib import SequenceMatcher
import requests
import time
import asyncio
from utils.config import CONFIG



# Semaphore 控制并发负载
sem = asyncio.Semaphore(CONFIG.get("vllm_max_concurrent_requests", 32))
timeout = CONFIG.get("vllm_timeout", 60)
api_url = CONFIG.get("vllm_api_url", "http://localhost:8000/v1/chat/completions")


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
        with sem:  # 控制并发
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
        # with sem:  # 限制并发请求
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
