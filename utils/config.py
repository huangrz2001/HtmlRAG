# -*- coding: utf-8 -*-
"""
通用配置加载模块：从上级目录的 config.json 文件中加载配置，控制全局统一的配置信息
"""
import json
import os
import aiohttp
import asyncio
current_file_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG = None
CONFIG_PATH = os.path.join(current_file_dir, "..", "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
else:
    raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")



_session = None
sem = asyncio.Semaphore(CONFIG.get("vllm_max_concurrent_requests", 32))

async def get_aiohttp_session():
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=CONFIG.get("vllm_timeout", 60))
        _session = aiohttp.ClientSession(timeout=timeout, headers={"Content-Type": "application/json"})
    return _session

async def close_aiohttp_session():
    global _session
    if _session and not _session.closed:
        await _session.close()