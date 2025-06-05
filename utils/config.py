# -*- coding: utf-8 -*-
"""
通用配置加载模块：从上级目录的 config.json 文件中加载配置，控制全局统一的配置信息
"""
import json
import os
import aiohttp
import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler

# ======================== 配置加载 ========================
current_file_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(current_file_dir, "..", "config.json")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# ======================== 日志配置 ========================
LOG_LEVEL = CONFIG.get("log_level", "INFO").upper()
LOG_DIR = CONFIG.get("log_dir", os.path.join(current_file_dir, "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "app.log")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("GlobalLogger")
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（按天轮转，最多保留7天）
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",         # 每天凌晨创建新日志
        interval=1,
        backupCount=7,           # 最多保留 7 天的日志
        encoding="utf-8",
        utc=False
    )
    file_handler.suffix = "%Y-%m-%d"     # app.log.2025-06-05
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
logger.info("="*100)
logger.info("✅ 全局配置和日志系统初始化完成，服务器时区GMT-4注意转换（+12h为北京时间）")

# ======================== aiohttp 会话管理 ========================
_session = None
sem = asyncio.Semaphore(CONFIG.get("vllm_max_concurrent_requests", 32))

async def get_aiohttp_session():
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=CONFIG.get("vllm_timeout", 60))
        _session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        logger.info("新建 aiohttp 会话")
    return _session

async def close_aiohttp_session():
    global _session
    if _session and not _session.closed:
        await _session.close()
        logger.info("aiohttp 会话已关闭")
