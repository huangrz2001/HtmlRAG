import json
import os

# 获取当前脚本文件的绝对路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 构建相对于当前文件的配置文件路径 (上级目录下的 config.json)
CONFIG_PATH = os.path.join(current_file_dir, "..", "config.json")

# 确保路径存在并加载配置
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
else:
    raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")