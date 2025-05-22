# -*- coding: utf-8 -*-
"""
HTML 批量清洗与结构简化脚本

核心功能：
1. 清除特殊 Markdown 段（如```html```)
2. 去除标签之间的换行符与多余空格
3. 保留原始目录结构，输出到 `_cleaned` 目录
4. 合并所有 HTML 为一个总文件 _.html（可选）

依赖模块：utils.html_utils.clean_html
"""

import os
import re
import shutil
from utils.html_utils import process_html_file

# ---------------------------
# 全目录批量处理入口
# ---------------------------

source_root = "./总知识库"
target_root = source_root + "_cleaned"

os.makedirs(target_root, exist_ok=True)  # 创建目标根目录

html_files = []
combined_output_path = "_.html"
combined_content = ""

for dirpath, _, filenames in os.walk(source_root):
    rel_path = os.path.relpath(dirpath, source_root)
    target_dir = os.path.join(target_root, rel_path)
    os.makedirs(target_dir, exist_ok=True)

    for filename in filenames:
        if filename.endswith(".html"):
            source_file = os.path.join(dirpath, filename)
            target_file = os.path.join(target_dir, filename)

            try:
                simplified_html = process_html_file(source_file, target_file)
                html_files.append((source_file, target_file))

                # 为合并文档添加标题标记
                file_name = os.path.basename(target_file)
                title = f"<html>{file_name}</html>\n"
                combined_content += title + simplified_html + "\n\n"

                print(f"📎 已合并：{target_file}")
            except Exception as e:
                print(f"❌ 处理失败 {source_file}：{e}")

# ---------------------------
# 可选：合并为 _.html 文件
# ---------------------------

# try:
#     with open(combined_output_path, 'w', encoding='utf-8') as f:
#         f.write(combined_content)
#     print(f"✅ 合并文件保存至 {combined_output_path}")
# except Exception as e:
#     print(f"❌ 合并保存失败：{e}")
