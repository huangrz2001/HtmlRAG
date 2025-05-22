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
from utils.html_utils import clean_html, expand_table_spans

# ---------------------------
# 基础清洗函数
# ---------------------------

def clean_special_markdown_tags(html_content):
    """去除 markdown 风格的 ```html``` 块"""
    html_content = re.sub(r'\n*```html\n*', '', html_content)
    html_content = re.sub(r'\n*```\n*', '', html_content)
    return html_content

def remove_tag_newlines(html_content):
    """移除标签之间的换行符（如 >\n -> >）"""
    html_content = re.sub(r'>\n+', '>', html_content)
    html_content = re.sub(r'[^\s]<', lambda m: m.group(0).replace('\n', ''), html_content)
    return html_content

def remove_tag_whitespace(html_content):
    """移除标签边缘的多余空格"""
    html_content = re.sub(r'>\s+', '>', html_content)
    html_content = re.sub(r'\s+<', '<', html_content)
    return html_content

# ---------------------------
# 单文件清洗入口
# ---------------------------

def process_html_file(source_path, target_path):
    """
    对单个 HTML 文件进行清洗，保留 <time> 开头标签，写入目标路径。
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # 提取起始 <time> 标签（格式固定）
    time_pattern = r"^\s*<time[^>]*?>.*?</time>"
    time_match = re.match(time_pattern, html, flags=re.IGNORECASE | re.DOTALL)

    time_tag = ""
    remaining_html = html

    if time_match:
        time_tag = time_match.group(0).strip()
        remaining_html = html[time_match.end():].lstrip()

    # 对剩余内容进行清洗
    simplified_html = clean_html(remaining_html, keep_att=False)
    simplified_html = clean_special_markdown_tags(simplified_html)
    simplified_html = remove_tag_newlines(simplified_html)
    simplified_html = remove_tag_whitespace(simplified_html)
    simplified_html = expand_table_spans(simplified_html)


    # 将 <time> 标签补回开头
    final_html = time_tag + simplified_html

    # 写入目标路径
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"✅ 已处理 {source_path} → {target_path}")
    print(f"原始长度: {len(html)}, 清理后长度: {len(final_html)}")
    return final_html


# ---------------------------
# 全目录批量处理入口
# ---------------------------

source_root = "./测试知识库"
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
