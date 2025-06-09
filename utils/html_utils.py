# -*- coding: utf-8 -*-
"""
HTML 清洗与结构化分块模块

本模块用于处理网页抓取或原始 HTML 文件，为下游知识检索与向量化系统（如 RAG）提供高质量结构化输入。
支持表格还原、标题域包装、语义块切分、冗余标签剔除、合并单元格展开等操作。

核心功能概览：
------------------------------------------------
1. HTML 清洗（结构保留 + 标签净化）：
   - `clean_html`: 清洗入口，结合简化、域包装与字符规范化等步骤。
   - `simplify_html_keep_table`: 保留表格结构，去除样式脚本与冗余属性。
   - `warp_domains`: 对标题与表格结构进行语义封装（如 h2_domain、table_domain）。
   - `expand_table_spans`: 展平所有 rowspan/colspan 合并单元格，转换为矩阵表格。
   - `clean_xml`, `clean_html_text`: 清理 XML 声明、markdown 标签与多余换行空格。

2. HTML 结构化切块：
   - `build_block_tree`: 将清洗后的 HTML 拆解为语义块，支持按最大词数与最小内容量控制。
   - 自动记录路径信息与层级深度，保留重要结构节点（标题、表格等）。

3. 单文件处理入口：
   - `process_html_file`: 执行完整清洗 + 表格展开 + time 标签保留，并保存到目标路径。

参数与适用说明：
------------------------------------------------
- `max_node_words`: 控制每个切分块的最大词数（适配向量生成限制，如 tokenizer 长度）。
- `min_node_words`: 控制保留的最小块单位，避免碎片。
- `zh_char`: 为 True 时，使用字符长度而非词数作为分割依据（适配中文纯文本）。

"""



import re
import json
import os
import bs4
from typing import List, Tuple, Dict
from collections import defaultdict
from bs4 import BeautifulSoup, Comment, Tag
import copy




# ======================== 提取 <time> 标签辅助函数 ========================
def parse_time_tag(html: str):
    time_pattern = r"^\s*<time[^>]*?>(.*?)</time>"
    time_match = re.match(time_pattern, html, flags=re.IGNORECASE | re.DOTALL)
    time_value = ""
    if time_match:
        time_value = time_match.group(1).strip()
        html = html[time_match.end():].lstrip()
    return time_value, html


# ===================== HTML 清洗核心 =====================
def simplify_html_keep_table(soup, keep_attr=False):
    """保留表格结构的 HTML 简化版本"""
    TABLE_PROTECTED_TAGS = {'table', 'colgroup', 'col', 'thead', 'tbody', 'tr', 'td', 'th'}
    ALWAYS_KEEP_ATTRS = {'data-block-type'}

    for script in soup(["script", "style"]):
        script.decompose()

    if not keep_attr:
        for tag in soup.find_all(True):
            class_list = tag.get("class", [])
            if isinstance(class_list, str):
                class_list = class_list.split()
            for cls in class_list:
                match = re.match(r"heading-h(\d)", cls)
                if match:
                    level = match.group(1)
                    tag.attrs["data-block-type"] = f"heading{level}"
                    break

            if tag.name in TABLE_PROTECTED_TAGS:
                tag.attrs = {
                    k: v for k, v in tag.attrs.items()
                    if k in {'colspan', 'rowspan'} | ALWAYS_KEEP_ATTRS
                }
            else:
                tag.attrs = {
                    k: v for k, v in tag.attrs.items() if k in ALWAYS_KEEP_ATTRS
                }

    # 移除空标签（表格标签若有子元素则保留）
    while True:
        removed = False
        for tag in soup.find_all():
            if tag.name in TABLE_PROTECTED_TAGS and tag.contents:
                continue
            if not tag.text.strip() and not tag.contents:
                tag.decompose()
                removed = True
        if not removed:
            break

    for tag in soup.find_all("a"):
        tag.attrs.pop("href", None)

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 去除冗余包装标签
    def concat_text(text):
        return re.sub(r"[\n\t ]+", "", text)

    for tag in soup.find_all():
        if tag.name not in TABLE_PROTECTED_TAGS:
            children = [c for c in tag.contents if isinstance(c, Tag)]
            if len(children) == 1:
                tag_text = tag.get_text()
                child_text = children[0].get_text()
                if concat_text(tag_text) == concat_text(child_text):
                    if any(attr in tag.attrs for attr in ALWAYS_KEEP_ATTRS):
                        continue
                    tag.replace_with_children()

    # 清除空行
    return "\n".join(line for line in str(soup).split("\n") if line.strip())


def warp_domains(html: str) -> str:
    """将 HTML 中的 <hX> 标签和 <table> 标签进行包装"""
    """将具有 data-block-type 属性的标签转换为标准 <hX> 标签，并根据标题结构包装内容"""
    soup = BeautifulSoup(html, 'html.parser')

    def convert_headings(soup):
        block_map = {f"heading{i}": f"h{i}" for i in range(1, 7)}
        for tag in soup.find_all(attrs={"data-block-type": True}):
            block_type = tag.get("data-block-type")
            if block_type in block_map:
                new_tag = soup.new_tag(block_map[block_type])
                for child in list(tag.contents):
                    new_tag.append(child.extract())
                tag.replace_with(new_tag)

    def wrap_heading_domains(soup):
        def get_level(tag_name):
            return int(tag_name[1]) if tag_name.startswith("h") and tag_name[1:].isdigit() else None

        def process_nodes(nodes):
            result, i = [], 0
            while i < len(nodes):
                node = nodes[i]
                if isinstance(node, Tag) and node.name.startswith('h') and node.name[1].isdigit():
                    level = get_level(node.name)
                    j, children = i + 1, [node]
                    while j < len(nodes):
                        next_node = nodes[j]
                        next_level = get_level(next_node.name) if isinstance(next_node, Tag) else None
                        if next_level is not None and next_level <= level:
                            break
                        children.append(next_node)
                        j += 1
                    wrapped = process_nodes(children[1:])
                    wrapper = soup.new_tag("div", **{'class': f'h{level}_domain'})
                    wrapper.append(children[0])
                    for c in wrapped:
                        wrapper.append(c)
                    result.append(wrapper)
                    i = j
                else:
                    result.append(node)
                    i += 1
            return result

        body_nodes = list(soup.contents)
        soup.clear()
        has_heading = any(isinstance(n, Tag) and n.name in [f"h{i}" for i in range(1, 7)] for n in body_nodes)
        if has_heading:
            for node in process_nodes(body_nodes):
                soup.append(node)
        else:
            wrapper = soup.new_tag("div", **{'class': 'isolated_domain'})
            for node in body_nodes:
                wrapper.append(node)
            soup.append(wrapper)

    def wrap_table_domains(soup):
        """对所有 table 标签外包一层 <div class='table_domain'>"""
        for table in soup.find_all("table"):
            if not table.find_parent("div", class_="table_domain"):
                wrapper = soup.new_tag("div", **{'class': 'table_domain'})
                table.insert_before(wrapper)
                wrapper.append(table.extract())

    convert_headings(soup)
    wrap_heading_domains(soup)
    wrap_table_domains(soup)  # <-- 新增对表格的处理
    return str(soup)


def expand_table_spans(html: str) -> str:
    """
    展开 HTML 中的表格合并单元格（colspan 和 rowspan），生成标准矩阵表格。
    忽略 rowspan=0 / colspan=0 占位单元格，避免错位。
    """
    soup = BeautifulSoup(html, "html.parser")

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        grid = []  # 二维网格 grid[row][col] = cell
        max_cols = 0

        for row_idx, row in enumerate(rows):
            if len(grid) <= row_idx:
                grid.append([])

            col_idx = 0
            for cell in row.find_all(["td", "th"]):
                # 获取并跳过无效 rowspan/colspan
                try:
                    rowspan = int(cell.get("rowspan", 1))
                except:
                    rowspan = 1
                try:
                    colspan = int(cell.get("colspan", 1))
                except:
                    colspan = 1

                if rowspan == 0 or colspan == 0:
                    continue  # 忽略占位空格

                # 找下一个空白列
                while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                # 清除合并属性
                cell.attrs.pop("rowspan", None)
                cell.attrs.pop("colspan", None)

                for r in range(rowspan):
                    while row_idx + r >= len(grid):
                        grid.append([])

                    while len(grid[row_idx + r]) < col_idx + colspan:
                        grid[row_idx + r].append(None)

                    for c in range(colspan):
                        if r == 0 and c == 0:
                            grid[row_idx + r][col_idx + c] = cell
                        else:
                            grid[row_idx + r][col_idx + c] = copy.copy(cell)

                col_idx += colspan
                max_cols = max(max_cols, col_idx)

        # 构建新的表格
        new_table = soup.new_tag("table")
        for row_cells in grid:
            tr = soup.new_tag("tr")
            for cell in row_cells[:max_cols]:
                if cell is not None:
                    tr.append(cell)
                else:
                    empty = soup.new_tag("td")
                    empty.string = ""
                    tr.append(empty)
            new_table.append(tr)

        table.replace_with(new_table)

    return str(soup)


def clean_xml(html: str) -> str:
    """移除 XML/Doctype 声明"""
    html = re.sub(r"<\?xml.*?>", "", html)
    html = re.sub(r"<!DOCTYPE.*?>", "", html, flags=re.IGNORECASE)
    return html


def clean_html_text(html_content):
    """去除 markdown 风格的 ```html``` 块"""
    html_content = re.sub(r'\n*```html\n*', '', html_content)
    html_content = re.sub(r'\n*```\n*', '', html_content)

    """移除标签之间的换行符（如 >\n -> >）"""
    html_content = re.sub(r'>\n+', '>', html_content)
    html_content = re.sub(r'[^\s]<', lambda m: m.group(0).replace('\n', ''), html_content)
    
    """移除标签边缘的多余空格"""
    html_content = re.sub(r'>\s+', '>', html_content)
    html_content = re.sub(r'\s+<', '<', html_content)
    return html_content


def clean_html(html: str, keep_att=False) -> str:
    """主入口函数：清洗 HTML，保留结构并统一化"""
    soup = BeautifulSoup(html, 'html.parser')
    html = simplify_html_keep_table(soup, keep_att)
    html = warp_domains(html)
    html = clean_xml(html)
    html = clean_html_text(html)
    return html



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
    simplified_html = expand_table_spans(simplified_html)


    # 将 <time> 标签补回开头
    final_html = time_tag + simplified_html

    # 写入目标路径
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"✅ 已处理 {source_path} → {target_path}")
    print(f"原始长度: {len(html)}, 清理后长度: {len(final_html)}")
    return final_html



# ===================== HTML 分块函数 =====================
def build_block_tree(
    html: str,
    max_node_words: int = 512,
    min_node_words: int = 32,
    zh_char: bool = False
) -> Tuple[List[Tuple[bs4.element.Tag, List[str], bool]], str]:
    """将 HTML 分割成结构化块（Tag, 路径, 是否叶子）"""
    soup = BeautifulSoup(html, 'html.parser')
    word_count = len(soup.get_text()) if zh_char else len(soup.get_text().split())

    if word_count < min_node_words:
        return [], str(soup)

    if word_count > max_node_words:
        possible_trees = [(soup, [])]
        target_trees = []
        while possible_trees:
            tree, path = possible_trees.pop(0)
            tag_children = defaultdict(int)
            bare_word_count = 0
            for child in tree.contents:
                if isinstance(child, bs4.element.Tag):
                    tag_children[child.name] += 1
            _tag_children = {k: 0 for k in tag_children}

            for child in tree.contents:
                if isinstance(child, bs4.element.Tag):
                    if child.name in {'table', 'tbody'}:
                        words = len(child.get_text()) if zh_char else len(child.get_text().split())
                        if words >= min_node_words:
                            target_trees.append((child, path + [child.name], True))
                        continue

                    if tag_children[child.name] > 1:
                        new_name = f"{child.name}{_tag_children[child.name]}"
                        _tag_children[child.name] += 1
                        child.name = new_name
                    else:
                        new_name = child.name

                    new_path = path + [new_name]
                    words = len(child.get_text()) if zh_char else len(child.get_text().split())
                    if words < min_node_words:
                        continue
                    if words > max_node_words and len(new_path) < 64:
                        possible_trees.append((child, new_path))
                    else:
                        target_trees.append((child, new_path, True))
                else:
                    bare_word_count += len(str(child)) if zh_char else len(str(child).split())

            if not tag_children and bare_word_count >= min_node_words:
                target_trees.append((tree, path, True))
            elif bare_word_count > max_node_words:
                target_trees.append((tree, path, False))
    else:
        soup_children = [c for c in soup.contents if isinstance(c, bs4.element.Tag)]
        if len(soup_children) == 1:
            if len(soup_children[0].get_text()) >= min_node_words:
                return [(soup_children[0], [soup_children[0].name], True)], str(soup)
            else:
                return [], str(soup)
        else:
            new_soup = bs4.BeautifulSoup("", 'html.parser')
            new_tag = new_soup.new_tag("html")
            new_soup.append(new_tag)
            valid_children = []
            for child in soup_children:
                if len(child.get_text()) >= min_node_words:
                    new_tag.append(child)
                    valid_children.append(child)
            if valid_children:
                return [(new_tag, ["html"], True)], str(soup)
            else:
                return [], str(soup)

    return target_trees, str(soup)
