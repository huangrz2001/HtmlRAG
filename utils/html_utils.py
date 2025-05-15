# -*- coding: utf-8 -*-
"""
HTML 清洗与结构化处理模块

核心功能：
- clean_html: 对原始 HTML 进行清洗（保留表格结构、标题域包装等）
- build_block_tree: 将清洗后的 HTML 拆解成结构化的文本块（用于向量生成）
"""

import re
import json
import os
import bs4
from typing import List, Tuple, Dict
from collections import defaultdict
from bs4 import BeautifulSoup, Comment, Tag


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


def process_html(html: str) -> str:
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

    convert_headings(soup)
    wrap_heading_domains(soup)
    return str(soup)


def clean_xml(html: str) -> str:
    """移除 XML/Doctype 声明"""
    html = re.sub(r"<\?xml.*?>", "", html)
    html = re.sub(r"<!DOCTYPE.*?>", "", html, flags=re.IGNORECASE)
    return html


def clean_html(html: str, keep_att=False) -> str:
    """主入口函数：清洗 HTML，保留结构并统一化"""
    soup = BeautifulSoup(html, 'html.parser')
    html = simplify_html_keep_table(soup, keep_att)
    html = process_html(html)
    html = clean_xml(html)
    return html


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
