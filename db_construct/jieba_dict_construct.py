import os
import re
from collections import Counter
from bs4 import BeautifulSoup

def extract_text_from_html(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            text = soup.get_text()
            print(f"✔️ 成功提取文本：{html_path}（{len(text)} 字）")
            return text
    except Exception as e:
        print(f"❌ 读取失败: {html_path}, 原因: {e}")
        return ""

def collect_all_html_texts(root_dir):
    texts = []
    print(f"🔍 正在遍历目录: {root_dir}")
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".html"):
                full_path = os.path.join(root, file)
                text = extract_text_from_html(full_path)
                if text:
                    texts.append(text)
    print(f"📄 共收集到 {len(texts)} 个 HTML 文本")
    return texts

def split_sentences(text):
    # 中文及中英标点断句
    return re.split(r"[。！？；，,.!?;:、\n\r]+", text)

def is_chinese_word(word):
    return bool(word) and all('\u4e00' <= ch <= '\u9fff' for ch in word)


def extract_phrases_by_frequency(texts, ngram_range=(2, 5), top_k=500):
    counter = Counter()
    print("📊 正在进行 n-gram 词频统计（按句截断，纯中文）...")

    for text in texts:
        sentences = split_sentences(text)
        for sentence in sentences:
            sentence = re.sub(r"[^\u4e00-\u9fff]", "", sentence)  # 仅保留中文
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(sentence) - n + 1):
                    gram = sentence[i:i + n]
                    if is_chinese_word(gram):
                        counter[gram] += 1

    top_phrases = counter.most_common(top_k)
    print(f"✅ 高频短语提取完毕（分句后），共返回前 {len(top_phrases)} 项")
    return top_phrases




def save_to_jieba_dict(phrases, output_path="user_dict.txt", default_freq=10000):
    print(f"📝 正在保存词典到 {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for phrase, freq in phrases:
            f.write(f"{phrase} {freq or default_freq} n\n")
    print(f"✅ 词典保存成功，可用于 jieba / IK 分词器")


def filter_keep_longest_only(phrases):
    print("📊 正在过滤冗余词 ...")
    phrases_sorted = sorted(phrases, key=lambda x: (-len(x[0]), -x[1]))
    kept = []
    for phrase, freq in phrases_sorted:
        if any(phrase in longer for longer, _ in kept if phrase != longer):
            continue  # 是已保留词的子串，跳过
        kept.append((phrase, freq))
    return kept

def filter_by_freq_ratio(phrases, threshold=0.8):
    print("📊 正在过滤冗余词 ...")
    phrases_sorted = sorted(phrases, key=lambda x: -len(x[0]))  # 先长词后短词
    phrase_map = {phrase: freq for phrase, freq in phrases}
    filtered = {}

    for phrase in phrase_map:
        if any(
            phrase in longer and phrase != longer and
            phrase_map.get(longer, 0) >= phrase_map[phrase] * threshold
            for longer in phrase_map
        ):
            continue  # 被某个更长词吸收，跳过
        filtered[phrase] = phrase_map[phrase]

    return list(filtered.items())



# ===== 主流程入口 =====
if __name__ == "__main__":
    html_root_dir = "巨量千川知识库all"  # ❗ 请替换为实际 HTML 目录
    texts = collect_all_html_texts(html_root_dir)
    phrases = extract_phrases_by_frequency(texts, ngram_range=(2, 12), top_k=1000)
    filtered_phrases = filter_by_freq_ratio(phrases)
    # filtered_phrases = phrases
    save_to_jieba_dict(filtered_phrases, output_path="user_dict.txt")
    # 额外生成 IK 分词器使用的纯词条词典
    ik_dict_path = "my_dict.dic"
    with open("user_dict.txt", "r", encoding="utf-8") as fin, open(ik_dict_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split()
            if parts:
                word = parts[0]
                fout.write(word + "\n")
    print(f"✅ IK 分词器词典已生成: {ik_dict_path}（仅包含词条）")
