
---

# 📘 中文 HTML 结构化知识库检索系统（Milvus + Elasticsearch）

本项目用于将大规模 HTML 网页内容（如电商平台运营规则、政策说明等）结构化为高质量、可检索的知识块。系统支持：

* 🔍 **Elasticsearch 关键字检索**：支持中文 IK 分词器（可 fallback）
* 🧠 **Milvus 语义向量检索**：基于 BGE/BCE 嵌入模型
* 📝 **ChatGLM 摘要生成**（可选）
* 🔁 **BCE Reranker 精排模型**：进一步优化向量检索结果排序
* 📦 **块级存储与全局索引**：支持 HTML 分块结构管理，跨文件唯一定位
* 🧱 **目录结构保留 + 批量处理**

---

## 📁 项目结构

```
.
├── utils/
│   ├── db_utils.py             # 数据库构建、Milvus/ES 插入与查询、精排器支持
│   ├── html_utils.py           # HTML 文档结构化工具（提取块树结构、子域）
│   ├── text_process_utils.py   # 文本清洗、去重、分词、摘要等辅助函数
│   └── __init__.py
├── html_clean.py               # Step 1: 清洗 HTML 格式与结构
├── block_construct.py          # Step 2: 将 HTML 结构划分为文档块（含摘要）
├── db_construct.py             # Step 3: 插入已分块的 JSON 文档到 ES 与 Milvus
├── jieba_dict_construct.py     # 构建供 Elasticsearch 使用的自定义词典
├── query.py                    # 启动查询模式（Milvus + reranker）
├── requirements.txt            # Python 依赖
└── README.md
```

---

## 🧱 数据清洗与知识库构建流程

本系统采用「清洗 → 分块 → 入库」三步预处理流程：

### ① HTML 清洗与结构标记：`html_clean.py`

处理原始 HTML 文档，完成：

* 删除 `<script>`、样式等无用结构
* 标记 `<h1>`\~`<h6>`、表格、段落等结构块
* 提取 `<time>` 时间标签（如存在）
* 输出结构规整后的 HTML 至 `*_cleaned/` 目录

---

### ② 文档结构化分块与摘要：`block_construct.py`

将结构化 HTML 文档切分为语义块，提取以下字段：

* `chunk_idx`：局部块编号
* `title`：标题（从结构块或首句提取）
* `page_name`, `page_url`
* `text`, `summary`, `time`

📌 **提前离线生成摘要（ChatGLM）以节省 I/O 与推理开销**

输出目录：

```
data/
└── xxx_cleaned_block/
    ├── a/b/c.json
    └── ...
```

---

### ③ 构建 Elasticsearch 与 Milvus 索引：`db_construct.py`

该模块会遍历所有 `*.json` 文档块，并写入数据库：

#### ✅ Elasticsearch

* 优先使用 IK 分词器，自动 fallback 到 `standard`
* 支持字段搜索 + 分词查询
* 索引字段：
  * `text`, `summary`, `title`, `page_url`, `page_name`, `time`, `chunk_idx`, `global_chunk_idx`

#### ✅ Milvus

* 使用指定嵌入模型编码 text 字段
* 字段配置支持中文长文本（最长支持 9000 字）
* 自动维护 `global_chunk_idx`和局部`chunk_idx`（用于多文件跨块全局定位）


---

## 🔍 查询系统（query.py）

支持用户输入自然语言问题，系统执行：

* 嵌入语义 → Milvus 检索相似文档块
* 可选：使用 BCE reranker 精排
* 输出包含：标题、时间、摘要、内容等信息


---

## 💡 字段设计说明（支持中文内容）

| 字段名                | 类型         | 用途                | 最大长度 |
| ------------------ | ---------- | ----------------- | ------ |
| `chunk_idx`        | INT64    | 文档内局部编号           | —      |
| `global_chunk_idx` | INT64 (主键)   | 全局唯一编号（自动维护）      | —      |
| `text`             | VARCHAR    | 文本内容，最长中文达 8482 字 | `待定` |
| `title`            | VARCHAR    | 提取标题              | `512`  |
| `summary`          | VARCHAR    | 自动摘要              | `4096` |
| `page_url`         | VARCHAR    | HTML 文件路径或原始链接    | `512` |
| `page_name`        | VARCHAR    | 页面文件名或上级目录名       | `512`  |
| `time`             | VARCHAR    | 起始时间标签内容（如存在）     | `128`   |

---

## 📦 模型使用

| 模块           | 模型名称                                           |
| ------------ | ------------------------------------------------ |
| 向量嵌入（Milvus） | `bge-base-zh`, `bge-m3`, `bce-embedding-base_v1` |
| 精排 Reranker  | `bce-reranker-base_v1`              |
| 摘要生成模型       | `chatglm3-6b`                            |

---

## ⚙️ 环境与依赖

确保部署：
* [x] Milvus 2.x 向量服务
* [x] Elasticsearch 8.x，若需 IK 分词器需预先离线安装
* [x] 所有 HuggingFace 模型路径本地化或提前缓存
以及其他python包

---

## 🧩 其他模块说明

* `utils/html_utils.py`：构建 HTML DOM 树结构，用于高质量分块（基于标签等级、子域范围）
* `utils/db_utils.py`：
  * 支持分离的 Milvus 插入、ES 插入函数
* `jieba_dict_construct.py`：用于将 `user_dict.txt` 构建为 `.dic` 格式供 Elasticsearch 使用

---
## 📌 TODO 

* [ ] 支持多轮 QA 与上下文对话（意图指代）
* [ ] 高层级信息利用（路径标题摘要等）


---