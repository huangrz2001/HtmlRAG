
---

# 📘 中文 HTML 结构化知识库检索系统（支持 Milvus + Elasticsearch）

本项目旨在将大规模 HTML 网页内容（如电商平台运营规则、产品说明、平台政策等）自动结构化为高质量的知识块，支持关键字与语义双检索，适用于构建中文领域的智能问答与运营知识系统。

## ✅ 系统核心能力
* 🔍 **Elasticsearch 关键字检索**
  * 支持中文 IK 分词器
  * 支持标题/正文/摘要多字段搜索
  * 提供字段权重配置

* 🧠 **Milvus 向量语义检索**
  * 基于 BGE/BCE 等中文 Embedding 模型
  * 支持语义 Top-K 召回
  * 提供摘要增强与页面去重机制

* ✏️ **ChatGLM 智能摘要生成**
  * 针对规则/操作/信息类文档进行摘要压缩
  * 可选接入本地模型或 vLLM 服务

* ❓ **ChatGLM 用户问题生成**
  * 基于块内容生成潜在用户提问
  * 提高检索覆盖面与 QA 适配性

* 🔁 **Reranker 精排模块**
  * 使用 BCE 模型基于摘要路径、语义内容等信息进行精排
  * 融合 ES + 向量检索候选结果，提升最终排序质量

* 📦 **块级结构化与索引管理**
  * 支持 HTML 自动切块、跨文件唯一标识（chunk\_idx + file\_idx）
  * 文档清洗、表格展开、标题封装等结构化处理

* ⚙️ **统一 API 部署与 vLLM 支持**
  * 提供 FastAPI 接口封装
  * 支持 ChatGLM、Yi 等模型通过 vLLM 接入，支持摘要/问句生成与 Query Rewriting

---
## 📁 项目结构

```
.
├── utils/
│   ├── db_utils.py             # 向量索引构建、文档块插入与删除、索引重建、多源融合检索、reranker 精排等
│   ├── html_utils.py           # HTML属性清洗、表格消解、表格和标题域包装、语义块切分树等
│   ├── text_process_utils.py   # 从 HTML 清洗结果中提取结构化文档块，生成摘要与问句，并提供查询构建、文本清洗与去重能力等
│   ├── llm_api.py              # 文档块摘要生成（支持ChatGLM与vLLM）、潜在问题生成、query重写等 LLM 工具函数
│   ├── jieba_util.py           # HTML 文本高频短语提取与自定义词典生成
│   ├── config.py               # 配置全局统一的参数项（如路径、模型地址等）
├──finetune/
│   ├── trained_reranker_single/ # 保存已训练好的模型 reranker
│   ├── wandb/                   # wandb 日志目录（可选）
│   ├── build_dataset.py         # 构建 reranker 微调数据集（潜在问题->初查->大模型API，正负问句对）
│   ├── reranker_ft.py           # reranker 模型训练主脚本（单卡全参微调）
│   ├── reranker_qa_dataset.json # reranker 训练用的 QA 样本（已构造问句对 + 标签）
├── step1_html_clean.py               # Step 1: 清洗 HTML 格式与结构
├── step2_block_construct.py          # Step 2: 将 HTML 结构划分为文档块（含摘要）
├── step3_db_construct.py             # Step 3: 插入已分块的 JSON 文档到 ES 与 Milvus
├── step4_query.py                    # Step 4: 根据query查询数据库
├── main.py                           # fastAPI支持，使用uvicorn启动
├── test_rewrite_api.py               # 测试API并发调用负载能力
├── test_rewrite_vllm.py              # 测试原生vLLM并发负载能力
├── rewrite_vllm.py                   # vllm负载测试
├── run_vllm.sh                       # 启动vllm服务（ChatGLM4-9B）
├── config.json                       # 全局的配置文件
└── README.md
```


## 🧱 数据清洗与知识库构建流程
本系统采用「清洗 → 分块 → 入库」三步预处理流程：

### ① HTML 清洗与结构标记：`step1_html_clean.py`
处理原始 HTML 文档，完成：
* 删除 `<script>`、样式等无用结构
* 标记 `<h1>`\~`<h6>`、表格结构块，包裹`domain`属性
* 提取 `<time>` 时间标签（如果存在）
* 输出结构规整后的 HTML 至 `*_cleaned/` 目录（内容可精简到原先的1/10以内）
---

### ② 文档结构化分块与摘要：`step2_block_construct.py`
将结构化 HTML 文档切分为语义块并保存，生成以下字段：
* `chunk_idx`：局部块编号
* `title`：标题（从结构块或首句提取）
* `page_name`, `page_url`
* `text`, `summary`, `time`
* 输出结构规整后的 block 至 `*_block/` 目录（切块已经保存为完整json，可以直接批量插入）
---

### ③ 构建 Elasticsearch 与 Milvus 索引：`step3_db_construct.py`
该模块会遍历所有 `*.json` 文档块，并写入数据库：
#### ✅ Elasticsearch
* 优先使用 IK 分词器，自动 fallback 到 `standard`
* 支持字段搜索 + 分词查询
* 索引字段：
  * `text`, `summary`, `title`, `page_url`, `page_name`, `time`, `chunk_idx`, `global_chunk_idx`
#### ✅ Milvus
* 使用指定嵌入模型编码 text 字段
* 字段配置支持中文长文本
* 自动维护 `global_chunk_idx`和局部`chunk_idx`（用于多文件跨块全局定位）

### 🔍 查询：`step4_query.py`
支持用户输入自然语言问题，系统执行：

* 嵌入语义 → Milvus 检索相似文档块
* 可选：使用 BCE reranker 精排
* 输出包含：标题、时间、摘要、内容等信息


---
| 字段名                | 类型             | 用途说明                        | 最大长度                       |
| ------------------ | -------------- | --------------------------- | -------------------------- |
| `global_chunk_idx` | `INT64`（主键，自增） | 全局唯一块编号，用于 Milvus 主键        | —                          |
| `document_index`         | `INT64`        | 所属文件编号（便于整文件删除与管理）          | —                          |
| `chunk_idx`        | `INT64`        | 文档内局部编号（同一 HTML 文件内的块编号）    | —                          |
| `vector`           | `FLOAT_VECTOR` | 向量表示（BGE/BCE 输出）            | `dim=768`                  |
| `text`             | `VARCHAR`      | 文本块正文内容，支持较长段落              | `20000`（Milvus）            |
| `title`            | `VARCHAR`      | 文本块的标题或首段 heading，便于快速定位    | `512`                      |
| `summary`          | `VARCHAR`      | 自动生成的块级摘要，用于 reranker 与预览显示 | `4096`                     |
| `question`         | `VARCHAR`      | 自动生成的代表性问题，用于 QA 覆盖与训练数据增强  | `1024`                     |
| `page_url`         | `VARCHAR`      | 原始 HTML 文件路径或来源 URL         | `1024`（Milvus）/ `text`（ES） |
| `page_name`        | `VARCHAR`      | 文件名或目录名（页面归属）               | `512`                      |
| `time`             | `VARCHAR`      | HTML 开头 `<time>` 标签内容（如存在）  | `128`                      |

---

## 📦 模型使用

| 模块           | 模型名称                                           |
| ------------ | ------------------------------------------------ |
| 向量嵌入 |  `bce-embedding-base_v1`    |
| 精排 Reranker  | `bce-reranker-base_v1`              |
| 摘要和重写模型       | `THUDM/glm-4-9b-chat`                            |
---


## ⚙️ 环境与依赖
本项目依赖如下核心组件与服务，请确保环境完整：
### ✅ 基础服务依赖
* **Milvus 2.x**
  * 向量检索服务，用于存储结构化文档块的嵌入向量
  * 地址：`host:19530`
* **Elasticsearch 8.x**
  * 关键字检索服务，需安装 IK 中文分词器插件
  * 地址：`host:9200`
* **vLLM 模型本地推理服务**
  * 用于摘要生成 / 问题生成 / Query 重写等任务
  * 支持部署 ChatGLM、Yi 等模型，建议 GPU 加速运行
  * 默认监听：`http://localhost:8000/v1/chat/completions`

### ✅ Python 依赖环境
请使用 Python ≥ 3.8，推荐使用 Conda 环境：
```bash
conda env create -f environment.yml
conda activate htmlrag
bash run_vllm.sh                                        # 启动vLLM服务
uvicorn main:app --host 0.0.0.0 --port 8080 --reload    # 启动API
```

主要依赖组件（已在 `environment.yml` 中包含）：

| 库名                    | 用途说明               |
| --------------------- | ------------------ |
| `transformers`        | 本地加载 ChatGLM 等模型   |
| `pymilvus`            | 与 Milvus 向量库通信     |
| `elasticsearch`       | 与 ES 检索系统通信        |
| `jieba`               | 中文分词、关键词提取         |
| `scikit-learn`        | 文本相似度、去重、TF-IDF 支持 |
| `beautifulsoup4`      | HTML 清洗与结构提取       |
| `requests`            | 调用 vLLM HTTP 接口    |
| `fastapi` + `uvicorn` | 提供统一插入 / 检索 API 服务 |
| `vllm`                | 提供本地模型的加速推理服务 |


---