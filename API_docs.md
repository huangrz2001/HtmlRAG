
---

## 🔎 RAG 检索服务接口文档

> 🌐 **Base URL**：`http://192.168.7.179:80`  
> 🌐 **RAG docs**：`http://localhost:80/rag/docs`  
> 🌐 **问答对CURD docs**：`http://localhost:80/qa/docs`  
> 🌐 **文档CURD及query重写 docs**：`http://localhost:80/document/docs`
---

### 📡 `/query` - 查询参考内容

**完整路径**：`POST http://192.168.7.179:80/rag/query`

**描述**：根据用户提出的问题，在已向量化的文档库中检索匹配内容

#### 📤 请求体（QueryRequest）

```json
{
  "question": "如何优化广告投放效果？",
  "limit": 16
}
```
#### ✅ 返回值（成功）
```json
{
  "status": "success",
  "results": [
    {
      "text":"方向  问题点  怎么做？...",
      "page_url":"总知识库_cleaned/巨量本地推帮助中心/功能介绍/巨量本地推直播全域推广产品手册.html",
      "page_name":"巨量本地推直播全域推广产品手册",
      "chunk_idx":6,
      "title":"五、怎么投放效果好？​ 表格行1-7",
      "summary":"进行全域推广时，...",
      "time":"2025-04-23 17:58:33"
    }
  ],
  "reference_contents": [
    "[内容1]:...",
    "[内容1]:...",
    ...
  ],
  "processing_time": 1.4188416004180908
}
```
#### ❌ 返回值（失败）

```json
{
  "status": "failed",
  "processing_time": 0.4188416004180908
}
```

---

## 💬 问答对CURD 服务接口文档

> 🌐 **Base URL**：`http://192.168.7.179:80/qa`

---

### 📥 `/insert` - 插入问答对

**完整路径**：`POST http://192.168.7.179:80/qa/insert`

**描述**：将问答数据插入向量数据库，用于后续直接复用。

#### 📤 请求体

```json
{
  "qa_index": 10086,
  "question": "如何优化广告投放效果？"
}
```
#### ✅ 返回值（成功）
```json
{
  "status": "inserted",
  "qa_index": 10086,
  "question": "如何优化广告投放效果？"
}
```
#### ❌ 返回值（失败）

```json
{
  "status": "failed",
  "qa_index": 10086,
  "qa_index": "如何优化广告投放效果？"
}
```
---

### 🔍 `/search` - 检索语义相似问题

**完整路径**：`POST http://192.168.7.179:80/qa/search`

**描述**：在向量数据库中基于语义查找与输入问题相似的问句。

#### 📤 请求体

```json
{
  "question": "如何优化广告投放效果？",
  "top_k": 5
}
```
#### ✅ 返回值（成功）
```json
{
  [
    {
      "qa_index": 10086,
      "question": "如何优化广告投放效果？",
      "distance": 0.987527362731,
    },

    {
      ...
    },
    ...
  ]
}
```
#### ❌ 返回值（失败）

```json
{
  "status": "failed"
}
```
---

### 🗑️ `/delete` - 删除问答对

**完整路径**：`POST http://192.168.7.179:80/qa/delete`

**描述**：根据索引号从向量库中删除某条问答记录。

#### 📤 请求体

```json
{
  "qa_index": 10086
}
```
#### ✅ 返回值（成功）
```json
{
  "status": "deleted",
  "qa_index": 10086,
  "delete_count": 1
}
```
#### ❌ 返回值（失败）

```json
{
  "status": "failed",
  "qa_index": 10086,
  "error": "..."
}
```

---

## 📄 文档CURD及query重写 服务接口文档

> 🌐 **Base URL**：`http://192.168.7.179:80/document`

---

### 📥 `/add` - 新增文档

**完整路径**：`POST http://192.168.7.179:80/document/add`

**描述**：向文档库插入一个 HTML 页面内容，文档将被解析为结构化语义块，用于向量化检索。

#### 📤 请求体（InsertRequest）

```json
{
  "document_index": 4008893141271707648,
  "resource_id": 4008893141271707648,
  "page_url": "分类一/_全球购_发布违禁商品_信息_细则.html"
}
```

#### ✅ 返回值（成功）

```json
{
  "result": "ok",
  "document_index": 4008893141271707648,
  "inserted_chunks_milvus": 7,
  "inserted_chunks_es": 7
}
```
#### ❌ 返回值（失败）

```json
{
  "result": "fail",
  "error": "HTML 文件下载失败"
}
```

---

### 🗑️ `/delete` - 删除文档

**完整路径**：`POST http://192.168.7.179:80/document/delete`

**描述**：根据文档索引，从文档库中删除该页面对应的所有语义块，并且删除保存的文件。

#### 📤 请求体（DeleteRequest）

```json
{
  "document_index": 4008893141271707648,
  "page_url": "分类一/_全球购_发布违禁商品_信息_细则.html"
}
```

#### ✅ 返回值（成功）

```json
{
  "result": "ok",
  "document_index": 4008893141271707648,
  "deleted_chunks_milvus": 7,
  "deleted_chunks_es": 7
}
```
#### ❌ 返回值（失败）

```json
{
  "result": "fail",
  "error": "删除失败: 向量库连接失败"
}
```

---

### 🔁 `/query_rewrite` - 重写 Query

**完整路径**：`POST http://192.168.7.179:80/document/query_rewrite`

**描述**：基于历史对话语境和当前 query 生成语义更完整的新查询，用于提升多轮问答准确性。

#### 📤 请求体（RewriteRequest）

```json
{
  "dialogue": 
  [
    {"speaker": "user", "text": "我打算做一场新品直播，有什么建议？"}, 
    {"speaker": "assistant", "text": "建议提前3天预热，准备预售链接并邀请达人带播。"}, 
    {"speaker": "user", "text": "我们预算紧张，请问达人费用能省吗？"}, 
    {"speaker": "assistant", "text": "可以选择平台免费达人共创服务，或用中腰部达人进行合作。"}, 
    {"speaker": "user", "text": "那如果我们就是小商家呢？"}], 
  "final_query": "那如果我们就是小商家呢？"
}

```

#### ✅ 返回值（成功）

```json
{
  "status": "ok",
  "rewritten_query": "若是小商家进行新品直播推广，在预算有限的情况下应如何规划达人合作？"
}
``` 

#### ❌ 返回值（失败）

```json
{
  "status": "fail",
  "error": "重写失败: VLLM 请求超时"
}
```

### 🩺 `/ping` - 健康检查

**完整路径**：`GET http://192.168.7.179:80/document/ping`

**描述**：服务存活探测接口，用于部署监控与健康检查。

#### 📥 返回值

```json
{
  "status": "ok"
}
```
