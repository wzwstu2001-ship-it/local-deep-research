# LDR + LightRAG + open-webui 融合设计

- **日期**：2026-08-28（2026-08-30 修订：文档上传入口从 LDR WebUI 改为 open-webui）
- **状态**：待评审（Draft for review）
- **作者**：Claude Code（经用户逐项确认）

## 1. 背景与目标

把三个独立项目融合为一套本地优先的 AI 研究与知识库系统：

| 项目 | 融合后的角色 |
|---|---|
| **LDR**（local-deep-research，Flask） | 核心「大脑」与 open-webui 的**问答后端**：agentic 深度研究、搜索、报告生成、原文存储；对外暴露 OpenAI 兼容 chat 接口 + 上传/检索 API 供 open-webui 调用 |
| **LightRAG**（FastAPI 独立服务） | 知识库引擎：建图 + 多模式检索（取代 LDR 原有 FAISS 向量检索） |
| **open-webui**（Svelte，fork） | 统一前台：聊天 + 文档上传入口（fork 新增上传组件，不重写 LDR/LightRAG 完整 UI） |

**核心目标**：

1. LightRAG 的检索方式成为 LDR 的检索工具之一（`search_lightrag`）。
2. LDR 默认的本地知识库构建方式（FAISS 向量化）整体替换为 LightRAG。
3. open-webui 作为统一前台，承载聊天与文档上传；保留关键特征（尤以 LightRAG 的知识图谱探索器为要）。
4. LDR 保留文档原文（SQLCipher），用于引用溯源。

## 2. 非目标（明确不做）

- 不把 LDR 全套 UI 重写进 open-webui（仅新增一个文档上传组件，研究/报告等能力由 LDR 的问答后端承载，不搬 UI）。
- 不把知识图谱探索器内嵌进 open-webui（保留 LightRAG 自带 WebUI，通过入口链接访问）。
- 不融合两套数据库：LDR 的 SQLCipher 与 LightRAG 的 storage 各自独立。
- 不保留 LDR 的 FAISS 向量化路径（全局关闭，非并存）。
- 不使用 open-webui 自带的 Documents/RAG（会自建第三套索引，违背「检索全走 LightRAG」）。

## 3. 现状摘要

### 3.1 LDR

- Flask + Socket.IO 应用，入口 `web/app.py::main()`，默认端口 5000。
- 知识库链路：`upload_to_collection`（抽文本）→ 存 SQLCipher `Document.text_content` → `LibraryRAGService.index_document`（chunk → embed → FAISS）→ `LibraryRAGService.search`（向量相似度 top-k）。
- 向量库仅 FAISS（`vector_stores/facade.py::VectorIndex`）；embedding 支持 sentence_transformers / ollama / openai 兼容。
- 检索工具挂载点：`web_search_engines/retriever_registry.py::retriever_registry.register(...)`——任何 LangChain `BaseRetriever` 注册后自动成为 search engine / `search_<name>` 工具，供 LangGraph agent 与 MCP 调用。
- LLM 侧原生支持 `llamacpp` / `openai_endpoint`（OpenAI 兼容 base_url）。
- 对外 API：`/api/v1`（CSRF 豁免 JSON）、`LDRClient`、`ldr-mcp`（MCP server，含 `search` / `quick_research` / `analyze_documents` 等）。

### 3.2 LightRAG

- FastAPI 服务（`lightrag_server.py`），含 REST + Ollama 兼容 API 与 React WebUI（含知识图谱探索器）。
- 检索：`POST /query`（多模式：`local` / `global` / `hybrid` / `mix` / `naive`），返回 entities / relations / chunks。
- 文档插入：`POST /documents/text`、`POST /documents/texts`、`POST /documents/upload`；异步 pipeline 处理，`GET /documents/track_status/{track_id}` 跟踪文档处理状态（`pending → processing → processed/failed`）。
- 图 API：`GET /graphs?label=...&max_depth=...`、`GET /graph/label/list` 等（`graph_routes.py`）。
- 向量库可插拔（NanoVectorDB 默认，可换 Faiss/Milvus/Qdrant/Postgres/Redis/OpenSearch）。

### 3.3 open-webui

- Svelte 前端 + FastAPI 后端；连 OpenAI 兼容接口做聊天（本方案连 LDR 暴露的 OpenAI 兼容接口，LDR 是问答后端）。
- 工具接入：v0.6.0+ 支持 MCP（经 `mcpo` 代理）；也支持自定义 Tools（OpenAPI spec）与 Pipelines。
- 自带 Documents/RAG，但本方案**不使用**（见 §6）；改为 fork 前端新增自定义上传组件，直接调 LDR 后端 API。

## 4. 总体架构

```
open-webui（统一前台，端口 3000，fork）
  ├─ 聊天/问答 → LDR OpenAI 兼容接口（LDR 是问答后端，不直连 llama.cpp）
  ├─ 自定义上传组件 → LDR /api/v1 upload（原文入库 + 转 LightRAG，手动按钮）
  └─ 入口链接 → LightRAG WebUI（知识图谱探索器）
        │
LDR（Flask，核心，端口 5000）
  ├─ OpenAI 兼容 chat 接口（供 open-webui 连接）→ LangGraph agentic 研究
  ├─ 原文存 SQLCipher（Document.text_content）
  ├─ LightRAGClient + LightRAGRetriever（HTTP 适配层）
  ├─ 对外 /api/v1：upload + track_status
  ├─ LLM provider：llamacpp / openai_endpoint（内部调底模型推理）
  └─ HTTP ──────────────────────────▶ LightRAG（FastAPI，独立服务）
                                          ├─ 图存储 + 多模式检索
                                          ├─ /query、/documents/*
                                          └─ 自有 WebUI（仅作知识图谱探索器）
```

**关键约束**：LightRAG 必须是**独立服务**（非进程内 import），因为方案要求保留其自带 WebUI 的图探索器；进程内嵌入会失去该 UI。与用户现状一致（LightRAG 已跑成服务）。

## 5. 关键设计决策（ADR）

| # | 决策 | 结论 | 理由 |
|---|---|---|---|
| D1 | 集成拓扑 | LDR 为核心 | 用户明确「LightRAG 成为 LDR 的工具」 |
| D2 | 知识库替换深度 | 索引+检索都换，保留 LDR 原文 | 拿到图谱能力，同时保留引用链/加密 |
| D3 | 路由方式 | 全局开关，取消 FAISS | LightRAG `naive` 模式即 FAISS 等价物，且是超集（多图检索） |
| D4 | UI 集成 | fork open-webui，仅新增上传组件 | 用户「基本不用 LDR 前端」；上传入口放 open-webui，问答走后端 LDR（OpenAI 兼容接口），不搬 UI |
| D5 | 文档管理归属 | 唯一入口 = open-webui（自定义组件，经 LDR 后端 API） | 上传经 LDR 后端保原文/引用链；避免 split-brain（见 §6） |
| D6 | LightRAG 部署形态 | 独立服务（HTTP） | 保留图 WebUI；与现状一致 |

## 6. 文档管理归属（单一数据流）

**文档上传的唯一入口是 open-webui（自定义上传组件），但物理上必须经过 LDR 后端 API**，否则原文不进 SQLCipher、引用链断。

| 入口 | 结果 |
|---|---|
| open-webui 自定义组件（调 LDR API）✅ | 原文进 SQLCipher + 转交 LightRAG 建图，引用链完整（**主入口**） |
| LDR WebUI 上传 ✅（保留） | 原文进 SQLCipher + 转交 LightRAG，引用链完整；但用户「基本不用」，降级为兜底 |
| LightRAG WebUI 上传 ❌ | 仅 LightRAG 存，LDR 无原文 → 引用链断、LDR 不知情 |
| open-webui 自带 Documents/RAG ❌ | 第三套索引，与「检索全走 LightRAG」矛盾 |

**落地约束**：

- LightRAG WebUI 仅保留「知识图谱探索器」用途；其文档上传/管理 UI 需禁用/隐藏，或明确标注「请去 open-webui 上传」。
- open-webui 禁用自带 Documents/RAG；上传组件直接调 LDR 的 `/api/v1` upload 接口。
- LDR WebUI 不作为上传主入口（可保留作调试/兜底，不投入前端开发）。

## 7. 数据流

### 7.1 上传文档

1. open-webui 自定义组件 → LDR `/api/v1` upload（文件/文本，携带用户凭证）。
2. LDR `upload_to_collection`（校验、抽文本）。
3. LDR 存原文到 SQLCipher `Document.text_content`（保留，供引用）。
4. LDR 调 LightRAG `POST /documents/text`（text + file_paths），**不再走 FAISS**。
5. LightRAG 异步入 pipeline 建图；LDR 返回 `track_id` 给 open-webui。
6. open-webui 轮询 LDR `/api/v1` track_status（内部封装 LightRAG `GET /documents/track_status/{track_id}`），显示「处理中 → 完成/失败」。

### 7.2 检索

1. 用户在 open-webui 提问 → open-webui 将对话发到 LDR 的 OpenAI 兼容 chat 接口。
2. LDR 的 LangGraph agent 判断需要检索 → `LightRAGRetriever` → LightRAG `POST /query`（`mode=mix`）。
3. 结果（entities/relations/chunks）映射回 LDR 的 citation 格式（title/url/snippet + source）。
4. 经 `retriever_registry.register("lightrag", retriever, is_local=True, username=...)` 成为 `search_lightrag` 工具，供 LDR agent 调用。

### 7.3 深度研究

- 不变：LDR 的 LangGraph agent 照常运行；知识库检索从 FAISS 切换到 `search_lightrag`；其余 web 搜索引擎不变。open-webui 仅作为其前台（对话经 LDR OpenAI 兼容接口直达 agent）。

## 8. 组件与改动点

### 8.1 LightRAG 侧（几乎零改动）

- 仅作为服务被调用；最多补一条配置记录（工作目录、服务地址、`MAX_TOTAL_TOKENS` 约束）。
- 约束：`MAX_TOTAL_TOKENS` 必须 < llama.cpp 的 `n_ctx_slot`（当前 `-c 65536 -np 2` → `n_ctx_slot=32768`）。

### 8.2 LDR 侧（核心改动，均在 `local-deep-research` 仓库）

1. 新增 `LightRAGClient`：封装 `POST /documents/text`、`POST /query`、`GET /documents/track_status/{track_id}`（httpx）。✅ 已完成
2. 新增 `LightRAGRetriever(BaseRetriever)`：内部调用 `LightRAGClient`，把返回映射成 LangChain `Document`（含 source 元数据）。✅ 已完成
3. 改造 `research_library/services/library_rag_service.py::LibraryRAGService`：`index_document` → 调 `LightRAGClient` 插入；`search` → 调 `LightRAGClient` 查询。✅ 已完成
4. 注册：`retriever_registry.register("lightrag", retriever, is_local=True)`，使 LangGraph agent + MCP 可见。✅ 已完成
5. 配置：新增 `LDR_LIGHTRAG_BASE_URL`（默认指向现有 LightRAG 服务）。✅ 已完成
6. 闲置/移除 FAISS 路径：`local_search_*` 配置项、`vector_stores/`、`embeddings/`。✅ 已完成（闲置）
7. **新增**：暴露 `/api/v1` 的 upload + track_status 接口（CSRF 豁免 JSON），供 open-webui 上传组件与轮询调用；补全认证（open-webui → LDR 的凭证传递，见 §13 #6）。
8. **新增**：`register_lightrag_engine()` 运行时注册（web app + MCP 进程）。✅ 已完成
9. **新增**：OpenAI 兼容 chat 接口（`/v1/chat/completions`），让 open-webui 把它当模型后端连接；内部路由到 LDR 的 LangGraph agent（研究/检索/报告），再调底层 LLM provider 推理。

### 8.3 open-webui 侧（fork，改动点）

> 端口说明：open-webui 默认端口为 8080，与 llama.cpp 的 8080 冲突；部署时需将 open-webui 改到其它端口（如 3000）。

1. fork open-webui，新增自定义上传组件（Svelte）：文件选择 → 调 LDR `/api/v1` upload → 轮询 track_status → 展示「处理中/完成/失败」。
2. 禁用 open-webui 自带 Documents/RAG（隐藏或移除入口）。
3. 连接 LDR 的 OpenAI 兼容接口（`http://127.0.0.1:5000/v1`）作为模型后端，所有对话走 LDR（不再直连 llama.cpp）。
4. 入口链接到 LightRAG WebUI（知识图谱探索器）。

## 9. 语义变化与错误处理

- **异步语义**：LightRAG 插入为异步 pipeline，上传后**非立即可检索**（原 FAISS 同步）。open-webui 上传组件需显示「处理中」，用 `track_id` 轮询 LDR `/api/v1` track_status（状态 `pending → processed`）。
- **内容去重**：LightRAG 对内容做 hash 去重（相同内容即使文件名不同也判重，标记 `DUPLICATE:content_hash` 且状态 FAILED）；LDR 上传接口需向 open-webui 透传该结果，避免重复上传被误判为「成功」。
- **检索失败不静默兜底**：FAISS 已取消；LightRAG 不可用时应返回明确错误，不做静默降级。
- **上下文超限**：`MAX_TOTAL_TOKENS < n_ctx_slot`（见 §8.1），避免再次出现 `24872 > 8192` 类错误。
- **egress 安全**：`LightRAGRetriever` 注册 `is_local=True`，被 LDR egress 策略归类为本地，不触发联网限制。

## 10. 安全与隐私

- LDR 原文仍在 SQLCipher 加密库中（保留现有加密/权限体系）。
- LightRAG storage 为本地存储（工作目录），不新增外网依赖。
- open-webui 仅连本地 llama.cpp / LDR / LightRAG，无外部数据出口（web 搜索仍由 LDR 既有 egress 策略管控）。
- open-webui → LDR 的 API 调用需认证（凭证不落 open-webui 日志；见 §13 #6）。

## 11. 测试策略

- **单元测试**（mock httpx）：
  - `LightRAGClient`：插入/查询/轮询的请求构造与响应解析。
  - `LightRAGRetriever`：LightRAG 返回 → LangChain `Document` 映射（含 citation 格式）。
  - `LibraryRAGService` 改造：index/search 正确路由到 `LightRAGClient`，不再触碰 FAISS。
- **集成测试**：LDR + 真实 LightRAG 服务（或 mock server）联调：上传 → 等待处理 → 检索 → 引用链完整（`scripts/lightrag_smoke.py` 已覆盖 insert → track_status → query 三步）。
- **回归**：LDR 现有 `tests/` 中与 `library_rag_service`、`retriever_registry`、`search_engine_factory` 相关的用例需更新/通过。

## 12. 分阶段实施

- **Phase 1（核心，已完成）**：LDR 关 FAISS、接 LightRAG（client + retriever + service 改造 + 运行时注册）、保留原文；LightRAG 作为服务；联调三步全绿（`scripts/lightrag_smoke.py`）。
- **Phase 2（open-webui 接入，主路径）**：LDR 暴露 OpenAI 兼容 chat 接口（问答后端）+ `/api/v1` upload + track_status（含认证）；open-webui fork + 上传组件 + 禁用自带 Documents + 连接 LDR 作为模型后端；LightRAG WebUI 文档管理 UI 禁用处理。

## 13. 风险与开放问题

1. **函数调用依赖**（LDR agent 内部）：需验证 `qwen3.6-35b-a3b` 经 llama.cpp（`--jinja`）是否稳定支持 function calling；若不稳定，LDR agent 的工具调用（检索/研究）可能退化。此为 LDR 既有能力，不再是 open-webui 接入的新风险。
2. **异步处理反馈**：open-webui 上传组件需「处理中 + 轮询 track_status」；上传线是 open-webui 侧 UI 工作，检索线不涉及。
3. **引用格式映射**：LightRAG 返回的 source 结构与 LDR 现有 citation（title/url/snippet）不完全一致，需一层适配；字段覆盖需验证。
4. **LightRAG 工作目录与 LDR 用户隔离**：多用户场景下，LDR 按用户隔离，而 LightRAG 为单一服务，需确认工作目录/workspace 隔离策略（当前单用户可暂缓）。
5. **open-webui fork 维护成本**：fork 后需跟进上游版本，上传组件需适配版本升级；评估是否可改用 open-webui 的 plugin/Pipelines 机制替代 fork。
6. **open-webui → LDR 认证**：`/api/v1` upload 需携带用户凭证；确定用 API key / 服务账号 / OAuth，且凭证不回写 open-webui 日志。
