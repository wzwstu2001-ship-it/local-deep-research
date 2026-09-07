# LDR + LightRAG + open-webui 融合设计

- **日期**：2026-08-28（2026-09-03 修订：检索改为 agent 自主组合、FAISS 恢复、新增会话级隔离、图谱入口改前端直连；2026-09-06 修订：agent 自由推理——寒暄/常识直接回答不检索，检索仅按需，LightRAG 仅为可选检索工具）
- **状态**：待评审
- **作者**：wzw

## 1. 背景与目标

把三个独立项目融合为一套本地优先的 AI 研究与知识库系统：

| 项目 | 融合后的角色 |
|---|---|
| **open-webui**（Svelte，fork） | 统一前台：聊天 + 文档上传组件 + 知识图谱查看入口；对外只连 LDR 与 LightRAG 两个后端 |
| **LDR**（local-deep-research，Flask） | 主要后端「大脑」：agentic 深度研究、搜索、报告生成、原文存储；负责用户上传文档的 **FAISS 向量检索**（会话级隔离）；对外暴露 OpenAI 兼容 chat 接口 + 上传/检索 API 供 open-webui 调用 |
| **LightRAG**（FastAPI 独立服务） | 全局**基础文档库**（由后端人员维护并上传）：建图 + 向量/图谱检索 + 自带 WebUI 知识图谱探索器 |

**核心目标**：

1. LDR 的向量检索**不禁用**——用户在 open-webui 上传的文档参与 LDR 的 FAISS 向量检索。
2. LightRAG 数据库由后端人员维护、在 LightRAG 自身 UI 上传，作为**全局共享的基础文档库**，不经 LDR。
3. LDR 由 LangGraph **agent 自由推理**：寒暄 / 自我介绍 / 常识等无需检索的问题**直接回答**（不调用任何检索工具）；需要事实、时效或来源支撑时，agent 才**自主选择** 1-3 种检索方式（组合，非互斥）：网搜（`search_*`）/ 本地向量库（`search_library`，FAISS）/ LightRAG 向量+图谱检索（`search_lightrag`）。LightRAG 仅为可选检索方式之一，不改变 agent 原本的推理与工具调用逻辑。
4. open-webui 前端直连 LightRAG 图 API，提供**查看知识图谱**的入口。
5. **会话级隔离**：不同 chat 会话只能检索该会话自己上传的文档，不能检索其他会话上传的文档（LightRAG 全局基础库对所有会话共享，不受此约束）。
6. LDR 保留文档原文（SQLCipher），用于引用溯源。

## 2. 非目标（明确不做）

- 不把 LDR 全套 UI 重写进 open-webui（仅新增文档上传组件与图谱查看入口；研究/报告等能力由 LDR 问答后端承载，不搬 UI）。
- 不融合两套数据库：LDR 的 SQLCipher 与 LightRAG 的 storage 各自独立。
- 不使用 open-webui 自带的 Documents/RAG（会自建第三套索引，与「用户文档走 LDR FAISS、基础文档走 LightRAG」的分工矛盾）。
- 不新增「按文档过滤图」的 LightRAG 服务端能力：LightRAG workspace 是实例级而非请求级，图谱按全局展示；会话级隔离只作用于 LDR 的 FAISS 检索。
- 不在 LDR 内部做「三选一路由」判定节点——检索路由由 agent 的 tool choice 机制天然承载（见 §5 D2）。

## 3. 现状摘要

### 3.1 LDR

- Flask + Socket.IO 应用，入口 `web/app.py::main()`，默认端口 5000。SQLCipher 分库（每用户一个加密库）。
- 知识库链路：`upload_to_collection`（抽文本）→ 存 SQLCipher `Document.text_content` → `LibraryRAGService.index_document`（chunk → embed → FAISS）→ `LibraryRAGService.search`（向量相似度 top-k）。
- 文档/collection 多对多：`library.py` 已有 `Document`、`Collection`、`DocumentCollection` 三模型（无 user_id，隔离靠分库）。
- **FAISS 现状**：`LibraryRAGService.search()` 的 FAISS 语义搜索入口在 Phase 1 已删除（切到 LightRAG），但 `VectorIndex`、`FaissVectorStore.search`、写锁、`FAISSIndexVerifier`、reconcile 等代码完整保留，可复用。`index_document` 有 FAISS 分支但当前为死代码（工厂恒传非 None 的 lightrag_client）。需切回 FAISS。
- 检索工具挂载点：`web_search_engines/retriever_registry.py::retriever_registry.register(...)`——任何 LangChain `BaseRetriever` 注册后自动成为 search engine / `search_<name>` 工具，供 LangGraph agent 与 MCP 调用。`LightRAGRetriever`（注册名 `lightrag`、`is_local=True`）独立存在，走 `client.query_data`。
- LLM 侧原生支持 `llamacpp` / `openai_endpoint`（OpenAI 兼容 base_url）。
- 对外 API：`/api/v1`（CSRF 豁免 JSON）、`LDRClient`、`ldr-mcp`、`/openai_compat`（OpenAI 兼容 chat 接口，已实现 Task 1-3）。
- `ChatSession`（`database/models/chat.py`）**无** collection_id / user_id，是会话级隔离的主要缺口。

### 3.2 LightRAG

- FastAPI 服务（`lightrag_server.py`），含 REST + Ollama 兼容 API 与 React WebUI（含知识图谱探索器）。
- 检索：`POST /query`（多模式：`local` / `global` / `hybrid` / `mix` / `naive`）、`POST /query/data`，返回 entities / relations / chunks。
- 文档插入：`POST /documents/text`、`POST /documents/texts`、`POST /documents/upload`；异步 pipeline，`GET /documents/track_status/{track_id}` 跟踪状态。
- 图 API（`graph_routes.py`）：`GET /graph/label/list`、`GET /graph/label/popular`、`GET /graph/label/search`、`GET /graphs?label=&max_depth=&max_nodes=`、`GET /graph/entity/exists`。认证 `combined_auth`（X-API-Key / Bearer JWT / 开放三种）；CORS 默认 `*`。**前端直连即可，无需新增接口。**
- 向量库可插拔（NanoVectorDB 默认，可换 Faiss/Milvus/Qdrant/Postgres/Redis/OpenSearch）。

### 3.3 open-webui

- Svelte 前端 + FastAPI 后端；连 OpenAI 兼容接口做聊天（`.env` 配 `OPENAI_API_BASE_URLS` / `OPENAI_API_KEYS`）。
- 工具接入：v0.6.0+ 支持 MCP；也支持自定义 Tools（OpenAPI spec）与 Pipelines。
- 自带 Documents/RAG，但本方案**不使用**（见 §6）。
- Chat 表有 `id`（UUID）+ `user_id`，无「每 chat 独立文档集合」原生机制，需靠透传 chat_id 到 LDR 实现会话隔离。

## 4. 总体架构

```
open-webui（统一前台，fork，默认端口 8080 冲突需改）
  ├─ 聊天/问答 ──────────────▶ LDR OpenAI 兼容接口（/v1/chat/completions，透传 chat_id）
  ├─ 文档上传组件 ──────────▶ LDR /api/v1 upload（透传 chat_id → 会话专属 collection）
  ├─ 图谱查看页 ────────────▶ LightRAG 图 API（前端直连，GET /graphs、/graph/label/list）
  └─ （禁用自带 Documents/RAG）
        │
LDR（Flask，核心后端，端口 5000）
  ├─ OpenAI 兼容 chat 接口 → LangGraph agent 自由推理（直接回答 / 按需检索）
  ├─ 用户文档原文存 SQLCipher（Document.text_content）
  ├─ 用户文档 FAISS 向量索引（会话级隔离：collection_id 过滤）
  ├─ LightRAGRetriever（HTTP 适配层，检索全局基础库）
  ├─ 对外 /api/v1：upload + track_status（CSRF 豁免）
  └─ LLM provider：llamacpp / openai_endpoint

LightRAG（FastAPI，独立服务，全局基础文档库）
  ├─ 图存储 + 向量存储 + 多模式检索
  ├─ /query、/query/data、/documents/*
  ├─ /graphs、/graph/label/list（图 API，前端直连）
  └─ 自有 WebUI（后端人员在此上传基础文档 + 知识图谱探索器）
```

**检索工具候选池**（agent 自主组合，见 §5 D2）：

| 工具 | 注册名 | 后端 | 数据范围 |
|---|---|---|---|
| 网搜 | `search_<engine>` | 各搜索引擎 | 公开互联网 |
| 本地向量库 | `search_library` | LDR FAISS | 当前会话上传的文档（隔离） |
| LightRAG | `search_lightrag` | LightRAG `/query` | 全局基础文档库（共享） |

**关键约束**：LightRAG 必须是**独立服务**（非进程内 import），因为方案要求保留其自带 WebUI 的图探索器与基础文档上传；进程内嵌入会失去该 UI。

## 5. 关键设计决策（ADR）

| # | 决策 | 结论 | 理由 |
|---|---|---|---|
| D1 | 集成拓扑 | open-webui 前端 + LDR 主后端 + LightRAG 独立基础库 | 用户明确三项目分工 |
| D2 | 回答与检索方式 | **agent 自由推理**：寒暄/常识直接回答（不检索）；需检索时自主组合 1-3 种，非三选一、非全局开关 | 用户明确「LDR 不一定经过检索才能回答，可直接推理」；LangGraph 的 tool choice 机制天然承载，无需新增判定节点 |
| D3 | FAISS 去向 | **恢复启用**，服务用户上传文档 | 用户明确「LDR 向量检索不禁用」；FAISS 代码完整可复用 |
| D4 | LightRAG 定位 | 全局基础文档库，后端人员在 LightRAG UI 上传，不经 LDR | 用户明确「LightRAG 数据库由后端人员维护并上传作为基础文档」 |
| D5 | 会话级隔离 | 复用 LDR 既有 `Collection` 维度：每个 chat 一个专属 collection，open-webui 透传 chat_id，LDR 建/取 collection 并按之过滤上传与检索 | 用户明确「不同会话只能检索该会话上传的文档」；collection 已是 LDR 原生文档分组维度，改动最小 |
| D6 | 图谱入口 | open-webui 前端直连 LightRAG 图 API | 用户明确「前端直连即可」；LightRAG 图 API 已存在，无需新增 |
| D7 | LightRAG 部署形态 | 独立服务（HTTP） | 保留图 WebUI 与基础文档上传；与现状一致 |

## 6. 文档管理归属（两条数据流）

文档分两类，走**两条互不交叉的数据流**：

| 文档类型 | 上传入口 | 归属 | 检索方式 | 隔离范围 |
|---|---|---|---|---|
| **用户文档** | open-webui 上传组件（调 LDR API，透传 chat_id） | LDR（SQLCipher 原文 + FAISS 索引） | `search_library`（FAISS） | **会话级隔离** |
| **基础文档** | LightRAG 自身 WebUI（后端人员维护） | LightRAG storage | `search_lightrag`（向量+图谱） | 全局共享 |

| 入口 | 结果 |
|---|---|
| open-webui 上传组件 → LDR `/api/v1` upload ✅ | 原文进 SQLCipher + 建 FAISS 索引（按 chat 专属 collection），**主入口** |
| LightRAG WebUI 上传 ✅ | 进 LightRAG 全局基础库，供 `search_lightrag`；**后端人员专用** |
| LDR WebUI 上传（保留兜底） | 原文进 SQLCipher + FAISS，但需显式指定 collection；用户「基本不用」 |
| open-webui 自带 Documents/RAG ❌ | 第三套索引，违背分工 |

**落地约束**：

- LightRAG WebUI 保留「基础文档上传」与「知识图谱探索器」两个用途（后端人员用）。
- open-webui 禁用自带 Documents/RAG；上传组件直接调 LDR 的 `/api/v1` upload 接口并透传 chat_id。
- LDR WebUI 不作为上传主入口（可保留作调试/兜底）。

## 7. 数据流

### 7.1 用户上传文档（→ LDR FAISS，会话隔离）

1. open-webui 上传组件 → LDR `/api/v1` upload（文件/文本，携带用户凭证 + `chat_id`）。
2. LDR 依据 `chat_id` 找到（或新建）该会话的专属 `Collection`（`ChatSession.collection_id` 外键）。
3. LDR 抽文本、存原文到 SQLCipher `Document.text_content`（保留，供引用）。
4. LDR `LibraryRAGService.index_document` 走 **FAISS**（chunk → embed → 写入向量索引），文档关联到该 collection。
5. 返回结果给 open-webui（FAISS 同步，无需 track_status 轮询；见 §9）。

### 7.2 基础文档上传（→ LightRAG，全局共享）

1. 后端人员在 LightRAG WebUI 上传基础文档。
2. LightRAG 异步入 pipeline 建图 + 建向量索引。
3. 全局所有会话经 `search_lightrag` 检索，不受会话隔离约束。

### 7.3 回答与检索（agent 自由推理）

1. 用户在 open-webui 提问 → open-webui 将对话发到 LDR 的 OpenAI 兼容 chat 接口（透传 chat_id）。
2. LDR 的 LangGraph agent 先判断是否需要检索：寒暄 / 自我介绍 / 常识等可直接回答的问题，直接基于自身知识回答，**不调用任何检索工具**。
3. 需要事实 / 时效 / 来源支撑时，agent **自主选择 1-3 个工具**：
   - `search_<engine>`（网搜）→ 公开互联网；
   - `search_library`（FAISS）→ 仅当前 chat 专属 collection 内文档；
   - `search_lightrag`（LightRAG `POST /query`，`mode=mix`）→ 全局基础库。
4. 各工具结果映射回 LDR 的 citation 格式（title/url/snippet + source）。
5. agent 综合多路结果生成回答（可引用多个来源）。

### 7.4 图谱查看（前端直连）

1. open-webui 图谱页直接调用 LightRAG 图 API：`GET /graph/label/list` 取标签列表 → `GET /graphs?label=&max_depth=&max_nodes=` 取子图 → 渲染。
2. 不经 LDR 代理、不经 open-webui 后端转发；认证用 X-API-Key / Bearer JWT（或开放模式）。
3. 图谱为全局基础库的图谱，不按会话过滤（见 §2）。

### 7.5 深度研究

- LDR 的 LangGraph agent 自由推理：寒暄/常识直接回答（不检索）；需检索时照常运行，知识库检索从单一 `search_lightrag` 扩展为「FAISS + LightRAG + 网搜」自主组合；其余 web 搜索引擎不变。open-webui 仅作为其前台。

## 8. 组件与改动点

### 8.1 LightRAG 侧（零改动）

- 仅作为服务被调用；图 API、检索 API、文档插入 API 均已存在。
- 约束：`MAX_TOTAL_TOKENS` 必须 < llama.cpp 的 `n_ctx_slot`（当前 `-c 65536 -np 2` → `n_ctx_slot=32768`）。

### 8.2 LDR 侧（核心改动，均在 `local-deep-research` 仓库）

1. **切回 FAISS**：`library_rag_service.py::LibraryRAGService` 的 `index_document` / `search` 恢复走 FAISS（`VectorIndex` / `FaissVectorStore.search`），`collection_id` 参数真正生效（当前「只保留签名未使用」）。`rag_service_factory.py::get_rag_service` 三个构造点恢复 FAISS 构造（`_build_lightrag_client` 不再恒非 None，或改为不传 lightrag_client 走 FAISS 分支）。
2. **保留 LightRAGRetriever**：`web_search_engines/engines/lightrag_retriever.py::LightRAGRetriever` 独立保留，走 `client.query_data`，注册名 `lightrag`、`is_local=True`——与 FAISS 并存，供 agent 组合调用。
3. **会话级隔离**：`database/models/chat.py::ChatSession` 加 `collection_id` 外键（指向 `Collection`）；上传接口依据透传的 `chat_id` 建/取专属 collection；`LibraryRAGService.search` 按 collection_id 过滤文档。
4. **OpenAI 兼容接口透传 chat_id + 走自由推理 agent**：`openai_compat_routes.py` 接收 open-webui 传来的 chat_id（或 session 标识），路由到对应 chat 的 collection；聊天请求走 `search_strategy="langgraph-agent"`（自由推理，可直接回答或按需检索），而非 `source_based`（强制搜索）。
5. **落地遗留 fix**（Phase 1 SDD final review 发现、fix agent 显示 stopped 需确认）：① `app_factory.py` CSRF 豁免 tuple 加 `"openai_compat"`；② `openai_compat_routes.py` 复用 `_scrub_error_fields` 防信息泄露。
6. 新增 `LightRAGClient` 封装（`POST /documents/text`、`POST /query`、`GET /documents/track_status/{track_id}`）——保留，供 `LightRAGRetriever` 使用。✅ 已完成
7. 注册 `retriever_registry.register("lightrag", ...)` ✅ 已完成；`search_library` 为 LDR 既有工具，恢复 FAISS 后自动可用。
8. 配置：`LDR_LIGHTRAG_BASE_URL`（默认指向现有 LightRAG 服务）。✅ 已完成

### 8.3 open-webui 侧（fork，改动点）

> 端口说明：open-webui 默认端口为 8080，与 llama.cpp 的 8080 冲突；部署时需将 open-webui 改到其它端口。

1. 连接 LDR 的 OpenAI 兼容接口（`http://127.0.0.1:5000/v1`）作为模型后端，所有对话走 LDR；`.env` 配 `OPENAI_API_BASE_URLS` / `OPENAI_API_KEYS`。
2. 新增文档上传组件（Svelte）：文件选择 → 调 LDR `/api/v1` upload → 展示结果；上传时透传当前 chat 的 chat_id。
3. 新增图谱查看页（`src/routes/(app)/graph/+page.svelte`）：前端直连 LightRAG 图 API，渲染标签列表 + 子图；侧边栏 `UserMenu.svelte` 加入口链接。
4. 禁用 open-webui 自带 Documents/RAG（隐藏或移除入口）。

## 9. 语义变化与错误处理

- **检索语义变化**：用户文档检索恢复 FAISS 同步语义（上传后立即可检索）；LightRAG 基础文档仍为异步 pipeline（上传后非立即可检索），但基础文档由后端维护，open-webui 上传线不涉及 track_status。
- **会话隔离强制**：`search_library`（FAISS）必须按当前 chat 的 collection 过滤；若透传的 chat_id 缺失或无法解析，不得回退为「全库检索」，应返回空结果或明确错误（fail-closed）。
- **LightRAG 全局库不受隔离**：`search_lightrag` 对所有会话开放；会话隔离只约束用户上传文档。
- **检索失败不静默兜底**：FAISS 不可用返回明确错误；LightRAG 不可用返回明确错误；两者皆不可用时 agent 仍可走网搜，但不得把「本地检索失败」静默伪装成「无结果」。
- **上下文超限**：`MAX_TOTAL_TOKENS < n_ctx_slot`（见 §8.1）。
- **egress 安全**：`LightRAGRetriever` 与 FAISS `search_library` 均注册 `is_local=True`，被 LDR egress 策略归类为本地，不触发联网限制。

## 10. 安全与隐私

- LDR 原文仍在 SQLCipher 加密库中（保留现有加密/权限体系）。
- **会话级隔离**是本方案新增的关键安全边界：同一用户内不同 chat 的文档互不可见；通过 collection_id 过滤实现，需在检索路径强制校验，防止越权检索。
- LightRAG storage 为本地存储（工作目录），不新增外网依赖。
- open-webui 仅连本地 LDR / LightRAG，无外部数据出口（web 搜索仍由 LDR 既有 egress 策略管控）。
- open-webui → LDR 的 API 调用需认证（凭证不落 open-webui 日志；见 §13 #6）；open-webui → LightRAG 图 API 的认证见 §7.4。

## 11. 测试策略

- **单元测试**（mock）：
  - `LibraryRAGService` 切回 FAISS：index/search 正确走 FAISS，且 `collection_id` 过滤生效（不同 collection 互不可见）。
  - `LightRAGRetriever`：LightRAG 返回 → LangChain `Document` 映射（含 citation 格式），与 FAISS 并存不冲突。
  - 会话隔离：`ChatSession.collection_id` 关联、上传按 chat 建 collection、检索按 collection 过滤。
  - OpenAI 兼容接口：透传 chat_id → 解析到正确 collection。
- **集成测试**：LDR + 真实/mock LightRAG 服务联调：基础文档上传 → `search_lightrag` 检索；用户文档上传 → `search_library` 隔离检索。
- **回归**：LDR 现有 `tests/` 中与 `library_rag_service`、`retriever_registry`、`search_engine_factory`、`chat` 模型相关的用例需更新/通过；CSRF/scrub fix 需补 regression test。

## 12. 分阶段实施

- **Phase 1（核心，已完成）**：LDR 接 LightRAG（client + retriever + 注册）、保留原文；`search_lightrag` 工具可用。
- **Phase 2（本次框架调整，主路径）**：① LDR 切回 FAISS（用户文档检索恢复）；② 会话级隔离（ChatSession.collection_id + 上传/检索按 collection 过滤）；③ OpenAI 兼容接口透传 chat_id；④ 落地 CSRF/scrub fix。
- **Phase 3（open-webui 接入）**：open-webui 连接 LDR 作为模型后端 + 上传组件（透传 chat_id）+ 图谱页（前端直连 LightRAG 图 API）+ 禁用自带 Documents/RAG。

## 13. 风险与开放问题

1. **函数调用依赖**（LDR agent 内部）：需验证 `qwen3.6-35b-a3b` 经 llama.cpp（`--jinja`）是否稳定支持 function calling；若不稳定，agent 的多工具组合检索可能退化。此为 LDR 既有能力。
2. **引用格式映射**：LightRAG 返回的 source 结构与 LDR 现有 citation（title/url/snippet）不完全一致，需一层适配；字段覆盖需验证。
3. **会话隔离的 chat_id 来源**：open-webui 需在聊天与上传请求中一致地透传 chat_id；若两者标识不一致（如聊天用 conversation id、上传用另一个），隔离会失效，需在设计上传组件时统一。
4. **LightRAG 工作目录与多用户**：LightRAG 为单一服务、全局基础库；多用户共享基础文档是预期行为，无需按用户隔离基础库。
5. **open-webui fork 维护成本**：fork 后需跟进上游版本，上传组件与图谱页需适配版本升级；评估是否可改用 open-webui 的 plugin/Pipelines 机制替代 fork。
6. **open-webui → LDR 认证**：`/api/v1` upload 与 OpenAI 兼容接口需携带用户凭证；确定用 API key / 服务账号 / OAuth，且凭证不回写 open-webui 日志。
7. **图谱前端直连的跨域与认证**：LightRAG CORS 默认 `*`，前端直连可行；若启用 `combined_auth`，需在 open-webui 前端妥善持有 X-API-Key / JWT，避免密钥暴露在前端可被任意读取。
