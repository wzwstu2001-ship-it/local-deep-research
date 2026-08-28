# LDR + LightRAG + open-webui 融合设计

- **日期**：2026-08-28
- **状态**：待评审（Draft for review）
- **作者**：Claude Code（经用户逐项确认）

## 1. 背景与目标

把三个独立项目融合为一套本地优先的 AI 研究与知识库系统：

| 项目 | 融合后的角色 |
|---|---|
| **LDR**（local-deep-research，Flask） | 核心「大脑」：agentic 深度研究、搜索、报告生成、原文存储 |
| **LightRAG**（FastAPI 独立服务） | 知识库引擎：建图 + 多模式检索（取代 LDR 原有 FAISS 向量检索） |
| **open-webui**（Svelte） | 统一聊天前台 + 导航入口（不重写 LDR/LightRAG 的完整 UI） |

**核心目标**：

1. LightRAG 的检索方式成为 LDR 的检索工具之一（`search_lightrag`）。
2. LDR 默认的本地知识库构建方式（FAISS 向量化）整体替换为 LightRAG。
3. open-webui 作为统一聊天前台，保留关键特征（尤以 LightRAG 的知识图谱探索器为要）。
4. LDR 保留文档原文（SQLCipher），用于引用溯源与 UI 展示。

## 2. 非目标（明确不做）

- 不把 LDR 全套 UI 重写进 open-webui。
- 不把知识图谱探索器内嵌进 open-webui（保留 LightRAG 自带 WebUI，通过入口链接访问）。
- 不融合两套数据库：LDR 的 SQLCipher 与 LightRAG 的 storage 各自独立。
- 不保留 LDR 的 FAISS 向量化路径（全局关闭，非并存）。

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
- 文档插入：`POST /documents/text`、`POST /documents/texts`、`POST /documents/upload`；异步 pipeline 处理，`POST /documents/scan` + `GET /documents/scan/status/{track_id}` 跟踪状态。
- 图 API：`GET /graphs?label=...&max_depth=...`、`GET /graph/label/list` 等（`graph_routes.py`）。
- 向量库可插拔（NanoVectorDB 默认，可换 Faiss/Milvus/Qdrant/Postgres/Redis/OpenSearch）。

### 3.3 open-webui

- Svelte 前端 + FastAPI 后端；连 OpenAI 兼容接口做聊天。
- 工具接入：v0.6.0+ 支持 MCP（经 `mcpo` 代理）；也支持自定义 Tools（OpenAPI spec）与 Pipelines。
- 自带 Documents/RAG，但本方案**不使用**（见 §6）。

## 4. 总体架构

```
open-webui（聊天前台，端口 3000）
  ├─ 连接 llama.cpp（127.0.0.1:8080/v1）做日常聊天
  ├─ [phase 2] Tools → LDR /api/v1（深度研究 / 知识库检索，函数调用触发）
  └─ 入口链接 → LDR WebUI（研究/库管理） + LightRAG WebUI（知识图谱）
        │
LDR（Flask，核心，端口 5000）
  ├─ LangGraph agentic 研究
  ├─ 原文存 SQLCipher（Document.text_content）
  ├─ LightRAGClient + LightRAGRetriever（HTTP 适配层）
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
| D4 | UI 集成 | 方案 A（轻集成） | 不追求所有特性上 UI，图探索器保留原 UI |
| D5 | 文档管理归属 | 唯一入口 = LDR WebUI | 避免 split-brain，保住原文/引用链（见 §6） |
| D6 | LightRAG 部署形态 | 独立服务（HTTP） | 保留图 WebUI；与现状一致 |

## 6. 文档管理归属（单一数据流）

**文档上传/管理的唯一入口是 LDR WebUI。**

| 入口 | 结果 |
|---|---|
| LDR WebUI 上传 ✅ | 原文进 SQLCipher + 转交 LightRAG 建图，引用链完整 |
| LightRAG WebUI 上传 ❌ | 仅 LightRAG 存，LDR 无原文 → 引用链断、LDR 不知情 |
| open-webui 自带 Documents/RAG ❌ | 第三套索引，与「检索全走 LightRAG」矛盾 |

**落地约束**：

- LightRAG WebUI 仅保留「知识图谱探索器」用途；其文档上传/管理 UI 需禁用/隐藏，或明确标注「请去 LDR 管理」。
- open-webui 不使用自带 Documents/RAG；文档管理入口链接到 LDR（phase 2 可用 Tools 调 LDR 上传接口）。

## 7. 数据流

### 7.1 上传文档

1. LDR WebUI → `upload_to_collection`（校验、抽文本）。
2. LDR 存原文到 SQLCipher `Document.text_content`（保留，供引用）。
3. LDR 调 LightRAG `POST /documents/text`（text + file_paths），**不再走 FAISS**。
4. LightRAG 异步入 pipeline 建图；LDR 记 `track_id`，前端显示「处理中」并轮询 `scan/status/{track_id}`。

### 7.2 检索

1. 研究 agent 检索 → `LightRAGRetriever` → LightRAG `POST /query`（`mode=mix`）。
2. 结果（entities/relations/chunks）映射回 LDR 的 citation 格式（title/url/snippet + source）。
3. 经 `retriever_registry.register("lightrag", retriever, is_local=True, username=...)` 成为 `search_lightrag` 工具。

### 7.3 深度研究

- 不变：LDR 的 LangGraph agent 照常运行；知识库检索从 FAISS 切换到 `search_lightrag`；其余 web 搜索引擎不变。

## 8. 组件与改动点

### 8.1 LightRAG 侧（几乎零改动）

- 仅作为服务被调用；最多补一条配置记录（工作目录、服务地址、`MAX_TOTAL_TOKENS` 约束）。
- 约束：`MAX_TOTAL_TOKENS` 必须 < llama.cpp 的 `n_ctx_slot`（当前 `-c 65536 -np 2` → `n_ctx_slot=32768`）。

### 8.2 LDR 侧（核心改动，均在 `local-deep-research` 仓库）

1. 新增 `LightRAGClient`：封装 `POST /documents/text`、`POST /query`、状态轮询（httpx）。
2. 新增 `LightRAGRetriever(BaseRetriever)`：内部调用 `LightRAGClient`，把返回映射成 LangChain `Document`（含 source 元数据）。
3. 改造 `research_library/services/library_rag_service.py::LibraryRAGService`：
   - `index_document` → 调 `LightRAGClient` 插入（替换 chunk→embed→FAISS）。
   - `search` → 调 `LightRAGClient` 查询（替换 FAISS 检索）。
4. 注册：`web_search_engines/retriever_registry.py::retriever_registry.register("lightrag", retriever, is_local=True)`，使 LangGraph agent + MCP 可见。
5. 配置：新增 `LDR_LIGHTRAG_BASE_URL`（默认指向现有 LightRAG 服务）；沿用 `settings/manager.py` 的 `LDR_*` env → DB → JSON 默认值机制。
6. 闲置/移除 FAISS 路径：`local_search_*` 配置项、`vector_stores/`、`embeddings/`（embedding 若 LightRAG 独立管理，则 LDR 侧不再需要）。

### 8.3 open-webui 侧（轻量）

> 端口说明：open-webui 默认端口为 8080，与 llama.cpp 的 8080 冲突；部署时需将 open-webui 改到其它端口（如 3000）。

1. 配置 llama.cpp 连接（OpenAI 兼容，`127.0.0.1:8080/v1`）做聊天。
2. 入口链接到 LDR WebUI（研究/库管理）与 LightRAG WebUI（知识图谱）。
3. [phase 2] 定义 Tools 指向 LDR `/api/v1`，让聊天内可触发研究/KB 检索。

## 9. 语义变化与错误处理

- **异步语义**：LightRAG 插入为异步 pipeline，上传后**非立即可检索**（原 FAISS 同步）。LDR 前端需显示「处理中」，用 `track_id` + `scan/status` 轮询。
- **检索失败不静默兜底**：FAISS 已取消；LightRAG 不可用时应返回明确错误，不做静默降级。
- **上下文超限**：`MAX_TOTAL_TOKENS < n_ctx_slot`（见 §8.1），避免再次出现 `24872 > 8192` 类错误。
- **egress 安全**：`LightRAGRetriever` 注册 `is_local=True`，被 LDR egress 策略归类为本地，不触发联网限制。

## 10. 安全与隐私

- LDR 原文仍在 SQLCipher 加密库中（保留现有加密/权限体系）。
- LightRAG storage 为本地存储（工作目录），不新增外网依赖。
- open-webui 仅连本地 llama.cpp / LDR / LightRAG，无外部数据出口（web 搜索仍由 LDR 既有 egress 策略管控）。

## 11. 测试策略

- **单元测试**（mock httpx）：
  - `LightRAGClient`：插入/查询/轮询的请求构造与响应解析。
  - `LightRAGRetriever`：LightRAG 返回 → LangChain `Document` 映射（含 citation 格式）。
  - `LibraryRAGService` 改造：index/search 正确路由到 `LightRAGClient`，不再触碰 FAISS。
- **集成测试**：LDR + 真实 LightRAG 服务（或 mock server）联调：上传 → 等待处理 → 检索 → 引用链完整。
- **回归**：LDR 现有 `tests/` 中与 `library_rag_service`、`retriever_registry`、`search_engine_factory` 相关的用例需更新/通过。

## 12. 分阶段实施

- **Phase 1（核心，本次）**：LDR 关 FAISS、接 LightRAG（client + retriever + service 改造）、文档管理归属 LDR、保留原文；LightRAG 作为服务；open-webui 最小接入（llama.cpp 连接 + 入口链接）。
- **Phase 2（可选增量）**：open-webui Tools 函数调用（研究/KB 检索）；LightRAG WebUI 文档管理 UI 的禁用/隐藏处理；LDR `/api/v1` 的 OpenAPI（或 MCP + mcpo）接入。

## 13. 风险与开放问题

1. **函数调用依赖**（phase 2）：需验证 `qwen3.6-35b-a3b` 经 llama.cpp（`--jinja`）是否稳定支持 function calling；若不稳定，open-webui 的 Tools 触发研究/KB 检索可能不可靠，退化为「入口链接」纯导航模式。
2. **异步处理反馈**：LDR 前端需新增「处理中」状态展示，替代原同步索引体验。
3. **引用格式映射**：LightRAG 返回的 source 结构与 LDR 现有 citation（title/url/snippet）不完全一致，需一层适配；字段覆盖需在 Phase 1 验证。
4. **LightRAG 工作目录与 LDR 用户隔离**：多用户场景下，LDR 按用户隔离，而 LightRAG 为单一服务，需确认工作目录/workspace 隔离策略（当前单用户可暂缓）。
