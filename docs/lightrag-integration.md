# LightRAG 集成说明（Phase 1）

LDR 的本地知识库检索已从 FAISS 整体切换到 LightRAG（HTTP 服务）。本文记录 Phase 1 的落地状态、配置与语义变化。设计背景见 `docs/superpowers/specs/2026-08-28-ldr-lightrag-openwebui-integration-design.md`。

## 架构

- 新增 `LightRAGClient`（`web_search_engines/engines/lightrag_client.py`）：httpx 封装 `/documents/text`、`/query`、`/query/data`、`/documents/track_status/{track_id}`。
- 新增 `LightRAGRetriever`（`web_search_engines/engines/lightrag_retriever.py`）：`BaseRetriever` 子类，把 `/query/data` 的 chunks 映射成 LangChain `Document`，经 `register_lightrag_retriever()` 挂入 `retriever_registry`。
- 改造 `LibraryRAGService`（`research_library/services/library_rag_service.py`）：`index_document` 改为「读原文 → POST 到 LightRAG」；`search` 改为「`/query/data` → 映射回 `SearchResult`」。
- 接线 `rag_service_factory.py`：读 `lightrag.base_url` 设置构造 `LightRAGClient` 传入 service。

## 配置

- 设置项 `lightrag.base_url`，环境变量覆盖 `LDR_LIGHTRAG_BASE_URL`，默认 `http://127.0.0.1:9621`（LightRAG 服务的 `HOST=127.0.0.1` / `PORT=9621`）。
- LightRAG 的 `MAX_TOTAL_TOKENS` 必须小于 llama.cpp 的 `n_ctx_slot`（当前 `-c 65536 -np 2` → `n_ctx_slot=32768`），否则会出现 `request (N tokens) exceeds the available context size` 类错误。

## 异步语义

LightRAG 插入是异步 pipeline，上传后**非立即**可检索（原 FAISS 为同步）。`index_document` 返回 `track_id`，前端应轮询 `GET /documents/track_status/{track_id}`（状态 `pending → processed`）。

## 内容去重（content_hash）

LightRAG 对插入内容做 hash 去重：内容相同（即使 `file_source` 不同）会被标记为 `DUPLICATE:content_hash`，文档状态置 `FAILED` 且不重复建图。`/documents/text` 另按 `file_source` 判重，重复文件名返回 409。因此重跑联调（如 `scripts/lightrag_smoke.py`）时，内容与文件名都必须每次唯一。

## 错误处理

检索失败**不静默兜底**：LightRAG 不可达或返回错误时抛出 `LightRAGUnavailableError`。`search` 在 service 未配置 `lightrag_client` 时抛出 `RuntimeError`。

## FAISS 路径状态

FAISS / `vector_stores/` / `embeddings/` / `local_search_*` 相关机制**闲置**（保留但不再被 `index_document` / `search` 调用），彻底删除为延后项（spec §8.2.6）。测 FAISS 索引自愈与多模型切换的 `test_index_finalize_self_heal.py`、`test_multi_model_switch_revert.py` 已标记 `skip`。

## 运行时注册（已落地）

`search_lightrag` 工具通过 `register_lightrag_engine()`（`rag_service_factory.py`）在启动时注册到共享命名空间；`web/app.py::main()` 与 `mcp/server.py::run_server()` 均调用，覆盖 web app 进程（LangGraph agent）与 MCP 进程。

## 开放问题（Phase 1 已知限制）

1. `collection_id` 与 `force_reindex` 在 LightRAG 路径下语义弱化：Phase 1 为单 workspace，`collection_id` 保留签名但不分区（spec 开放问题 #4 多用户隔离待定）。
2. **异步处理反馈**：`index_document` 返回 `track_id`，但 LDR 前端「处理中」状态 + `track_status` 轮询尚未落地（spec §9）。
3. **引用格式端到端验证**：LightRAG 返回的 source 结构到 LDR citation 的字段覆盖仅在单测层验证，未做真实服务联调（spec 开放问题 #3）。
4. open-webui 接入（Tools 函数调用、LightRAG WebUI 文档管理 UI 禁用）为 Phase 2。
