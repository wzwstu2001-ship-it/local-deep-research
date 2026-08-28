# LDR + LightRAG 集成实现计划（Phase 1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 LDR 的本地知识库检索从 FAISS 整体切换到 LightRAG（HTTP 服务），让 LightRAG 的检索成为 LDR 的 `search_lightrag` 工具，同时保留 LDR 的原文存储（SQLCipher）用于引用溯源。

**Architecture:** LDR 新增两个 HTTP 适配组件——`LightRAGClient`（封装 `/documents/text`、`/query`、`/query/data`、`/documents/scan/status/{track_id}`）与 `LightRAGRetriever(BaseRetriever)`（把 `/query/data` 的 chunks 映射成 LangChain `Document`）。`LibraryRAGService.index_document` 改为「读原文 → POST 到 LightRAG」，`search` 改为「`/query/data` → 映射回 `SearchResult`」。检索工具经 `retriever_registry.register("lightrag", ...)` 挂入既有的 `create_search_engine` → `RetrieverSearchEngine` 链路。FAISS 路径不再被 index/search 调用（闲置，不删除）。

**Tech Stack:** Python（httpx、LangChain `BaseRetriever`、loguru、pytest）、LightRAG FastAPI 服务（HTTP）、SQLAlchemy/SQLCipher（原文存储）。

**Spec:** `docs/superpowers/specs/2026-08-28-ldr-lightrag-openwebui-integration-design.md`

## Global Constraints

- 后端代码、注释、日志消息、commit message 一律英文（AGENTS.md「Code Style」）。计划正文用中文，代码块与 commit 用英文。
- Python：PEP 8、4 空格缩进、类型注解；日志用 LDR 既有 logger（`from loguru import logger` 或 `..security.secure_logging.logger`），不要新引入 `print`。
- 测试用 mock（`httpx.MockTransport`），不依赖真实 LightRAG 服务。
- LightRAG 服务默认地址 `http://127.0.0.1:9621`（`HOST=127.0.0.1`、`PORT=9621`，见 LightRAG `.env`）。
- 检索失败不静默兜底：LightRAG 不可达时抛出 `LightRAGUnavailableError`（spec §9）。
- 改动全部落在 `local-deep-research` 仓库（spec §8.2）。

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/local_deep_research/web_search_engines/engines/lightrag_client.py` | 新增 | `LightRAGClient`：HTTP 封装 + `LightRAGUnavailableError` |
| `src/local_deep_research/web_search_engines/engines/lightrag_retriever.py` | 新增 | `LightRAGRetriever(BaseRetriever)` + `register_lightrag_retriever()` |
| `src/local_deep_research/research_library/services/library_rag_service.py` | 修改 | `index_document` / `search` 改走 LightRAG；`__init__` 增加 `lightrag_client` 注入 |
| `src/local_deep_research/research_library/services/rag_service_factory.py` | 修改 | 读 `lightrag.base_url` 设置，构造 `LightRAGClient` 传入 service |
| `tests/web_search_engines/engines/test_lightrag_client.py` | 新增 | client 单测 |
| `tests/web_search_engines/engines/test_lightrag_retriever.py` | 新增 | retriever + 注册单测 |
| `tests/research_library/services/test_library_rag_service_lightrag.py` | 新增 | service index/search 改造单测 |
| `tests/research_library/services/test_rag_service_factory_lightrag.py` | 新增 | factory 接线单测 |

**测试命令约定**（本计划所有测试命令均在 `local-deep-research` 仓库根目录、项目 venv 内运行）：

```bash
pytest tests/web_search_engines/engines/test_lightrag_client.py -v
```

---

### Task 1: `LightRAGClient`

**Files:**
- Create: `src/local_deep_research/web_search_engines/engines/lightrag_client.py`
- Test: `tests/web_search_engines/engines/test_lightrag_client.py`

**Interfaces:**
- Produces: `LightRAGClient(base_url, timeout, transport)`、`LightRAGUnavailableError`、`DEFAULT_LIGHTRAG_BASE_URL = "http://127.0.0.1:9621"`。
  - `insert_text(text: str, file_source: str | None = None) -> dict`（POST `/documents/text`，返回 `{status, message, track_id}`）
  - `query(query: str, mode: str = "mix", top_k: int = 60) -> dict`（POST `/query`，返回 `{response, references}`）
  - `query_data(query: str, mode: str = "mix", top_k: int = 60) -> dict`（POST `/query/data`，返回 `{status, message, data, metadata}`）
  - `scan_status(track_id: str) -> dict`（GET `/documents/scan/status/{track_id}`）
  - `close() -> None`

- [ ] **Step 1: 写失败测试**

`tests/web_search_engines/engines/test_lightrag_client.py`：

```python
"""Unit tests for LightRAGClient."""
import json

import httpx
import pytest

from local_deep_research.web_search_engines.engines.lightrag_client import (
    DEFAULT_LIGHTRAG_BASE_URL,
    LightRAGClient,
    LightRAGUnavailableError,
)


def _client(handler) -> LightRAGClient:
    return LightRAGClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )


def test_default_base_url():
    assert DEFAULT_LIGHTRAG_BASE_URL == "http://127.0.0.1:9621"


def test_insert_text_posts_payload_and_returns_json():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"status": "success", "message": "ok", "track_id": "t1"}
        )

    result = _client(handler).insert_text("hello world", file_source="doc.txt")
    assert captured["path"] == "/documents/text"
    assert captured["body"] == {"text": "hello world", "file_source": "doc.txt"}
    assert result["track_id"] == "t1"


def test_query_posts_mode_and_top_k():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "answer"})

    result = _client(handler).query("q", mode="mix", top_k=30)
    assert captured["body"] == {"query": "q", "mode": "mix", "top_k": 30}
    assert result["response"] == "answer"


def test_query_data_returns_full_body():
    payload = {
        "status": "success",
        "message": "ok",
        "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
        "metadata": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    assert _client(handler).query_data("q") == payload


def test_scan_status_gets_by_track_id():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json={"track_id": "t1", "status": "completed"})

    result = _client(handler).scan_status("t1")
    assert captured["path"] == "/documents/scan/status/t1"
    assert result["status"] == "completed"


def test_connection_error_raises_lightrag_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LightRAGUnavailableError):
        _client(handler).query("q")


def test_http_error_status_raises_lightrag_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    with pytest.raises(LightRAGUnavailableError):
        _client(handler).query_data("q")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/web_search_engines/engines/test_lightrag_client.py -v`
Expected: 全部 FAIL（`ModuleNotFoundError: ...lightrag_client`）

- [ ] **Step 3: 写最小实现**

`src/local_deep_research/web_search_engines/engines/lightrag_client.py`：

```python
"""HTTP client for the LightRAG service (document insert, query, scan status)."""

from typing import Any, Dict, Optional

import httpx

from ...security.secure_logging import logger

DEFAULT_LIGHTRAG_BASE_URL = "http://127.0.0.1:9621"
DEFAULT_TIMEOUT = 300.0


class LightRAGUnavailableError(RuntimeError):
    """Raised when the LightRAG service cannot be reached or returns an error."""


class LightRAGClient:
    """Thin HTTP wrapper over the LightRAG REST API used by LDR."""

    def __init__(
        self,
        base_url: str = DEFAULT_LIGHTRAG_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = self._client.post(path, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.error(f"LightRAG request {path} failed: {exc}")
            raise LightRAGUnavailableError(
                f"LightRAG request {path} failed: {exc}"
            ) from exc

    def insert_text(
        self, text: str, file_source: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._post("/documents/text", {"text": text, "file_source": file_source})

    def query(
        self, query: str, mode: str = "mix", top_k: int = 60
    ) -> Dict[str, Any]:
        return self._post("/query", {"query": query, "mode": mode, "top_k": top_k})

    def query_data(
        self, query: str, mode: str = "mix", top_k: int = 60
    ) -> Dict[str, Any]:
        return self._post("/query/data", {"query": query, "mode": mode, "top_k": top_k})

    def scan_status(self, track_id: str) -> Dict[str, Any]:
        try:
            resp = self._client.get(f"/documents/scan/status/{track_id}")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.error(f"LightRAG scan_status {track_id} failed: {exc}")
            raise LightRAGUnavailableError(
                f"LightRAG scan_status {track_id} failed: {exc}"
            ) from exc
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/web_search_engines/engines/test_lightrag_client.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/web_search_engines/engines/lightrag_client.py \
        tests/web_search_engines/engines/test_lightrag_client.py
git commit -m "feat(lightrag): add LightRAGClient HTTP wrapper"
```

---

### Task 2: `LightRAGRetriever` 与 `register_lightrag_retriever`

**Files:**
- Create: `src/local_deep_research/web_search_engines/engines/lightrag_retriever.py`
- Test: `tests/web_search_engines/engines/test_lightrag_retriever.py`

**Interfaces:**
- Consumes: `LightRAGClient`（Task 1）、`retriever_registry`（既有）。
- Produces:
  - `LightRAGRetriever(client: LightRAGClient, mode: str = "mix", top_k: int = 60)`，重写 `_get_relevant_documents(query) -> list[Document]`。
  - `register_lightrag_retriever(client: LightRAGClient, name: str = "lightrag", username: str | None = None) -> None`

- [ ] **Step 1: 写失败测试**

`tests/web_search_engines/engines/test_lightrag_retriever.py`：

```python
"""Unit tests for LightRAGRetriever and its registry helper."""
import httpx

from langchain_core.documents import Document

from local_deep_research.web_search_engines.engines.lightrag_client import (
    LightRAGClient,
)
from local_deep_research.web_search_engines.engines.lightrag_retriever import (
    LightRAGRetriever,
    register_lightrag_retriever,
)
from local_deep_research.web_search_engines.retriever_registry import (
    retriever_registry,
)


def _client(payload: dict) -> LightRAGClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return LightRAGClient(base_url="http://test", transport=httpx.MockTransport(handler))


_CHUNKS_PAYLOAD = {
    "status": "success",
    "message": "ok",
    "data": {
        "entities": [],
        "relationships": [],
        "chunks": [
            {
                "content": "chunk one text",
                "file_path": "/documents/a.txt",
                "chunk_id": "chunk-1",
                "reference_id": "1",
            },
            {
                "content": "chunk two text",
                "file_path": "/documents/b.txt",
                "chunk_id": "chunk-2",
                "reference_id": "2",
            },
        ],
        "references": [
            {"reference_id": "1", "file_path": "/documents/a.txt"},
            {"reference_id": "2", "file_path": "/documents/b.txt"},
        ],
    },
    "metadata": {"query_mode": "mix"},
}


def test_maps_chunks_to_documents():
    retriever = LightRAGRetriever(client=_client(_CHUNKS_PAYLOAD))
    docs = retriever._get_relevant_documents("q")
    assert [d.page_content for d in docs] == ["chunk one text", "chunk two text"]
    assert docs[0].metadata["source"] == "/documents/a.txt"
    assert docs[0].metadata["reference_id"] == "1"
    assert docs[0].metadata["title"] == "a.txt"


def test_invoke_returns_documents():
    retriever = LightRAGRetriever(client=_client(_CHUNKS_PAYLOAD))
    docs = retriever.invoke("q")
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)


def test_register_lightrag_retriever_is_local_and_resolvable():
    retriever_registry.clear()
    register_lightrag_retriever(client=_client(_CHUNKS_PAYLOAD))
    assert retriever_registry.get("lightrag") is not None
    assert retriever_registry.get_metadata("lightrag") == {"is_local": True}


def test_register_custom_name():
    retriever_registry.clear()
    register_lightrag_retriever(client=_client(_CHUNKS_PAYLOAD), name="kb")
    assert retriever_registry.get("kb") is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/web_search_engines/engines/test_lightrag_retriever.py -v`
Expected: FAIL（`ModuleNotFoundError: ...lightrag_retriever`）

- [ ] **Step 3: 写最小实现**

`src/local_deep_research/web_search_engines/engines/lightrag_retriever.py`：

```python
"""LangChain BaseRetriever backed by LightRAG /query/data."""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ...security.secure_logging import logger
from ..retriever_registry import retriever_registry
from .lightrag_client import LightRAGClient


class LightRAGRetriever(BaseRetriever):
    """Retrieve document chunks from LightRAG as LangChain Documents."""

    client: LightRAGClient
    mode: str = "mix"
    top_k: int = 60

    def __init__(
        self,
        client: LightRAGClient,
        mode: str = "mix",
        top_k: int = 60,
    ) -> None:
        super().__init__(client=client, mode=mode, top_k=top_k)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        result = self.client.query_data(query, mode=self.mode, top_k=self.top_k)
        if result.get("status") != "success":
            logger.error(f"LightRAG query_data failed: {result.get('message')}")
            return []
        chunks = result.get("data", {}).get("chunks", [])
        return [self._chunk_to_document(chunk) for chunk in chunks]

    @staticmethod
    def _chunk_to_document(chunk: dict) -> Document:
        file_path = chunk.get("file_path", "")
        title = Path(file_path).name if file_path else ""
        return Document(
            page_content=chunk.get("content", ""),
            metadata={
                "source": file_path,
                "title": title,
                "reference_id": chunk.get("reference_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
            },
        )


def register_lightrag_retriever(
    client: LightRAGClient,
    name: str = "lightrag",
    username: Optional[str] = None,
) -> None:
    """Register a LightRAGRetriever in the shared (or per-user) registry."""
    retriever_registry.register(
        name, LightRAGRetriever(client=client), is_local=True, username=username
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/web_search_engines/engines/test_lightrag_retriever.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/web_search_engines/engines/lightrag_retriever.py \
        tests/web_search_engines/engines/test_lightrag_retriever.py
git commit -m "feat(lightrag): add LightRAGRetriever and registry helper"
```

---

### Task 3: `LibraryRAGService.search` 走 LightRAG

**Files:**
- Modify: `src/local_deep_research/research_library/services/library_rag_service.py`（`__init__` 增加 `lightrag_client` 参数；`search` 方法改走 client）
- Test: `tests/research_library/services/test_library_rag_service_lightrag.py`

**Interfaces:**
- Consumes: `LightRAGClient`（Task 1）、`SearchResult`（`vector_stores.facade`，既有）。
- Produces: `LibraryRAGService.__init__(..., lightrag_client: Optional[LightRAGClient] = None)`；`search(query, collection_id, top_k) -> List[SearchResult]`（签名不变，来源改为 LightRAG）。

- [ ] **Step 1: 写失败测试**

`tests/research_library/services/test_library_rag_service_lightrag.py`：

```python
"""Unit tests for the LightRAG-backed search path of LibraryRAGService."""
import httpx

from local_deep_research.web_search_engines.engines.lightrag_client import (
    LightRAGClient,
)
from local_deep_research.research_library.services.library_rag_service import (
    LibraryRAGService,
)


def _client(payload: dict) -> LightRAGClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return LightRAGClient(base_url="http://test", transport=httpx.MockTransport(handler))


def _service_with_client(client: LightRAGClient) -> LibraryRAGService:
    svc = LibraryRAGService.__new__(LibraryRAGService)
    svc.lightrag_client = client
    return svc


def test_search_maps_lightrag_chunks_to_search_results():
    payload = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [
                {
                    "content": "found text",
                    "file_path": "/documents/a.txt",
                    "chunk_id": "chunk-1",
                    "reference_id": "1",
                }
            ],
            "references": [{"reference_id": "1", "file_path": "/documents/a.txt"}],
        },
        "metadata": {},
    }
    svc = _service_with_client(_client(payload))
    results = svc.search("query", "collection-id", top_k=10)
    assert len(results) == 1
    assert results[0].text == "found text"
    assert results[0].metadata["source"] == "/documents/a.txt"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/research_library/services/test_library_rag_service_lightrag.py -v`
Expected: FAIL（`search` 仍走 FAISS，或 `lightrag_client` 属性不存在）

- [ ] **Step 3: 写最小实现**

在 `library_rag_service.py` 顶部 import 区加：

```python
from ...web_search_engines.engines.lightrag_client import LightRAGClient
```

在 `LibraryRAGService.__init__` 签名末尾（`db_password` 之后）加：

```python
        lightrag_client: Optional["LightRAGClient"] = None,
```

在 `__init__` 体内（`self.rag_index_record = None` 附近）加：

```python
        self.lightrag_client = lightrag_client
```

把 `search` 方法体替换为：

```python
    def search(
        self, query: str, collection_id: str, top_k: int
    ) -> List[SearchResult]:
        """Semantic search via LightRAG (replaces FAISS vector search).

        ``collection_id`` is retained for signature compatibility but is not
        used yet: Phase 1 runs a single LightRAG workspace (see spec open
        question #4 on per-user/collection workspace isolation).
        """
        if self.lightrag_client is None:
            raise RuntimeError("LightRAG client is not configured for this service")
        result = self.lightrag_client.query_data(query, mode="mix", top_k=top_k)
        if result.get("status") != "success":
            logger.error(f"LightRAG search failed: {result.get('message')}")
            return []
        chunks = result.get("data", {}).get("chunks", [])
        results = []
        for i, chunk in enumerate(chunks):
            file_path = chunk.get("file_path", "")
            results.append(
                SearchResult(
                    chunk_id=i,
                    text=chunk.get("content", ""),
                    distance=0.0,
                    metric="cosine",
                    metadata={
                        "source": file_path,
                        "reference_id": chunk.get("reference_id", ""),
                        "chunk_id": chunk.get("chunk_id", ""),
                    },
                )
            )
        return results
```

> 说明：`SearchResult.chunk_id` 是 `int`（原 FAISS 内部 id），LightRAG 的 `chunk_id` 是字符串；这里用循环下标占位，真正用于引用的是 `metadata["source"]` 与 `metadata["reference_id"]`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/research_library/services/test_library_rag_service_lightrag.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/research_library/services/library_rag_service.py \
        tests/research_library/services/test_library_rag_service_lightrag.py
git commit -m "feat(lightrag): route LibraryRAGService.search through LightRAG"
```

---

### Task 4: `LibraryRAGService.index_document` 走 LightRAG 插入

**Files:**
- Modify: `src/local_deep_research/research_library/services/library_rag_service.py`（`index_document` 方法）
- Test: `tests/research_library/services/test_library_rag_service_lightrag.py`（追加）

**Interfaces:**
- Consumes: `LightRAGClient.insert_text`（Task 1）、`Document` / `DocumentCollection`（既有 ORM）。
- Produces: `index_document(document_id, collection_id, force_reindex=False) -> Dict[str, Any]`，返回 `{"status": "success", "chunk_count": 0, "track_id": ..., "message": ...}`。

- [ ] **Step 1: 写失败测试**

在 `test_library_rag_service_lightrag.py` 追加（用最小 `__new__` + mock session 隔离）：

```python
from unittest.mock import patch

from local_deep_research.models import Document, DocumentCollection


def test_index_document_posts_text_to_lightrag():
    import json as _json

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _json.loads(request.content)
        return httpx.Response(
            200, json={"status": "success", "message": "ok", "track_id": "t9"}
        )

    client = LightRAGClient(base_url="http://test", transport=httpx.MockTransport(handler))
    svc = LibraryRAGService.__new__(LibraryRAGService)
    svc.lightrag_client = client

    fake_doc = Document(id="doc-1", text_content="full text body", title="My Title")
    doc_collection = DocumentCollection(
        document_id="doc-1", collection_id="coll-1", indexed=False, chunk_count=0
    )

    class _Session:
        def query(self, model):
            return _Query(model)

    class _Query:
        def __init__(self, model):
            self.model = model

        def filter_by(self, **kw):
            if self.model is Document:
                return _Result(fake_doc)
            return _Result(doc_collection)

    class _Result:
        def __init__(self, value):
            self.value = value

        def first(self):
            return self.value

        def update(self, values):
            self.values = values

    session = _Session()

    import local_deep_research.research_library.services.library_rag_service as mod

    with patch.object(mod, "get_user_db_session") as mock_ctx:
        mock_ctx.return_value.__enter__.return_value = session
        mock_ctx.return_value.__exit__.return_value = False
        result = svc.index_document("doc-1", "coll-1")

    assert result["status"] == "success"
    assert result["track_id"] == "t9"
    assert captured["body"] == {"text": "full text body", "file_source": "My Title"}
```

> 说明：若 `get_user_db_session` / ORM 模型的 mock 约定与既有 fixture 不同，执行者应参照 `tests/research_library/services/test_library_rag_service.py` 里现有的 session mock 写法，但**断言对象不变**：LightRAG 收到 `text`、返回 `track_id`、`DocumentCollection.indexed` 被置 True。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/research_library/services/test_library_rag_service_lightrag.py -v`
Expected: 新增用例 FAIL（`index_document` 仍走 chunk→embed→FAISS）

- [ ] **Step 3: 写最小实现**

在 `index_document` 方法**开头**（`_hold_faiss_write_lock` 之前）插入 LightRAG 分支，返回后不再走 FAISS：

```python
    def index_document(
        self, document_id: str, collection_id: str, force_reindex: bool = False
    ) -> Dict[str, Any]:
        # LightRAG path: submit the document text to the LightRAG service and
        # mark it submitted. LightRAG processes asynchronously, so chunk_count
        # is not known at enqueue time (0 = pending). collection_id is kept for
        # signature compatibility; Phase 1 uses a single workspace.
        if self.lightrag_client is not None:
            return self._index_document_via_lightrag(
                document_id, collection_id, force_reindex
            )
        collection_name = f"collection_{collection_id}"
        index_hash = self._get_index_hash(
            collection_name, self.embedding_model, self.embedding_provider
        )
        index_path = self._get_index_path(index_hash)
        with _hold_faiss_write_lock(self.username, str(index_path)):
            return self._index_document_locked(
                document_id, collection_id, force_reindex
            )
```

新增私有方法（放在 `index_document` 附近）：

```python
    def _index_document_via_lightrag(
        self, document_id: str, collection_id: str, force_reindex: bool
    ) -> Dict[str, Any]:
        """Submit one document's text to LightRAG (replaces chunk+embed+FAISS)."""
        with get_user_db_session(self.username, self.db_password) as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                return {"status": "error", "error": "Document not found"}
            if not document.text_content:
                return {"status": "error", "error": "Document has no text content"}
            file_source = (
                document.title
                or document.filename
                or document.original_url
                or f"document_{document_id}"
            )
            ensure_in_collection(session, document_id, collection_id)
            doc_collection = (
                session.query(DocumentCollection)
                .filter_by(document_id=document_id, collection_id=collection_id)
                .first()
            )
            if doc_collection is not None and doc_collection.indexed and not force_reindex:
                return {
                    "status": "skipped",
                    "message": "Document already submitted for this collection",
                    "chunk_count": doc_collection.chunk_count,
                }
            text = document.text_content

        try:
            resp = self.lightrag_client.insert_text(text, file_source=file_source)
        except Exception as exc:
            logger.exception("Error submitting document to LightRAG")
            return {"status": "error", "error": f"Operation failed: {type(exc).__name__}"}

        track_id = resp.get("track_id", "")
        status = resp.get("status", "failure")
        if status not in ("success", "partial_success"):
            return {
                "status": "error",
                "error": f"LightRAG insert failed: {resp.get('message')}",
            }

        with get_user_db_session(self.username, self.db_password) as session:
            session.query(DocumentCollection).filter_by(
                document_id=document_id, collection_id=collection_id
            ).update(
                {
                    "indexed": True,
                    "chunk_count": 0,
                    "last_indexed_at": datetime.now(UTC),
                }
            )
            session.commit()

        logger.info(
            f"Submitted document {document_id} to LightRAG (track_id={track_id})"
        )
        return {
            "status": "success",
            "chunk_count": 0,
            "track_id": track_id,
            "message": "submitted to LightRAG (processing asynchronously)",
        }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/research_library/services/test_library_rag_service_lightrag.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/research_library/services/library_rag_service.py \
        tests/research_library/services/test_library_rag_service_lightrag.py
git commit -m "feat(lightrag): route LibraryRAGService.index_document through LightRAG"
```

---

### Task 5: 工厂接线（`lightrag.base_url` 设置 → `LightRAGClient`）

**Files:**
- Modify: `src/local_deep_research/research_library/services/rag_service_factory.py`
- Test: `tests/research_library/services/test_rag_service_factory_lightrag.py`

**Interfaces:**
- Consumes: `LightRAGClient`（Task 1）、`get_settings_manager` / `settings.get_setting`（既有）。
- Produces: `get_rag_service(...)` 构造的 `LibraryRAGService` 带有 `lightrag_client`（base_url 来自 `lightrag.base_url`，env 覆盖为 `LDR_LIGHTRAG_BASE_URL`）。

- [ ] **Step 1: 写失败测试**

`tests/research_library/services/test_rag_service_factory_lightrag.py`：

```python
"""Unit tests for LightRAG wiring in the RAG service factory."""
from local_deep_research.research_library.services import rag_service_factory as factory


def test_build_lightrag_client_from_setting():
    class FakeSettings:
        def get_setting(self, key, default=None):
            if key == "lightrag.base_url":
                return "http://127.0.0.1:9999"
            return default

    client = factory._build_lightrag_client(FakeSettings())
    assert client._client.base_url == "http://127.0.0.1:9999"


def test_build_lightrag_client_default():
    class FakeSettings:
        def get_setting(self, key, default=None):
            return default

    client = factory._build_lightrag_client(FakeSettings())
    assert client._client.base_url == "http://127.0.0.1:9621"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/research_library/services/test_rag_service_factory_lightrag.py -v`
Expected: FAIL（`_build_lightrag_client` 不存在）

- [ ] **Step 3: 写最小实现**

在 `rag_service_factory.py` 顶部 import 区加：

```python
from ...web_search_engines.engines.lightrag_client import (
    DEFAULT_LIGHTRAG_BASE_URL,
    LightRAGClient,
)
```

新增模块级 helper（放在 `_get_default_text_separators` 之前）：

```python
def _build_lightrag_client(settings) -> LightRAGClient:
    """Build a LightRAGClient from the `lightrag.base_url` setting.

    The env override is `LDR_LIGHTRAG_BASE_URL` (handled by SettingsManager's
    check_env mapping of dotted keys to `LDR_<UPPER_SNAKE>` names).
    """
    base_url = settings.get_setting("lightrag.base_url") or DEFAULT_LIGHTRAG_BASE_URL
    return LightRAGClient(base_url=base_url)
```

在 `get_rag_service` 内，**三处** `LibraryRAGService(...)` 调用各自加 `lightrag_client=_build_lightrag_client(settings)` 参数。例如第一处（collection 已存 settings 分支）：

```python
                return LibraryRAGService(
                    username=username,
                    ...
                    db_password=db_password,
                    lightrag_client=_build_lightrag_client(settings),
                )
```

> 三处构造点都要加同一参数；其余参数保持不变。执行者用 `grep -n "LibraryRAGService(" src/local_deep_research/research_library/services/rag_service_factory.py` 定位全部构造点。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/research_library/services/test_rag_service_factory_lightrag.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/research_library/services/rag_service_factory.py \
        tests/research_library/services/test_rag_service_factory_lightrag.py
git commit -m "feat(lightrag): wire LightRAGClient into RAG service factory"
```

---

### Task 6: 回归 + 文档（FAISS 闲置标注）

**Files:**
- Modify: `src/local_deep_research/research_library/services/library_rag_service.py`（模块 docstring 补注）
- 新增/更新：LDR 仓库集成说明文档（若已有则追加，否则在 `docs/` 下新建 `lightrag-integration.md`）

**Interfaces:**
- 无新接口。验证既有相关测试不回归。

- [ ] **Step 1: 运行受影响的既有测试目录**

Run:

```bash
pytest tests/research_library/services/ -v
pytest tests/web_search_engines/engines/ -v
pytest tests/retriever_integration/ -v
```

Expected: 除 FAISS 专属用例（若被 `index_document`/`search` 改造影响）外应全绿。若有失败，逐条确认是「旧 FAISS 断言不再成立」还是「真回归」；FAISS 专属断言需按新语义更新（如 `search` 不再依赖 `_get_vector_index`）。

- [ ] **Step 2: 标注 FAISS 闲置（代码注释，不删模块）**

在 `library_rag_service.py` 模块 docstring 末尾补一行：

```python
# NOTE: as of LightRAG integration, index/search route through LightRAGClient.
# The FAISS/vector_stores/embeddings machinery (and local_search_* settings) is
# now IDLE — retained but not called by index_document/search. Removal is a
# deferred follow-up (spec §8.2.6 "闲置/移除").
```

- [ ] **Step 3: 记录异步语义与开放问题（文档）**

在集成文档中追加三条（与 spec §9 / §13 一致）：

1. 上传后文档**非立即**可检索；`index_document` 返回 `track_id`，前端应轮询 `GET /documents/scan/status/{track_id}`（状态 `running→completed`）。
2. `collection_id` 与 `force_reindex` 目前在 LightRAG 路径下语义弱化：Phase 1 单 workspace，`collection_id` 保留签名但不分区（spec 开放问题 #4）。
3. `LDR_LIGHTRAG_BASE_URL` 默认 `http://127.0.0.1:9621`；LightRAG 的 `MAX_TOTAL_TOKENS` 必须小于 llama.cpp `n_ctx_slot`（当前 `-c 65536 -np 2` → 32768）。

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(lightrag): mark FAISS idle and record async-status semantics"
```

---

## Self-Review 结论

- **Spec 覆盖**：§8.2.1（client）→ Task 1；§8.2.2（retriever）→ Task 2；§8.2.3（service 改造）→ Task 3+4；§8.2.4（注册）→ Task 2 的 `register_lightrag_retriever`；§8.2.5（配置）→ Task 5；§8.2.6（FAISS 闲置）→ Task 6；§9（异步/失败/上下文）→ Task 6 文档 + Task 1 的 `LightRAGUnavailableError`。§6（文档管理归属 LDR）与 open-webui 接入为部署/配置层面，属 Phase 2 或运维动作，不在本计划代码任务内（与 spec §12 的 Phase 1 边界一致）。
- **未覆盖项（明确延后）**：open-webui Tools 函数调用（phase 2）、LightRAG WebUI 文档上传 UI 禁用（phase 2）、FAISS/`vector_stores`/`embeddings`/`local_search_*` 的彻底删除（spec §8.2.6 允许「闲置」）、`search_lightrag` 工具的应用启动时自动注册（§7.2 运行时接线点需在执行阶段用 `grep -rn "register_multiple\|retriever_registry.register" src/local_deep_research/web/` 定位应用初始化处后加一行 `register_lightrag_retriever(...)`，此处不写死未核实的文件行号）。
- **占位符扫描**：无 TBD/TODO；所有代码块含真实实现。
- **类型一致性**：`LightRAGClient` 的方法名在 Task 1 定义、Task 2/3/4 使用一致；`lightrag_client` 参数名在 Task 3/4/5 一致；`SearchResult` 字段（`chunk_id/text/distance/metric/metadata`）与 `vector_stores/facade.py` 定义一致；`Document` 来自 `langchain_core.documents`，与 `RetrieverSearchEngine` 所用一致。
