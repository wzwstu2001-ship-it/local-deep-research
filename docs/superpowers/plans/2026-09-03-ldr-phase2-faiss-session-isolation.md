# LDR Phase 2 — FAISS 切回 + 会话级隔离 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 LDR 用户文档检索从「Phase 1 接入的 LightRAG」切回本地 FAISS，并在检索路径上强制会话级隔离（每个 chat 只能检索该 chat 专属 collection 的文档），同时透传 open-webui 的 chat_id 并落地 Phase 1 遗留的 CSRF/scrub fix。

**Architecture:** 复用 LDR 既有的 `Collection` 维度作为隔离单元：`ChatSession` 新增 `collection_id` 外键指向专属 `Collection`；openai_compat 接口透传 chat_id → 建/取该 chat 的 collection → 注入 `settings_snapshot` 的两个约定 key（`_session_collection_id` + `search.tool`）。检索侧双层 fail-closed：agent 工具列表过滤掉全库 `library` 与其他 `collection_*` 工具（第一道）；`LibraryRAGSearchEngine.search` 读取 `_session_collection_id` 只搜该 collection（第二道）。FAISS 恢复通过把 `LibraryRAGService.search` 的 `lightrag_client is None` 分支改走 `VectorIndex.search`，并移除 factory 三处 `lightrag_client=`。

**Tech Stack:** Python 3.12 / Flask / SQLAlchemy + Alembic / SQLite (SQLCipher) / FAISS (faiss-cpu) / LangGraph agent / pytest。

**Spec:** `docs/superpowers/specs/2026-08-28-ldr-lightrag-openwebui-integration-design.md`（本计划实现其 §8.2 LDR 侧改动点 + §12 Phase 2；§9 定义 fail-closed 语义）。

## Global Constraints

- **会话隔离 fail-closed（spec §9，逐字语义）**：`search_library`（FAISS）必须按当前 chat 的 collection 过滤；若透传的 `chat_id` 缺失或无法解析，不得回退为「全库检索」，应返回空结果或明确错误。
- **检索方式（spec §7.3）**：agent 自主组合 1-3 种检索（`search_*` 网搜 / `search_library` 本地 FAISS / `search_lightrag` LightRAG），不是三选一。Phase 2 不得禁用 FAISS。
- **LightRAG 全局库保留（spec §8.1）**：`search_lightrag` 工具（`LightRAGRetriever`，注册名 `lightrag`）与 `_build_lightrag_client` / `register_lightrag_engine` 必须保留，仅 `LibraryRAGService` 的默认构造不再强制走 LightRAG。
- **commit 规则（CLAUDE.md）**：subject/body 用英文，每条以 `Co-Authored-By: Claude Code <noreply@anthropic.com>` 结尾。
- **测试子集原则**：只运行 mirror 变更模块的 `tests/<dir>`，不跑全量（~7000 tests）。后端用 pytest。

---

### Task 1: 恢复 FAISS 语义搜索

**Files:**
- Modify: `src/local_deep_research/research_library/services/library_rag_service.py:1449-1484`（`search()`）
- Modify: `src/local_deep_research/research_library/services/rag_service_factory.py:300,326,351`（移除三处 `lightrag_client=`）
- Test: `tests/research_library/services/test_library_rag_service.py`
- Test: `tests/research_library/services/test_rag_service_factory.py`

**Interfaces:**
- Consumes: `VectorIndex.search(query: str, top_k: int, session: Optional[Session] = None) -> List[SearchResult]`（`vector_stores/facade.py:431`）；`self.get_current_index_info(collection_id) -> Optional[Dict]`（同文件 1486 行，返回 None 表示无索引）；`self._get_vector_index(collection_id, collection_name, *, reset_stale_state=False) -> VectorIndex`（同文件 1317 行，read path 传 `False` 不创建不 quarantine）。
- Produces: `LibraryRAGService.search(query, collection_id, top_k) -> List[SearchResult]` 在 `lightrag_client is None` 时走 FAISS 且只搜 `collection_id`；`lightrag_client is not None` 时行为不变（Phase 1 LightRAG 分支）。
- 后置任务依赖：Task 4 的 `LibraryRAGSearchEngine.search` 和 `CollectionSearchEngine.search` 都直接构造**不带 `lightrag_client`** 的 `LibraryRAGService` 并调 `.search(query, collection_id, limit)`，因此本任务是它们能重新工作的前置条件。

- [ ] **Step 1: 写失败测试**

在 `tests/research_library/services/test_library_rag_service.py` 末尾新增两个测试：

```python
def test_search_routes_to_faiss_when_no_lightrag_client():
    """Without a LightRAG client, search() must run FAISS against THIS
    collection only — not raise, and not fall back to a broader search."""
    from local_deep_research.research_library.services.library_rag_service import (
        LibraryRAGService,
    )
    from local_deep_research.vector_stores.facade import SearchResult

    svc = LibraryRAGService(
        username="testuser",
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence_transformers",
    )
    assert svc.lightrag_client is None

    fake_index = Mock()
    fake_index.search.return_value = [
        SearchResult(
            chunk_id=1,
            text="the matched chunk text",
            distance=0.1,
            metric="cosine",
            metadata={"source": "doc.pdf"},
            document_title="doc.pdf",
            source_id="doc-1",
            source_type="document",
        )
    ]
    with patch.object(svc, "get_current_index_info", return_value={"embedding_model": "m"}):
        with patch.object(svc, "_get_vector_index", return_value=fake_index) as mock_gvi:
            results = svc.search("query", "col-abc", top_k=5)

    # FAISS branch must scope to the exact collection, on the read path.
    mock_gvi.assert_called_once_with(
        "col-abc", "collection_col-abc", reset_stale_state=False
    )
    assert len(results) == 1
    assert results[0].text == "the matched chunk text"


def test_search_fails_closed_when_collection_has_no_index():
    """A collection with no FAISS index yields empty results — never a
    whole-library fallback."""
    from local_deep_research.research_library.services.library_rag_service import (
        LibraryRAGService,
    )

    svc = LibraryRAGService(
        username="testuser",
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence_transformers",
    )
    with patch.object(svc, "get_current_index_info", return_value=None):
        assert svc.search("query", "col-abc", top_k=5) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/research_library/services/test_library_rag_service.py::test_search_routes_to_faiss_when_no_lightrag_client tests/research_library/services/test_library_rag_service.py::test_search_fails_closed_when_collection_has_no_index -v`

Expected: 第一个测试 FAIL —— `RuntimeError: LightRAG client is not configured for this service`（当前 `search()` 在 `lightrag_client is None` 时直接 raise）。

- [ ] **Step 3: 恢复 `search()` 的 FAISS 分支**

把 [library_rag_service.py:1449-1484](src/local_deep_research/research_library/services/library_rag_service.py#L1449-L1484) 的 `search()` 替换为（保留原 LightRAG 映射逻辑不变，只改入口条件并追加 FAISS 分支）：

```python
    def search(
        self, query: str, collection_id: str, top_k: int
    ) -> List[SearchResult]:
        """Semantic search over this collection.

        Two backends coexist:

        * ``lightrag_client is not None`` (Phase 1 opt-in) routes to LightRAG's
          ``query_data`` and maps its chunks onto ``SearchResult``. Kept for the
          ``search_lightrag`` retriever path, which sets the client explicitly.
        * otherwise (the default) runs FAISS against THIS collection's index
          only — ``collection_id`` is authoritative and there is no whole-library
          fallback (fail-closed, spec §9).
        """
        if self.lightrag_client is not None:
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
                        document_title=Path(file_path).name if file_path else None,
                        source_id=chunk.get("reference_id", "") or None,
                        source_type="document",
                    )
                )
            return results

        # FAISS path. Fail closed: no index for this collection → no results.
        if self.get_current_index_info(collection_id) is None:
            return []
        collection_name = f"collection_{collection_id}"
        vector_index = self._get_vector_index(
            collection_id, collection_name, reset_stale_state=False
        )
        return vector_index.search(query, top_k)
```

- [ ] **Step 4: 移除 factory 三处 `lightrag_client=`**

在 [rag_service_factory.py](src/local_deep_research/research_library/services/rag_service_factory.py) 删除第 300、326、351 行各一处 `lightrag_client=_build_lightrag_client(settings),`（连同其上一行末尾的逗号衔接保持语法正确 —— 第 300/326/351 行都是某次 `LibraryRAGService(...)` 构造的最后一个参数）。构造后 `lightrag_client` 落到 `__init__` 默认值 `None`，从而走 FAISS。

> 保留 `_build_lightrag_client`（29 行）与 `register_lightrag_engine`（39 行）—— 它们服务 `LightRAGRetriever` / `search_lightrag`，与 FAISS 并存。

- [ ] **Step 5: 新增 factory 测试**

在 `tests/research_library/services/test_rag_service_factory.py` 末尾新增：

```python
def test_get_rag_service_returns_faiss_service_by_default(
    patch_settings, mock_db_session
):
    """get_rag_service() must NOT wire a LightRAG client by default — the
    returned service routes search() through FAISS."""
    svc = get_rag_service(settings=patch_settings)
    assert svc.lightrag_client is None
```

> 若现有 `get_rag_service` 签名/调用方式与本测试夹具不符（例如还需要 collection_id 参数或额外的 session patch），以文件内既有 `get_rag_service` 调用样例为准，仅调整夹具参数，不改变断言（`svc.lightrag_client is None`）。

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/research_library/services/test_library_rag_service.py tests/research_library/services/test_rag_service_factory.py tests/research_library/services/test_library_rag_service_lightrag.py tests/research_library/services/test_rag_service_factory_lightrag.py -v`

Expected: 全部 PASS。尤其 `test_library_rag_service_lightrag.py` 的 search 映射测试（手动 `svc.lightrag_client = client`）仍走 LightRAG 分支、继续通过。

- [ ] **Step 7: Commit**

```bash
git add src/local_deep_research/research_library/services/library_rag_service.py \
        src/local_deep_research/research_library/services/rag_service_factory.py \
        tests/research_library/services/test_library_rag_service.py \
        tests/research_library/services/test_rag_service_factory.py
git commit -m "feat(library): restore FAISS search path for library rag service

Default LibraryRAGService.search() now runs FAISS vector search scoped to
the requested collection instead of raising when no LightRAG client is set.
rag_service_factory stops wiring a LightRAG client by default; the
search_lightrag retriever keeps building its client independently.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: `ChatSession.collection_id` 字段 + migration 0031

**Files:**
- Modify: `src/local_deep_research/database/models/chat.py:99-100`（`ChatSession` 加列）
- Create: `src/local_deep_research/database/migrations/versions/0031_add_chat_session_collection_id.py`
- Test: `tests/database/test_migration_0031.py`
- Modify: `tests/database/test_alembic_migrations.py`（把 `"0031"` 加入 `NON_REVERSIBLE_REVISIONS`，因为本 migration 的 downgrade 是 `NotImplementedError`）

**Interfaces:**
- Consumes: `Column(String(36), ForeignKey("collections.id", ondelete="SET NULL"))` 所需的 `ForeignKey` 已在 [chat.py:18](src/local_deep_research/database/models/chat.py#L18) import。
- Produces: ORM 字段 `ChatSession.collection_id: Optional[str]`；DB 列 `chat_sessions.collection_id`（nullable，FK → `collections.id` ON DELETE SET NULL，单列 index `ix_chat_sessions_collection_id`）。
- 后置任务依赖：Task 3 的 `ChatService.get_or_create_session_collection` 读写 `ChatSession.collection_id`；Task 5 透传 chat_id 时调用该 helper。

- [ ] **Step 1: 写失败测试**

新建 `tests/database/test_migration_0031.py`：

```python
"""Tests for migration 0031: Add chat_sessions.collection_id."""

from importlib import import_module

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text

from local_deep_research.database.alembic_runner import (
    get_alembic_config,
    run_migrations,
)


@pytest.fixture
def fully_migrated_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/migrated_0031.db")
    run_migrations(engine)
    yield engine
    engine.dispose()


def test_collection_id_column_exists(fully_migrated_engine):
    insp = inspect(fully_migrated_engine)
    cols = {c["name"]: c for c in insp.get_columns("chat_sessions")}
    assert "collection_id" in cols
    assert cols["collection_id"]["nullable"] is True


def test_collection_id_fk_points_to_collections_set_null(fully_migrated_engine):
    with fully_migrated_engine.connect() as conn:
        rows = list(
            conn.execute(
                text("PRAGMA foreign_key_list(chat_sessions)")
            ).mappings()
        )
    fks = [
        r
        for r in rows
        if r["from"] == "collection_id" and r["table"] == "collections"
    ]
    assert len(fks) == 1
    assert fks[0]["on_delete"] == "SET NULL"


def test_collection_id_single_column_index_exists(fully_migrated_engine):
    insp = inspect(fully_migrated_engine)
    idx = {i["name"]: i["column_names"] for i in insp.get_indexes("chat_sessions")}
    assert "ix_chat_sessions_collection_id" in idx
    assert idx["ix_chat_sessions_collection_id"] == ["collection_id"]


def test_downgrade_raises_not_implemented(fully_migrated_engine):
    config = get_alembic_config(fully_migrated_engine)
    with fully_migrated_engine.begin() as conn:
        config.attributes["connection"] = conn
        with pytest.raises(NotImplementedError):
            command.downgrade(config, "0030")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/database/test_migration_0031.py -v`

Expected: 全部 FAIL —— migration 模块 `0031_add_chat_session_collection_id` 尚不存在（`run_migrations` 到 head 后列不存在）。

- [ ] **Step 3: 给 `ChatSession` 加列**

在 [chat.py:99-100](src/local_deep_research/database/models/chat.py#L99-L100)（`message_count = Column(...)` 之后、`# Relationships` 之前）插入：

```python
    # Session-scoped document isolation: each chat owns a dedicated
    # Collection; retrieval may only search THAT collection. NULL means "no
    # dedicated collection yet" — ChatService lazily creates one on first
    # upload/query. ON DELETE SET NULL detaches the chat from a deleted
    # collection without cascading the delete into the chat itself.
    collection_id = Column(
        String(36),
        ForeignKey("collections.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
```

- [ ] **Step 4: 新建 migration 0031**

创建 `src/local_deep_research/database/migrations/versions/0031_add_chat_session_collection_id.py`：

```python
"""Add chat_sessions.collection_id (FK -> collections.id).

Session-scoped document isolation: each chat points at its dedicated
Collection, and retrieval must only search that collection (spec §9,
fail-closed). The FK is added via batch_alter_table because SQLite's
op.add_column cannot carry an inline ForeignKey ("No support for ALTER of
constraints") — the same pattern as migration 0010's research_history
chat_session_id FK.

Revision ID: 0031
Revises: 0030
Create Date: 2026-09-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy import inspect

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None

_INDEX_NAME = "ix_chat_sessions_collection_id"
_FK_NAME = "fk_chat_sessions_collection_id"


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table(table_name):
        return False
    return column_name in {c["name"] for c in inspector.get_columns(table_name)}


def _index_exists(index_name: str, table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table(table_name):
        return False
    return index_name in {i["name"] for i in inspector.get_indexes(table_name)}


def _fk_exists() -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("chat_sessions"):
        return False
    return any(
        fk.get("name") == _FK_NAME
        for fk in inspector.get_foreign_keys("chat_sessions")
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("collections") or not inspector.has_table(
        "chat_sessions"
    ):
        return

    if not _column_exists("chat_sessions", "collection_id"):
        op.add_column(
            "chat_sessions",
            sa.Column("collection_id", sa.String(36), nullable=True),
        )
        logger.info("0031: added chat_sessions.collection_id")

    if not _index_exists(_INDEX_NAME, "chat_sessions"):
        op.create_index(
            _INDEX_NAME, "chat_sessions", ["collection_id"]
        )
        logger.info("0031: added index %s", _INDEX_NAME)

    if not _fk_exists():
        with op.batch_alter_table("chat_sessions", schema=None) as batch_op:
            batch_op.create_foreign_key(
                _FK_NAME,
                "collections",
                ["collection_id"],
                ["id"],
                ondelete="SET NULL",
            )
        logger.info("0031: added FK chat_sessions.collection_id -> collections.id")


def downgrade() -> None:
    raise NotImplementedError(
        "0031 (chat_sessions.collection_id) is not reversible: SQLite ALTER "
        "TABLE forbids dropping an indexed FK column cleanly, and "
        "batch_alter_table must rebuild chat_sessions — the FK target of "
        "chat_messages.session_id and research_history.chat_session_id. "
        "Recreate the dev database to roll back."
    )
```

- [ ] **Step 5: 把 0031 加入 `NON_REVERSIBLE_REVISIONS`**

在 `tests/database/test_alembic_migrations.py` 找到 `NON_REVERSIBLE_REVISIONS`（含 `"0010"` 的集合），追加 `"0031"`，使 downgrade 阶梯测试跳过本 revision（downgrade 抛 `NotImplementedError`）。

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/database/test_migration_0031.py tests/database/test_alembic_migrations.py -v`

Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add src/local_deep_research/database/models/chat.py \
        src/local_deep_research/database/migrations/versions/0031_add_chat_session_collection_id.py \
        tests/database/test_migration_0031.py \
        tests/database/test_alembic_migrations.py
git commit -m "feat(chat): add chat_sessions.collection_id FK with migration 0031

Adds the per-session collection pointer that session-scoped document
isolation keys off. The FK targets collections.id with ON DELETE SET NULL;
downgrade is intentionally not reversible.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: `ChatService.get_or_create_session_collection`

**Files:**
- Modify: `src/local_deep_research/chat/service.py`（新增方法，`ChatService` 类内，放在 `get_session` 之后）
- Test: `tests/chat/test_chat_service.py`

**Interfaces:**
- Consumes: `ChatSession.collection_id`（Task 2 新增）；`Collection` model（`from ..database.models.library import Collection`）；`get_user_db_session(self.username)`；`ChatSessionNotFound`（同文件已定义）。
- Produces: `ChatService.get_or_create_session_collection(session_id: str) -> str` —— 返回该 chat 专属 collection 的 UUID，无则创建并回写 `ChatSession.collection_id`；`session_id` 无匹配时 raise `ChatSessionNotFound`。
- 后置任务依赖：Task 5 的 `openai_compat_routes.chat_completions` 调用此 helper 解析 chat_id → collection_id。

- [ ] **Step 1: 写失败测试**

在 `tests/chat/test_chat_service.py` 末尾新增（若文件内已有 `ChatService` 构造/`get_user_db_session` mock 的夹具，复用其风格；否则用如下自包含 mock）：

```python
def test_get_or_create_session_collection_creates_then_reuses():
    """First call creates a Collection and binds it to the session; the
    second call returns the same id without creating another."""
    from local_deep_research.chat.service import ChatService

    svc = ChatService("testuser")
    created = []

    class FakeCollection:
        def __init__(self, **kw):
            self.id = kw["id"]
            created.append(self.id)

    # Two ChatSession instances to simulate the before/after state across
    # the two calls (first sees collection_id=None, second sees the id).
    sessions = [
        type("S", (), {"collection_id": None})(),
        type("S", (), {"collection_id": "col-1"})(),
    ]

    class FakeQuery:
        def __init__(self, model):
            self.model = model
        def filter_by(self, **kw):
            return self
        def first(self):
            return sessions.pop(0)

    class FakeDB:
        def __init__(self):
            self._added = []
            self.committed = False
        def query(self, model):
            return FakeQuery(model)
        def add(self, obj):
            self._added.append(obj)
        def flush(self):
            pass
        def commit(self):
            self.committed = True
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    fake_db = FakeDB()

    with patch(
        "local_deep_research.chat.service.get_user_db_session",
        return_value=fake_db,
    ), patch(
        "local_deep_research.chat.service.Collection",
        side_effect=FakeCollection,
    ):
        cid1 = svc.get_or_create_session_collection("s1")
        cid2 = svc.get_or_create_session_collection("s1")

    assert cid1 == "col-1"
    assert cid2 == "col-1"
    assert len(created) == 1  # collection created exactly once
```

> 注意：`ChatService` 模块需从 `..database.models.library` 导入 `Collection`；测试里 patch 的是 `local_deep_research.chat.service.Collection`。若 `test_chat_service.py` 已有更贴近项目习惯的 DB mock 夹具（如 `monkeypatch` 掉 `get_user_db_session` 并返回 SQLite 内存 session），优先沿用该夹具并保留上面的断言意图。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/chat/test_chat_service.py::test_get_or_create_session_collection_creates_then_reuses -v`

Expected: FAIL —— `AttributeError: 'ChatService' object has no attribute 'get_or_create_session_collection'`。

- [ ] **Step 3: 实现 helper**

在 [chat/service.py](src/local_deep_research/chat/service.py) 的 `ChatService` 类内（`get_session` 方法之后）新增：

```python
    def get_or_create_session_collection(self, session_id: str) -> str:
        """Return (creating if needed) the chat's dedicated collection id.

        Session-scoped document isolation (spec §9): a chat owns exactly one
        Collection, and retrieval must only search that collection. The first
        upload/query through a chat lazily creates the collection and stamps
        ``ChatSession.collection_id``; later calls reuse it.

        Raises:
            ChatSessionNotFound: if no row matches ``session_id`` (route layer
                maps to 404 / openai_compat maps to 400).
        """
        from ..database.models.library import Collection

        try:
            with get_user_db_session(self.username) as db:
                chat = db.query(ChatSession).filter_by(id=session_id).first()
                if chat is None:
                    raise ChatSessionNotFound(session_id)

                if chat.collection_id:
                    return chat.collection_id

                collection = Collection(
                    id=str(uuid.uuid4()),
                    name=f"Chat {session_id[:8]}",
                    description="Session-scoped documents",
                    collection_type="user_collection",
                    is_public=False,
                    agent_enabled=True,
                )
                db.add(collection)
                db.flush()
                chat.collection_id = collection.id
                db.commit()
                logger.info(
                    f"Created collection {collection.id[:8]}... for chat "
                    f"{session_id[:8]}... (user {self.username})"
                )
                return collection.id
        except ChatSessionNotFound:
            raise
        except DB_EXCEPTIONS as exc:
            logger.exception("Error resolving chat session collection")
            raise ChatRepositoryError(
                f"DB error resolving collection for session {session_id[:8]}..."
            ) from exc
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/chat/test_chat_service.py -v`

Expected: 新增测试 PASS，文件内既有测试不受影响。

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/chat/service.py tests/chat/test_chat_service.py
git commit -m "feat(chat): add get_or_create_session_collection helper

Lazily creates and caches a chat's dedicated Collection, the isolation unit
that session-scoped retrieval keys off. First call creates; later calls
reuse the stamped ChatSession.collection_id.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: 检索层会话隔离（双层 fail-closed）

**Files:**
- Modify: `src/local_deep_research/web_search_engines/engines/search_engine_library.py:180-183`（`search()` 里 `get_all_collections()` 之后加过滤）
- Modify: `src/local_deep_research/advanced_search_system/strategies/langgraph_agent_strategy.py:476-479`（`_load_specialized_engine_tools` 循环内加过滤）
- Test: `tests/web_search_engines/engines/test_search_engine_library.py`
- Test: `tests/advanced_search_system/strategies/test_session_isolation_tool_filter.py`

**Interfaces:**
- Consumes: `settings_snapshot` dict（`_load_specialized_engine_tools` 的形参；`LibraryRAGSearchEngine.settings_snapshot` 由基类 `search_engine_base.py:363` 置为 `settings_snapshot or {}`）。约定 key：`_session_collection_id`（裸 key，collection UUID，由 Task 5 写入）。
- Produces: 当 `_session_collection_id` 存在时——(a) `LibraryRAGSearchEngine.search` 只搜该 collection，匹配不到则返回 `[]`；(b) `_load_specialized_engine_tools` 过滤掉 `library` 与所有非本 session 的 `collection_*` 工具。
- 后置任务依赖：Task 5 注入 `_session_collection_id` 后，本层过滤即生效。

- [ ] **Step 1: 写失败测试（engine 过滤）**

在 `tests/web_search_engines/engines/test_search_engine_library.py` 追加：

```python
def test_search_filters_to_session_collection_when_scoped():
    """With _session_collection_id set, search() must only query that
    collection — never the whole library (spec §9 fail-closed)."""
    from local_deep_research.web_search_engines.engines.search_engine_library import (
        LibraryRAGSearchEngine,
    )

    settings = {"_username": "testuser", "_session_collection_id": "col-2"}
    all_collections = [
        {"id": "col-1", "name": "other"},
        {"id": "col-2", "name": "mine"},
    ]

    with patch(
        "local_deep_research.web_search_engines.engines.search_engine_library.get_setting_from_snapshot",
        return_value=None,
    ), patch(
        "local_deep_research.web_search_engines.engines.search_engine_library.get_server_url",
        return_value="http://localhost:5000",
    ), patch(
        "local_deep_research.web_search_engines.engines.search_engine_library.LibraryService"
    ) as mock_service:
        mock_service.return_value.get_all_collections.return_value = all_collections

        engine = LibraryRAGSearchEngine(settings_snapshot=settings)
        with patch(
            "local_deep_research.web_search_engines.engines.search_engine_library.get_user_db_session"
        ) as mock_session:
            mock_rag_index = Mock()
            mock_rag_index.embedding_model = "all-MiniLM-L6-v2"
            mock_rag_index.embedding_model_type = Mock(value="sentence_transformers")
            mock_rag_index.chunk_size = 1000
            mock_rag_index.chunk_overlap = 200
            mock_session.return_value.__enter__.return_value.query.return_value.filter_by.return_value.first.return_value = mock_rag_index

            with patch(
                "local_deep_research.web_search_engines.engines.search_engine_library.LibraryRAGService"
            ) as mock_rag_service:
                mock_inst = Mock()
                mock_inst.get_rag_stats.return_value = {"indexed_documents": 0}
                mock_rag_service.return_value.__enter__.return_value = mock_inst
                mock_rag_service.return_value.__exit__.return_value = None

                results = engine.search("test query")

    assert results == []


def test_search_returns_empty_when_session_collection_missing():
    """If _session_collection_id names no known collection, search must NOT
    fall back to the whole library — return []."""
    from local_deep_research.web_search_engines.engines.search_engine_library import (
        LibraryRAGSearchEngine,
    )

    settings = {"_username": "testuser", "_session_collection_id": "col-missing"}
    all_collections = [{"id": "col-1", "name": "other"}]

    with patch(
        "local_deep_research.web_search_engines.engines.search_engine_library.get_setting_from_snapshot",
        return_value=None,
    ), patch(
        "local_deep_research.web_search_engines.engines.search_engine_library.get_server_url",
        return_value="http://localhost:5000",
    ), patch(
        "local_deep_research.web_search_engines.engines.search_engine_library.LibraryService"
    ) as mock_service:
        mock_service.return_value.get_all_collections.return_value = all_collections
        engine = LibraryRAGSearchEngine(settings_snapshot=settings)
        assert engine.search("test query") == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/web_search_engines/engines/test_search_engine_library.py::test_search_filters_to_session_collection_when_scoped tests/web_search_engines/engines/test_search_engine_library.py::test_search_returns_empty_when_session_collection_missing -v`

Expected: 第一个 FAIL（当前会遍历所有 collections，命中 `col-1`）；第二个 FAIL（当前会回退全库）。

- [ ] **Step 3: 给 `LibraryRAGSearchEngine.search` 加过滤**

在 [search_engine_library.py:180-183](src/local_deep_research/web_search_engines/engines/search_engine_library.py#L180-L183) 的 `collections = library_service.get_all_collections()` 之后、`if not collections:` 之前插入：

```python
            # Session isolation (spec §9, fail-closed): when the caller scoped
            # this run to a chat's dedicated collection, search ONLY that
            # collection. A scoped id that matches nothing yields an empty list
            # — never a whole-library fallback.
            session_collection_id = self.settings_snapshot.get(
                "_session_collection_id"
            )
            if session_collection_id:
                collections = [
                    c
                    for c in collections
                    if c.get("id") == session_collection_id
                ]
```

- [ ] **Step 4: 写失败测试（agent 工具过滤）**

新建 `tests/advanced_search_system/strategies/test_session_isolation_tool_filter.py`：

```python
"""Agent tool-list filtering enforces session isolation (spec §9)."""

from unittest.mock import Mock, patch

import pytest

from local_deep_research.advanced_search_system.strategies.langgraph_agent_strategy import (
    _load_specialized_engine_tools,
)


def _eligible(*names):
    return {name: {"agent_enabled": True} for name in names}


def test_scoped_session_drops_library_and_other_collections():
    """With _session_collection_id set, only that collection's tool survives;
    the whole-library tool and every other collection_* tool are dropped."""
    eligible = _eligible("library", "collection_col-1", "collection_col-2", "web")
    settings_snapshot = {"_session_collection_id": "col-2"}

    with patch(
        "local_deep_research.advanced_search_system.strategies.langgraph_agent_strategy.list_eligible_engine_configs",
        return_value=eligible,
    ):
        tools = _load_specialized_engine_tools(
            skip_engine=None,
            model=Mock(),
            settings_snapshot=settings_snapshot,
            collector=Mock(),
        )

    names = [t.name for t in tools]
    assert "search_collection_col-2" in names
    assert "search_library" not in names
    assert "search_collection_col-1" not in names


def test_unscoped_session_keeps_library():
    """Without _session_collection_id, the library tool remains available."""
    eligible = _eligible("library", "collection_col-1")
    with patch(
        "local_deep_research.advanced_search_system.strategies.langgraph_agent_strategy.list_eligible_engine_configs",
        return_value=eligible,
    ):
        tools = _load_specialized_engine_tools(
            skip_engine=None,
            model=Mock(),
            settings_snapshot={},
            collector=Mock(),
        )
    names = [t.name for t in tools]
    assert "search_library" in names
    assert "search_collection_col-1" in names
```

- [ ] **Step 5: 运行测试确认失败**

Run: `pytest tests/advanced_search_system/strategies/test_session_isolation_tool_filter.py -v`

Expected: FAIL —— `search_library` 与 `search_collection_col-1` 仍出现在 tool 列表（当前无过滤）。

- [ ] **Step 6: 给 `_load_specialized_engine_tools` 加过滤**

在 [langgraph_agent_strategy.py:476-479](src/local_deep_research/advanced_search_system/strategies/langgraph_agent_strategy.py#L476-L479) 的循环内、`if name == skip_engine: continue` 之后插入：

```python
            # Session isolation (spec §9, fail-closed): when this run is scoped
            # to a chat's dedicated collection, drop the whole-library tool and
            # every OTHER collection tool. The LLM then never SEES the forbidden
            # names (the core silent-expansion fix), and can only retrieve this
            # chat's documents.
            session_collection_id = (settings_snapshot or {}).get(
                "_session_collection_id"
            )
            if session_collection_id:
                if name == "library":
                    continue
                if name.startswith("collection_") and (
                    name != f"collection_{session_collection_id}"
                ):
                    continue
```

- [ ] **Step 7: 运行测试确认通过**

Run: `pytest tests/web_search_engines/engines/test_search_engine_library.py tests/advanced_search_system/strategies/test_session_isolation_tool_filter.py -v`

Expected: 全部 PASS。

- [ ] **Step 8: Commit**

```bash
git add src/local_deep_research/web_search_engines/engines/search_engine_library.py \
        src/local_deep_research/advanced_search_system/strategies/langgraph_agent_strategy.py \
        tests/web_search_engines/engines/test_search_engine_library.py \
        tests/advanced_search_system/strategies/test_session_isolation_tool_filter.py
git commit -m "feat(search): enforce session-scoped collection isolation

LibraryRAGSearchEngine.search filters to the chat's dedicated collection
when _session_collection_id is set, and the agent tool list drops the
whole-library tool plus every other collection_* tool. Both fail closed so
a scoped run can never fall back to whole-library retrieval.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: openai_compat 透传 chat_id + scrub fix

**Files:**
- Modify: `src/local_deep_research/web/routes/openai_compat_routes.py:16,26-62`（import `_scrub_error_fields`；`chat_completions` 接收 chat_id、注入 scope、scrub）
- Test: `tests/web/test_openai_compat_routes.py`（新增 chat_id 场景 + scrub 场景；更新既有 `test_chat_completions_returns_openai_shape` 以携带 chat_id）

**Interfaces:**
- Consumes: `ChatService(username).get_or_create_session_collection(chat_id) -> str`（Task 3）；`_load_user_context_into_params(params, username)`（[web/api.py:93](src/local_deep_research/web/api.py#L93)，成功后 `params["settings_snapshot"]` 必为 dict）；`_scrub_error_fields(results)`（[web/api.py:41](src/local_deep_research/web/api.py#L41)）。
- Produces: 端点把 `_session_collection_id` 与 `search.tool = "collection_<id>"` 注入 `settings_snapshot`，使 Task 4 的两层过滤与 `_resolve_engine_name`（优先 `search.tool`）生效。chat_id 缺失/不可解析 → 400。
- 后置任务依赖：无（本任务依赖 Task 3 与 Task 4）。

- [ ] **Step 1: 写失败测试**

在 `tests/web/test_openai_compat_routes.py` 追加（并修改既有 `test_chat_completions_returns_openai_shape` 见 Step 2）：

```python
def test_chat_completions_requires_chat_id(client):
    """Missing chat_id fails closed with 400 (spec §9) — never whole-library."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "ldr", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400


def test_chat_completions_injects_session_collection_scope(client):
    """chat_id resolves to a collection and scopes the run to it."""
    def _load(params, username, allow_default_settings=False):
        params["username"] = username
        params["settings_snapshot"] = {"_username": username}
        return None

    with patch(
        "local_deep_research.web.routes.openai_compat_routes._load_user_context_into_params",
        side_effect=_load,
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_chat_svc, patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={"summary": "scoped answer"},
    ) as mock_qs:
        mock_chat_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ldr",
                "chat_id": "chat-1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    _args, kwargs = mock_qs.call_args
    snapshot = kwargs["settings_snapshot"]
    assert snapshot["_session_collection_id"] == "col-abc"
    assert snapshot["search.tool"] == "collection_col-abc"


def test_chat_completions_scrubs_error_summary(client):
    """An Error:-prefixed summary is scrubbed before leaving the boundary."""
    def _load(params, username, allow_default_settings=False):
        params["username"] = username
        params["settings_snapshot"] = {"_username": username}
        return None

    with patch(
        "local_deep_research.web.routes.openai_compat_routes._load_user_context_into_params",
        side_effect=_load,
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_chat_svc, patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={"summary": "Error: internal failure details"},
    ):
        mock_chat_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ldr",
                "chat_id": "chat-1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    content = resp.get_json()["choices"][0]["message"]["content"]
    assert not content.startswith("Error:")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/web/test_openai_compat_routes.py -v`

Expected: 新增三个测试 FAIL；同时既有的 `test_chat_completions_returns_openai_shape` 也会 FAIL（它不传 chat_id，新逻辑返回 400）。

- [ ] **Step 3: 实现 chat_id 透传 + scope + scrub**

修改 [openai_compat_routes.py](src/local_deep_research/web/routes/openai_compat_routes.py)：

(1) 第 16 行 import 增加 `_scrub_error_fields`：

```python
from ..api import (
    _load_user_context_into_params,
    _scrub_error_fields,
    api_access_control,
)
```

(2) 替换 `chat_completions` 的 `try` 块（第 44-56 行）：

```python
    try:
        # Import here to avoid the research-stack import cycle, matching the
        # existing /api/v1/quick_summary handler.
        from ...api.research_functions import quick_summary

        # Session isolation (spec §9, fail-closed): open-webui forwards the
        # conversation id as chat_id (accept a few spellings). Missing chat_id
        # is a 400 — a headless call must never silently search the library.
        chat_id = (
            data.get("chat_id")
            or data.get("session_id")
            or (data.get("metadata") or {}).get("chat_id")
        )
        if not chat_id:
            return jsonify({"error": {"message": "chat_id is required"}}), 400

        username = get_current_username()
        params = {"temperature": data.get("temperature", 0.7)}
        error = _load_user_context_into_params(params, username)
        if error is not None:
            return error

        # Resolve (creating if needed) the chat's dedicated collection and
        # scope retrieval to it. Fail closed: an unresolvable chat_id is a 400.
        from ...chat.service import ChatService

        try:
            collection_id = ChatService(username).get_or_create_session_collection(
                chat_id
            )
        except Exception:
            logger.exception("Failed to resolve chat collection for chat_id")
            return (
                jsonify({"error": {"message": "chat_id could not be resolved"}}),
                400,
            )

        params["settings_snapshot"]["_session_collection_id"] = collection_id
        params["settings_snapshot"]["search.tool"] = f"collection_{collection_id}"

        result = quick_summary(query, **params)
        _scrub_error_fields(result)
        return jsonify(chat_completion_response(result.get("summary", ""), model))
```

- [ ] **Step 4: 更新既有测试以携带 chat_id**

修改 `tests/web/test_openai_compat_routes.py` 的 `test_chat_completions_returns_openai_shape`，给请求体加 `"chat_id": "chat-1"`，并增加对 `ChatService` 的 patch（返回 `"col-abc"`）：

```python
def test_chat_completions_returns_openai_shape(client):
    with patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={"summary": "LightRAG is a graph-based RAG framework."},
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_chat_svc:
        mock_chat_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ldr",
                "chat_id": "chat-1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "ldr"
    assert body["choices"][0]["message"]["content"] == (
        "LightRAG is a graph-based RAG framework."
    )
    assert body["choices"][0]["finish_reason"] == "stop"
```

> 既有的 `test_chat_completions_requires_a_user_message` 与 `test_chat_completions_rejects_missing_messages` 不传 chat_id，但它们分别因「非空列表 / messages 缺失」在 chat_id 检查**之前**就返回 400，因此不受影响。

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/web/test_openai_compat_routes.py -v`

Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/local_deep_research/web/routes/openai_compat_routes.py \
        tests/web/test_openai_compat_routes.py
git commit -m "feat(openai-compat): thread chat_id into session-scoped retrieval

chat_completions now accepts open-webui's chat_id, resolves (creating if
needed) the chat's dedicated collection, and injects _session_collection_id
plus search.tool so the agent and library search stay scoped. Missing or
unresolvable chat_id fails closed with 400. Reuses _scrub_error_fields so
internal error text never leaks (CWE-209).

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: CSRF 豁免 fix（app_factory）

**Files:**
- Modify: `src/local_deep_research/web/app_factory.py:890`（豁免 tuple 加 `"openai_compat"`）
- Test: `tests/web/test_app_factory.py`

**Interfaces:**
- Consumes: `app.extensions["csrf"]`（Flask-WTF `CSRFProtect`，[app_factory.py:271](src/local_deep_research/web/app_factory.py#L271) 已初始化）；blueprint 名 `"openai_compat"`（openai_compat_routes.py 定义）。
- Produces: `openai_compat` blueprint 被 `csrf.exempt`，headless POST（无浏览器 cookie/token）不再被 400 CSRFError 拒绝。

- [ ] **Step 1: 写失败测试**

在 `tests/web/test_app_factory.py` 追加（若该文件用真实 app fixture 则直接断言；否则用如下最小复现）：

```python
def test_openai_compat_blueprint_is_csrf_exempt():
    """openai_compat must be CSRF-exempt — open-webui POSTs have no browser
    cookie/token and would otherwise be rejected with a 400 CSRFError."""
    from local_deep_research.web import app_factory

    app = app_factory.create_app()  # or the project's app-builder entry point
    csrf = app.extensions.get("csrf")
    assert csrf is not None

    # Flask-WTF keeps exempt views keyed by endpoint name.
    endpoint = "openai_compat.chat_completions"
    assert endpoint in getattr(csrf, "_exempt_views", {})
```

> `create_app` 的具体入口名与 `_exempt_views` 内部表示以项目实际为准（可 grep `def create_app` 与 `_exempt_views` 确认）。断言意图不变：`openai_compat` 的 endpoint 出现在 CSRF 豁免集合中。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/web/test_app_factory.py::test_openai_compat_blueprint_is_csrf_exempt -v`

Expected: FAIL —— 豁免集合中无 `openai_compat` 相关 endpoint（当前 tuple 仅 `("api_v1",)`）。

- [ ] **Step 3: 修改豁免 tuple**

在 [app_factory.py:890](src/local_deep_research/web/app_factory.py#L890) 把：

```python
        for bp_name in ("api_v1",):
```

改为：

```python
        for bp_name in ("api_v1", "openai_compat"):
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/web/test_app_factory.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/web/app_factory.py tests/web/test_app_factory.py
git commit -m "fix(web): exempt openai_compat blueprint from CSRF protection

open-webui calls /v1/chat/completions server-to-server without a browser
cookie or CSRF token; the global CSRFProtect was rejecting it with a 400
CSRFError, breaking the integration end-to-end.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: 新增「按 chat_id 上传」入口

> Pre-flight review 补入：spec §8.2 点3b「上传接口依据透传的 chat_id 建/取专属 collection」在原 6 任务中无对应实现，此处补齐（依赖 Task 3）。

**Files:**
- Modify: `src/local_deep_research/research_library/routes/rag_routes.py`（新增 `chat_upload` 路由，紧跟 `upload_to_collection` 之后）
- Test: `tests/research_library/routes/test_rag_routes_chat_upload.py`

**Interfaces:**
- Consumes: `ChatService(username).get_or_create_session_collection(session_id) -> str`（Task 3，抛 `ChatSessionNotFound`）；`upload_to_collection(collection_id)`（既有视图，同文件 1721 行）
- Produces: `POST /library/api/collections/chat/upload`（form: `chat_id` + `files`）→ 解析 chat_id → collection_id → 委托 `upload_to_collection`。chat_id 缺失→400，chat 不存在→404（fail-closed，spec §9）。
- 后置依赖：Phase 3 open-webui 上传组件透传 chat_id 调用此端点。

- [ ] **Step 1: 写失败测试**

新建 `tests/research_library/routes/test_rag_routes_chat_upload.py`（复用 `_route_helpers_rag` 的 `_auth_client` / `_create_app`，同目录 `test_rag_routes_upload_coverage.py` 的认证模式）：

```python
"""Coverage for the chat-scoped upload entry (session isolation, spec §8.2.3)."""

import functools
from io import BytesIO
from unittest.mock import patch

import pytest

from ._route_helpers_rag import (
    _auth_client as _shared_auth_client,
    _create_app,
)

_auth_client = functools.partial(_shared_auth_client, disable_real_limiter=True)


@pytest.fixture
def app():
    return _create_app()


def test_chat_upload_requires_chat_id(app):
    with _auth_client(app) as (client, ctx):
        resp = client.post(
            "/library/api/collections/chat/upload",
            data={"files": [(BytesIO(b"x"), "a.txt")]},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 400


def test_chat_upload_resolves_collection_and_delegates(app):
    with _auth_client(app) as (client, ctx), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_svc, patch(
        "local_deep_research.research_library.routes.rag_routes.upload_to_collection"
    ) as mock_upload:
        mock_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        mock_upload.return_value = ({"success": True}, 200)

        resp = client.post(
            "/library/api/collections/chat/upload",
            data={"chat_id": "chat-1", "files": [(BytesIO(b"x"), "a.txt")]},
            content_type="multipart/form-data",
        )

    mock_svc.return_value.get_or_create_session_collection.assert_called_once_with(
        "chat-1"
    )
    mock_upload.assert_called_once_with("col-abc")
    assert resp.status_code == 200


def test_chat_upload_unknown_chat_returns_404(app):
    from local_deep_research.chat.service import ChatSessionNotFound

    with _auth_client(app) as (client, ctx), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_svc:
        mock_svc.return_value.get_or_create_session_collection.side_effect = (
            ChatSessionNotFound("chat-1")
        )
        resp = client.post(
            "/library/api/collections/chat/upload",
            data={"chat_id": "chat-1", "files": [(BytesIO(b"x"), "a.txt")]},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/research_library/routes/test_rag_routes_chat_upload.py -v`
Expected: 全部 FAIL —— `/library/api/collections/chat/upload` 路由不存在（404 Not Found）。

- [ ] **Step 3: 实现路由**

在 [rag_routes.py](src/local_deep_research/research_library/routes/rag_routes.py) 的 `upload_to_collection` 之后新增：

```python
@rag_bp.route(
    "/api/collections/chat/upload", methods=["POST"]
)
@login_required
@upload_rate_limit_user
@upload_rate_limit_ip
def chat_upload():
    """Upload files to the chat's dedicated collection (session isolation).

    open-webui (Phase 3) posts files with a chat_id; LDR resolves (creating
    if needed) that chat's collection and reuses the existing collection
    upload path. Missing or unresolvable chat_id fails closed — a headless
    upload never lands in the whole-library collection (spec §9).
    """
    chat_id = request.form.get("chat_id") or request.args.get("chat_id")
    if not chat_id:
        return jsonify({"success": False, "error": "chat_id is required"}), 400

    # Import here to avoid the research-stack import cycle.
    from ...chat.service import ChatService, ChatSessionNotFound

    try:
        collection_id = ChatService(
            session["username"]
        ).get_or_create_session_collection(chat_id)
    except ChatSessionNotFound:
        return jsonify({"success": False, "error": "chat not found"}), 404

    return upload_to_collection(collection_id)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/research_library/routes/test_rag_routes_chat_upload.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/research_library/routes/rag_routes.py \
        tests/research_library/routes/test_rag_routes_chat_upload.py
git commit -m "feat(library): add chat-scoped upload entry for session isolation

POST /library/api/collections/chat/upload resolves a chat_id to its
dedicated collection (creating it if needed) and delegates to the existing
collection upload path. Missing chat_id is 400, unknown chat 404 — fail
closed so uploads never fall into the whole-library collection.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## 测试运行汇总（执行者快速参照）

只跑 mirror 变更模块的目录，不跑全量：

```bash
pytest tests/research_library/services/test_library_rag_service.py \
      tests/research_library/services/test_rag_service_factory.py \
      tests/research_library/services/test_library_rag_service_lightrag.py \
      tests/research_library/services/test_rag_service_factory_lightrag.py -v   # Task 1

pytest tests/database/test_migration_0031.py tests/database/test_alembic_migrations.py -v  # Task 2

pytest tests/chat/test_chat_service.py -v                                          # Task 3

pytest tests/web_search_engines/engines/test_search_engine_library.py \
      tests/advanced_search_system/strategies/test_session_isolation_tool_filter.py -v  # Task 4

pytest tests/web/test_openai_compat_routes.py -v                                   # Task 5

pytest tests/web/test_app_factory.py -v                                            # Task 6

pytest tests/research_library/routes/test_rag_routes_chat_upload.py -v             # Task 7
```

跨切面回归（里程碑或收尾时跑，非每次）：`pytest tests/web tests/chat tests/database tests/web_search_engines/engines tests/advanced_search_system/strategies -v`。
