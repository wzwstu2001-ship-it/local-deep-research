# LDR OpenAI 兼容 Chat 接口实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 LDR 新增 OpenAI 兼容的 `POST /v1/chat/completions` 接口，让 open-webui 把它当作「模型后端」连接，内部桥接到 LDR 的 `quick_summary`（agentic 研究/检索/报告路径）。

**Architecture:** 这是 Phase 2 三个独立子系统中的第一个（LDR 侧）。open-webui 连接 `http://<ldr-host>:5000/v1` 后，每次对话经本接口把 OpenAI chat 消息折算成一条 `quick_summary` 调用，返回 OpenAI `chat.completion` 格式的响应。认证复用现有 `/api/v1` 的 session 认证栈（`api_access_control` + `api_rate_limit`）；open-webui 的 Bearer-token 认证是 spec §13 #6 的独立开放问题，不在本计划内。MVP 只做非流式响应；`stream=true` 流式输出列为后续。

**Tech Stack:** Flask blueprint、`quick_summary`（`local_deep_research.api.research_functions`）、pytest（`unittest.mock`）。

**Spec:** `docs/superpowers/specs/2026-08-28-ldr-lightrag-openwebui-integration-design.md`（§8.2 第 9 点、§4 架构图）

## Global Constraints

- Python 3.10+（代码使用 `str | None` 联合类型语法）。
- 测试用 venv 解释器：`.venv/Scripts/python.exe -m pytest`（直接用系统 `python` 会 `ModuleNotFoundError: loguru`）。
- 端点完整路径必须是 `/v1/chat/completions`（open-webui 连接时填 base URL `http://127.0.0.1:5000/v1`）。
- 认证复用 `api_access_control`（session 认证，`get_current_username()` 为空 → 401）与 `api_rate_limit`；不新增 Bearer-token 逻辑（spec §13 #6 后续）。
- 代码风格：PEP 8，4 空格缩进，type annotations，`loguru` logger，注释与 commit 用英语。
- 非流式 MVP：请求体 `stream` 字段忽略，始终返回单个完整 `chat.completion`。

---

## Task 1: 纯函数——messages 解析与 OpenAI 响应构造

**Files:**
- Create: `src/local_deep_research/web/openai_compat.py`
- Test: `tests/web/test_openai_compat.py`

**Interfaces:**
- Consumes: 无（纯函数，零依赖）。
- Produces:
  - `last_user_message(messages: list[dict[str, Any]]) -> str | None` —— 返回最后一条非空 `user` 消息的去除首尾空白后的内容，无可用 user 消息时返回 `None`。
  - `chat_completion_response(content: str, model: str) -> dict[str, Any]` —— 构造 OpenAI `chat.completion` 响应 dict。

- [ ] **Step 1: 写失败测试**

创建 `tests/web/test_openai_compat.py`：

```python
"""Unit tests for the OpenAI-compat chat helpers."""

from local_deep_research.web.openai_compat import (
    chat_completion_response,
    last_user_message,
)


def test_last_user_message_extracts_final_user_turn():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "  what is LightRAG?  "},
    ]
    assert last_user_message(messages) == "what is LightRAG?"


def test_last_user_message_ignores_non_user_roles():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "assistant", "content": "no user here"},
    ]
    assert last_user_message(messages) is None


def test_last_user_message_skips_blank_user_turns():
    messages = [
        {"role": "user", "content": "   "},
        {"role": "user", "content": "actual question"},
    ]
    assert last_user_message(messages) == "actual question"


def test_chat_completion_response_shape():
    resp = chat_completion_response("an answer", model="ldr")
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "ldr"
    choices = resp["choices"]
    assert len(choices) == 1
    assert choices[0]["message"] == {"role": "assistant", "content": "an answer"}
    assert choices[0]["finish_reason"] == "stop"
    assert "usage" in resp
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/web/test_openai_compat.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'local_deep_research.web.openai_compat'`

- [ ] **Step 3: 写最小实现**

创建 `src/local_deep_research/web/openai_compat.py`：

```python
"""Helpers for the OpenAI-compatible chat endpoint.

Open-webui connects to LDR as an OpenAI-compatible "model backend", so these
helpers translate between the OpenAI chat wire format and LDR's internal
question-answering entry point (``quick_summary``).
"""

from __future__ import annotations

import time
import uuid
from typing import Any


def last_user_message(messages: list[dict[str, Any]]) -> str | None:
    """Return the trimmed content of the final non-empty ``user`` message.

    Returns ``None`` when the conversation carries no usable user turn.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def chat_completion_response(content: str, model: str) -> dict[str, Any]:
    """Build an OpenAI ``chat.completion`` response body for ``content``."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/web/test_openai_compat.py -v`
Expected: PASS，4 passed

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/web/openai_compat.py tests/web/test_openai_compat.py
git commit -m "feat(openai-compat): add chat request/response helpers"
```

---

## Task 2: blueprint + `/v1/chat/completions` 端点

**Files:**
- Create: `src/local_deep_research/web/routes/openai_compat_routes.py`
- Test: `tests/web/test_openai_compat_routes.py`

**Interfaces:**
- Consumes: Task 1 的 `last_user_message`、`chat_completion_response`；`web.api` 的 `api_access_control`、`_load_user_context_into_params`；`security.decorators.require_json_body`；`security.rate_limiter.api_rate_limit`、`get_current_username`；`api.research_functions.quick_summary`（函数内局部 import）。
- Produces: `openai_compat_bp = Blueprint("openai_compat", __name__, url_prefix="/v1")`，路由 `POST /chat/completions`。

- [ ] **Step 1: 写失败测试**

创建 `tests/web/test_openai_compat_routes.py`：

```python
"""Tests for the OpenAI-compatible chat endpoint (minimal Flask app)."""

from contextlib import contextmanager
from unittest.mock import patch

import flask
import pytest

from local_deep_research.web.routes.openai_compat_routes import openai_compat_bp


@contextmanager
def _fake_db(username):
    yield None


@pytest.fixture
def client():
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(openai_compat_bp)
    # api_access_control (defined in web/api.py) calls web.api's own
    # get_current_username / get_user_db_session; the endpoint body calls the
    # module-level names imported into this routes module.
    with patch(
        "local_deep_research.web.api.get_current_username", return_value="testuser"
    ), patch(
        "local_deep_research.web.api.get_user_db_session", side_effect=_fake_db
    ), patch(
        "local_deep_research.web.routes.openai_compat_routes.get_current_username",
        return_value="testuser",
    ), patch(
        "local_deep_research.web.routes.openai_compat_routes._load_user_context_into_params",
        return_value=None,
    ):
        yield app.test_client()


def test_chat_completions_returns_openai_shape(client):
    with patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={"summary": "LightRAG is a graph-based RAG framework."},
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "ldr", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "ldr"
    assert body["choices"][0]["message"]["content"] == (
        "LightRAG is a graph-based RAG framework."
    )
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_requires_a_user_message(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "be helpful"}]},
    )
    assert resp.status_code == 400


def test_chat_completions_rejects_missing_messages(client):
    resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 400
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/web/test_openai_compat_routes.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'local_deep_research.web.routes.openai_compat_routes'`

- [ ] **Step 3: 写最小实现**

创建 `src/local_deep_research/web/routes/openai_compat_routes.py`：

```python
"""OpenAI-compatible chat endpoint for open-webui.

Open-webui connects to LDR as an OpenAI-compatible model backend. This
blueprint exposes ``/v1/chat/completions``, which folds an OpenAI chat
request into a single LDR ``quick_summary`` call (the agentic
research/retrieval path) and returns an OpenAI ``chat.completion`` body.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from loguru import logger

from ...security.decorators import require_json_body
from ...security.rate_limiter import api_rate_limit, get_current_username
from ..api import _load_user_context_into_params, api_access_control
from ..openai_compat import chat_completion_response, last_user_message

openai_compat_bp = Blueprint("openai_compat", __name__, url_prefix="/v1")


@openai_compat_bp.route("/chat/completions", methods=["POST"])
@api_access_control
@api_rate_limit
@require_json_body(error_message="messages are required")
def chat_completions():
    data = request.json or {}
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return (
            jsonify({"error": {"message": "messages must be a non-empty list"}}),
            400,
        )

    query = last_user_message(messages)
    if query is None:
        return (
            jsonify({"error": {"message": "no non-empty user message found"}}),
            400,
        )

    model = data.get("model") or "ldr"

    try:
        # Import here to avoid the research-stack import cycle, matching the
        # existing /api/v1/quick_summary handler.
        from ...api.research_functions import quick_summary

        username = get_current_username()
        params = {"temperature": data.get("temperature", 0.7)}
        error = _load_user_context_into_params(params, username)
        if error is not None:
            return error

        result = quick_summary(query, **params)
        return jsonify(chat_completion_response(result.get("summary", ""), model))
    except TimeoutError:
        logger.exception("OpenAI-compat chat request timed out")
        return jsonify({"error": {"message": "request timed out"}}), 504
    except Exception:
        logger.exception("Error in OpenAI-compat chat_completions")
        return jsonify({"error": {"message": "internal error"}}), 500
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/web/test_openai_compat_routes.py -v`
Expected: PASS，3 passed

- [ ] **Step 5: Commit**

```bash
git add src/local_deep_research/web/routes/openai_compat_routes.py tests/web/test_openai_compat_routes.py
git commit -m "feat(openai-compat): add /v1/chat/completions endpoint"
```

---

## Task 3: 在 app_factory 注册 blueprint

**Files:**
- Modify: `src/local_deep_research/web/app_factory.py`（在「Register API v1 blueprint」附近加注册）
- Modify: `src/local_deep_research/web/routes/route_registry.py`（文档同步，可选但建议）

**Interfaces:**
- Consumes: Task 2 的 `openai_compat_bp`。
- Produces: 完整 app 挂载 `/v1/chat/completions` 路由。

- [ ] **Step 1: 修改 app_factory.py 注册 blueprint**

在 `src/local_deep_research/web/app_factory.py` 的「Register API v1 blueprint」块（`app.register_blueprint(api_blueprint)` 之后）追加：

```python
    # Register OpenAI-compatible chat blueprint (open-webui model backend)
    from .routes.openai_compat_routes import openai_compat_bp

    app.register_blueprint(openai_compat_bp)  # Already has url_prefix='/v1'
    logger.info("OpenAI-compatible chat routes registered successfully")
```

- [ ] **Step 2: 同步 route_registry.py 文档**

在 `src/local_deep_research/web/routes/route_registry.py` 的 `ROUTE_REGISTRY` dict 末尾追加一个条目（与现有条目同格式）：

```python
    "openai_compat": {
        "blueprint": "openai_compat_bp",
        "url_prefix": "/v1",
        "routes": [
            (
                "POST",
                "/chat/completions",
                "chat_completions",
                "OpenAI-compatible chat (open-webui model backend)",
            ),
        ],
    },
```

- [ ] **Step 3: 验证**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/web/test_openai_compat.py tests/web/test_openai_compat_routes.py -v
ruff check src/local_deep_research/web/openai_compat.py src/local_deep_research/web/routes/openai_compat_routes.py
```
Expected: 7 passed，ruff 无报错。（完整 app 的路由挂载用 `flask routes` 或 `curl http://127.0.0.1:5000/v1/chat/completions` 手动验证一次。）

- [ ] **Step 4: Commit**

```bash
git add src/local_deep_research/web/app_factory.py src/local_deep_research/web/routes/route_registry.py
git commit -m "feat(openai-compat): register /v1 blueprint in app factory"
```

---

## Self-Review

**1. Spec coverage:** spec §8.2 第 9 点（LDR 暴露 OpenAI 兼容 chat 接口）由 Task 1–3 完整覆盖。spec §8.2 第 7 点（`/api/v1` upload + track_status）与 open-webui fork、LightRAG WebUI 禁用**不在本计划**——它们是 Phase 2 的另外两个子系统，需各自独立 plan（open-webui 尚未 clone）。spec §13 #6（open-webui → LDR 认证，Bearer token）标记为后续，本计划复用现有 session 认证。

**2. Placeholder scan:** 无 TBD/TODO；所有测试与实现均有真实代码。Task 3 因是纯接线（一行注册 + 文档），未写「失败测试」，验证用 lint + 回归 + 手动路由检查替代——这是 writing-plans 允许的「把配置/脚手架 fold 进其交付物所在 task」。

**3. Type consistency:** `last_user_message` 返回 `str | None`，`chat_completion_response` 返回 `dict[str, Any]`，在 Task 1 定义、Task 2 消费，签名一致。`openai_compat_bp` 在 Task 2 定义、Task 3 消费，名称一致。patch 目标路径（`web.api.get_current_username` vs `openai_compat_routes.get_current_username`）已分别区分 `api_access_control` 内部调用与 endpoint 正文调用。

## 遗留（本计划之外）

- `stream=true` 流式响应（SSE）。
- Bearer-token 认证（spec §13 #6）。
- `model` 字段当前只回显，不实际切换底层模型（桥接 `quick_summary`，由用户 settings 决定）。
