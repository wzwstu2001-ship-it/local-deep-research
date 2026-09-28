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
def app():
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(openai_compat_bp)
    return app


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
        "local_deep_research.web.routes.openai_compat_routes.get_openai_compat_username",
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


def test_chat_completions_requires_a_user_message(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "be helpful"}]},
    )
    assert resp.status_code == 400


def test_chat_completions_rejects_missing_messages(client):
    resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 400


def test_chat_completions_requires_chat_id(client):
    """Missing chat_id fails closed with 400 (spec §9) — never whole-library."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "ldr", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400


def test_chat_completions_unresolvable_chat_id_fails_closed(client):
    """An unresolvable chat_id is a 400, never a whole-library fallback."""
    def _load(params, username, allow_default_settings=False):
        params["username"] = username
        params["settings_snapshot"] = {"_username": username}

    with patch(
        "local_deep_research.web.routes.openai_compat_routes._load_user_context_into_params",
        side_effect=_load,
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_chat_svc:
        mock_chat_svc.return_value.get_or_create_session_collection.side_effect = (
            RuntimeError("db down")
        )
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ldr",
                "chat_id": "chat-missing",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 400
    assert "chat_id could not be resolved" in resp.get_json()["error"]["message"]


def test_chat_completions_injects_session_collection_scope(client):
    """chat_id resolves to a collection and scopes the run to it."""
    def _load(params, username, allow_default_settings=False):
        params["username"] = username
        params["settings_snapshot"] = {"_username": username}

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
        mock_chat_svc.return_value.session_collection_has_index.return_value = False
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
    # An UNINDEXED collection must NOT become the primary: that would replace
    # the agent's web search and yield "No sources" for an empty collection.
    assert "search.tool" not in snapshot


def test_chat_completions_promotes_indexed_collection_to_primary(client):
    """An indexed session collection becomes the run's primary (search.tool),
    so the agent's web_search retrieves the uploads instead of relying on the
    LLM to pick a UUID-named collection tool."""
    def _load(params, username, allow_default_settings=False):
        params["username"] = username
        params["settings_snapshot"] = {"_username": username}

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
        mock_chat_svc.return_value.session_collection_has_index.return_value = True
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
    """An Error:-prefixed summary has credential text scrubbed at the boundary."""
    leaked = (
        "Error: LLM call failed: "
        "https://api.example.com/v1?api_key=sk-SECRETKEY1234567890AB"
    )

    def _load(params, username, allow_default_settings=False):
        params["username"] = username
        params["settings_snapshot"] = {"_username": username}

    with patch(
        "local_deep_research.web.routes.openai_compat_routes._load_user_context_into_params",
        side_effect=_load,
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_chat_svc, patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={"summary": leaked},
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
    assert "sk-SECRETKEY1234567890AB" not in content
    assert content.startswith("Error:")


def test_chat_completions_open_mode_without_session(app):
    """Open mode: no Flask session cookie — a headless call must not 401."""
    client = app.test_client()
    with patch(
        "local_deep_research.web.routes.openai_compat_routes._load_user_context_into_params",
        side_effect=lambda p, u, **kw: p.update(username=u, settings_snapshot={}) or None,
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_svc, patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={"summary": "ok"},
    ):
        mock_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "ldr", "chat_id": "chat-1",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200


def test_list_models_returns_ldr(app):
    """GET /v1/models exposes the single 'ldr' model for open-webui's picker."""
    client = app.test_client()
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["ldr"]


def test_chat_completions_stream_returns_sse_with_citations(client):
    """stream=true returns SSE whose delta.annotations carry url_citations.

    open-webui's streaming path extracts ``url_citation`` from
    ``delta.annotations``, so the merged sources (FAISS + LightRAG + searxng)
    render as citation cards. This pins that LDR no longer drops ``sources``.

    The mock summary references both sources with ``[N]`` markers — the
    strict citation filter downstream drops every source the answer body
    never named, so without markers the citation panel would (correctly)
    come back empty.
    """
    with patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={
            "summary": "回答正文参见 [1] 与 [2]",
            "sources": [
                {"title": "设备操作规程.pdf", "url": "/library/document/1"},
                {"title": "外部网页", "link": "https://example.com"},
            ],
        },
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_chat_svc:
        mock_chat_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        mock_chat_svc.return_value.session_collection_has_index.return_value = False
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ldr",
                "chat_id": "chat-1",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.content_type
    body = resp.get_data(as_text=True)
    assert "data: [DONE]" in body
    assert '"url_citation"' in body
    assert "/library/document/1" in body
    assert "https://example.com" in body


def test_chat_completions_stream_with_empty_sources(client):
    """Empty sources still yield a well-formed SSE stream with no annotations."""
    with patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={"summary": "无引用回答", "sources": []},
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_chat_svc:
        mock_chat_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        mock_chat_svc.return_value.session_collection_has_index.return_value = False
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ldr",
                "chat_id": "chat-1",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "data: [DONE]" in body
    assert '"url_citation"' not in body
