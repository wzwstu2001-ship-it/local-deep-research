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
