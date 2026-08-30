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


def test_track_status_gets_by_track_id():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "track_id": "t1",
                "documents": [{"status": "processed"}],
                "total_count": 1,
                "status_summary": {"processed": 1},
            },
        )

    result = _client(handler).track_status("t1")
    assert captured["path"] == "/documents/track_status/t1"
    assert result["status_summary"] == {"processed": 1}


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
