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
        "local_deep_research.research_library.routes.rag_routes._upload_to_collection_impl"
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
    mock_upload.assert_called_once_with("col-abc", "openwebui")
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


def test_chat_upload_open_mode_without_session(app):
    """Open mode: chat_upload must run without a login session."""
    client = app.test_client()
    with patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_svc, patch(
        "local_deep_research.research_library.routes.rag_routes._upload_to_collection_impl"
    ) as mock_upload:
        mock_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        mock_upload.return_value = ({"success": True}, 200)
        resp = client.post(
            "/library/api/collections/chat/upload",
            data={"chat_id": "chat-1", "files": [(BytesIO(b"x"), "a.txt")]},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    mock_svc.return_value.get_or_create_session_collection.assert_called_once_with("chat-1")


def test_capture_request_db_password_reads_g_user_password_open_mode(app):
    """Open mode root cause: with no Flask session, capture_request_db_password
    returns ``g.user_password`` (set by get_user_db_session via the service-user
    env password). This is what lets chat_upload trigger auto-indexing."""
    from flask import g

    from local_deep_research.database.session_passwords import (
        capture_request_db_password,
    )

    with app.test_request_context("/"):
        g.user_password = "openwebui-local-password"
        assert (
            capture_request_db_password("openwebui")
            == "openwebui-local-password"
        )
