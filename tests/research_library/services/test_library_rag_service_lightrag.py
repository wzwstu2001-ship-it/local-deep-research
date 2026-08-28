"""Unit tests for the LightRAG-backed search/index paths of LibraryRAGService."""
import json

import httpx
from unittest.mock import MagicMock

from local_deep_research.database.models.library import Document, DocumentCollection
from local_deep_research.research_library.services.library_rag_service import (
    LibraryRAGService,
)
from local_deep_research.web_search_engines.engines.lightrag_client import (
    LightRAGClient,
)


def _client(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return LightRAGClient(base_url="http://test", transport=httpx.MockTransport(handler))


def _service_with_client(client):
    svc = LibraryRAGService.__new__(LibraryRAGService)
    svc.lightrag_client = client
    svc.username = "test-user"
    svc._db_password = None
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


def test_index_document_posts_text_to_lightrag(mocker):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"status": "success", "message": "ok", "track_id": "t9"}
        )

    client = LightRAGClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    svc = _service_with_client(client)

    fake_doc = MagicMock()
    fake_doc.text_content = "full text body"
    fake_doc.title = "My Title"
    fake_doc.filename = None
    fake_doc.original_url = None

    doc_collection = MagicMock()
    doc_collection.indexed = False

    doc_query = MagicMock()
    doc_query.filter_by.return_value.first.return_value = fake_doc
    coll_query = MagicMock()
    coll_query.filter_by.return_value.first.return_value = doc_collection
    coll_query.filter_by.return_value.update.return_value = None

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.side_effect = lambda model: {
        Document: doc_query,
        DocumentCollection: coll_query,
    }[model]

    mocker.patch(
        "local_deep_research.research_library.services.library_rag_service.get_user_db_session",
        return_value=session,
    )
    mocker.patch(
        "local_deep_research.research_library.services.library_rag_service.ensure_in_collection",
    )

    result = svc.index_document("doc-1", "coll-1")

    assert result["status"] == "success"
    assert result["track_id"] == "t9"
    assert captured["body"] == {"text": "full text body", "file_source": "My Title"}
    coll_query.filter_by.return_value.update.assert_called_once()
