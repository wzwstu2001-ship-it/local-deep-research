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
