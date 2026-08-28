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
