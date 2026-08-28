"""Unit tests for LightRAG wiring in the RAG service factory."""
from local_deep_research.research_library.services import rag_service_factory as factory


def test_build_lightrag_client_from_setting():
    class FakeSettings:
        def get_setting(self, key, default=None):
            if key == "lightrag.base_url":
                return "http://127.0.0.1:9999"
            return default

    client = factory._build_lightrag_client(FakeSettings())
    assert str(client._client.base_url) == "http://127.0.0.1:9999"


def test_build_lightrag_client_default():
    class FakeSettings:
        def get_setting(self, key, default=None):
            return default

    client = factory._build_lightrag_client(FakeSettings())
    assert str(client._client.base_url) == "http://127.0.0.1:9621"
