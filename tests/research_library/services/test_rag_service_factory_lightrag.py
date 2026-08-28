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


def test_register_lightrag_engine_registers_shared_retriever(monkeypatch):
    from local_deep_research.web_search_engines.retriever_registry import (
        retriever_registry,
    )

    class FakeSettings:
        def get_setting(self, key, default=None):
            if key == "lightrag.base_url":
                return "http://127.0.0.1:9999"
            return default

    monkeypatch.setattr(factory, "get_settings_manager", lambda: FakeSettings())
    try:
        factory.register_lightrag_engine()
        retriever = retriever_registry.get("lightrag")
        assert retriever is not None, "lightrag retriever not registered"
        assert str(retriever.client._client.base_url) == "http://127.0.0.1:9999"
    finally:
        retriever_registry.clear()
