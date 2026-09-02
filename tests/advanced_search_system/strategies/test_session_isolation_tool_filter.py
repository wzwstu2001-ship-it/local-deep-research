"""Agent tool-list filtering enforces session isolation (spec §9)."""

from unittest.mock import Mock, patch

from local_deep_research.advanced_search_system.strategies.langgraph_agent_strategy import (
    _load_specialized_engine_tools,
)


def _eligible(*names):
    return {name: {"agent_enabled": True} for name in names}


def test_scoped_session_drops_library_and_other_collections():
    """With _session_collection_id set, only that collection's tool survives;
    the whole-library tool and every other collection_* tool are dropped."""
    eligible = _eligible("library", "collection_col-1", "collection_col-2", "web")
    settings_snapshot = {"_session_collection_id": "col-2"}

    with patch(
        "local_deep_research.web_search_engines.search_engines_config.list_eligible_engine_configs",
        return_value=eligible,
    ):
        tools = _load_specialized_engine_tools(
            skip_engine=None,
            model=Mock(),
            settings_snapshot=settings_snapshot,
            collector=Mock(),
        )

    names = [t.name for t in tools]
    assert "search_collection_col-2" in names
    assert "search_library" not in names
    assert "search_collection_col-1" not in names


def test_unscoped_session_keeps_library():
    """Without _session_collection_id, the library tool remains available."""
    eligible = _eligible("library", "collection_col-1")
    with patch(
        "local_deep_research.web_search_engines.search_engines_config.list_eligible_engine_configs",
        return_value=eligible,
    ):
        tools = _load_specialized_engine_tools(
            skip_engine=None,
            model=Mock(),
            settings_snapshot={},
            collector=Mock(),
        )
    names = [t.name for t in tools]
    assert "search_library" in names
    assert "search_collection_col-1" in names
