"""Tests for migration 0031: Add chat_sessions.collection_id.

Session-scoped document isolation: each chat points at its dedicated
Collection, and retrieval must only search that collection. The column is
nullable (a chat may have no collection yet — ChatService lazily creates one
on first upload/query) and carries an FK to collections.id with ON DELETE
SET NULL (deleting a collection detaches the chat without cascading into it).

Tests cover the fresh-install shape:
- collection_id exists and is nullable
- the FK points at collections.id with ON DELETE SET NULL (exactly one)
- the single-column index ix_chat_sessions_collection_id exists
- downgrade raises NotImplementedError (SQLite ALTER TABLE can't drop an
  indexed FK column cleanly; chat_sessions is an FK target of chat_messages
  and research_history)
"""

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text

from local_deep_research.database.alembic_runner import (
    get_alembic_config,
    run_migrations,
    stamp_database,
)


@pytest.fixture
def fully_migrated_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/migrated_0031.db")
    run_migrations(engine)
    yield engine
    engine.dispose()


def test_collection_id_column_exists(fully_migrated_engine):
    insp = inspect(fully_migrated_engine)
    cols = {c["name"]: c for c in insp.get_columns("chat_sessions")}
    assert "collection_id" in cols
    assert cols["collection_id"]["nullable"] is True


def test_collection_id_fk_points_to_collections_set_null(fully_migrated_engine):
    with fully_migrated_engine.connect() as conn:
        rows = list(
            conn.execute(
                text("PRAGMA foreign_key_list(chat_sessions)")
            ).mappings()
        )
    fks = [
        r
        for r in rows
        if r["from"] == "collection_id" and r["table"] == "collections"
    ]
    assert len(fks) == 1
    assert fks[0]["on_delete"] == "SET NULL"


def test_collection_id_single_column_index_exists(fully_migrated_engine):
    insp = inspect(fully_migrated_engine)
    idx = {
        i["name"]: i["column_names"] for i in insp.get_indexes("chat_sessions")
    }
    assert "ix_chat_sessions_collection_id" in idx
    assert idx["ix_chat_sessions_collection_id"] == ["collection_id"]


def test_downgrade_raises_not_implemented(fully_migrated_engine):
    config = get_alembic_config(fully_migrated_engine)
    with fully_migrated_engine.begin() as conn:
        config.attributes["connection"] = conn
        with pytest.raises(NotImplementedError):
            command.downgrade(config, "0030")


class TestUpgradeFrom0030:
    """0031's real work only runs on the upgrade path.

    A fresh install already carries collection_id via 0001's
    Base.metadata.create_all (the model change predates it), so the
    ``fully_migrated_engine`` fixture exercises only 0031's idempotent
    guards — the ADD COLUMN + batch_alter_table FK path never fires. Hand-
    build a minimal 0030-shape schema and stamp there to exercise that path
    (mirrors 0010's TestExistingDataBackfill).
    """

    @staticmethod
    def _build_0030_schema(engine):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT)"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE chat_sessions ("
                    " id TEXT PRIMARY KEY,"
                    " title TEXT,"
                    " status TEXT NOT NULL DEFAULT 'active',"
                    " message_count INTEGER NOT NULL DEFAULT 0,"
                    " created_at TEXT NOT NULL"
                    ")"
                )
            )
            # chat_messages references chat_sessions, so 0031's
            # batch_alter_table rebuild of chat_sessions exercises the
            # FK-target scenario (chat_sessions is a parent table).
            conn.execute(
                text(
                    "CREATE TABLE chat_messages ("
                    " id TEXT PRIMARY KEY,"
                    " session_id TEXT NOT NULL "
                    "  REFERENCES chat_sessions(id) ON DELETE CASCADE,"
                    " content TEXT NOT NULL"
                    ")"
                )
            )

    def test_upgrade_adds_column_index_and_fk(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path}/upgrade_0031.db")
        try:
            self._build_0030_schema(engine)
            stamp_database(engine, "0030")

            insp = inspect(engine)
            before = {c["name"] for c in insp.get_columns("chat_sessions")}
            assert "collection_id" not in before

            run_migrations(engine, target="0031")

            insp = inspect(engine)
            cols = {c["name"]: c for c in insp.get_columns("chat_sessions")}
            assert "collection_id" in cols
            assert cols["collection_id"]["nullable"] is True

            idx = {
                i["name"]: i["column_names"]
                for i in insp.get_indexes("chat_sessions")
            }
            assert "ix_chat_sessions_collection_id" in idx
            assert idx["ix_chat_sessions_collection_id"] == ["collection_id"]

            with engine.connect() as conn:
                rows = list(
                    conn.execute(
                        text("PRAGMA foreign_key_list(chat_sessions)")
                    ).mappings()
                )
            fks = [
                r
                for r in rows
                if r["from"] == "collection_id" and r["table"] == "collections"
            ]
            assert len(fks) == 1
            assert fks[0]["on_delete"] == "SET NULL"

            # The rebuild preserved chat_messages' own FK back to
            # chat_sessions (data/references survive batch_alter_table).
            with engine.connect() as conn:
                msg_fks = list(
                    conn.execute(
                        text("PRAGMA foreign_key_list(chat_messages)")
                    ).mappings()
                )
            assert any(r["table"] == "chat_sessions" for r in msg_fks)
        finally:
            engine.dispose()
