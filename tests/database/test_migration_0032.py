"""Tests for migration 0032: Add documents.original_filename.

Closes the gap where ``Document.original_filename`` was declared on the
ORM model (``database/models/library.py``) but never added to any
schema migration, so SQLAlchemy silently dropped the kwarg on insert
against any pre-existing install — leaving the indexer to fall through
the ``document.title or document.original_filename or document.filename``
chain to the sanitized ASCII ``filename`` and ship a Werkzeug-stripped
title to every citation.

Tests cover:
- Fresh-install path: column exists on ``documents``, VARCHAR(500),
  nullable.
- Pre-0032 legacy DB (column missing): upgrade adds the column and
  leaves existing rows' NULL value intact (no backfill, by design).
- Idempotency: re-running ``upgrade head`` against a freshly-migrated
  DB is a no-op (0032's own ``column_exists`` guard).
- Downgrade drops the column; ``documents.filename`` (sanitized value)
  is untouched, so old citations still render via that fallback chain.
"""

from importlib import import_module

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from local_deep_research.database.alembic_runner import (
    get_alembic_config,
    get_head_revision,
    run_migrations,
)


_MIGRATION_MODULE = (
    "local_deep_research.database.migrations.versions."
    "0032_add_document_original_filename"
)
_TABLE = "documents"
_COLUMN = "original_filename"


def _make_engine():
    """In-memory SQLite shared across connections.

    ``StaticPool`` keeps the connection alive so the multiple
    transactions alembic opens inside ``run_migrations`` see the same
    database state instead of each getting a fresh empty one.
    """
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _has_column(engine, name):
    return name in {
        c["name"] for c in inspect(engine).get_columns(_TABLE)
    }


def _column_type(engine, name):
    for c in inspect(engine).get_columns(_TABLE):
        if c["name"] == name:
            return c
    return None


def test_head_revision_is_0032():
    """Sanity: the migration is in the chain as ``head``."""
    assert get_head_revision() == "0032"


def test_module_loads():
    """Sanity: the migration module imports without side effects."""
    mod = import_module(_MIGRATION_MODULE)
    assert mod.revision == "0032"
    assert mod.down_revision == "0031"


def test_fresh_install_adds_column():
    """Baseline: a brand-new DB with the full schema picks up the
    column via ``Base.metadata.create_all`` in 0001. ``upgrade head``
    is then a no-op (0032's ``column_exists`` guard short-circuits).
    """
    engine = _make_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE documents (
                    id VARCHAR(64) PRIMARY KEY,
                    filename VARCHAR(500),
                    title VARCHAR(500),
                    file_size INTEGER NOT NULL,
                    original_filename VARCHAR(500),
                    created_at DATETIME
                )
                """
            )
        )

    assert _has_column(engine, _COLUMN)

    run_migrations(engine, "head")

    assert _has_column(engine, _COLUMN), (
        "upgrade head dropped original_filename on a fresh-install DB"
    )


def test_legacy_db_upgrade_adds_column():
    """The actual bug path: a DB that was created before this commit
    has ``documents`` minus ``original_filename``. After upgrade the
    column exists with NULL default; pre-existing rows keep their
    NULL value (no backfill — the upload code populates the column
    on every new upload, and re-uploading is the supported recovery
    path for legacy rows).
    """
    engine = _make_engine()
    with engine.begin() as conn:
        # Mirror the historical schema: no original_filename column.
        conn.execute(
            text(
                """
                CREATE TABLE documents (
                    id VARCHAR(64) PRIMARY KEY,
                    filename VARCHAR(500),
                    title VARCHAR(500),
                    file_size INTEGER NOT NULL,
                    created_at DATETIME
                )
                """
            )
        )
        # Seed one legacy row with only the sanitized filename (the
        # classic bug symptom in production).
        conn.execute(
            text(
                """
                INSERT INTO documents
                    (id, filename, title, file_size)
                VALUES ('doc-1', 'sanitized_only.pdf', NULL, 1024)
                """
            )
        )

    assert not _has_column(engine, _COLUMN)

    run_migrations(engine, "head")

    assert _has_column(engine, _COLUMN)
    col = _column_type(engine, _COLUMN)
    assert col is not None
    # VARCHAR(500), nullable — matches the ORM declaration.
    assert "VARCHAR" in str(col["type"]).upper()
    assert col["nullable"] is True

    # Pre-existing row keeps its sanitized filename, original_filename
    # stays NULL (no destructive backfill).
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT filename, original_filename FROM documents")
        ).fetchone()
    assert row[0] == "sanitized_only.pdf"
    assert row[1] is None


def test_upgrade_is_idempotent():
    """Re-running ``upgrade head`` after the column exists is a no-op
    (the migration's own ``column_exists`` guard returns early). This
    prevents ``alembic_runner.run_migrations`` from opening a write
    transaction on every cold engine reopen just to discover there's
    nothing to apply.
    """
    engine = _make_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE documents (
                    id VARCHAR(64) PRIMARY KEY,
                    filename VARCHAR(500),
                    title VARCHAR(500),
                    file_size INTEGER NOT NULL,
                    original_filename VARCHAR(500),
                    created_at DATETIME
                )
                """
            )
        )

    run_migrations(engine, "head")
    # Second call: must not raise (would otherwise fail with "column
    # already exists" or duplicate-add errors).
    run_migrations(engine, "head")

    assert _has_column(engine, _COLUMN)


def test_downgrade_drops_column_keeps_filename():
    """Downgrading past 0032 removes the column but does NOT touch
    ``filename`` — the sanitized fallback is still valid; legacy
    citations just lose access to the raw upload name.
    """
    engine = _make_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE documents (
                    id VARCHAR(64) PRIMARY KEY,
                    filename VARCHAR(500),
                    title VARCHAR(500),
                    file_size INTEGER NOT NULL,
                    original_filename VARCHAR(500),
                    created_at DATETIME
                )
                """
            )
        )

    run_migrations(engine, "head")
    assert _has_column(engine, _COLUMN)

    config = get_alembic_config(engine)
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        command.downgrade(config, "0031")

    assert not _has_column(engine, _COLUMN)
    # Sanitized filename column survives — citations still render.
    assert _has_column(engine, "filename")