"""Add chat_sessions.collection_id (FK -> collections.id).

Session-scoped document isolation: each chat points at its dedicated
Collection, and retrieval must only search that collection (spec section 9,
fail-closed). The FK is added via batch_alter_table because SQLite's
op.add_column cannot carry an inline ForeignKey ("No support for ALTER of
constraints") - the same pattern as migration 0010's research_history
chat_session_id FK.

Revision ID: 0031
Revises: 0030
Create Date: 2026-09-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy import inspect

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None

_INDEX_NAME = "ix_chat_sessions_collection_id"
_FK_NAME = "fk_chat_sessions_collection_id"


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table(table_name):
        return False
    return column_name in {c["name"] for c in inspector.get_columns(table_name)}


def _index_exists(index_name: str, table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table(table_name):
        return False
    return index_name in {i["name"] for i in inspector.get_indexes(table_name)}


def _fk_exists() -> bool:
    """Return whether the collection_id -> collections.id FK already exists.

    Fresh installs create chat_sessions via ``Base.metadata.create_all``
    (migration 0001), which emits the FK *unnamed* - the model's
    ``ForeignKey`` carries no explicit constraint name. Checking only the
    name would therefore cause 0031 to add the same FK a second time on
    fresh databases (the exact defect 0010's structural check guards
    against). Accept a named OR structurally-equivalent FK.
    """
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("chat_sessions"):
        return False

    for fk in inspector.get_foreign_keys("chat_sessions"):
        if fk.get("name") == _FK_NAME:
            return True
        if (
            fk.get("constrained_columns") == ["collection_id"]
            and fk.get("referred_table") == "collections"
            and fk.get("referred_columns") == ["id"]
        ):
            ondelete = (fk.get("options") or {}).get("ondelete")
            if (
                isinstance(ondelete, str)
                and ondelete.strip().upper() == "SET NULL"
            ):
                return True
    return False


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("collections") or not inspector.has_table(
        "chat_sessions"
    ):
        return

    if not _column_exists("chat_sessions", "collection_id"):
        op.add_column(
            "chat_sessions",
            sa.Column("collection_id", sa.String(36), nullable=True),
        )
        logger.info("0031: added chat_sessions.collection_id")

    if not _index_exists(_INDEX_NAME, "chat_sessions"):
        op.create_index(_INDEX_NAME, "chat_sessions", ["collection_id"])
        logger.info("0031: added index %s", _INDEX_NAME)

    if not _fk_exists():
        with op.batch_alter_table("chat_sessions", schema=None) as batch_op:
            batch_op.create_foreign_key(
                _FK_NAME,
                "collections",
                ["collection_id"],
                ["id"],
                ondelete="SET NULL",
            )
        logger.info(
            "0031: added FK chat_sessions.collection_id -> collections.id"
        )


def downgrade() -> None:
    raise NotImplementedError(
        "0031 (chat_sessions.collection_id) is not reversible: SQLite ALTER "
        "TABLE forbids dropping an indexed FK column cleanly, and "
        "batch_alter_table must rebuild chat_sessions - the FK target of "
        "chat_messages.session_id and research_history.chat_session_id. "
        "Recreate the dev database to roll back."
    )
