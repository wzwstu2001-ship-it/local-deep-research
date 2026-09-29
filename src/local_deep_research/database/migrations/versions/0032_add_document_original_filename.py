"""Add documents.original_filename column.

Preserves the raw (pre-sanitization) upload name on the Document row so
downstream consumers (chunks, citations, Library UI listing) can render
the human-readable filename in the user's language. sanitize_filename()
strips non-ASCII via werkzeug.secure_filename, so Document.filename alone
is ASCII-only and loses Chinese / accented / etc. upload names; the
indexer's preferred-fallback chain
``document.title or document.original_filename or document.filename``
fell through to the sanitized value on every existing install because
the column was only ever added to ``database/models/library.py`` —
never to the schema migration history.

Revision ID: 0032
Revises: 0031

Column Added
============
- original_filename (VARCHAR(500)): Raw upload filename. NULL on
  documents uploaded before this migration ran; populated on every
  upload via ``_upload_to_collection_impl`` going forward.

Migration Notes
===============
- Idempotent: skips the column if it already exists (fresh databases
  created after the model change will already have it via
  ``Base.metadata.create_all``).
- Uses SQLite batch mode (table recreation) for ALTER TABLE
  compatibility — same pattern as 0011 / 0031.
- No backfill: any pre-existing documents keep ``original_filename=NULL``
  and the indexer's ``filename`` fallback still surfaces a usable title
  (just without the original CJK characters). To re-derive the raw
  filename for an old document the user must re-upload it or call a
  future backfill that introspects a side store; out of scope here.

Downgrade Behavior
==================
``downgrade()`` drops ``original_filename``. Any persisted raw filenames
go with it; citations on already-indexed documents fall back to the
sanitized ``filename`` column — they remain correct, just less readable
for non-ASCII uploads.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None

_TABLE = "documents"
_COLUMN = "original_filename"


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)

    if not inspector.has_table(table_name):
        return False

    columns = {col["name"] for col in inspector.get_columns(table_name)}
    return column_name in columns


def upgrade() -> None:
    """Add ``original_filename`` to ``documents`` if missing."""
    bind = op.get_bind()
    inspector = inspect(bind)

    if not inspector.has_table(_TABLE):
        # Table doesn't exist yet — ``0001_initial_schema`` creates it
        # with all model columns via ``Base.metadata.create_all`` on a
        # fresh DB, so nothing to do here.
        return

    if column_exists(_TABLE, _COLUMN):
        # Column already present (fresh DB or already-run 0032). Idempotent
        # so re-running the migration against the same DB is a no-op.
        return

    # SQLite (and SQLCipher, which is also SQLite underneath) needs
    # batch mode for ALTER TABLE that mixes column add + constraint
    # logic; harmless on backends that don't need it.
    # nullable=True to match ``Document.original_filename`` declared
    # in ``database/models/library.py`` (String(500), nullable=True).
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.add_column(
            sa.Column(
                _COLUMN,
                sa.String(length=500),
                nullable=True,
            )
        )


def downgrade() -> None:
    """Drop ``original_filename`` from ``documents`` if present."""
    bind = op.get_bind()
    inspector = inspect(bind)

    if not inspector.has_table(_TABLE):
        return

    if not column_exists(_TABLE, _COLUMN):
        return

    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column(_COLUMN)