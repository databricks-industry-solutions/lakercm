"""add corrections JSONB column to document_extraction_reviews

Revision ID: 20260429_000024
Revises: 20260429_000023
Create Date: 2026-04-29 00:00:24.000000

Inline-edit support for the reviewer UI: when a reviewer marks an
extraction "incorrect" or "partially_correct", the field-level corrections
they typed in (keyed by `id:<idx>` from parseIdentifiers) are persisted
alongside the verdict + reasoning so curation downstream can train on
them. JSONB so we can index by key later without a schema change.

NULL-tolerant: existing rows have no corrections, and a "correct" verdict
keeps it NULL (no edits to record).
"""

from alembic import op


revision = "20260429_000024"
down_revision = "20260429_000023"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE public.document_extraction_reviews "
        "ADD COLUMN IF NOT EXISTS corrections JSONB"
    )


def downgrade():
    op.execute(
        "ALTER TABLE public.document_extraction_reviews "
        "DROP COLUMN IF EXISTS corrections"
    )
