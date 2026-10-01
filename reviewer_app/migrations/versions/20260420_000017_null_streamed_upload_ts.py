"""make upload_timestamp nullable and null it out for streamed-backfill rows.

Revision ID: 20260420_000017
Revises: 20260420_000016
Create Date: 2026-04-20 00:00:17.000000

Streamed docs (files written directly to the UC volume, bypassing
/api/documents/upload) have no true upload time — we only see them after
SDP has already extracted. The old backfill collapsed upload_timestamp and
processing_timestamp to extracted_at, producing bogus 0-second pipeline
deltas. Forward-fix: backfill now sets upload_timestamp = NULL,
processing_timestamp = extracted_at. This migration:

1. Drops the NOT NULL on upload_timestamp (rows from streamed backfills
   can legitimately lack one).
2. Nulls upload_timestamp for existing rows where it equals
   processing_timestamp — those are the collapsed-backfill artifacts.
"""

from alembic import op


revision = "20260420_000017"
down_revision = "20260420_000016"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE public.medical_documents "
        "ALTER COLUMN upload_timestamp DROP NOT NULL"
    )
    op.execute(
        """
        UPDATE public.medical_documents
        SET upload_timestamp = NULL
        WHERE processing_timestamp IS NOT NULL
          AND upload_timestamp = processing_timestamp
        """
    )


def downgrade():
    op.execute(
        """
        UPDATE public.medical_documents
        SET upload_timestamp = COALESCE(upload_timestamp, processing_timestamp, created_at)
        WHERE upload_timestamp IS NULL
        """
    )
    op.execute(
        "ALTER TABLE public.medical_documents "
        "ALTER COLUMN upload_timestamp SET NOT NULL"
    )
