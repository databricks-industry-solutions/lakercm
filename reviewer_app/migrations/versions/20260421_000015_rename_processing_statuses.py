"""rename processing_status values: pending->processing, ready->pending

Revision ID: 20260421_000015
Revises: 20260421_000014
Create Date: 2026-04-21 00:00:15.000000

Three-state semantic model for the reviewer UI:
  - processing      : file landed, extraction in flight (was 'pending')
  - pending         : extraction done, awaiting human review (was 'ready')
  - auto_verified   : extraction done, confidence >= threshold (unchanged)
  - failed          : extraction failed (unchanged)

The old 'processing' enum value was defined but never written. 'reviewed'
is a derived state (presence of a document_extraction_reviews row), not a
stored value. Existing rows are rewritten in place via a single CASE UPDATE
so the rename collides with nothing.
"""

from alembic import op


revision = "20260421_000015"
down_revision = "20260421_000014"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE public.medical_documents "
        "DROP CONSTRAINT IF EXISTS medical_documents_processing_status_check"
    )
    op.execute(
        """
        UPDATE public.medical_documents
        SET processing_status = CASE
            WHEN processing_status = 'pending' THEN 'processing'
            WHEN processing_status = 'ready'   THEN 'pending'
            ELSE processing_status
        END
        """
    )
    op.execute(
        "ALTER TABLE public.medical_documents "
        "ALTER COLUMN processing_status SET DEFAULT 'processing'"
    )
    op.execute(
        "ALTER TABLE public.medical_documents "
        "ADD CONSTRAINT medical_documents_processing_status_check "
        "CHECK (processing_status IN ('processing', 'pending', 'auto_verified', 'failed'))"
    )


def downgrade():
    op.execute(
        "ALTER TABLE public.medical_documents "
        "DROP CONSTRAINT IF EXISTS medical_documents_processing_status_check"
    )
    op.execute(
        """
        UPDATE public.medical_documents
        SET processing_status = CASE
            WHEN processing_status = 'processing' THEN 'pending'
            WHEN processing_status = 'pending'    THEN 'ready'
            ELSE processing_status
        END
        """
    )
    op.execute(
        "ALTER TABLE public.medical_documents "
        "ALTER COLUMN processing_status SET DEFAULT 'pending'"
    )
    op.execute(
        "ALTER TABLE public.medical_documents "
        "ADD CONSTRAINT medical_documents_processing_status_check "
        "CHECK (processing_status IN ('pending', 'processing', 'ready', 'auto_verified', 'failed'))"
    )
