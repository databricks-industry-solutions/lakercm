"""add auto_verified to medical_documents.processing_status CHECK constraint

Revision ID: 20260417_000010
Revises: 20260417_000009
Create Date: 2026-04-17 00:00:10.000000

Auto-verified docs (confidence >= auto_verdict_threshold) are flipped to this
status by /api/documents/sync-status so the Reviewer list hides them. They
live in the `gold_auto_verified_extractions` materialized view and are joined
into analytics via the SDP pipeline — no Lakebase review row is written.
"""

from alembic import op


revision = "20260417_000010"
down_revision = "20260417_000009"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE public.medical_documents "
        "DROP CONSTRAINT IF EXISTS medical_documents_processing_status_check"
    )
    op.execute(
        "ALTER TABLE public.medical_documents "
        "ADD CONSTRAINT medical_documents_processing_status_check "
        "CHECK (processing_status IN "
        "('pending', 'processing', 'ready', 'auto_verified', 'failed'))"
    )


def downgrade():
    op.execute(
        "ALTER TABLE public.medical_documents "
        "DROP CONSTRAINT IF EXISTS medical_documents_processing_status_check"
    )
    op.execute(
        "ALTER TABLE public.medical_documents "
        "ADD CONSTRAINT medical_documents_processing_status_check "
        "CHECK (processing_status IN "
        "('pending', 'processing', 'ready', 'failed'))"
    )
