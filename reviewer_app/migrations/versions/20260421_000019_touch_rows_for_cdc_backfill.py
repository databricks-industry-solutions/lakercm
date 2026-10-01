"""force WAL events on existing rows so the UC CDC sink backfills them.

Revision ID: 20260421_000019
Revises: 20260421_000018
Create Date: 2026-04-21 00:00:19.000000

Rows inserted before migration 18 made REPLICA IDENTITY FULL effective
never produced a CDC event the Lakebase->UC sink would accept, so the
UC mirror is missing them (lb_medical_documents_history had 5 distinct
ids vs ~20 in Lakebase). No-op UPDATE rewrites every row into the WAL
with the full image now that the identity setting is correct; the sink
then emits an `update` event per row and the mirror converges.

Idempotent: re-running is harmless (just produces more WAL traffic).
"""

from alembic import op


revision = "20260421_000019"
down_revision = "20260421_000018"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("UPDATE public.medical_documents SET updated_at = updated_at")
    op.execute("UPDATE public.document_extraction_reviews SET updated_at = updated_at")


def downgrade():
    pass
