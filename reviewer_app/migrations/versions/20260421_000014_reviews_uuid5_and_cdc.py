"""decouple reviews from medical_documents and enable CDC sync to UC

Revision ID: 20260421_000014
Revises: 20260420_000013
Create Date: 2026-04-21 00:00:14.000000

Two changes to make reviews streamable to UC and keyable by path:

1. Drop the document_id -> medical_documents(id) foreign key. Streamed
   documents (files landing directly in the UC volume) never get a
   medical_documents row, but still need human reviews keyed by
   uuid5(NAMESPACE_URL, document_path). The UNIQUE constraint on
   (document_id, reviewer_email) stays — one review per reviewer per doc.

2. REPLICA IDENTITY FULL on document_extraction_reviews so Lakebase ->
   UC change data capture includes the full row in each CDC event.
   Tables without REPLICA IDENTITY FULL are skipped by the sync.
   Applied to medical_documents too for symmetry in case we add a CDC
   flow for it later.
"""

from alembic import op


revision = "20260421_000014"
down_revision = "20260420_000013"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE public.document_extraction_reviews "
        "DROP CONSTRAINT IF EXISTS document_extraction_reviews_document_id_fkey"
    )
    op.execute("ALTER TABLE public.document_extraction_reviews REPLICA IDENTITY FULL")
    op.execute("ALTER TABLE public.medical_documents REPLICA IDENTITY FULL")


def downgrade():
    op.execute(
        "ALTER TABLE public.document_extraction_reviews REPLICA IDENTITY DEFAULT"
    )
    op.execute("ALTER TABLE public.medical_documents REPLICA IDENTITY DEFAULT")
    op.execute(
        "ALTER TABLE public.document_extraction_reviews "
        "ADD CONSTRAINT document_extraction_reviews_document_id_fkey "
        "FOREIGN KEY (document_id) REFERENCES public.medical_documents(id)"
    )
