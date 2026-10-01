"""create document_extraction_reviews table

Revision ID: 20260220_000005
Revises: 20260212_000004
Create Date: 2026-02-20 00:00:05.000000

Human review verdicts for document extraction quality.
One review per document per reviewer, with upsert semantics.
"""

from alembic import op

# revision identifiers, used by Alembic
revision = "20260220_000005"
down_revision = "20260220_000001"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS public.document_extraction_reviews (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES public.medical_documents(id),
            reviewer_email VARCHAR(320) NOT NULL,
            verdict VARCHAR(30) NOT NULL
                CHECK (verdict IN ('correct', 'partially_correct', 'incorrect')),
            reasoning TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_review_per_document_reviewer
                UNIQUE (document_id, reviewer_email)
        )
    """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_extraction_reviews_document_id
        ON public.document_extraction_reviews(document_id)
    """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_extraction_reviews_reviewer
        ON public.document_extraction_reviews(reviewer_email)
    """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_extraction_reviews_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """
    )

    op.execute(
        """
        CREATE TRIGGER trigger_update_extraction_reviews_updated_at
        BEFORE UPDATE ON public.document_extraction_reviews
        FOR EACH ROW
        EXECUTE FUNCTION update_extraction_reviews_updated_at()
    """
    )


def downgrade():
    op.execute(
        "DROP TRIGGER IF EXISTS trigger_update_extraction_reviews_updated_at "
        "ON public.document_extraction_reviews"
    )
    op.execute("DROP FUNCTION IF EXISTS update_extraction_reviews_updated_at")
    op.execute("DROP INDEX IF EXISTS public.idx_extraction_reviews_reviewer")
    op.execute("DROP INDEX IF EXISTS public.idx_extraction_reviews_document_id")
    op.execute("DROP TABLE IF EXISTS public.document_extraction_reviews")
