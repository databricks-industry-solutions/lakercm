"""collapse document_extraction_reviews to one row per document

Revision ID: 20260421_000022
Revises: 20260421_000021
Create Date: 2026-04-21 00:00:22.000000

Option B shared-review model: any reviewer's latest verdict is the
canonical one shown to everyone. Previously the unique constraint was
(document_id, reviewer_email), so Reviewer B's override of Reviewer A
inserted a *second* row instead of updating A's — causing the gold
review count to outrun the document count.

This migration:
  1. Keeps the most recently updated row per document_id, deletes the
     rest. Deletions propagate through logical replication (REPLICA
     IDENTITY FULL was set in 20260421_000018) so the CDC-synced copy
     and gold_review_analytics converge on the new cardinality.
  2. Swaps UNIQUE(document_id, reviewer_email) → UNIQUE(document_id).
     With the new constraint, upserts use ON CONFLICT (document_id),
     overwriting whatever prior verdict existed — including one from a
     different reviewer. `id` is preserved across the update, so CDC
     emits an UPDATE on the same PK and `_cdc_current` collapses
     cleanly.

created_at is left untouched on override, so review turnaround stays
"time to first review on this doc" rather than flipping each time a
later reviewer refines the verdict.
"""

from alembic import op


revision = "20260421_000022"
down_revision = "20260421_000021"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        DELETE FROM public.document_extraction_reviews r
        USING (
            SELECT id
            FROM (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY document_id
                        ORDER BY updated_at DESC, created_at DESC
                    ) AS rn
                FROM public.document_extraction_reviews
            ) ranked
            WHERE rn > 1
        ) dupes
        WHERE r.id = dupes.id
        """
    )

    op.execute(
        "ALTER TABLE public.document_extraction_reviews "
        "DROP CONSTRAINT IF EXISTS uq_review_per_document_reviewer"
    )

    op.execute(
        "ALTER TABLE public.document_extraction_reviews "
        "ADD CONSTRAINT uq_review_per_document UNIQUE (document_id)"
    )


def downgrade():
    op.execute(
        "ALTER TABLE public.document_extraction_reviews "
        "DROP CONSTRAINT IF EXISTS uq_review_per_document"
    )
    op.execute(
        "ALTER TABLE public.document_extraction_reviews "
        "ADD CONSTRAINT uq_review_per_document_reviewer "
        "UNIQUE (document_id, reviewer_email)"
    )
