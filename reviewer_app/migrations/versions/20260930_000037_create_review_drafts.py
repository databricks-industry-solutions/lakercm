"""create review_drafts table for unsubmitted review autosave

Revision ID: 20260930_000037
Revises: 20260929_000036
Create Date: 2026-09-30 00:00:37.000000

Backs draft autosave for the review form. Until now nothing in the form
persisted until Submit: a reviewer could pick a verdict, type reasoning, correct
three fields, press `j` to peek at the next document, and lose all of it.

A SEPARATE table, deliberately, not a `status` column on
public.document_extraction_reviews:

  * That table streams through the Lakebase change feed into silver/gold, where
    pipelines/silver/silver_reviews.sql constrains
    verdict IN ('correct','partially_correct','incorrect') -- which a half-filled
    draft violates by construction.
  * A `WHERE status='submitted'` filter on the feed would work, but it is a
    filter someone can forget, and it would put unsubmitted work one typo away
    from the accuracy stats.
  * document_extraction_reviews is ONE shared verdict row per document.
    A draft is per-(document, reviewer) in-progress work, like document_notes.

HOW THIS TABLE STAYS OUT OF THE FEED. Note that the CDF config covers the WHOLE
`public` schema (scripts/create_reverse_cdf.py creates one config for the
schema, and its FED_TABLES tuple is a wait-list for the deploy, NOT a filter).
What actually excludes a table is REPLICA IDENTITY: migration 000014 records
that tables without REPLICA IDENTITY FULL "are skipped by the sync". So this
table deliberately does NOT set it, exactly like document_notes.

That means: do not add REPLICA IDENTITY FULL to this table later without also
deciding what a draft row means in silver. Adding it is sufficient, on its own,
to start streaming unsubmitted reviews into the analytics layer.

Every column except the key is nullable or defaulted, because a draft is
partial by definition -- that partiality is the whole reason it cannot live in
the reviews table. The verdict CHECK still pins the vocabulary when a verdict IS
set, so a draft can never carry a value that would be rejected at submit time.

No agent grant. Unlike document_notes (which the agent READS via
get_active_review_context), the assistant has no need for another reviewer's
unsubmitted work, so the surface stays minimal. Adding a SELECT grant later is
a two-line migration if the in-document assistant ever wants draft context.
"""

from alembic import op

# revision identifiers, used by Alembic
revision = "20260930_000037"
down_revision = "20260929_000036"
branch_labels = None
depends_on = None


def upgrade():
    # Drop if pre-created outside Alembic (mirrors create_document_notes).
    op.execute("DROP TABLE IF EXISTS public.review_drafts")
    op.execute("""
        CREATE TABLE public.review_drafts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL,
            user_email VARCHAR(320) NOT NULL,
            verdict VARCHAR(32),
            reasoning TEXT,
            corrections JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_review_drafts_doc_user UNIQUE (document_id, user_email),
            CONSTRAINT ck_review_drafts_verdict CHECK (
                verdict IS NULL
                OR verdict IN ('correct', 'partially_correct', 'incorrect')
            )
        )
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_review_drafts_doc "
        "ON public.review_drafts(document_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_review_drafts_user "
        "ON public.review_drafts(user_email)"
    )
    # Permissive policy so all authenticated roles keep access if RLS is ever
    # enabled on this table (matches public.document_notes).
    op.execute(
        "CREATE POLICY lakercm_app_full_access "
        "ON public.review_drafts "
        "FOR ALL TO PUBLIC "
        "USING (true) WITH CHECK (true)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_review_drafts_user")
    op.execute("DROP INDEX IF EXISTS idx_review_drafts_doc")
    op.execute("DROP TABLE IF EXISTS public.review_drafts")
