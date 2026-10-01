"""drop document_embeddings, superseded by document_chunks

Revision ID: 20260929_000036
Revises: 20260929_000035
Create Date: 2026-09-29 00:00:36.000000

`public.document_embeddings` held one embedding per DOCUMENT over the curated
gold summary (label + identifier name:value pairs + document_name). It is
superseded by `public.document_chunks`, which indexes the real document text and
serves the same document-level discovery by collapsing chunk hits to each
document's best passage — over content the old table could not represent.

Nothing reads it any more: 000034 re-pointed `semantic_search_documents` at the
chunk table, and no reviewer endpoint was ever built against it.

WHY THIS IS SAFE TO DROP RATHER THAN DEPRECATE (checked live, 2026-09-29):
    dev  — the table existed with **0 rows**.
    prod — the table did **not exist at all** (000030 never ran there).
    So this retires dead code, not a working index. Its contents were in any case
    100% derived from gold_extraction_labels_sync and re-computable.

WHAT ELSE GOES WITH IT (same commit):
    jobs/embed_documents.py and the `lakercm_embed_documents` job block, whose
    `0 0/30 * * * ?` schedule was the LAST polling cron in the document path.
    ad2c018 moved ingest to file-arrival triggers; the chunk loader rides that
    same trigger as a task on lakercm_documents_refresh. After this, the whole
    path is event-driven end to end.

The downgrade recreates the table, its indexes and its grants so the chain is
structurally reversible — but it comes back EMPTY, because the job that populated
it is deleted in this commit. Repopulating means reverting the code, not just the
migration. Given both environments held zero rows, that is a faithful reversal.
"""

import os

from alembic import op

# revision identifiers, used by Alembic
revision = "20260929_000036"
down_revision = "20260929_000035"
branch_labels = None
depends_on = None

CHECKPOINT_ROLE = os.environ.get("CHECKPOINT_ROLE_NAME", "").strip()
AGENT_SP = os.environ.get("AGENT_SP_CLIENT_ID", "").strip()
REVIEWER_SP = os.environ.get("REVIEWER_SP_CLIENT_ID", "").strip()


def _role_exists(role: str) -> bool:
    if not role:
        return False
    bind = op.get_bind()
    res = bind.exec_driver_sql(
        "SELECT 1 FROM pg_roles WHERE rolname = %(r)s", {"r": role}
    )
    return res.scalar() is not None


def upgrade():
    # Grants and indexes go with the table; the policy does too.
    op.execute("DROP TABLE IF EXISTS public.document_embeddings")


def downgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.document_embeddings (
            document_path    VARCHAR(1000) PRIMARY KEY,
            document_id      UUID,
            label            VARCHAR(64),
            embedding_text   TEXT NOT NULL,
            embedding        vector(1024) NOT NULL,
            content_tsv      tsvector GENERATED ALWAYS AS (
                                 to_tsvector('english', COALESCE(embedding_text, ''))
                             ) STORED,
            confidence_score DOUBLE PRECISION,
            embedding_model  VARCHAR(128),
            extracted_at     TIMESTAMPTZ,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_hnsw "
        "ON public.document_embeddings USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_tsv "
        "ON public.document_embeddings USING gin (content_tsv)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_docid "
        "ON public.document_embeddings(document_id)"
    )
    for role, privs in (
        (CHECKPOINT_ROLE, "SELECT"),
        (AGENT_SP, "SELECT"),
        (REVIEWER_SP, "SELECT, INSERT, UPDATE"),
    ):
        if _role_exists(role):
            op.execute(f'GRANT USAGE ON SCHEMA public TO "{role}"')
            op.execute(f'GRANT {privs} ON public.document_embeddings TO "{role}"')
