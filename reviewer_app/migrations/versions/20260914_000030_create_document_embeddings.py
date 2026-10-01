"""create document_embeddings table for Lakebase-native document search

Revision ID: 20260914_000030
Revises: 20260530_000028
Create Date: 2026-09-14 00:00:30.000000

A dedicated, writable Lakebase table holding one embedding per document over the
CURATED gold extraction text (label + identifier name:value pairs + doc name) —
NEVER raw OCR (honors the agent's no-raw-text rule). Separate from
lakercm.gold_extraction_labels_sync (read-only synced replica — writes are
overwritten) and from public.store_vectors (LangGraph per-user memory, different
lifecycle).

Search is Lakebase-only:
  * SEMANTIC — pgvector `vector(1024)` + HNSW cosine index (matches the memory
    store's index shape; HNSW is the right choice at this corpus size).
  * KEYWORD  — a generated `content_tsv tsvector` + GIN index (Postgres FTS).
    This is the shippable in-Lakebase keyword backend. The Lakebase Search
    `lakebase_text` (BM25) extension is a drop-in upgrade once that Beta is
    enabled on the project (see docs/document-search.md); it is intentionally
    NOT created here so this migration stays deterministic (a failed
    CREATE EXTENSION would abort the migration transaction).

Populated by jobs/embed_documents.py (change-only upsert). Read by the agent's
semantic_search_documents tool (and, later, a reviewer endpoint).

Grants mirror migration 000027/000029: read for the agent's Postgres identities
(the checkpoint group role it sessions-as, + the agent SP as fallback), and
read/write for the reviewer SP (the embed job's write identity). All guarded so
they no-op on a fresh workspace where the roles don't exist yet.
"""

import os

from alembic import op

# revision identifiers, used by Alembic
revision = "20260914_000030"
down_revision = "20260530_000028"
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


def _grant(role: str, privs: str) -> None:
    if not _role_exists(role):
        print(
            f"[migration 20260914_000030] role {role!r} absent — "
            f"skipping GRANT {privs} on public.document_embeddings."
        )
        return
    quoted = f'"{role}"'
    op.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}")
    op.execute(f"GRANT {privs} ON public.document_embeddings TO {quoted}")


def upgrade():
    # pgvector is enabled by 20260419_000012; assert idempotently in case this
    # runs on a branch where that ordering differs.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("DROP TABLE IF EXISTS public.document_embeddings")
    op.execute("""
        CREATE TABLE public.document_embeddings (
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
    # SEMANTIC: HNSW cosine — same ops class the memory store uses.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_hnsw "
        "ON public.document_embeddings USING hnsw (embedding vector_cosine_ops)"
    )
    # KEYWORD: Postgres full-text (GIN over the generated tsvector).
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_tsv "
        "ON public.document_embeddings USING gin (content_tsv)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_embeddings_docid "
        "ON public.document_embeddings(document_id)"
    )
    # Permissive policy so access holds if RLS is ever enabled (matches
    # public.conversations / public.document_notes).
    op.execute(
        "CREATE POLICY lakercm_app_full_access "
        "ON public.document_embeddings "
        "FOR ALL TO PUBLIC "
        "USING (true) WITH CHECK (true)"
    )

    # Reads for the agent's identities; read/write for the reviewer SP (the
    # embed job's write identity).
    _grant(CHECKPOINT_ROLE, "SELECT")
    _grant(AGENT_SP, "SELECT")
    _grant(REVIEWER_SP, "SELECT, INSERT, UPDATE")


def downgrade():
    for role in (CHECKPOINT_ROLE, AGENT_SP, REVIEWER_SP):
        if _role_exists(role):
            op.execute(f'REVOKE ALL ON public.document_embeddings FROM "{role}"')
    op.execute("DROP INDEX IF EXISTS idx_document_embeddings_docid")
    op.execute("DROP INDEX IF EXISTS idx_document_embeddings_tsv")
    op.execute("DROP INDEX IF EXISTS idx_document_embeddings_hnsw")
    op.execute("DROP TABLE IF EXISTS public.document_embeddings")
