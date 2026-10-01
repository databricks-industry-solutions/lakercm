"""create document_chunks table for chunk-level Lakebase document search

Revision ID: 20260929_000034
Revises: 20260928_000033
Create Date: 2026-09-29 00:00:34.000000

A dedicated, writable Lakebase table holding one embedding per RETRIEVAL CHUNK,
produced by the pipeline's third parallel branch (pipelines/silver/
silver_doc_chunks.sql → ai_prep_search) and loaded by jobs/load_document_chunks.py.

This is the counterpart to public.document_embeddings, which holds one embedding
per DOCUMENT over the curated gold summary. The two answer different questions:
document_embeddings finds documents by what was extracted from them; this table
answers "what does THIS document say" and, aggregated to its best chunk per
document, also subsumes document-level discovery over the real document text.

Unlike document_embeddings, the text here is RAW parsed document content — that
is the point, and it is a deliberate reversal of that table's curated-text-only
design. Access is governed by the Postgres grants below, the same way every other
document table is: LakeRCM is a shared review queue, so no search tool filters by
user_email and any reviewer can retrieve any indexed document's passages. The
practical change from document_embeddings is that those passages are now raw
document text rather than a curated field summary.

Curated fields (label, confidence_score, verdict) are deliberately absent: the
agent LEFT JOINs lakercm.gold_extraction_labels_sync on document_path at query
time instead, so they never go stale here and the pipeline branch stays free of
any dependency on silver_classify_label.

Search is Lakebase-only, same two arms as 000030:
  * SEMANTIC — pgvector `vector(1024)` + HNSW cosine index.
  * KEYWORD  — a generated `content_tsv tsvector` + GIN index (Postgres FTS).
    The Lakebase Search `lakebase_bm25` index is a later, separate migration
    (000035), and only the KEYWORD arm adopts it — the vector arm keeps this
    migration's HNSW index, because `lakebase_ann` is IVF-based and would be built
    against an empty table at boot. Enabling Lakebase Search on the project
    restarts every compute and cannot be undone, so it must not gate this table
    landing. As in 000030, no CREATE EXTENSION for it here — a failed one would
    abort the migration transaction.

PRIMARY KEY is (document_path, chunk_id), not chunk_id alone: ai_prep_search
documents no global-uniqueness guarantee for chunk_id, and the loader replaces a
document's chunks as a set, so the composite key is both safe and the natural
access path.

Grants mirror 000030, with one addition: the reviewer SP also needs DELETE,
because the loader replaces each document's chunk set transactionally rather
than upserting row-wise (a re-parse renumbers chunks, so row-wise upsert would
orphan stale ones). All guarded so they no-op on a fresh workspace.
"""

import os

from alembic import op

# revision identifiers, used by Alembic
revision = "20260929_000034"
down_revision = "20260928_000033"
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
            f"[migration 20260929_000034] role {role!r} absent — "
            f"skipping GRANT {privs} on public.document_chunks."
        )
        return
    quoted = f'"{role}"'
    op.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}")
    op.execute(f"GRANT {privs} ON public.document_chunks TO {quoted}")


def upgrade():
    # pgvector is enabled by 20260419_000012; assert idempotently in case this
    # runs on a branch where that ordering differs.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("DROP TABLE IF EXISTS public.document_chunks")
    op.execute("""
        CREATE TABLE public.document_chunks (
            document_path     VARCHAR(1000) NOT NULL,
            chunk_id          VARCHAR(256)  NOT NULL,
            document_id       UUID,
            document_name     VARCHAR(512),
            user_email        VARCHAR(320),
            chunk_position    INTEGER,
            -- Nullable on purpose. The pipeline guards and EXPECTs only
            -- chunk_to_embed; chunk_to_retrieve is projected unchecked. Were this
            -- NOT NULL, one null passage would raise inside the loader's
            -- per-document transaction, roll the document back, and — because the
            -- anti-join keys on prepped_at, which never changes again — fail
            -- identically on every future run. A poison document forever.
            chunk_to_retrieve TEXT,
            chunk_to_embed    TEXT NOT NULL,
            embedding         vector(1024) NOT NULL,
            content_tsv       tsvector GENERATED ALWAYS AS (
                                  to_tsvector('english', COALESCE(chunk_to_embed, ''))
                              ) STORED,
            page_id           INTEGER,
            image_uri         TEXT,
            -- Populated only once ai_prep_search is called with the 'schema'
            -- option (document-level metadata extraction). Nullable now so
            -- adopting it later needs no migration.
            metadata          JSONB,
            embedding_model   VARCHAR(128),
            prepped_at        TIMESTAMPTZ,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (document_path, chunk_id)
        )
        """)
    # SEMANTIC: HNSW cosine — same ops class document_embeddings and the memory
    # store use.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_hnsw "
        "ON public.document_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    # KEYWORD: Postgres full-text (GIN over the generated tsvector).
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv "
        "ON public.document_chunks USING gin (content_tsv)"
    )
    # No index on user_email: LakeRCM is a shared review queue and NO search tool
    # filters by it (the only user_email predicate in the agent is the private
    # notepad). An index nothing queries would just cost write time on a table
    # that is loaded in bulk. No document_path index either — it is the PK's
    # leading column.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_docid "
        "ON public.document_chunks(document_id)"
    )
    # Permissive policy so access holds if RLS is ever enabled (matches
    # public.document_embeddings / public.document_notes).
    op.execute(
        "CREATE POLICY lakercm_app_full_access "
        "ON public.document_chunks "
        "FOR ALL TO PUBLIC "
        "USING (true) WITH CHECK (true)"
    )

    # Reads for the agent's identities; read/write/delete for the reviewer SP
    # (the chunk loader's write identity — DELETE is required for the
    # per-document replace, which 000030 did not need).
    _grant(CHECKPOINT_ROLE, "SELECT")
    _grant(AGENT_SP, "SELECT")
    _grant(REVIEWER_SP, "SELECT, INSERT, UPDATE, DELETE")


def downgrade():
    for role in (CHECKPOINT_ROLE, AGENT_SP, REVIEWER_SP):
        if _role_exists(role):
            op.execute(f'REVOKE ALL ON public.document_chunks FROM "{role}"')
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_docid")
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_tsv")
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_hnsw")
    op.execute("DROP TABLE IF EXISTS public.document_chunks")
