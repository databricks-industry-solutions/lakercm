"""adopt Lakebase Search on document_chunks (lakebase_ann + lakebase_bm25)

Revision ID: 20260929_000035
Revises: 20260929_000034
Create Date: 2026-09-29 00:00:35.000000

Upgrades the chunk index from pgvector HNSW + Postgres FTS to Lakebase Search.
Verified available on both Lakebase projects (2026-09-29): lakebase_vector 1.1.1,
lakebase_text 0.1.3, lakebase_tokenizer 0.1.1.

WHY THIS IS DEFENSIVE, NOT UNCONDITIONAL:
    Enabling Lakebase Search is a per-project UI toggle (project → Settings). It
    is irreversible, it restarts every compute in the project, and there is NO
    DAB field and NO API/CLI for it — the whole `databricks postgres` surface and
    the DAB schema were both checked. So a project can legitimately not have it,
    and this migration runs at reviewer-app boot on EVERY target.

    An unguarded CREATE EXTENSION on a project without the toggle aborts the
    migration transaction, which means `alembic upgrade head` fails and the
    reviewer app boots with no schema. That is why availability is probed first
    and the whole upgrade no-ops when the extensions are absent, leaving the
    000034 HNSW + GIN indexes in place. The agent detects which keyword backend
    is live at runtime, so both states are fully supported.

BOTH ARMS ADOPT LAKEBASE SEARCH. Lakebase Search went GA on 2026-09-18 (AWS and
Azure; still Beta on GCP), and the build semantics of the two indexes are
OPPOSITE — which is the whole reason this migration looks the way it does:

  * `lakebase_ann` (vector) — replaces 000034's HNSW index. It CAN be created on
    an empty table: unlike pgvector's ivfflat it does not need representative
    rows at build time, and it reflects inserts immediately, so it never needs a
    REINDEX. Creating it here, at reviewer-app boot before any chunk exists, is
    therefore correct.

    The win is NOT throughput at our corpus size — it is cold start. The index is
    storage-backed and survives scale-to-zero with no warmup (~1s cold start).
    dev's Lakebase project suspends after 5 minutes idle, so an HNSW index would
    be re-warmed constantly; lakebase_ann is unaffected. It also gets 32x
    compression from RaBitQ quantization. (The headline numbers — 1B+ vectors,
    50-100x faster builds, 97% recall at 71ms P99 on 100M LAION — are upside we
    do not need yet.) Same `<=>` operator and vector type, so no application
    change: it is a documented drop-in for pgvector.

  * `lakebase_bm25` (keyword) — the opposite. It computes corpus statistics (term
    frequency, document length, IDF) at BUILD time, NOT incrementally, so the
    documented order is to create it AFTER inserting data. This migration
    necessarily runs before any data exists, so the index created here starts with
    empty statistics.

    ⚠ THAT IS MADE CORRECT BY THE LOADER, NOT BY THIS MIGRATION:
    jobs/load_document_chunks.py ends every run that wrote chunks with
    `REINDEX INDEX CONCURRENTLY` on this index, so the first load rebuilds the
    statistics over real rows and every subsequent load keeps them current. That
    REINDEX is load-bearing, not a nicety — without it BM25 ranking is scored
    against an empty corpus and keyword relevance is silently wrong. (The docs
    suggest a nightly reindex for moderate write volumes; per-load is strictly
    fresher and cheap at this size.)

  * `idx_document_chunks_tsv` (GIN) is **KEPT** too. BM25's `<@>` operator
    replaces the *ranking* function but NOT the `@@` relevance predicate, which
    the agent keeps so a query with no real matches returns nothing rather than
    BM25-ranked noise. That predicate still needs GIN. The docs sanction running
    lakebase_bm25 "alongside" it.

Precedent: the FE `contextual-memory-lakebase-agent` project made the same call
in its ADR-0002 ("Lakebase Search over plain pgvector"), for exact-token
precision plus faster builds and no cold-start warmup.

lakebase_tokenizer is installed when available but NOT yet used. It is only
DOCUMENTED on GCP, yet it reads as available on our AWS projects — hence the
availability guard rather than an unconditional CREATE. Configurable whole-word
tokenization is the right lever for ICD-10 / CPT codes and member IDs in the
keyword arm, but no config is published for medical codes and choosing one needs
measurement against real queries, not a guess in a migration.
"""

from alembic import op

# revision identifiers, used by Alembic
revision = "20260929_000035"
down_revision = "20260929_000034"
branch_labels = None
depends_on = None

ANN_INDEX = "idx_document_chunks_ann"
BM25_INDEX = "idx_document_chunks_bm25"


def _available(ext: str) -> bool:
    """Cheap fast path only — NOT a safety check. See _try()."""
    bind = op.get_bind()
    res = bind.exec_driver_sql(
        "SELECT 1 FROM pg_available_extensions WHERE name = %(n)s", {"n": ext}
    )
    return res.scalar() is not None


def _try(sql: str, label: str) -> bool:
    """Run sql inside a SAVEPOINT. True on success; on failure roll back to the
    savepoint and return False, leaving the migration's transaction USABLE.

    This is the actual safety mechanism, and it exists because
    pg_available_extensions is not a reliable test for whether Lakebase Search is
    ENABLED on the project: the extensions may well be listed as available on a
    project whose (UI-only, irreversible) toggle has never been flipped. Were the
    availability probe the only guard, such a project would pass it, then abort
    the whole transaction on CREATE EXTENSION — and because migrations run at
    reviewer-app boot, the app would come up with NO SCHEMA AT ALL.

    A savepoint turns every step below into "adopt it if it works, otherwise keep
    what 000034 built", which is the behaviour this migration needs on every
    target.
    """
    bind = op.get_bind()
    try:
        with bind.begin_nested():
            bind.exec_driver_sql(sql)
        return True
    except Exception as e:
        print(f"[migration 20260929_000035] {label} unavailable, skipping: {e}")
        return False


def upgrade():
    keep_msg = (
        "[migration 20260929_000035] Lakebase Search not adopted — keeping the "
        "pgvector HNSW + GIN indexes from 000034. The agent detects the live "
        "keyword backend at runtime, so this is a supported state, not a broken "
        "one. To adopt: enable it in the Lakebase UI (project > Settings > "
        "Lakebase Search; one-way, restarts all computes) and redeploy."
    )

    if not (_available("lakebase_vector") and _available("lakebase_text")):
        print(keep_msg)
        return

    # CASCADE pulls in pgvector, already installed by 000012 — harmless.
    if not _try(
        "CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE", "lakebase_vector"
    ):
        print(keep_msg)
        return
    if not _try("CREATE EXTENSION IF NOT EXISTS lakebase_text", "lakebase_text"):
        print(keep_msg)
        return

    # Optional, and only DOCUMENTED on GCP even though it reads as available on
    # AWS. Unused for now either way, so a failure here is inconsequential.
    if _available("lakebase_tokenizer"):
        _try("CREATE SCHEMA IF NOT EXISTS tokenizer_ext", "tokenizer_ext schema")
        _try(
            "CREATE EXTENSION IF NOT EXISTS lakebase_tokenizer "
            "WITH SCHEMA tokenizer_ext",
            "lakebase_tokenizer",
        )

    # Create the replacements before dropping anything, so a failure leaves a
    # fully-indexed table rather than an unindexed one.
    ann_ok = _try(
        f"CREATE INDEX IF NOT EXISTS {ANN_INDEX} "
        "ON public.document_chunks USING lakebase_ann (embedding vector_cosine_ops)",
        ANN_INDEX,
    )
    # If this fails on the GENERATED content_tsv column, the fallback is to drop
    # GENERATED from 000034 and have the loader write to_tsvector() explicitly, as
    # the docs' example does. The agent stays on ts_rank meanwhile.
    _try(
        f"CREATE INDEX IF NOT EXISTS {BM25_INDEX} "
        "ON public.document_chunks USING lakebase_bm25 (content_tsv)",
        BM25_INDEX,
    )

    # Drop HNSW ONLY once lakebase_ann exists. Same column, same ops class, same
    # `<=>` operator, so keeping both would just double write cost — but dropping
    # it after a failed ANN create would leave the vector arm with no index at all.
    # GIN is never dropped: it still backs the ts_rank fallback's `@@` predicate.
    if ann_ok:
        op.execute("DROP INDEX IF EXISTS idx_document_chunks_hnsw")


def downgrade():
    # Restore HNSW first so the vector arm is never left without an index.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_hnsw "
        "ON public.document_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(f"DROP INDEX IF EXISTS {BM25_INDEX}")
    op.execute(f"DROP INDEX IF EXISTS {ANN_INDEX}")
    # Extensions are left installed: dropping lakebase_vector would cascade to
    # pgvector and take the embedding column with it.
