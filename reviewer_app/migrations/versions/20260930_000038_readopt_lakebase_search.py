"""re-attempt Lakebase Search adoption after the project toggle was enabled

Revision ID: 20260930_000038
Revises: 20260930_000037
Create Date: 2026-09-30 00:00:38.000000

000035 adopts Lakebase Search, and it is written to no-op safely when the
project's (UI-only) toggle is off — otherwise CREATE EXTENSION would abort the
migration transaction and the reviewer app would boot with no schema at all.
That defensive design is right, and it has one consequence nobody gets a warning
about:

    THE NO-OP IS RECORDED AS SUCCESS. Once alembic stamps 20260929_000035,
    enabling the toggle later changes nothing. The extensions are never
    installed, idx_document_chunks_ann and idx_document_chunks_bm25 are never
    created, and the app runs on the 000034 pgvector HNSW + GIN fallback
    indefinitely. Nothing reports a problem, because the fallback is a supported
    state -- the agent detects the live keyword backend at runtime.

Verified on dev 2026-09-30, which is exactly this state: alembic head was already
past 000035; pg_extension held only vector 0.8.0; lakebase_vector / lakebase_text
/ lakebase_tokenizer all read AVAILABLE with installed_version NULL; and
document_chunks carried only idx_document_chunks_hnsw (29 MB) and
idx_document_chunks_tsv (6.5 MB). A CREATE EXTENSION probe in a rolled-back
transaction succeeded for all three, which is the only reliable test that the
toggle is on -- pg_available_extensions lists them either way, as 000035's own
docstring warns.

So this migration re-runs the adoption. It is the same sequence as 000035, with
the same savepoint guards, so it is equally safe on a target whose toggle is
still off (prod, today): it prints and returns, leaving 000034's indexes alone.

ONE THING IS BETTER HERE THAN IN 000035. That migration necessarily ran before
any chunk existed, so the BM25 index it created started with statistics over an
empty corpus -- correct only because jobs/load_document_chunks.py REINDEXes it
after every load. This one runs against a POPULATED table, so
`CREATE INDEX ... USING lakebase_bm25` computes real term frequencies and IDF
immediately. The loader's REINDEX still keeps them current; it is simply no
longer load-bearing for the very first query.

Idempotent: CREATE EXTENSION IF NOT EXISTS / CREATE INDEX IF NOT EXISTS
throughout, so a target that already adopted via 000035 passes through untouched.
"""

from alembic import op

# revision identifiers, used by Alembic
revision = "20260930_000038"
down_revision = "20260930_000037"
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

    Identical to 000035's, and load-bearing for the same reason: these migrations
    run at reviewer-app boot, so an aborted transaction means the app comes up
    with no schema. pg_available_extensions is not a reliable test for whether
    the toggle is on, so the savepoint is what actually makes this safe.
    """
    bind = op.get_bind()
    try:
        with bind.begin_nested():
            bind.exec_driver_sql(sql)
        return True
    except Exception as e:
        print(f"[migration 20260930_000038] {label} unavailable, skipping: {e}")
        return False


def upgrade():
    keep_msg = (
        "[migration 20260930_000038] Lakebase Search still not available — "
        "keeping the pgvector HNSW + GIN indexes. This is a supported state, not "
        "a broken one. To adopt: enable it in the Lakebase UI (project > "
        "Settings > Lakebase Search; one-way, restarts all computes) and "
        "redeploy — this migration is idempotent and a later revision can "
        "re-attempt it the same way."
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
    _try(
        f"CREATE INDEX IF NOT EXISTS {BM25_INDEX} "
        "ON public.document_chunks USING lakebase_bm25 (content_tsv)",
        BM25_INDEX,
    )

    # Drop HNSW ONLY once lakebase_ann exists. Same column, same ops class, same
    # `<=>` operator, so keeping both just doubles write cost — but dropping it
    # after a failed ANN create would leave the vector arm with no index at all.
    # GIN is never dropped: it still backs the ts_rank fallback's `@@` predicate,
    # which BM25's `<@>` ranking operator does not replace.
    if ann_ok:
        op.execute("DROP INDEX IF EXISTS idx_document_chunks_hnsw")
        print(
            "[migration 20260930_000038] adopted Lakebase Search: "
            f"{ANN_INDEX} created, HNSW dropped."
        )


def downgrade():
    # Mirror of 000035's downgrade: restore the pgvector arm, then drop the
    # Lakebase Search indexes. The extensions are left installed — dropping them
    # is project-wide and not this revision's business.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_hnsw "
        "ON public.document_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(f"DROP INDEX IF EXISTS {BM25_INDEX}")
    op.execute(f"DROP INDEX IF EXISTS {ANN_INDEX}")
