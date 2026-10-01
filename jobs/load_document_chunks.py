# Databricks notebook source
# MAGIC %md
# MAGIC # LakeRCM — Load retrieval chunks into Lakebase pgvector
# MAGIC
# MAGIC Populates `public.document_chunks` (Lakebase) so the agent can answer
# MAGIC "what does THIS document say" and rank documents by their best-matching
# MAGIC passage — hybrid vector + keyword search, **entirely in Lakebase**.
# MAGIC
# MAGIC Source is the pipeline's third parallel branch,
# MAGIC `<catalog>.<schema>.silver_doc_chunks` (`ai_prep_search` off
# MAGIC `bronze_doc_parsed`), which produces chunk TEXT only — `ai_prep_search`
# MAGIC does not embed.
# MAGIC
# MAGIC What it does:
# MAGIC   1. Read each document's already-loaded prep time from Lakebase.
# MAGIC   2. Spark-read `silver_doc_chunks` and pick documents whose prep time is
# MAGIC      newer (or absent) — an anti-join per document, NOT a global
# MAGIC      high-water mark. A global mark would skip a document that was
# MAGIC      prepped earlier but not yet committed when a previous run died.
# MAGIC   3. Embed `chunk_to_embed` via the shared FM helper
# MAGIC      (databricks-gte-large-en, 1024-dim).
# MAGIC   4. Replace each document's chunk SET transactionally (DELETE + INSERT).
# MAGIC      A re-parse renumbers chunk_ids, so a row-wise upsert would leave
# MAGIC      stale chunks behind.
# MAGIC
# MAGIC The text here is RAW parsed document content — that is
# MAGIC the point of this corpus. Access is governed by the Postgres grants, like
# MAGIC every other document table: LakeRCM is a shared review queue, so no search
# MAGIC tool filters by `user_email`.
# MAGIC
# MAGIC Run identity must hold INSERT/UPDATE/**DELETE** on public.document_chunks
# MAGIC (the reviewer SP, granted by migration 000034).
# MAGIC
# MAGIC Triggered by the `load_chunks` task of `lakercm_documents_refresh`, which
# MAGIC rides that job's file-arrival trigger — there is deliberately no cron.

# COMMAND ----------

# MAGIC %pip install -r ../agent_app/requirements-eval.txt -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` and `spark` are injected by the Databricks runtime
import os
import sys

dbutils.widgets.text("batch_size", "128", "Embedding batch size")
dbutils.widgets.text("max_docs", "5000", "Max documents to (re)load per run")
dbutils.widgets.dropdown(
    "full_refresh", "false", ["false", "true"], "Reload every document"
)
# Optional. Blank (the default, and what the jobs bundle now passes) means "connect
# as whoever this task runs as", which is the only identity the Lakebase credential
# is valid for. A value here is NOT a free choice of role: Postgres still requires
# the run-as principal to be a MEMBER of it, so naming an unrelated role fails with
# "OAuth: User is not authorized" rather than dropping privilege.
dbutils.widgets.text("pg_user", "", "Lakebase role for writes (blank = caller)")

# Inject the workspace/data-plane config (same idiom as jobs/triage_review_queue.py).
# Set BEFORE importing services (config/embeddings read env at import).
for _env, _wid in (
    ("DATABRICKS_CATALOG", "catalog"),
    ("LAKERCM_SCHEMA", "schema"),
    ("LAKERCM_EMBEDDING_ENDPOINT", "embedding_endpoint"),
    ("BUNDLE_VAR_lakebase_pg_host", "lakebase_pg_host"),
    ("LAKEBASE_PROJECT_ID", "lakebase_project_id"),
    ("LAKEBASE_BRANCH_ID", "lakebase_branch_id"),
    ("LAKEBASE_ENDPOINT_ID", "lakebase_endpoint_id"),
):
    dbutils.widgets.text(_wid, "", _wid)
    _val = dbutils.widgets.get(_wid).strip()
    if _val:
        os.environ[_env] = _val

batch_size = max(1, int(dbutils.widgets.get("batch_size") or "128"))
max_docs = max(1, int(dbutils.widgets.get("max_docs") or "5000"))
full_refresh = dbutils.widgets.get("full_refresh") == "true"
pg_user_override = dbutils.widgets.get("pg_user").strip()

CATALOG = os.environ.get("DATABRICKS_CATALOG", "")
SCHEMA = os.environ.get("LAKERCM_SCHEMA", "lakercm")
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.silver_doc_chunks"

# COMMAND ----------

agent_app = os.path.abspath(os.path.join(os.getcwd(), "..", "agent_app"))
if agent_app not in sys.path:
    sys.path.insert(0, agent_app)
os.chdir(agent_app)

import psycopg  # noqa: E402
from databricks.sdk import WorkspaceClient  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from services.embeddings import EMBEDDING_MODEL, embed_texts  # noqa: E402

PROJECT = os.environ.get("LAKEBASE_PROJECT_ID", "lakercm")
BRANCH = os.environ.get("LAKEBASE_BRANCH_ID", "prod")
ENDPOINT = os.environ.get("LAKEBASE_ENDPOINT_ID", "primary")
ENDPOINT_NAME = f"projects/{PROJECT}/branches/{BRANCH}/endpoints/{ENDPOINT}"


def _pgvector_literal(vec):
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _is_zero_vector(vec) -> bool:
    """True when the embed call failed.

    services.embeddings.embed_texts never raises — on any error it returns
    zero-vectors of the right shape. Persisting one would mark the document
    loaded at its current prep time, and the per-document anti-join would then
    never retry it. So a zero vector aborts its whole document instead.
    """
    return not any(float(x) != 0.0 for x in vec)


# COMMAND ----------

w = WorkspaceClient()
cred = w.postgres.generate_database_credential(endpoint=ENDPOINT_NAME)


def _connect_identity() -> str:
    """The Postgres role this task can actually authenticate as.

    generate_database_credential() above mints a token for the identity THIS code
    is authenticated as -- the job's run-as principal on a job, the developer on a
    laptop. Postgres then requires the login role to BE that identity (or a role it
    is a member of), so the only safe default is to ask who we are rather than to
    guess.

    Both previous defaults were wrong, and each failed differently:

      * The `pg_user` task parameter named the REVIEWER SP, chosen because
        migration 000034 grants it the INSERT/UPDATE/DELETE this loader needs. But
        the job runs as the DEPLOYER SP, which is not a member of the reviewer
        role, so the credential was rejected outright:
            OAuth: User is not authorized
        Least privilege cannot be expressed by naming a role the caller cannot
        assume; it needs either role membership or a different run-as.

      * Falling back to DATABRICKS_CLIENT_ID looked right but is not set to the
        run-as principal on serverless -- it resolves to an ephemeral Spark
        identity that has no Postgres role at all:
            password authentication failed for user 'spark-d20c4e77-...'

    current_user.me() returns the authenticated principal itself, which is exactly
    who the token belongs to. An explicit override is still honoured for the case
    it was meant for -- a deliberate role the caller IS a member of -- but it is no
    longer the default, and a mismatch is now reported instead of being discovered
    as a connection error.
    """
    me = ""
    try:
        me = (w.current_user.me().user_name or "").strip()
    except Exception as e:  # noqa: BLE001 - fall through to the override/env
        print(f"  WARNING: could not resolve the current identity: {e}")

    if pg_user_override:
        if me and pg_user_override != me:
            print(
                f"  WARNING: pg_user={pg_user_override!r} differs from this task's "
                f"identity {me!r}. The database credential belongs to the latter, so "
                "this only works if that identity is a MEMBER of the former; "
                "otherwise expect 'OAuth: User is not authorized'."
            )
        return pg_user_override

    if me:
        return me
    # Last resort. Known to be an ephemeral Spark identity on serverless, so it is
    # ordered last rather than removed -- it is still correct on runtimes that set
    # it to the run-as principal.
    return os.environ.get("DATABRICKS_CLIENT_ID") or ""


pg_user = _connect_identity()
print(f"  connecting to Lakebase as {pg_user!r}")
conn = psycopg.connect(
    host=os.environ.get("BUNDLE_VAR_lakebase_pg_host") or os.environ.get("PGHOST", ""),
    port=5432,
    dbname="databricks_postgres",
    user=pg_user,
    password=cred.token,
    sslmode="require",
)
conn.autocommit = False

# Per-document prep time already in Lakebase, as epoch seconds. Comparing epoch
# ints sidesteps tz/precision mismatches between a Postgres TIMESTAMPTZ and a
# Spark timestamp, which an equality test on datetimes would trip over.
loaded: dict[str, int] = {}
if not full_refresh:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT document_path,
                   FLOOR(EXTRACT(EPOCH FROM MAX(prepped_at)))::BIGINT
            FROM public.document_chunks
            GROUP BY document_path
            """)
        loaded = {r[0]: int(r[1]) for r in cur.fetchall() if r[1] is not None}

# autocommit is off, so the read above opened a transaction. Close it before the
# Spark collect below, which can run for minutes — otherwise the connection sits
# idle-in-transaction and holds back autovacuum on the chunk table.
conn.commit()

print(f"{len(loaded)} document(s) already loaded (full_refresh={full_refresh}).")

# COMMAND ----------

# This task runs on `run_if: ALL_DONE`, so it also runs when the documents
# pipeline FAILED — and on a first deploy, or a workspace where the
# ai_prep_search preview is off, silver_doc_chunks may not exist at all. Without
# this guard spark.table() raises, the task fails, retries twice and emails, on
# every single file arrival. A missing source is a no-op, not an error.
if not spark.catalog.tableExists(CHUNKS_TABLE):
    print(
        f"{CHUNKS_TABLE} does not exist yet — nothing to load. This is expected "
        "before the documents pipeline has created it (first deploy, or the "
        "ai_prep_search preview not enabled). Exiting cleanly."
    )
    dbutils.notebook.exit("no-source-table")

src = spark.table(CHUNKS_TABLE)

doc_times = (
    src.groupBy("document_path")
    .agg(F.max("prepped_at").alias("prepped_at"))
    .select("document_path", F.unix_timestamp("prepped_at").alias("prepped_epoch"))
    .collect()
)

# Per-document target generation. silver_doc_chunks is an APPEND-only streaming
# table, so a re-parse of the same document_path appends a second set of chunks
# alongside the first. Selecting rows by path alone would insert the union of
# both generations, and since each row carries its own prepped_at the MAX would
# then look current — leaving that document permanently mixed. So carry the
# target epoch and keep only rows that match it.
want_epoch: dict[str, int] = {
    r["document_path"]: int(r["prepped_epoch"])
    for r in doc_times
    if r["prepped_epoch"] is not None
}

stale_paths = [p for p, epoch in want_epoch.items() if loaded.get(p, -1) < epoch][
    :max_docs
]

print(f"{len(stale_paths)} document(s) need (re)loading.")

# COMMAND ----------

insert_sql = """
    INSERT INTO public.document_chunks
        (document_path, chunk_id, document_id, document_name, user_email,
         chunk_position, chunk_to_retrieve, chunk_to_embed, embedding,
         page_id, image_uri, embedding_model, prepped_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, NOW())
    ON CONFLICT (document_path, chunk_id) DO UPDATE SET
        document_id       = EXCLUDED.document_id,
        document_name     = EXCLUDED.document_name,
        user_email        = EXCLUDED.user_email,
        chunk_position    = EXCLUDED.chunk_position,
        chunk_to_retrieve = EXCLUDED.chunk_to_retrieve,
        chunk_to_embed    = EXCLUDED.chunk_to_embed,
        embedding         = EXCLUDED.embedding,
        page_id           = EXCLUDED.page_id,
        image_uri         = EXCLUDED.image_uri,
        embedding_model   = EXCLUDED.embedding_model,
        prepped_at        = EXCLUDED.prepped_at,
        updated_at        = NOW()
"""

# document_id links a chunk to the reviewer app's own row so the UI can deep-link,
# so the reviewer pane can deep-link a cited chunk.
doc_ids: dict[str, str] = {}
with conn.cursor() as cur:
    cur.execute("SELECT file_path, id FROM public.medical_documents")
    doc_ids = {r[0]: r[1] for r in cur.fetchall()}

documents_written = 0
chunks_written = 0
documents_skipped = 0

# Pull chunks a slice of documents at a time so the IN list and the driver-side
# collect both stay bounded on a large backlog.
DOC_SLICE = 50

for slice_start in range(0, len(stale_paths), DOC_SLICE):
    batch_paths = stale_paths[slice_start : slice_start + DOC_SLICE]
    rows = src.filter(F.col("document_path").isin(batch_paths)).collect()

    # Keep only the generation we intend to load (see want_epoch above).
    by_doc: dict[str, list] = {}
    dropped_old = 0
    for r in rows:
        path = r["document_path"]
        ts = r["prepped_at"]
        if ts is None or int(ts.timestamp()) != want_epoch.get(path):
            dropped_old += 1
            continue
        by_doc.setdefault(path, []).append(r)
    if dropped_old:
        print(f"  skipped {dropped_old} chunk row(s) from superseded prep generations")

    for path, chunk_rows in by_doc.items():
        texts = [(r["chunk_to_embed"] or "") for r in chunk_rows]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(embed_texts(texts[start : start + batch_size]))

        if any(_is_zero_vector(v) for v in vectors):
            # Leave the document unloaded so the next run retries it, rather
            # than committing a document whose prep time says it is current.
            print(f"  SKIP {path}: embedding failed for at least one chunk")
            documents_skipped += 1
            continue

        try:
            with conn.cursor() as cur:
                # Replace the SET: a re-parse renumbers chunk_ids, so deleting
                # first is what prevents orphans. The ON CONFLICT above is a
                # belt-and-braces guard for duplicate chunk_ids within one prep.
                cur.execute(
                    "DELETE FROM public.document_chunks WHERE document_path = %s",
                    (path,),
                )
                for r, text, vec in zip(chunk_rows, texts, vectors):
                    cur.execute(
                        insert_sql,
                        (
                            path,
                            r["chunk_id"],
                            doc_ids.get(path),
                            r["document_name"],
                            r["user_email"],
                            r["chunk_position"],
                            r["chunk_to_retrieve"],
                            text,
                            _pgvector_literal(vec),
                            r["page_id"],
                            r["image_uri"],
                            EMBEDDING_MODEL,
                            r["prepped_at"],
                        ),
                    )
                    chunks_written += 1
            conn.commit()
            documents_written += 1
        except Exception as e:
            conn.rollback()
            print(f"  FAILED {path}: {e}")
            documents_skipped += 1

    print(f"  {documents_written}/{len(stale_paths)} document(s) loaded")

# Backfill document_id for any chunk whose medical_documents row did not exist
# when it was loaded. A bulk/external drop is parsed as soon as it lands, which
# can precede the reviewer app registering the document — and since prepped_at
# never changes again, the anti-join would leave that link null forever, breaking
# the deep-link the search tools' citations rely on. Idempotent and cheap.
with conn.cursor() as cur:
    cur.execute("""
        UPDATE public.document_chunks c
        SET document_id = d.id, updated_at = NOW()
        FROM public.medical_documents d
        WHERE c.document_id IS NULL AND d.file_path = c.document_path
        """)
    backfilled = cur.rowcount
conn.commit()
if backfilled:
    print(f"Backfilled document_id on {backfilled} previously-unlinked chunk(s).")

# Rebuild the Lakebase Search BM25 index when this run actually added chunks.
#
# THIS IS LOAD-BEARING, not an optimization. lakebase_bm25 computes its corpus
# statistics (term frequency, document length, IDF) at BUILD time, NOT
# incrementally, and the documented order is to create the index AFTER inserting
# data. Migration 000035 necessarily runs at reviewer-app boot, before any chunk
# exists, so that index starts with statistics over an empty corpus. This REINDEX
# is what makes the first load correct and every later load current — without it
# BM25 ranks against nothing and keyword relevance is silently wrong. (The docs
# suggest a nightly reindex at moderate write volumes; per-load is strictly
# fresher and cheap at this corpus size.)
#
# lakebase_ann needs no equivalent: it reflects inserts immediately.
#
# Skipped entirely when Lakebase Search is not enabled on the project (000035
# then created no BM25 index and the agent stays on Postgres FTS).
BM25_INDEX = "idx_document_chunks_bm25"
with conn.cursor() as cur:
    cur.execute(
        "SELECT 1 FROM pg_class WHERE relname = %s AND relkind = 'i'", (BM25_INDEX,)
    )
    has_bm25 = cur.fetchone() is not None
conn.commit()

if has_bm25 and chunks_written:
    # CONCURRENTLY so the rebuild does not take an ACCESS EXCLUSIVE lock on an
    # index the agent is reading live; it cannot run inside a transaction block.
    #
    # Whether lakebase_bm25 (a custom index access method) supports the
    # CONCURRENTLY option is NOT documented anywhere. So fall back to a plain
    # REINDEX rather than giving up: this rebuild is what makes BM25 ranking
    # correct at all, so skipping it silently is much worse than a brief lock on
    # one index of a demo-sized table.
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"REINDEX INDEX CONCURRENTLY public.{BM25_INDEX}")
        print(f"Rebuilt {BM25_INDEX} concurrently (BM25 stats are build-time only).")
    except Exception as e:
        print(f"  CONCURRENTLY reindex failed ({e}); retrying as a plain REINDEX")
        try:
            with conn.cursor() as cur:
                cur.execute(f"REINDEX INDEX public.{BM25_INDEX}")
            print(f"Rebuilt {BM25_INDEX} (plain REINDEX, brief lock).")
        except Exception as e2:
            # Only now give up. Stale statistics degrade ranking; they do not
            # corrupt results, and the next load retries. Never fail the load.
            print(f"  WARNING: BM25 reindex failed, ranking may be stale: {e2}")
    finally:
        conn.autocommit = False
elif not has_bm25:
    print("Lakebase Search BM25 index absent — agent uses Postgres FTS. No reindex.")

conn.close()
print(
    f"\nDone. Loaded {chunks_written} chunk(s) across {documents_written} "
    f"document(s) into public.document_chunks; {documents_skipped} skipped."
)
