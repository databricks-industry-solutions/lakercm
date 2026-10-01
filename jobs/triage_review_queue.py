# Databricks notebook source
# MAGIC %md
# MAGIC # LakeRCM — Triage the review queue
# MAGIC
# MAGIC Takes the first pass over documents the pipeline HELD, so a reviewer
# MAGIC opens a document that already carries the reason it was held and the
# MAGIC shortlist of codes the terminology would accept.
# MAGIC
# MAGIC What it writes is only the SHORTLIST, never a chosen value. Picking a
# MAGIC candidate needs the document read, which is the agent's job through the
# MAGIC reviewer pane (and the reviewer's to approve). This job makes the
# MAGIC deterministic half — what the terminology permits — available before
# MAGIC anyone opens the document.
# MAGIC
# MAGIC * `missing_member_id` is recorded as `declined`, not `pending`. It cannot
# MAGIC   be resolved, and leaving it pending would make the queue look like it
# MAGIC   holds work that is waiting to happen.
# MAGIC * A share of documents (`holdout_rate`) has its proposals computed and
# MAGIC   stored but never shown, so "review got faster" can be measured against
# MAGIC   a control arm instead of asserted. The split is derived from the
# MAGIC   document id, so it is stable across runs.
# MAGIC * Idempotent: a document that already has ANY proposal row is skipped, so
# MAGIC   a re-run never re-proposes over a reviewer's decision.
# MAGIC
# MAGIC Run identity needs INSERT on `public.document_review_proposals` (the
# MAGIC reviewer SP, which owns it) plus read on the gold layer.

# COMMAND ----------

# MAGIC # A LEAN install, not reviewer_app/requirements.txt. That file pins
# MAGIC # fastapi/uvicorn/alembic and OpenTelemetry gRPC exporters whose native
# MAGIC # extensions clash with the serverless runtime's grpc/pyarrow and abort
# MAGIC # the Python process with SIGABRT on import. This job needs none of the
# MAGIC # app/web/otel stack — only Lakebase (psycopg) to write proposals, the SDK
# MAGIC # for the warehouse read, and pydantic-settings for config. Same lean
# MAGIC # approach as jobs/load_document_chunks.py.
# MAGIC %pip install -q "pydantic-settings>=2.0.0" "databricks-sdk>=0.70.0" "psycopg[binary]>=3.1.0" "psycopg-pool>=3.3.0" pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` is injected by the Databricks runtime
import json
import os
import sys

dbutils.widgets.text("max_docs", "500", "Max held documents to triage per run")
dbutils.widgets.text("holdout_rate", "0.2", "Share withheld as the control arm")
dbutils.widgets.dropdown(
    "dry_run", "false", ["false", "true"], "Compute but do not write"
)
dbutils.widgets.text("warehouse_id", "", "SQL warehouse for the gold reads")

# Inject the workspace/data-plane config BEFORE importing services (config reads
# env at import). Same idiom as jobs/load_document_chunks.py.
for _env, _wid in (
    ("DATABRICKS_CATALOG", "catalog"),
    ("LAKERCM_SCHEMA", "schema"),
    ("BUNDLE_VAR_lakebase_pg_host", "lakebase_pg_host"),
    ("LAKEBASE_PROJECT_ID", "lakebase_project_id"),
    ("LAKEBASE_BRANCH_ID", "lakebase_branch_id"),
    ("LAKEBASE_ENDPOINT_ID", "lakebase_endpoint_id"),
    ("DATABRICKS_WAREHOUSE_ID", "warehouse_id"),
):
    dbutils.widgets.text(_wid, "", _wid)
    _val = dbutils.widgets.get(_wid).strip()
    if _val:
        os.environ[_env] = _val

# LakeRCMDatabase (via config) reads PGHOST / PGUSER / PGDATABASE / ENDPOINT_NAME
# by those exact names — not the BUNDLE_VAR_* / LAKEBASE_* the loop above set —
# and an empty PGHOST makes the pool connect to no host and time out.
#
# The connect identity is the RUN identity, resolved from current_user, NOT a
# service principal passed in: a Lakebase OAuth token can only authenticate as
# the principal that minted it (verified — connecting as the reviewer SP with a
# user's token is rejected "OAuth: User is not authorized"), and the run
# identity here (the Lakebase project owner for a laptop-triggered run) holds
# write on public.document_review_proposals. A scheduled run inherits the job
# owner's identity, which must likewise hold write.
from databricks.sdk import WorkspaceClient as _WC  # noqa: E402

_host = dbutils.widgets.get("lakebase_pg_host").strip()
if _host:
    os.environ["PGHOST"] = _host
os.environ.setdefault("PGDATABASE", "databricks_postgres")
_proj = os.environ.get("LAKEBASE_PROJECT_ID", "lakercm")
_branch = os.environ.get("LAKEBASE_BRANCH_ID", "prod")
_endpoint = os.environ.get("LAKEBASE_ENDPOINT_ID", "primary")
os.environ["ENDPOINT_NAME"] = (
    f"projects/{_proj}/branches/{_branch}/endpoints/{_endpoint}"
)
os.environ["PGUSER"] = _WC().current_user.me().user_name

max_docs = max(1, int(dbutils.widgets.get("max_docs") or "500"))
holdout_rate = float(dbutils.widgets.get("holdout_rate") or "0.2")
dry_run = dbutils.widgets.get("dry_run") == "true"

if not 0.0 <= holdout_rate < 1.0:
    # 1.0 would withhold every proposal, leaving nothing for a reviewer to see
    # and no treatment arm to compare the control against.
    raise ValueError(
        f"holdout_rate must be in [0.0, 1.0), got {holdout_rate}. "
        "1.0 would withhold everything and measure nothing."
    )

# COMMAND ----------

reviewer_app = os.path.abspath(os.path.join(os.getcwd(), "..", "reviewer_app"))
if reviewer_app not in sys.path:
    sys.path.insert(0, reviewer_app)
os.chdir(reviewer_app)

from databricks.sdk import WorkspaceClient  # noqa: E402

from config import settings  # noqa: E402
from services.lakehouse_db import LakeRCMDatabase  # noqa: E402
from services.remediation import remediate_document  # noqa: E402
from services import review_proposals as rp  # noqa: E402

workspace_client = WorkspaceClient()
warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID") or settings.get_warehouse_id()
db = LakeRCMDatabase(workspace_client=workspace_client)

print(f"catalog={settings.catalog} schema={settings.lakercm_schema}")
print(f"warehouse={warehouse_id} max_docs={max_docs} holdout={holdout_rate}")
print(f"dry_run={dry_run}")

# COMMAND ----------

# The deterministic inputs: every held document's reasons + flagged codes in one
# read, and the terminology to resolve them against.
held = rp.fetch_held_documents(workspace_client, warehouse_id, limit=max_docs)
terminology = rp.load_terminology(workspace_client, warehouse_id)
print(f"held documents in gold: {len(held)}")
print(f"terminology codes: {len(terminology)}")

# Documents written straight to the volume reach gold without ever touching
# Lakebase; the backfill is what gives them a medical_documents row to hang a
# proposal on. The reviewer app does it on its way to rendering a list, so until
# now this job silently depended on somebody having opened the app first — which
# inverts the point of precomputing proposals for a reviewer who has not arrived
# yet. Idempotent, so on an already-synced corpus it is a no-op.
db.backfill_streamed_documents()

# Resolve exactly the held paths to live Lakebase documents. Keyed by file_path,
# which is what gold calls document_path. Scoped to the held set on purpose — a
# bare "N newest documents" window silently stops overlapping the held set once
# the corpus outgrows it (see lookup_documents_for_triage).
candidates = db.lookup_documents_for_triage([path for path, _, _ in held])
by_path = {(row.get("file_path") or ""): row for row in candidates}
awaiting = sum(1 for row in candidates if not row.get("has_proposal"))
print(f"held documents resolved in Lakebase: {len(by_path)} of {len(held)}")
print(f"documents with no proposal yet: {awaiting}")

# COMMAND ----------

stats = {
    "held_documents": len(held),
    "triaged": 0,
    "skipped_already_triaged": 0,
    "skipped_not_in_lakebase": 0,
    "proposals_written": 0,
    "withheld": 0,
    "declined": 0,
    "by_reason": {},
    "by_resolution": {},
}

for document_path, reasons, codes in held:
    row = by_path.get(document_path)
    if row is None:
        # The gold row has no live Lakebase document (never synced, or deleted).
        # Counted apart from "already triaged" so a systematic path mismatch is
        # visible rather than looking like "everything is already done" — both
        # cases used to increment skipped_already_triaged, which is exactly how
        # a zero-proposal run got mistaken for an idempotent no-op.
        stats["skipped_not_in_lakebase"] += 1
        continue
    if row.get("has_proposal"):
        # Already has a proposal (pending or dispositioned): re-running the job
        # must not re-propose over a reviewer's work.
        stats["skipped_already_triaged"] += 1
        continue
    document_id = row.get("document_id")

    remediations = remediate_document(reasons, codes, terminology)
    if not remediations:
        # Held, but nothing the terminology can speak to.
        continue

    for rem in remediations:
        stats["by_reason"][rem.review_reason] = (
            stats["by_reason"].get(rem.review_reason, 0) + 1
        )
        stats["by_resolution"][rem.resolution] = (
            stats["by_resolution"].get(rem.resolution, 0) + 1
        )
        if rem.resolution == "not_resolvable":
            stats["declined"] += 1

    if dry_run:
        stats["triaged"] += 1
        stats["proposals_written"] += len(remediations)
        continue

    written = rp.record_remediations(
        db,
        str(document_id),
        remediations,
        source=rp.SOURCE_TRIAGE,
        holdout_rate=holdout_rate,
    )
    stats["triaged"] += 1
    stats["proposals_written"] += len(written)
    stats["withheld"] += sum(1 for row in written if row.get("withheld"))

print(json.dumps(stats, indent=2))

# COMMAND ----------

# A run that triaged nothing is reported, not failed: on a corpus where every
# held document has already been triaged that is the correct outcome, and this
# job is scheduled.
summary = (
    f"triaged {stats['triaged']} of {stats['held_documents']} held documents; "
    f"{stats['proposals_written']} proposals "
    f"({stats['withheld']} withheld, {stats['declined']} declined)"
)
print(summary)
dbutils.notebook.exit(json.dumps({"summary": summary, **stats}))
