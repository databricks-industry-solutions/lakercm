# Databricks notebook source
# MAGIC %md
# MAGIC # Generate synthetic claims documents
# MAGIC
# MAGIC Writes realistic, entirely fictional referrals, prior-authorization requests,
# MAGIC denial notices, explanations of benefits, lab reports and progress notes as
# MAGIC PDFs into the documents input volume, where the LakeRCM pipeline reads
# MAGIC them.
# MAGIC
# MAGIC * **Facts come from code** (`docgen.py`): people, clinics and addresses from
# MAGIC   Faker, NPIs that fail the NPI check digit, payers from the fictional payer
# MAGIC   list, and codes from the same reference sets the pipeline validates against.
# MAGIC   About a third of documents carry a deliberate problem (a non-billable or
# MAGIC   malformed code, or a missing member ID) so they land in the review queue.
# MAGIC * **The narrative comes from Claude Opus 5.5** on the Foundation Model APIs. A
# MAGIC   reply that isn't well-formed JSON, or that names a real insurer, is rejected
# MAGIC   and retried.
# MAGIC * **Ground truth is kept.** Each document gets a row in the manifest table, so
# MAGIC   extraction and classification can be scored against what was generated.
# MAGIC
# MAGIC Run it from the `lakercm-generate-synthetic-documents` job, or here with the
# MAGIC widgets below.

# COMMAND ----------

# MAGIC %pip install -q -r ./requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils`, `spark` and `display` come from the Databricks runtime
import collections
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

for _name, _default, _label in (
    ("catalog", "", "Catalog"),
    ("schema", "lakercm", "Schema"),
    ("count", "20", "Documents to generate"),
    ("review_share", "0.3", "Share with a deliberate problem (0-1)"),
    ("model_endpoint", "databricks-claude-opus-5-5", "Model serving endpoint"),
    ("output_path", "", "Output folder (blank: the documents input volume)"),
    (
        "manifest_table",
        "",
        "Manifest table (blank: <catalog>.<schema>.synthetic_document_manifest)",
    ),
    ("pipeline_id", "", "Documents pipeline to start afterwards (blank: none)"),
    ("seed", "", "Seed (blank: a new batch)"),
    ("workers", "4", "Parallel model calls"),
    ("flush_every", "100", "Append the manifest every N documents"),
):
    dbutils.widgets.text(_name, _default, _label)


def arg(name: str) -> str:
    return dbutils.widgets.get(name).strip()


catalog, schema = arg("catalog"), arg("schema")
if not catalog or not schema:
    raise ValueError("Set the catalog and schema widgets.")
count = int(arg("count") or 20)
review_share = float(arg("review_share") or 0.3)
model = arg("model_endpoint") or "databricks-claude-opus-5-5"
output_path = arg("output_path") or f"/Volumes/{catalog}/{schema}/documents_input/"
manifest_table = (
    arg("manifest_table") or f"{catalog}.{schema}.synthetic_document_manifest"
)
pipeline_id = arg("pipeline_id")
seed = int(arg("seed") or time.time())
workers = max(1, int(arg("workers") or 4))
flush_every = max(1, int(arg("flush_every") or 100))

# The notebook runs from synthetic_data/; docgen imports the reference content
# from scripts/.
here = os.getcwd()
for _path in (here, os.path.abspath(os.path.join(here, "..", "scripts"))):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import docgen  # noqa: E402

print(f"Generating {count} documents with {model} (seed {seed}) into {output_path}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient  # noqa: E402

w = WorkspaceClient()
scenarios = docgen.build_scenarios(count, seed=seed, review_share=review_share)
os.makedirs(output_path, exist_ok=True)


def make(scenario):
    narrative = docgen.generate_narrative(w, scenario, model=model)
    path = os.path.join(output_path, docgen.file_name(scenario))
    with open(path, "wb") as fh:
        fh.write(docgen.render_pdf(scenario, narrative))
    return docgen.manifest_row(scenario, path, model)


from pyspark.sql.types import StringType, StructField, StructType  # noqa: E402


def _flush(batch):
    """Append a batch of manifest rows to the table.

    The manifest is written INCREMENTALLY, in batches, not once at the end.
    make() writes each PDF to the volume before its row is returned, so a run
    that is cut short (the job timeout is a hard SIGKILL that no finally-block
    survives) leaves those PDFs on the volume. If the manifest were written only
    at the end, every one of them would be an untracked orphan — exactly what a
    2000-count run produced, and what verify_corpus_integrity then flags.
    Flushing per batch bounds the orphans to at most one unflushed batch instead
    of the whole run, and keeps the manifest consistent with the volume at every
    batch boundary.
    """
    if not batch:
        return
    fields = list(batch[0])
    df = spark.createDataFrame(
        [[r[f] for f in fields] for r in batch],
        StructType([StructField(f, StringType()) for f in fields]),
    )
    df.write.mode("append").option("mergeSchema", "true").saveAsTable(manifest_table)


rows, failures, pending = [], [], []
by_type = collections.Counter()
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(make, s): s for s in scenarios}
    for future in as_completed(futures):
        scenario = futures[future]
        try:
            row = future.result()
            rows.append(row)
            pending.append(row)
            by_type[(row["doc_type"], row["difficulty"])] += 1
            print(f"wrote {docgen.file_name(scenario)} ({scenario.difficulty})")
            if len(pending) >= flush_every:
                _flush(pending)
                print(f"  flushed {len(pending)} manifest rows ({len(rows)} total)")
                pending = []
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failures.append((scenario.doc_id, str(exc)))
            print(f"FAILED {scenario.doc_id}: {exc}")

_flush(pending)  # the remainder

if not rows:
    raise RuntimeError(f"No documents were generated: {failures[:3]}")

print(
    f"{len(rows)} documents written, {len(failures)} failed; manifest: {manifest_table}"
)
for (doc_type, difficulty), n in sorted(by_type.items()):
    print(f"  {doc_type:26s} {difficulty:7s} {n}")

# COMMAND ----------

PIPELINE_DONE = {"COMPLETED", "FAILED", "CANCELED"}


def _await_pipeline(update_id: str, wait_seconds: int = 1800) -> str:
    """Block until the documents pipeline update finishes; return its state."""
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        state = str(
            w.pipelines.get_update(
                pipeline_id=pipeline_id, update_id=update_id
            ).update.state
        ).rsplit(".", 1)[-1]
        if state in PIPELINE_DONE:
            return state
        time.sleep(15)
    return "TIMEOUT"


if pipeline_id:
    try:
        update = w.pipelines.start_update(pipeline_id=pipeline_id)
        print(f"Started documents pipeline update {update.update_id}")
        state = _await_pipeline(update.update_id)
        print(f"  documents pipeline: {state}")
        # Extraction alone does not reach the apps: they read the Lakebase COPY of
        # gold_extraction_labels, which is TRIGGERED on dev. Refreshing it here is
        # what makes a batch show up in the reviewer app and the agent's tools
        # without a manual step. Grants stay the deploy's job
        # (scripts/refresh_forward_sync.py asserts them there), so a copy created
        # for the very first time may need one deploy before the apps can read it.
        if state == "COMPLETED":
            import refresh_forward_sync as fsync

            fsync.refresh(w, fsync.synced_table_id(catalog, schema), wait_seconds=900)
    except SystemExit as exc:
        # refresh() exits on a missing or failed copy. The documents are written
        # either way, so report it instead of failing the run.
        print(f"Lakebase copy not refreshed: {exc}")
    except Exception as exc:  # noqa: BLE001 - an update may already be running
        print(
            f"Pipeline not started ({exc}); it picks the files up on its next update."
        )

display(spark.table(manifest_table))
