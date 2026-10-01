# Databricks notebook source
# MAGIC %md
# MAGIC # Bind an MLflow experiment to its UC OTel trace tables
# MAGIC
# MAGIC One-time, idempotent provisioning. An MLflow 3 GenAI experiment must be
# MAGIC bound to a Unity Catalog trace-storage location for its OTel span/log
# MAGIC tables to materialize on first write. The DAB (`resources/00_mlflow.yml`)
# MAGIC creates the experiment *objects*; this notebook applies the UC binding
# MAGIC plus the load-bearing `mlflow.monitoring.sqlWarehouseId` tag (which
# MAGIC `eval/dataset.py::_ensure_sql_warehouse_for_traces` reads so eval/GEPA can
# MAGIC read its own rollout traces back — without it you get the
# MAGIC "composite 0.0 / 0 paired samples" failure).
# MAGIC
# MAGIC This finally makes real the provisioning notebook the observability README
# MAGIC referenced (`recreate_experiment.py`) but which never existed in git.
# MAGIC
# MAGIC **Idempotent + no-op-safe:** re-running an already-bound experiment skips
# MAGIC the (immutable) bind and only refreshes the warehouse tag. It never raises
# MAGIC — a missing schema/warehouse on an alternate workspace logs + skips rather
# MAGIC than failing the deploy. Driven by `lakercm_bind_experiments`, one task
# MAGIC per experiment (prod reuses `agent_traces_otel_*`; eval gets fresh
# MAGIC `agent_eval_otel_*`).

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.11" databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` is injected by the Databricks runtime
import json

import mlflow

dbutils.widgets.text("experiment_path", "", "Experiment workspace path")
dbutils.widgets.text("catalog", "", "UC catalog for OTel tables")
dbutils.widgets.text("schema", "", "UC schema for OTel tables")
dbutils.widgets.text("table_prefix", "", "OTel table prefix (e.g. agent_traces)")
dbutils.widgets.text("warehouse_id", "", "Monitoring SQL warehouse id")
dbutils.widgets.dropdown(
    "smoke_trace",
    "false",
    ["false", "true"],
    "Write a smoke trace to verify the binding",
)

experiment_path = dbutils.widgets.get("experiment_path").strip()
catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
table_prefix = dbutils.widgets.get("table_prefix").strip()
warehouse_id = dbutils.widgets.get("warehouse_id").strip()
smoke_trace = dbutils.widgets.get("smoke_trace").strip() == "true"

if not experiment_path:
    raise ValueError("experiment_path is required")

mlflow.set_tracking_uri("databricks")
summary: dict = {
    "experiment_path": experiment_path,
    "target": f"{catalog}.{schema}.{table_prefix}_otel_*",
    "bound": False,
    "warehouse_tagged": False,
    "smoke_trace_written": False,
    "note": "",
}

# COMMAND ----------

# UnityCatalog is the trace-location type mlflow.set_experiment(trace_location=...)
# expects (per the Databricks UC-tracing docs). `table_prefix` names the tables
# `<table_prefix>_otel_spans` / `_otel_logs` / `_otel_annotations` — this is how
# the production tables came to be named `agent_traces_otel_*`. (NOTE: the raw
# `UCSchemaLocation` lacks `.table_prefix` and is NOT accepted by set_experiment.)
try:
    from mlflow.entities.trace_location import UnityCatalog
except Exception:  # noqa: BLE001 — fall back across mlflow 3.x module layouts
    from mlflow.tracing.destination import UnityCatalog


def _already_bound(exp) -> bool:
    """True if the experiment already has a UC trace-storage binding.

    The binding is immutable; re-binding would raise. We detect it via the
    `databricksTraceStorageTable` tag the platform writes on bound experiments
    (the same tag the purge script keys off of)."""
    tags = getattr(exp, "tags", None) or {}
    return any(
        k in tags
        for k in (
            "databricksTraceStorageTable",
            "mlflow.experiment.traceStorageLocation",
        )
    )


# COMMAND ----------

# Ensure the experiment exists (DAB should have created it) and resolve it.
exp = mlflow.get_experiment_by_name(experiment_path)
if exp is None:
    summary["note"] = "experiment did not exist — created it (DAB may not have run yet)"
    mlflow.set_experiment(experiment_path)
    exp = mlflow.get_experiment_by_name(experiment_path)

experiment_id = exp.experiment_id
summary["experiment_id"] = experiment_id
print(f"experiment_id={experiment_id} for {experiment_path}")

# COMMAND ----------

# 1) Bind the UC trace location (immutable; skip if already bound).
if _already_bound(exp):
    summary["bound"] = True
    summary["note"] = (
        summary["note"] + "; " if summary["note"] else ""
    ) + "already bound — skipped (immutable)"
    print("Already bound to a UC trace location; skipping bind.")
elif not (catalog and schema and table_prefix):
    summary["note"] = (
        summary["note"] + "; " if summary["note"] else ""
    ) + "catalog/schema/table_prefix missing — skipped bind"
    print("Missing UC target params; skipping bind (no-op-safe).")
else:
    loc = UnityCatalog(
        catalog_name=catalog,
        schema_name=schema,
        table_prefix=table_prefix,
    )
    try:
        # mlflow>=3.11 get-or-bind: attaches the UC trace location to the
        # experiment. (Binding is allowed before the first trace write.)
        mlflow.set_experiment(experiment_name=experiment_path, trace_location=loc)
        summary["bound"] = True
        print(f"Bound {experiment_path} -> {catalog}.{schema}.{table_prefix}_otel_*")
    except TypeError as e:
        # set_experiment lacks trace_location => mlflow<3.11 in this runtime.
        summary["note"] = (
            summary["note"] + "; " if summary["note"] else ""
        ) + f"set_experiment(trace_location=) unsupported: {e}"
        print(f"WARN: {summary['note']}")
    except Exception as e:  # noqa: BLE001 — never fail the deploy
        summary["note"] = (
            summary["note"] + "; " if summary["note"] else ""
        ) + f"bind failed (non-fatal): {type(e).__name__}: {e}"
        print(f"WARN: {summary['note']}")

# COMMAND ----------

# 2) Set the monitoring SQL warehouse tag (mutable; idempotent). Load-bearing for
#    eval read-back via _ensure_sql_warehouse_for_traces.
if warehouse_id:
    try:
        from mlflow.tracking import MlflowClient

        MlflowClient().set_experiment_tag(
            experiment_id, "mlflow.monitoring.sqlWarehouseId", warehouse_id
        )
        summary["warehouse_tagged"] = True
        print(f"Tagged mlflow.monitoring.sqlWarehouseId={warehouse_id}")
    except Exception as e:  # noqa: BLE001 — never fail the deploy
        summary["note"] = (
            summary["note"] + "; " if summary["note"] else ""
        ) + f"warehouse tag failed (non-fatal): {type(e).__name__}: {e}"
        print(f"WARN: {summary['note']}")
else:
    print("No warehouse_id provided; skipping monitoring tag.")

# COMMAND ----------

# 3) Optional smoke trace — write one trivial trace to confirm the binding
#    actually routes to the UC tables (proves the <table_prefix>_otel_* tables
#    materialize). Default off; enabled on the EVAL bind task only — the eval
#    experiment is the right home for synthetic/test traces, so this never
#    pollutes the production experiment.
if smoke_trace:
    try:
        mlflow.set_experiment(experiment_name=experiment_path)

        @mlflow.trace(name="bind_smoke_trace")
        def _smoke():
            return {"ok": True, "experiment": experiment_path}

        _smoke()
        # Flush before the notebook exits — the async trace exporter queues the
        # span and the process would otherwise end before it's written to UC.
        try:
            mlflow.flush_trace_async_logging(terminate=True)
        except Exception:  # noqa: BLE001 — API name varies; sleep is the backstop
            pass
        import time as _t

        _t.sleep(20)  # give Zerobus ingest a moment to materialize the row
        summary["smoke_trace_written"] = True
        print(f"Wrote + flushed bind_smoke_trace to {experiment_path}")
    except Exception as e:  # noqa: BLE001 — never fail the deploy
        summary["note"] = (
            summary["note"] + "; " if summary["note"] else ""
        ) + f"smoke trace failed (non-fatal): {type(e).__name__}: {e}"
        print(f"WARN: {summary['note']}")

# COMMAND ----------

print(json.dumps(summary, indent=2))
dbutils.notebook.exit(json.dumps(summary))
