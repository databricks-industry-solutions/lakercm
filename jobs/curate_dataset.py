# Databricks notebook source
# MAGIC %md
# MAGIC # LakeRCM Eval Dataset Curation
# MAGIC
# MAGIC Pulls "interesting" traces (HUMAN feedback, failed scorers, eval.include tag) from the
# MAGIC last N days and merges them into the UC-managed eval dataset.

# COMMAND ----------

# MAGIC %pip install -r ../agent_app/requirements-eval.txt -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` is injected by the Databricks runtime
import os
import sys

dbutils.widgets.text("days", "7", "Look-back window (days)")
dbutils.widgets.text("max_traces", "1000", "Max traces to scan")
dbutils.widgets.dropdown(
    "init", "false", ["false", "true"], "Init dataset (seed smoke set)"
)
dbutils.widgets.dropdown(
    "register_scorers", "false", ["false", "true"], "Register/update production scorers"
)
dbutils.widgets.dropdown(
    "reset_scorers",
    "false",
    ["false", "true"],
    "Reset production scorers (delete-then-recreate; repairs undeserializable registry)",
)

days = int(dbutils.widgets.get("days"))
max_traces = int(dbutils.widgets.get("max_traces"))
init_first = dbutils.widgets.get("init") == "true"
do_register_scorers = dbutils.widgets.get("register_scorers") == "true"
do_reset_scorers = dbutils.widgets.get("reset_scorers") == "true"

# Inject the workspace config the apps get from app.yaml but jobs don't
# (otherwise config.py defaults apply — catalog='main' etc.). Set BEFORE
# importing config/eval, since config.Settings reads env at import.
for _env, _wid in (
    ("DATABRICKS_CATALOG", "catalog"),
    ("LAKERCM_SCHEMA", "schema"),
    ("LLM_ENDPOINT", "llm_endpoint"),
    ("MLFLOW_EXPERIMENT_NAME", "mlflow_experiment"),
):
    dbutils.widgets.text(_wid, "", _wid)
    _val = dbutils.widgets.get(_wid).strip()
    if _val:
        os.environ[_env] = _val

# COMMAND ----------

agent_app = os.path.abspath(os.path.join(os.getcwd(), "..", "agent_app"))
if agent_app not in sys.path:
    sys.path.insert(0, agent_app)
os.chdir(agent_app)

from eval.dataset import curate_from_traces, init_dataset, stats  # noqa: E402

# Repair the production scheduled-scorer registry first if requested — this is
# the documented self-heal for a registry left undeserializable (delete every
# scheduled scorer by name, re-register fresh from production_schedule()).
if do_reset_scorers:
    print("\nResetting production scorer schedule (delete-then-recreate)...")
    from eval.scorers import reset_scorers  # noqa: E402

    reset_scorers()
    print("Scorers reset + re-registered.")

if init_first:
    print("Initializing dataset (seeding smoke set)...")
    init_dataset()

added = curate_from_traces(days=days, max_traces=max_traces)
print(f"Curated {added} interesting trace(s) over the last {days} day(s).")

if do_register_scorers and not do_reset_scorers:
    print("\nRegistering production scorer schedule...")
    from eval.scorers import register_scorers  # noqa: E402

    register_scorers()
    print("Scorers registered.")

print("\nDataset stats:")
for k, v in stats().items():
    print(f"  {k}: {v}")
