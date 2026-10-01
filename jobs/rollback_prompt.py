# Databricks notebook source
# MAGIC %md
# MAGIC # LakeRCM Prompt Rollback
# MAGIC
# MAGIC Re-points the `champion` alias to a prior prompt version — the one-command
# MAGIC revert for a bad promotion. MLflow prompt versions are immutable, so the
# MAGIC prior version is always recoverable; this is a single alias swap, logged as
# MAGIC an auditable MLflow run tagged with the reason.
# MAGIC
# MAGIC Provide either an explicit `to_version`, or leave it blank with
# MAGIC `previous=true` to revert to the version immediately before the current
# MAGIC champion. Also callable programmatically by the auto-rollback monitor.

# COMMAND ----------

# MAGIC %pip install -r ../agent_app/requirements-eval.txt -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` is injected by the Databricks runtime
import os
import sys

dbutils.widgets.text("to_version", "", "Target version (blank → use previous)")
dbutils.widgets.dropdown(
    "previous", "false", ["false", "true"], "Revert to version before current"
)
dbutils.widgets.text("reason", "manual_revert", "Rollback reason (audit tag)")

to_version_raw = dbutils.widgets.get("to_version").strip()
use_previous = dbutils.widgets.get("previous") == "true"
reason = dbutils.widgets.get("reason") or "manual_revert"

# Inject the workspace config the apps get from app.yaml but jobs don't
# (otherwise config.py defaults apply — catalog='main' etc.). Set BEFORE
# importing config/eval, since config.Settings reads env at import.
for _env, _wid in (
    ("DATABRICKS_CATALOG", "catalog"),
    ("LAKERCM_SCHEMA", "schema"),
    ("MLFLOW_EXPERIMENT_NAME", "mlflow_experiment"),
    # Eval jobs must NOT re-register the production scorer schedule on the eval
    # experiment. A base_parameters entry alone never reaches the process — it
    # MUST be threaded here into os.environ for config.Settings to pick it up.
    ("LAKERCM_SKIP_SCORER_REGISTRATION", "skip_scorers"),
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

from eval.rollback import rollback  # noqa: E402
from eval.rollback import _previous_version  # noqa: E402

if to_version_raw:
    target = int(to_version_raw)
elif use_previous:
    target = _previous_version()
else:
    raise ValueError(
        "Provide a to_version, or set previous=true to revert to the prior version."
    )

new_version = rollback(target, reason=reason)
print(f"Champion rolled back to v{new_version} (reason={reason}).")
