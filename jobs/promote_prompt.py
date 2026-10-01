# Databricks notebook source
# MAGIC %md
# MAGIC # LakeRCM Prompt Promotion
# MAGIC
# MAGIC Re-runs eval for the current `champion` and the supplied candidate version on the
# MAGIC held-out split, applies the CI-mode gate (hard Safety floor + Correctness floor +
# MAGIC audit non-regression + paired-bootstrap composite-Δ CI lower bound > 0, refusing to
# MAGIC decide under `min_eval_samples`), and on pass swaps the `champion` alias.
# MAGIC
# MAGIC On block: the alias is left untouched. A real run raises (job fails); a `--dry-run`
# MAGIC reports the gate decision and exits 0 (a block is a valid informational outcome).

# COMMAND ----------

# MAGIC %pip install -r ../agent_app/requirements-eval.txt -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` is injected by the Databricks runtime
import os
import sys

dbutils.widgets.text("candidate_version", "", "Candidate prompt version")
dbutils.widgets.dropdown(
    "dry_run", "false", ["false", "true"], "Dry run (no alias change)"
)

candidate_version = int(dbutils.widgets.get("candidate_version"))
dry_run = dbutils.widgets.get("dry_run") == "true"

# Inject the workspace config the apps get from app.yaml but jobs don't
# (otherwise config.py defaults apply — catalog='main' etc.). Set BEFORE
# importing config/eval, since config.Settings reads env at import.
# No Lakebase env: promote's eval uses trace-replay (recorded tool outputs),
# so the agent's tools never touch Postgres.
for _env, _wid in (
    ("DATABRICKS_CATALOG", "catalog"),
    ("LAKERCM_SCHEMA", "schema"),
    ("LLM_ENDPOINT", "llm_endpoint"),
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

from eval.promote import promote  # noqa: E402

exit_code = promote(candidate_version=candidate_version, dry_run=dry_run)
# A dry-run that blocks (min-samples / no significant improvement) is a valid
# informational result, not a job failure — the gate summary is already printed.
# Only a REAL promotion that blocks or errors should fail the job.
if exit_code != 0 and not dry_run:
    raise RuntimeError(
        f"Promotion blocked or errored (exit={exit_code}). Alias unchanged."
    )
print(f"promote complete: exit_code={exit_code} dry_run={dry_run}")
