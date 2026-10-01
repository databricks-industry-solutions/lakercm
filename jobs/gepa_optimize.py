# Databricks notebook source
# MAGIC %md
# MAGIC # LakeRCM GEPA Prompt Optimization
# MAGIC
# MAGIC Runs `mlflow.genai.optimize_prompts(...)` with `GepaPromptOptimizer` over a 70/30
# MAGIC train/holdout split of the eval dataset, registers the optimized prompt as a new
# MAGIC version, sets the `candidate` alias to that version, and runs `run_eval.py`
# MAGIC against the holdout for an auditable MLflow run.
# MAGIC
# MAGIC Promotion (alias swap to `champion`) is a separate job: `promote_prompt.py`.

# COMMAND ----------

# MAGIC %pip install -r ../agent_app/requirements-eval.txt -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` is injected by the Databricks runtime
import os
import sys

dbutils.widgets.text(
    "reflection_model", "databricks-claude-sonnet-4-5", "Reflection model"
)

reflection_model = dbutils.widgets.get("reflection_model").strip() or None

# Inject the workspace config the apps get from app.yaml but jobs don't (the
# job notebooks run with config.py defaults otherwise — catalog='main' etc.).
# Set BEFORE importing config/eval, since config.Settings reads env at import.
# No Lakebase env: GEPA uses trace-replay (recorded tool outputs from the
# experiment's OTel traces), so the agent's tools never touch Postgres.
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

# Notebooks deploy to <bundle>/files/jobs/; the importable package + its
# requirements live in <bundle>/files/agent_app/. Resolve agent_app explicitly
# (getcwd() is .../files/jobs at runtime).
agent_app = os.path.abspath(os.path.join(os.getcwd(), "..", "agent_app"))
if agent_app not in sys.path:
    sys.path.insert(0, agent_app)
os.chdir(agent_app)

from eval.optimize import optimize  # noqa: E402

new_version = optimize(reflection_model=reflection_model)
print(f"\nNew prompt version: {new_version}")
