# Databricks notebook source
# MAGIC %md
# MAGIC # LakeRCM Eval Dataset Generation (agent_eval_v2)
# MAGIC
# MAGIC Builds the authored, fixture-grounded golden eval dataset from
# MAGIC `eval/dataset_gen.py`. Each record embeds the tool-output fixture the
# MAGIC agent's tools "return" for that question and derives its `expected_facts`
# MAGIC from that fixture — so trace-replay serves the fixture at eval time and
# MAGIC Correctness scores against fixture-grounded facts. No Lakebase, no live
# MAGIC traces: the records are authored, not mined.

# COMMAND ----------

# MAGIC %pip install -r ../agent_app/requirements-eval.txt -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` is injected by the Databricks runtime
import os
import sys

dbutils.widgets.text("dataset_name", "", "Dataset name (blank → config default)")

# Inject the workspace config the apps get from app.yaml but jobs don't (the
# job notebooks run with config.py defaults otherwise — catalog='main' etc.).
# Set BEFORE importing config/eval, since config.Settings reads env at import.
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

# Notebooks deploy to <bundle>/files/jobs/; the importable package lives in
# <bundle>/files/agent_app/. Resolve agent_app explicitly (getcwd() is
# .../files/jobs at runtime).
agent_app = os.path.abspath(os.path.join(os.getcwd(), "..", "agent_app"))
if agent_app not in sys.path:
    sys.path.insert(0, agent_app)
os.chdir(agent_app)

from eval.dataset_gen import V2_RECORDS, build_v2_dataset  # noqa: E402

dataset_name = dbutils.widgets.get("dataset_name").strip() or None
written = build_v2_dataset(dataset_name)
print(f"Built {len(V2_RECORDS)} authored records into {written}")

# COMMAND ----------

from eval.dataset import stats  # noqa: E402

print("\nDataset stats:")
for k, v in stats().items():
    print(f"  {k}: {v}")
