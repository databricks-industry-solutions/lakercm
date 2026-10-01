# Databricks notebook source
# MAGIC %md
# MAGIC # Register scheduled scorers for the LakeRCM agent
# MAGIC
# MAGIC Run this once after Feature 1 (MLflow experiment) lands, and any time
# MAGIC the scorer set below changes. Registers 6 scorers (4 built-in judges,
# MAGIC 2 domain Guidelines) against the agent's MLflow experiment.
# MAGIC Idempotent — re-running updates sampling config in place.
# MAGIC
# MAGIC Runs as the notebook owner (not the agent SP): scorer registration is
# MAGIC an admin operation on the experiment. Custom `@scorer` decorator
# MAGIC scorers from `eval/scorers.py` (no_sql_warehouse_regression,
# MAGIC latency_under_slo, tool_call_budget) are NOT supported by scheduled
# MAGIC scorers — they must be run locally via `mlflow.genai.evaluate(...)`
# MAGIC on a pulled trace set. Only LLM-as-judge scorers can run in the
# MAGIC scheduled pipeline today.

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.11" databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    Safety,
    ScorerSamplingConfig,
)

EXPERIMENT_PATH = "/Shared/lakercm/agent-traces"  # production experiment

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)
exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
experiment_id = exp.experiment_id
print(f"experiment_id: {experiment_id}")

# COMMAND ----------

NO_RAW_TEXT_LEAK = Guidelines(
    name="no_raw_text_leak",
    guidelines=(
        "The response must NOT include raw OCR text dumps from documents, "
        "raw SQL queries, database table or column names, internal tool names, "
        "or verbatim stack traces. It should surface only synthesized, "
        "user-readable information."
    ),
)

PROFESSIONAL_TONE = Guidelines(
    name="professional_tone",
    guidelines=(
        "The response must address the user professionally (they are a claims "
        "reviewer) and present data in scannable structures (tables, bullets, "
        "bold for key values). No casual/filler preamble before the content."
    ),
)

SCHEDULED = [
    (RetrievalGroundedness(), 0.2),
    (Safety(), 1.0),
    (RelevanceToQuery(), 0.2),
    (Correctness(), 0.1),
    (NO_RAW_TEXT_LEAK, 1.0),
    (PROFESSIONAL_TONE, 0.1),
]

existing = {s.name: s for s in mlflow.genai.list_scorers(experiment_id=experiment_id)}
print(f"already registered: {list(existing.keys())}")

for scorer_obj, rate in SCHEDULED:
    name = scorer_obj.name
    try:
        if name in existing:
            existing[name].update(
                sampling_config=ScorerSamplingConfig(sample_rate=rate)
            )
            print(f"updated: {name} (sample_rate={rate})")
        else:
            reg = scorer_obj.register(experiment_id=experiment_id)
            reg.start(sampling_config=ScorerSamplingConfig(sample_rate=rate))
            print(f"registered+started: {name} (sample_rate={rate})")
    except Exception as e:
        print(f"FAILED {name}: {type(e).__name__}: {e}")

# COMMAND ----------

print("Currently scheduled scorers on this experiment:")
for s in mlflow.genai.list_scorers(experiment_id=experiment_id):
    rate = s.sample_rate if hasattr(s, "sample_rate") else "?"
    print(f"  - {s.name} (sample_rate={rate})")
