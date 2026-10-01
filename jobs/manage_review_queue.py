# Databricks notebook source
# MAGIC %md
# MAGIC # LakeRCM Review Queue Management
# MAGIC
# MAGIC Opens / syncs MLflow 3 expert-feedback **review queues** (labeling
# MAGIC sessions) over the agent's traces. Actions:
# MAGIC   - `ensure_schemas` — create/refresh the LakeRCM label schemas
# MAGIC   - `create` — open a queue over recent (or filtered) traces + assign reviewers
# MAGIC   - `sync`   — fold collected labels/expectations into the eval dataset
# MAGIC   - `list`   — list existing queues
# MAGIC
# MAGIC This job is NOT scheduled — run it on demand. It creates Review App
# MAGIC artifacts in the workspace when action=create/sync/ensure_schemas.

# COMMAND ----------

# MAGIC %pip install -r ../agent_app/requirements-eval.txt -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ruff: noqa: F821  # `dbutils` is injected by the Databricks runtime
import os
import sys

dbutils.widgets.dropdown(
    "action", "list", ["list", "ensure_schemas", "create", "sync"], "Action"
)
dbutils.widgets.text("name", "lakercm-expert-review", "Queue (session) name")
dbutils.widgets.text("assigned_users", "", "Reviewer emails (comma-separated)")
dbutils.widgets.text("filter_string", "", "search_traces filter (blank = recent)")
dbutils.widgets.text("max_traces", "50", "Max traces to add")
dbutils.widgets.text("to_dataset", "", "Sync target dataset (blank = eval default)")

action = dbutils.widgets.get("action").strip()
name = dbutils.widgets.get("name").strip()
assigned_users = [
    u.strip() for u in dbutils.widgets.get("assigned_users").split(",") if u.strip()
]
filter_string = dbutils.widgets.get("filter_string").strip() or None
max_traces = int(dbutils.widgets.get("max_traces") or "50")
to_dataset = dbutils.widgets.get("to_dataset").strip() or None

# Inject the workspace config the apps get from app.yaml but jobs don't. Set
# BEFORE importing config/eval, since config.Settings reads env at import.
for _env, _wid in (
    ("DATABRICKS_CATALOG", "catalog"),
    ("LAKERCM_SCHEMA", "schema"),
    ("MLFLOW_EXPERIMENT_NAME", "mlflow_experiment"),
    ("LAKERCM_EVAL_DATASET", "eval_dataset"),
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

from eval import review_queues  # noqa: E402

if action == "ensure_schemas":
    names = review_queues.ensure_label_schemas()
    print(f"Ensured {len(names)} label schemas: {names}")

elif action == "create":
    if not assigned_users:
        raise ValueError("assigned_users is required for action=create")
    summary = review_queues.create_review_queue(
        name=name,
        assigned_users=assigned_users,
        filter_string=filter_string,
        max_traces=max_traces,
    )
    print("Created review queue:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nOpen the Review App to label: {summary.get('url')}")

elif action == "sync":
    result = review_queues.sync_review_queue(name, to_dataset=to_dataset)
    print(f"Synced review queue {result['session_name']} → {result['to_dataset']}")

else:  # list
    queues = review_queues.list_review_queues()
    print(f"{len(queues)} review queue(s):")
    for q in queues:
        print(f"  - {q['session_name']}  ({q.get('url')})")
