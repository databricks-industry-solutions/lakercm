"""MLflow 3 review queues (expert-feedback labeling) for LakeRCM.

Wraps `mlflow.genai.labeling` + `mlflow.genai.label_schemas` (the Review App
labeling API — https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/expert-feedback/review-queues)
into a small, domain-shaped surface:

  - ensure_label_schemas()  → create/refresh the LakeRCM label schemas
  - create_review_queue(...) → open a labeling session over selected traces and
                               assign expert reviewers; returns the Review App URL
  - sync_review_queue(...)   → fold collected labels (verdicts + expected facts)
                               back into the eval dataset the GEPA loop consumes
  - list_review_queues()     → enumerate existing sessions

Design notes
------------
- Traces are already tagged with `mlflow.trace.user` / `mlflow.trace.session`
  (services/observability.set_session_context), so review queues attach to the
  live agent traces with no extra instrumentation.
- The schemas intentionally mirror the production scorers in
  `eval/scorer_set.py` (grounding, guideline adherence, correctness/expected
  facts) so expert labels can be compared against — and can seed expectations
  for — the automated judges.
- Nothing here is created until a caller explicitly runs it (see
  jobs/manage_review_queue.py). Importing this module has no side effects.

Version: the labeling / label_schemas API is Databricks-only and stabilized in
mlflow 3.16 (`pip install "mlflow[databricks]>=3.16"`). This module imports it
behind a guard so it can be imported (and unit-tested) even where the running
mlflow predates the API; callers get a clear error instead of an AttributeError.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import mlflow

logger = logging.getLogger(__name__)

# Guarded import of the labeling API. Kept import-safe on older mlflow so the
# module (and its tests) load regardless; _require() gates actual use.
try:  # pragma: no cover - import shape depends on installed mlflow
    from mlflow.genai import label_schemas as _label_schemas
    from mlflow.genai import labeling as _labeling

    _LABELING_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    _label_schemas = None
    _labeling = None
    _LABELING_IMPORT_ERROR = _e


_MIN_MLFLOW = "3.16"


def _require():
    """Raise a clear error if the labeling API is unavailable."""
    if _labeling is None or _label_schemas is None:
        raise RuntimeError(
            "mlflow.genai labeling API unavailable "
            f"(import error: {_LABELING_IMPORT_ERROR}). Review queues need "
            f'`pip install "mlflow[databricks]>={_MIN_MLFLOW}"` on a Databricks '
            "tracking URI. This is installed by agent_app/requirements-eval.txt "
            "for the review-queue job."
        )


# -----------------------------------------------------------------------------
# Label schemas — the questions expert reviewers answer per trace.
#
# `key` is the stable schema name (id). `kind` maps to a label_schemas Input
# factory. `schema_type` is "feedback" (a judgement about the response) or
# "expectation" (ground truth that can seed the Correctness judge / dataset).
# -----------------------------------------------------------------------------
_QUALITY = "lakercm_response_quality"
_GROUNDED = "lakercm_grounded_in_data"
_GUIDELINES = "lakercm_guideline_adherence"
_EXPECTED_FACTS = "lakercm_expected_facts"
_NOTES = "lakercm_reviewer_notes"

_SCHEMA_DEFS: list[dict[str, Any]] = [
    {
        "name": _QUALITY,
        "schema_type": "feedback",
        "title": "Overall response quality",
        "instruction": "How good is the assistant's answer for a claims reviewer?",
        "kind": ("categorical", ["Excellent", "Good", "Fair", "Poor"]),
    },
    {
        "name": _GROUNDED,
        "schema_type": "feedback",
        "title": "Grounded in the document data",
        "instruction": (
            "Did the answer stick to the retrieved gold data without inventing "
            "values or leaking raw OCR text?"
        ),
        "kind": ("categorical", ["Fully", "Partially", "No"]),
    },
    {
        "name": _GUIDELINES,
        "schema_type": "feedback",
        "title": "Guideline adherence",
        "instruction": (
            "Professional tone, scannable structure, no internal details (tool "
            "names, SQL, table/column names)? Pass or fail."
        ),
        "kind": ("categorical", ["Pass", "Fail"]),
    },
    {
        "name": _EXPECTED_FACTS,
        "schema_type": "expectation",
        "title": "Expected facts",
        "instruction": (
            "List the key facts the answer should contain. These seed the "
            "Correctness judge and the eval dataset."
        ),
        "kind": ("text_list", None),
    },
    {
        "name": _NOTES,
        "schema_type": "feedback",
        "title": "Reviewer notes",
        "instruction": "Anything else worth capturing for this trace.",
        "kind": ("text", None),
    },
]


def _build_input(kind: tuple[str, Any]):
    """Map a (kind, options) spec to a label_schemas Input instance."""
    kind_name, options = kind
    if kind_name == "categorical":
        return _label_schemas.InputCategorical(options=list(options))
    if kind_name == "text_list":
        return _label_schemas.InputTextList()
    if kind_name == "text":
        return _label_schemas.InputText()
    raise ValueError(f"unknown schema input kind: {kind_name}")


def ensure_label_schemas() -> list[str]:
    """Create or refresh all LakeRCM label schemas. Idempotent.

    Returns the list of schema names, ordered as reviewers should see them.
    """
    _require()
    names: list[str] = []
    for spec in _SCHEMA_DEFS:
        _label_schemas.create_label_schema(
            name=spec["name"],
            type=spec["schema_type"],
            title=spec["title"],
            instruction=spec["instruction"],
            input=_build_input(spec["kind"]),
            enable_comment=True,
            overwrite=True,
        )
        names.append(spec["name"])
    logger.info("Ensured %d label schemas: %s", len(names), names)
    return names


def _experiment_id() -> str:
    """Resolve the agent experiment id.

    Mirrors eval.dataset._experiment_id (kept local so this module has no heavy
    import): prefer the numeric MLFLOW_EXPERIMENT_ID (SP-traversable), else the
    MLFLOW_EXPERIMENT_NAME path. Eval jobs set these; the bundle wires them.
    """
    exp_id = os.environ.get("MLFLOW_EXPERIMENT_ID", "").strip()
    if exp_id:
        exp = mlflow.get_experiment(exp_id)
        if exp is None:
            raise RuntimeError(f"MLflow experiment id not found: {exp_id}")
        return exp.experiment_id
    path = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
    if not path:
        raise RuntimeError(
            "Set MLFLOW_EXPERIMENT_ID or MLFLOW_EXPERIMENT_NAME to target an "
            "experiment for the review queue."
        )
    exp = mlflow.get_experiment_by_name(path)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {path}")
    return exp.experiment_id


def _select_traces(filter_string: Optional[str], max_traces: int) -> list:
    """Pull traces for review, newest first. `filter_string` uses MLflow's
    search_traces syntax (e.g. "attributes.status = 'ERROR'"); None → recent."""
    kwargs: dict[str, Any] = {
        "experiment_ids": [_experiment_id()],
        "max_results": max(1, min(max_traces, 500)),
        "order_by": ["timestamp DESC"],
        "return_type": "list",
    }
    if filter_string:
        kwargs["filter_string"] = filter_string
    return mlflow.search_traces(**kwargs)


def create_review_queue(
    name: str,
    assigned_users: list[str],
    *,
    filter_string: Optional[str] = None,
    max_traces: int = 50,
    label_schemas: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Open a review queue (labeling session) over selected traces.

    Args:
        name: session name (also the Review App queue title).
        assigned_users: reviewer emails to assign.
        filter_string: MLflow search_traces filter; None → most recent traces.
        max_traces: cap on traces added (clamped to 500).
        label_schemas: schema names to attach; None → all LakeRCM schemas
            (created/refreshed first).

    Returns a summary dict incl. the Review App `url`. Does not run any eval.
    """
    _require()
    schemas = label_schemas or ensure_label_schemas()

    session = _labeling.create_labeling_session(
        name=name,
        assigned_users=assigned_users,
        label_schemas=schemas,
    )

    traces = _select_traces(filter_string, max_traces)
    if traces:
        session.add_traces(traces)

    summary = {
        "session_name": session.name,
        "labeling_session_id": getattr(session, "labeling_session_id", None),
        "url": getattr(session, "url", None),
        "assigned_users": assigned_users,
        "label_schemas": schemas,
        "trace_count": len(traces) if traces is not None else 0,
    }
    logger.info("Created review queue %r with %d traces", name, summary["trace_count"])
    return summary


def _find_session(name: str):
    for s in _labeling.get_labeling_sessions():
        if s.name == name:
            return s
    return None


def sync_review_queue(session_name: str, to_dataset: Optional[str] = None) -> dict:
    """Sync a review queue's collected labels/expectations into a dataset.

    The dataset then feeds the offline eval + GEPA loop (eval/dataset.py). If
    `to_dataset` is None, uses LAKERCM_EVAL_DATASET from the environment.
    """
    _require()
    dataset = to_dataset or os.environ.get("LAKERCM_EVAL_DATASET", "").strip()
    if not dataset:
        raise RuntimeError(
            "No target dataset: pass to_dataset or set LAKERCM_EVAL_DATASET."
        )
    session = _find_session(session_name)
    if session is None:
        raise RuntimeError(f"Review queue not found: {session_name}")
    session.sync(to_dataset=dataset)
    logger.info("Synced review queue %r → dataset %s", session_name, dataset)
    return {"session_name": session_name, "to_dataset": dataset}


def list_review_queues() -> list[dict]:
    """Enumerate existing review queues (labeling sessions)."""
    _require()
    out: list[dict] = []
    for s in _labeling.get_labeling_sessions():
        out.append(
            {
                "session_name": s.name,
                "labeling_session_id": getattr(s, "labeling_session_id", None),
                "assigned_users": getattr(s, "assigned_users", None),
                "url": getattr(s, "url", None),
            }
        )
    return out
