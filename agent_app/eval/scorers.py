"""
Production-monitoring scorer registration.

The actual scorer definitions live in `eval/scorer_set.py` so the same set
runs both in production (sampled) and offline (`run_eval.py`, 100% coverage).
This file is the side-effecting registration script:

    python -m eval.scorers register   # idempotent register/refresh
    python -m eval.scorers reset      # delete-then-recreate (use when the
                                       # registry is in an undeserializable
                                       # state — see self-heal below)

The scheduler runs each scorer as a background job against new production
traces and writes Feedback assessments back to the trace record.

Self-heal: if MLflow's stored scorer artifacts can't be deserialized under
the current runtime (raised as ``AtLeastOneUndeserializableScorerError``),
``register_scorers`` deletes every scheduled-name and re-registers from
``production_schedule()``. This unblocks the agent boot path that calls
this from ``services/observability.init_tracing`` without operator action.
"""

from __future__ import annotations

import logging
from typing import Iterable

import mlflow
from mlflow.genai.scorers import ScorerSamplingConfig

from config import settings
from eval.scorer_set import production_schedule

logger = logging.getLogger(__name__)


def _resolve_experiment_id() -> str:
    """Resolve the MLflow experiment id, preferring the path-independent
    env value. Mirrors ``services.observability.init_tracing`` precedence:
    a numeric experiment_id always wins because the agent SP can't always
    traverse path-based experiments."""
    if not str(mlflow.get_tracking_uri()).startswith("databricks"):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if settings.mlflow_experiment_id:
        return settings.mlflow_experiment_id
    mlflow.set_experiment(settings.mlflow_experiment_name)
    exp = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    return exp.experiment_id


def _scorer_names() -> list[str]:
    """Single source of truth for which scorer names this agent owns."""
    return [scorer_obj.name for scorer_obj, _ in production_schedule()]


def _delete_scorers(names: Iterable[str], experiment_id: str) -> None:
    """Best-effort delete by name. ``not found`` and other failures are
    logged at info — the next register pass recreates them anyway."""
    for name in names:
        try:
            mlflow.genai.scorers.delete_scorer(name=name, experiment_id=experiment_id)
            logger.info("Deleted scorer: %s", name)
        except Exception as e:
            logger.info("Could not delete scorer %s (continuing): %s", name, e)


def _is_undeserializable_error(exc: BaseException) -> bool:
    """Match by class name so we don't pin to a specific MLflow import path
    (the symbol has moved between versions). The class name is stable."""
    return type(exc).__name__ == "AtLeastOneUndeserializableScorerError"


def register_scorers(experiment_path: str | None = None) -> None:
    """Register all scheduled scorers on the agent's MLflow experiment.

    Idempotent — re-registering with the same name updates in place. If the
    pre-flight ``list_scorers`` call raises ``AtLeastOneUndeserializableScorerError``
    (e.g. after an MLflow upgrade left stale artifacts), every scheduled
    scorer is deleted by name and re-registered fresh. Custom
    ``@scorer``-decorated scorers are rejected if the tracking URI is local.

    The ``experiment_path`` kwarg is preserved for backward compat — when
    omitted, falls through to ``settings.mlflow_experiment_name`` /
    ``settings.mlflow_experiment_id``.
    """
    if experiment_path:
        mlflow.set_experiment(experiment_path)
        exp = mlflow.get_experiment_by_name(experiment_path)
        experiment_id = exp.experiment_id
    else:
        experiment_id = _resolve_experiment_id()

    try:
        existing_by_name = {
            s.name: s for s in mlflow.genai.list_scorers(experiment_id=experiment_id)
        }
    except Exception as e:
        if _is_undeserializable_error(e):
            logger.warning(
                "list_scorers raised %s; deleting all scheduled scorers and "
                "recreating from production_schedule(). Cause: %s",
                type(e).__name__,
                e,
            )
            _delete_scorers(_scorer_names(), experiment_id)
            existing_by_name = {}
        else:
            raise

    for scorer_obj, sample_rate in production_schedule():
        name = scorer_obj.name
        try:
            if name in existing_by_name:
                existing_by_name[name].update(
                    sampling_config=ScorerSamplingConfig(sample_rate=sample_rate),
                )
                logger.info("Updated scorer: %s (sample_rate=%.2f)", name, sample_rate)
            else:
                registered = scorer_obj.register(experiment_id=experiment_id)
                registered.start(
                    sampling_config=ScorerSamplingConfig(sample_rate=sample_rate),
                )
                logger.info(
                    "Registered+started scorer: %s (sample_rate=%.2f)",
                    name,
                    sample_rate,
                )
        except Exception as e:
            logger.warning("Failed to register %s: %s", name, e)


def reset_scorers() -> None:
    """Force delete-then-recreate for every scheduled scorer.

    Use when the experiment registry is in a state where stored scorer
    artifacts can't be deserialized — typically signaled by
    ``AtLeastOneUndeserializableScorerError`` on the monitoring job.
    Equivalent to the self-heal branch of ``register_scorers`` but
    unconditional, so an operator can run it directly.
    """
    experiment_id = _resolve_experiment_id()
    logger.info("Resetting scheduled scorers on experiment_id=%s", experiment_id)
    _delete_scorers(_scorer_names(), experiment_id)
    register_scorers()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "register":
        register_scorers()
    elif cmd == "reset":
        reset_scorers()
    else:
        print("Usage: python -m eval.scorers {register|reset}")
