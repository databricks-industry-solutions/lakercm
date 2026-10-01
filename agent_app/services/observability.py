"""
MLflow 3 observability wiring for the LakeRCM agent.

Responsibilities:
- mlflow.langchain.autolog() at module import (covers LangGraph nodes, tools, LLM calls).
- Configure tracking URI + experiment on Databricks.
- Expose set_session_context() so each request can tag its root trace with
  session_id (thread_id) and user_id (user_email).
- Expose traced_sse_yield helper so streamed tokens stay attached to the
  root span instead of producing orphan traces.

Pitfall mitigations (per plan):
- Do NOT @mlflow.trace functions that are already auto-traced by the
  langchain integration — that creates duplicate spans.
- Tag values are capped at 250 chars and indexed; never put PII or large
  payloads in tags. Use metadata (immutable) for IDs, attributes for free-form.
- MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=true prevents trace export from
  blocking request latency (set in app.yaml).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import mlflow
from mlflow.entities import SpanType

from config import settings

logger = logging.getLogger(__name__)

_autolog_initialized = False


def init_tracing() -> None:
    """One-time MLflow tracing setup. Safe to call repeatedly.

    Resolution order:
      1. MLFLOW_EXPERIMENT_ID (numeric) — preferred, path-independent. The
         service principal that runs this app cannot traverse user-private
         workspace paths like /Users/<email>/, so resolving by name to such
         a path puts MLflow in no-op mode (every trace_id becomes the
         literal sentinel `MLFLOW_NO_OP_SPAN_TRACE_ID`).
      2. MLFLOW_EXPERIMENT_NAME — defaults to /Shared/lakercm/agent,
         which IS SP-traversable.

    Once the experiment is set, mlflow.set_active_model(name=<UC model>)
    binds every trace to the registered model so traces have a stable
    UC entity to attribute to (`logged_model_id` filter in the Trace UI).
    """
    global _autolog_initialized
    if _autolog_initialized:
        return

    # Configure MLflow's supported OpenTelemetry dual-export path before any
    # trace starts. This keeps MLflow as the owner of the existing tracer/context
    # behavior and avoids installing a second independent OTel SDK in agent_app.
    for env_name, value in (
        (
            "MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT",
            settings.mlflow_trace_enable_otlp_dual_export,
        ),
        (
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            settings.otel_exporter_otlp_traces_endpoint,
        ),
        ("OTEL_SERVICE_NAME", settings.otel_service_name),
        ("OTEL_RESOURCE_ATTRIBUTES", settings.otel_resource_attributes),
    ):
        if value:
            os.environ[env_name] = str(value)

    tracking_uri = settings.mlflow_tracking_uri
    try:
        mlflow.set_tracking_uri(tracking_uri)
    except Exception as e:
        logger.warning("Failed to set MLflow tracking URI=%s: %s", tracking_uri, e)

    experiment_id = settings.mlflow_experiment_id
    experiment_name = settings.mlflow_experiment_name
    set_via = None
    try:
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
            set_via = f"id={experiment_id}"
        else:
            mlflow.set_experiment(experiment_name)
            set_via = f"name={experiment_name}"
        logger.info("MLflow experiment set (%s)", set_via)
    except Exception as e:
        logger.warning(
            "Failed to set MLflow experiment (id=%s name=%s): %s",
            experiment_id,
            experiment_name,
            e,
        )

    # Active LoggedModel — singleton per agent, lazy-created on first boot
    # under the active experiment, reused on every subsequent boot. Traces
    # auto-attribute via mlflow.search_traces(filter="logged_model_id=...").
    # Separate from the UC registered_model declared in resources/00_mlflow.yml,
    # which is a governance entity (not used for trace attribution).
    model_name = settings.mlflow_logged_model_name
    if model_name:
        try:
            active = mlflow.set_active_model(name=model_name)
            logger.info(
                "MLflow active model: %s (model_id=%s)",
                model_name,
                getattr(active, "model_id", "?"),
            )
        except Exception as e:
            logger.warning("mlflow.set_active_model(%s) failed: %s", model_name, e)
    else:
        logger.info("LAKERCM_LOGGED_MODEL_NAME not set — skipping set_active_model")

    try:
        mlflow.langchain.autolog()
        logger.info("mlflow.langchain.autolog() enabled")
    except Exception as e:
        logger.warning("mlflow.langchain.autolog() failed: %s", e)

    # Idempotent production scorer registration. Without this, adding a new
    # scorer to eval/scorer_set.py requires a manual notebook run before it
    # reaches prod. The SDK's register/start are idempotent; failures are
    # swallowed so they never break agent boot. Self-heals on
    # AtLeastOneUndeserializableScorerError (see eval/scorers.py).
    if not settings.skip_scorer_registration:
        try:
            from eval.scorers import register_scorers

            register_scorers()
            logger.info("Production scorers registered/refreshed")
        except Exception as e:
            logger.warning("register_scorers failed at boot (non-fatal): %s", e)

    _autolog_initialized = True


def set_session_context(
    thread_id: Optional[str],
    user_email: Optional[str],
    app_version: str = "1.0.0",
    graph_node: Optional[str] = None,
) -> None:
    """Attach session + user context to the active trace.

    Uses MLflow's reserved metadata keys (mlflow.trace.session / mlflow.trace.user)
    so the UI automatically groups turns of a conversation and filters by user.
    """
    metadata = {}
    if thread_id:
        metadata["mlflow.trace.session"] = thread_id
    if user_email:
        metadata["mlflow.trace.user"] = user_email

    tags = {"app_version": app_version}
    if graph_node:
        tags["graph_node"] = graph_node

    try:
        mlflow.update_current_trace(metadata=metadata, tags=tags)
    except Exception as e:
        # Never let observability break a user request
        logger.debug("update_current_trace failed: %s", e)


def set_prompt_context(name: str, version: int, alias_used: str) -> None:
    """Tag the active trace with the prompt that produced it.

    Tags (indexed, filterable in the Trace UI):
      - prompt_version
      - prompt_alias_used     ("champion" | "candidate" | ...)
    Metadata (immutable, full-fidelity):
      - prompt.name
      - prompt.version
      - prompt.alias_used
    """
    metadata = {
        "prompt.name": name,
        "prompt.version": str(version),
        "prompt.alias_used": alias_used,
    }
    tags = {
        "prompt_version": str(version),
        "prompt_alias_used": alias_used,
    }
    try:
        mlflow.update_current_trace(metadata=metadata, tags=tags)
    except Exception as e:
        logger.debug("set_prompt_context failed: %s", e)


__all__ = [
    "init_tracing",
    "set_session_context",
    "set_prompt_context",
    "SpanType",
]
