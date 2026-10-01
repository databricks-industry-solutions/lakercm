"""MLflow tracing helpers for the agent app.

`trace_if_active` is the canonical "child-only" idiom: produce a child span
when the caller is already inside an active MLflow trace, otherwise emit
nothing. This prevents helpers like `lakehouse_db.execute_query` from
auto-promoting boot-time / lazy-init invocations into orphan root traces
(which pollute the experiment, distort scheduled-scorer distributions, and
obscure the real `agent_turn` / `agent_turn_agui` chain roots).

MLflow's `@mlflow.trace` decorator has no `parent_only` flag; the
[LangChain integration docs](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/app-instrumentation/manual-tracing/span-tracing)
suggest this guard pattern when conditional tracing is required.
"""

from contextlib import contextmanager

import mlflow


@contextmanager
def trace_if_active(name: str, span_type: str = "RETRIEVER"):
    """Yield an MLflow child span only if a parent span is active.

    Callers can check the yielded value: when there's no active parent
    (e.g. boot-time DDL, lazy schema probes), `span` is None and the
    caller should skip span population. When inside an active agent_turn,
    `span` is a real MLflow span and the caller can `set_attributes` /
    `set_inputs` / `set_outputs` on it.
    """
    if mlflow.get_current_active_span() is None:
        yield None
        return
    with mlflow.start_span(name=name, span_type=span_type) as span:
        yield span


@contextmanager
def tracing_disabled():
    """Globally suppress MLflow tracing for the duration of the block.

    Wraps `mlflow.tracing.disable()` / `mlflow.tracing.enable()` so
    boot-time code paths (prompt registry loads, PostgresSaver.setup(),
    autolog-emitted spans during graph construction) don't leak into
    the experiment as orphan root traces. Use sparingly — anything
    inside this block becomes invisible to MLflow observability.
    """
    mlflow.tracing.disable()
    try:
        yield
    finally:
        mlflow.tracing.enable()
