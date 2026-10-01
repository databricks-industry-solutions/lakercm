"""PHI-safe (opt-in) FastAPI OpenTelemetry instrumentation for the agent app.

The agent app ships UNINSTRUMENTED by default: its command is plain ``uvicorn``
(no ``opentelemetry-instrument``), ``opentelemetry-instrumentation-fastapi`` is
not a runtime dependency, and its telemetry is MLflow trace dual-export — not
ASGI HTTP spans. This module exists so that IF ASGI export is ever turned on
here, request-span query strings are stripped exactly like the reviewer app.

``instrument_fastapi`` is gated behind ``LAKERCM_AGENT_OTEL_FASTAPI``
(default off) so enabling ASGI instrumentation is a deliberate, tested choice
that never silently regresses AG-UI SSE streaming.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("lakercm.agent")

_FULL_URL_ATTRS = ("http.url", "url.full")
_ENABLE_FLAG = "LAKERCM_AGENT_OTEL_FASTAPI"
_TRUTHY = ("1", "true", "yes", "on")


def sanitize_request_span(span, scope) -> None:
    """OTel ASGI ``server_request_hook``: strip the query string from a span.

    STRIP/ALLOWLIST approach: the query component is dropped entirely (only the
    path is kept), so any current OR FUTURE query parameter is PHI-safe by
    default. ``http.route`` is left intact. Best-effort and fully guarded.
    """
    try:
        if span is None or not span.is_recording():
            return

        path = ""
        if isinstance(scope, dict):
            raw = scope.get("path")
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", "replace")
            path = (raw or "").split("?", 1)[0]

        span.set_attribute("url.query", "")
        if path:
            span.set_attribute("http.target", path)
            span.set_attribute("url.path", path)

        existing = getattr(span, "attributes", None) or {}
        for key in _FULL_URL_ATTRS:
            val = existing.get(key)
            if isinstance(val, str) and "?" in val:
                span.set_attribute(key, val.split("?", 1)[0])
    except Exception:  # noqa: BLE001 - never let sanitization raise
        return


def instrument_fastapi(app, tracer_provider=None, force: bool = False) -> None:
    """Opt-in, PHI-sanitized ASGI instrumentation for the agent app.

    No-op unless ``LAKERCM_AGENT_OTEL_FASTAPI`` is truthy (or ``force=True``,
    used by tests). ``tracer_provider`` is for tests only — production passes
    None (global provider).
    """
    if not force and os.getenv(_ENABLE_FLAG, "").strip().lower() not in _TRUTHY:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        logger.warning(
            "agent OTel FastAPI instrumentation requested but package missing"
        )
        return

    try:
        FastAPIInstrumentor.instrument_app(
            app,
            server_request_hook=sanitize_request_span,
            tracer_provider=tracer_provider,
        )
        logger.info(
            "agent OTel FastAPI instrumentation enabled "
            "(query-string PHI stripping active)"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("agent OTel FastAPI instrumentation failed: %s", e)
