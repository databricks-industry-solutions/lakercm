"""PHI-safe FastAPI OpenTelemetry instrumentation for the reviewer app.

The OTel ASGI/FastAPI instrumentation records the raw request target into span
attributes (``http.target`` / ``http.url`` under default HTTP semconv, and
``url.full`` / ``url.query`` under stable HTTP semconv). For LakeRCM those
query strings carry PHI/PII — reviewer search endpoints filter on document name
and reviewer email, e.g.
``/api/analytics/recent-reviews?search=Jane%20Doe&reviewer=nurse@hosp.org`` —
and with ``OTEL_TRACES_SAMPLER=always_on`` every request span is exported to
Unity Catalog indefinitely.

``sanitize_request_span`` is wired as the instrumentation's ``server_request_hook``
so it runs immediately after the span is created (before export) and STRIPS the
query component. It uses a STRIP/ALLOWLIST approach — the entire query is dropped
and only the path is kept — so any current OR FUTURE query parameter is PHI-safe
by default (never a denylist of known-bad param names).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Span attributes that can carry the raw request target (and thus the query).
# http.route (the templated route) is deliberately NOT touched — it never
# contains user input.
_FULL_URL_ATTRS = ("http.url", "url.full")


def sanitize_request_span(span, scope) -> None:
    """OTel ASGI ``server_request_hook``: strip the query string from a span.

    Best-effort and fully guarded — telemetry sanitization must NEVER break
    request handling.
    """
    try:
        if span is None or not span.is_recording():
            return

        # Path only, from the ASGI scope (authoritative; never contains query).
        path = ""
        if isinstance(scope, dict):
            raw = scope.get("path")
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", "replace")
            path = (raw or "").split("?", 1)[0]

        # Drop the raw query outright (stable HTTP semconv attribute).
        span.set_attribute("url.query", "")

        # Path-only variants of the query-bearing target attributes.
        if path:
            span.set_attribute("http.target", path)
            span.set_attribute("url.path", path)

        # Full-URL attributes: keep scheme://host/path, never the "?query".
        existing = getattr(span, "attributes", None) or {}
        for key in _FULL_URL_ATTRS:
            val = existing.get(key)
            if isinstance(val, str) and "?" in val:
                span.set_attribute(key, val.split("?", 1)[0])
    except Exception:  # noqa: BLE001 - never let sanitization raise
        return


def instrument_fastapi(app, tracer_provider=None) -> None:
    """Attach FastAPI OTel middleware, always with the PHI-stripping hook.

    Under plain ``uvicorn`` (telemetry off) the global tracer provider is a
    no-op proxy, so instrumentation and the hook are harmless; under
    ``opentelemetry-instrument`` (telemetry on) the hook runs against the real
    SDK span. ``tracer_provider`` is only for tests — production passes None so
    the global (``opentelemetry-instrument``-configured) provider is used.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        logger.warning("OpenTelemetry FastAPI instrumentation package not installed")
        return

    try:
        FastAPIInstrumentor.instrument_app(
            app,
            server_request_hook=sanitize_request_span,
            tracer_provider=tracer_provider,
        )
        logger.info(
            "OpenTelemetry FastAPI instrumentation enabled "
            "(query-string PHI stripping active)"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenTelemetry FastAPI instrumentation failed: %s", e)
