"""PHI redaction proof for the reviewer app's FastAPI OTel request spans.

Point 1 (BLOCKING): the OTel ASGI/FastAPI instrumentation records the raw
request target (http.target / http.url / url.full / url.query) into every
request span, and with OTEL_TRACES_SAMPLER=always_on every span is exported to
Unity Catalog. Reviewer search endpoints carry PHI/PII in the query string,
e.g. /api/analytics/recent-reviews?search=Jane Doe&reviewer=nurse@hosp.org.

``test_hook_strips_phi_query_from_all_exported_span_attributes`` FAILS before
the fix (the ``services.otel_fastapi`` module / ``server_request_hook`` did not
exist) and asserts that neither "Jane" nor "nurse@hosp.org" appears in ANY
exported span attribute. ``test_unsanitized_instrumentation_leaks_phi_control``
instruments the same request WITHOUT the hook and asserts the PHI IS present —
proving the assertion is meaningful and the hook is what removes it.
"""

from __future__ import annotations

import os
import sys

_REVIEWER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REVIEWER_DIR not in sys.path:
    sys.path.insert(0, _REVIEWER_DIR)

# A representative reviewer search request: document name (PHI) + reviewer email
# (PII), the exact shape the endpoint receives.
_PHI_PATH = "/api/analytics/recent-reviews"
_PHI_QUERY = "search=Jane%20Doe&reviewer=nurse@hosp.org"


def _make_exporter_provider():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


def _make_app():
    from fastapi import FastAPI

    app = FastAPI()

    @app.get(_PHI_PATH)
    def _recent_reviews(search: str = "", reviewer: str = ""):
        return {"ok": True}

    return app


def _all_span_attr_text(exporter) -> str:
    parts = []
    for span in exporter.get_finished_spans():
        for key, value in (span.attributes or {}).items():
            parts.append("%s=%s" % (key, value))
    return " || ".join(parts)


def _drive(app, provider, exporter):
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        resp = client.get("%s?%s" % (_PHI_PATH, _PHI_QUERY))
    provider.force_flush()
    return resp, _all_span_attr_text(exporter)


def test_hook_strips_phi_query_from_all_exported_span_attributes():
    from services.otel_fastapi import instrument_fastapi

    exporter, provider = _make_exporter_provider()
    app = _make_app()
    # Exactly how production wires it (server_request_hook attached).
    instrument_fastapi(app, tracer_provider=provider)

    resp, blob = _drive(app, provider, exporter)

    assert resp.status_code == 200
    assert exporter.get_finished_spans(), "expected at least one exported span"
    assert "Jane" not in blob, "PHI leaked into an exported span attr: %s" % blob
    assert "nurse@hosp.org" not in blob, "PII leaked into an exported span: %s" % blob


def test_unsanitized_instrumentation_leaks_phi_control():
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    exporter, provider = _make_exporter_provider()
    app = _make_app()
    # No server_request_hook — the pre-fix behavior.
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    _, blob = _drive(app, provider, exporter)

    assert "Jane" in blob, "control expected raw query in a span attr; got: %s" % blob
    assert "nurse@hosp.org" in blob, "control expected raw query; got: %s" % blob
