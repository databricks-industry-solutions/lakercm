"""PHI redaction proof for the agent app's (opt-in) FastAPI OTel request spans.

The agent app ships UNINSTRUMENTED (plain uvicorn; its telemetry is MLflow
dual-export, not ASGI HTTP spans). ``instrument_fastapi`` is opt-in behind
LAKERCM_AGENT_OTEL_FASTAPI. These tests prove that:
  1. When enabled, the PHI-stripping server_request_hook removes the query so
     neither "Jane" nor "nurse@hosp.org" reaches ANY exported span attribute.
  2. Without the hook the query IS recorded (control — proves the assertion is
     meaningful; FAILS before the fix, when the hook did not exist).
  3. It is OFF by default (opt-in) — no instrumentation without the flag/force.
"""

from __future__ import annotations

import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

_PHI_PATH = "/threads/abc"
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
    def _endpoint(search: str = "", reviewer: str = ""):
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
    # force=True bypasses the opt-in gate for the test; wires the real hook.
    instrument_fastapi(app, tracer_provider=provider, force=True)

    resp, blob = _drive(app, provider, exporter)

    assert resp.status_code == 200
    assert exporter.get_finished_spans(), "expected at least one exported span"
    assert "Jane" not in blob, "PHI leaked into an exported span attr: %s" % blob
    assert "nurse@hosp.org" not in blob, "PII leaked into an exported span: %s" % blob


def test_unsanitized_instrumentation_leaks_phi_control():
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    exporter, provider = _make_exporter_provider()
    app = _make_app()
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)  # no hook

    _, blob = _drive(app, provider, exporter)

    assert "Jane" in blob, "control expected raw query in a span attr; got: %s" % blob
    assert "nurse@hosp.org" in blob, "control expected raw query; got: %s" % blob


def test_instrumentation_is_off_by_default_opt_in():
    """No flag, no force -> no instrumentation (agent ships uninstrumented)."""
    from services.otel_fastapi import instrument_fastapi

    os.environ.pop("LAKERCM_AGENT_OTEL_FASTAPI", None)
    exporter, provider = _make_exporter_provider()
    app = _make_app()
    instrument_fastapi(app, tracer_provider=provider)  # gate closed -> no-op

    _, blob = _drive(app, provider, exporter)

    assert not exporter.get_finished_spans(), (
        "agent app must NOT be FastAPI-instrumented by default; got spans: %s" % blob
    )
