"""Regression tests for reviewer app psycopg OpenTelemetry SQL redaction."""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

_REVIEWER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REVIEWER_DIR not in sys.path:
    sys.path.insert(0, _REVIEWER_DIR)


from tests._isolation import IsolatedModules  # noqa: E402

# The config stub and the services.lakehouse_db imported against it are live
# only while this module runs; the test installed them and never removed them.
_ISOLATION = IsolatedModules()


def setup_module(module):
    _ISOLATION.start(purge_first_party=True)
    _install_config_stub()


def teardown_module(module):
    _ISOLATION.stop()


def _install_config_stub() -> None:
    cfg = types.ModuleType("config")
    cfg.settings = SimpleNamespace(  # type: ignore[attr-defined]
        endpoint_name="projects/p/branches/b/endpoints/e",
        db_pool_min_connections=1,
        db_pool_max_connections=2,
        auto_verdict_threshold=0.85,
        lakercm_schema="lakercm",
        automated_reviewer_email="<automated>",
    )
    sys.modules["config"] = cfg


def test_psycopg_composable_query_does_not_populate_sql_text_attributes():
    from opentelemetry.instrumentation import psycopg as otel_psycopg
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from psycopg import sql

    from services.lakehouse_db import _instrument_psycopg_safely

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    fake_integration = SimpleNamespace(
        _sem_conv_opt_in_mode_db="default",
        database_system="postgresql",
        database="databricks_postgres",
        span_attributes={},
        capture_parameters=False,
        enable_commenter=False,
        commenter_options={},
        enable_attribute_commenter=False,
        name="postgresql",
        _tracer=tracer,
        connect_module=None,
    )

    class FakeCursor:
        connection = object()

    composable = sql.SQL("SELECT {} FROM {}").format(
        sql.Identifier("claim_id"),
        sql.Identifier("claims"),
    )

    _instrument_psycopg_safely()
    cursor_tracer = otel_psycopg.CursorTracer(fake_integration)

    with tracer.start_as_current_span("db-test") as span:
        cursor_tracer._populate_span(span, FakeCursor(), composable, ("secret",))

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs.get("db.statement") in (None, "")
    assert attrs.get("db.query.text") in (None, "")
    assert attrs.get("db.statement.parameters") is None
    assert "claim_id" not in str(attrs)
    assert "claims" not in str(attrs)
    assert cursor_tracer.get_operation_name(FakeCursor(), [composable]) == "SQL"
    assert cursor_tracer.get_operation_name(FakeCursor(), ["SELECT 1"]) == "SELECT"
