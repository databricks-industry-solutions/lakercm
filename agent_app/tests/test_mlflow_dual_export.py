"""Regression tests for MLflow OTLP dual-export wiring."""

from __future__ import annotations

import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)


def _processor_names():
    from mlflow.tracing.provider import _get_span_processors

    return [type(processor).__name__ for processor in _get_span_processors()]


def test_mlflow_otlp_dual_export_keeps_native_mlflow_processor(monkeypatch, tmp_path):
    """An OTLP endpoint must add to, not replace, MLflow-native export."""
    import mlflow
    from config import Settings

    for name in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "MLFLOW_ENABLE_DUAL_EXPORT",
        "MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT",
    ):
        monkeypatch.delenv(name, raising=False)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4314")

    monkeypatch.setenv("MLFLOW_ENABLE_DUAL_EXPORT", "true")
    assert _processor_names() == ["OtelSpanProcessor"]

    monkeypatch.delenv("MLFLOW_ENABLE_DUAL_EXPORT")
    monkeypatch.setenv("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", "true")

    settings = Settings()
    assert settings.mlflow_trace_enable_otlp_dual_export == "true"
    assert _processor_names() == ["OtelSpanProcessor", "MlflowV3SpanProcessor"]
