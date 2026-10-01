"""Unit tests for eval.trace_replay — extraction + replay-tool serving.

Run from agent_app/:
  python3 -m pytest tests/test_trace_replay.py
"""

from __future__ import annotations

import json
import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.trace_replay import (  # noqa: E402
    _MISS_OUTPUT,
    build_replay_map,
    build_replay_tools,
    extract_tool_io,
)

# --- Fakes shaped like the real OTel span objects -------------------------


class _Span:
    def __init__(self, name, span_type, inputs, outputs):
        self.name = name
        # Mirror the real attributes dict (verified against agent_traces_otel_spans).
        self.attributes = {
            "mlflow.spanType": span_type,
            "mlflow.spanInputs": inputs,
            "mlflow.spanOutputs": outputs,
        }


class _Data:
    def __init__(self, spans):
        self.spans = spans


class _Trace:
    def __init__(self, spans):
        self.data = _Data(spans)


def _tool_output(content: str):
    # Shape of mlflow.spanOutputs for a LangChain ToolMessage.
    return {"content": content, "name": "x", "tool_call_id": "t1", "type": "tool"}


def _trace_with_tool(name, inputs, content):
    return _Trace(
        [
            _Span("LangGraph", "CHAIN", {}, {}),
            _Span(name, "TOOL", inputs, _tool_output(content)),
        ]
    )


# --- extract_tool_io ------------------------------------------------------


def test_extract_pulls_tool_content():
    tr = _trace_with_tool("get_review_statistics", {}, '{"total_reviews": 37}')
    m = extract_tool_io(tr)
    assert m[("get_review_statistics", "{}")] == '{"total_reviews": 37}'


def test_extract_keys_by_args():
    tr = _trace_with_tool("search_documents", {"limit": 20}, '{"count": 20}')
    m = extract_tool_io(tr)
    assert ("search_documents", json.dumps({"limit": 20}, sort_keys=True)) in m


def test_extract_ignores_non_tool_spans():
    tr = _Trace([_Span("llm", "LLM", {}, {"content": "hi"})])
    assert extract_tool_io(tr) == {}


def test_build_replay_map_merges_traces():
    traces = [
        _trace_with_tool("a", {}, "out_a"),
        _trace_with_tool("b", {"x": 1}, "out_b"),
    ]
    m = build_replay_map(traces)
    assert m[("a", "{}")] == "out_a"
    assert m[("b", json.dumps({"x": 1}, sort_keys=True))] == "out_b"


# --- build_replay_tools ---------------------------------------------------


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"desc for {name}"
        self.args_schema = None


def _invoke(tool, **kwargs):
    # StructuredTool is callable via .func for our purposes.
    return tool.func(**kwargs)


def _rmap():
    return {
        ("search_documents", json.dumps({"limit": 20}, sort_keys=True)): '{"count":20}'
    }


def test_replay_tool_serves_recorded_output():
    tools = build_replay_tools(_rmap(), base_tools=[_FakeTool("search_documents")])
    assert _invoke(tools[0], limit=20) == '{"count":20}'


def test_replay_tool_falls_back_by_name_on_args_miss():
    tools = build_replay_tools(_rmap(), base_tools=[_FakeTool("search_documents")])
    # Different args → no exact key → fall back to the recorded output for the tool.
    assert _invoke(tools[0], limit=5) == '{"count":20}'


def test_replay_tool_miss_returns_stub():
    tools = build_replay_tools({}, base_tools=[_FakeTool("get_review_statistics")])
    assert _invoke(tools[0]) == _MISS_OUTPUT


def test_replay_tools_preserve_names():
    base = [_FakeTool("a"), _FakeTool("b")]
    tools = build_replay_tools({}, base_tools=base)
    assert sorted(t.name for t in tools) == ["a", "b"]
