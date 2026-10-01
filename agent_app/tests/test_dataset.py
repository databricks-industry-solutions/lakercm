"""Unit tests for eval.dataset trace→record projection (curation crash fix).

Offline (light imports only). Run from agent_app/:
  python3 -m pytest tests/test_dataset.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

import json  # noqa: E402

from eval.dataset import (  # noqa: E402
    _curated_stratification_key,
    _records_from_traces,
    _trace_to_record,
    _user_messages_from_inputs,
)


def _fake_trace(
    messages, tool_name=None, expectations=None, trace_id="tr-1", inputs_as_str=False
):
    """A minimal stand-in for an mlflow Trace with the attributes the projection
    reads: data._get_root_span().inputs, data.spans, info.trace_id,
    search_assessments(type=...). `inputs_as_str` simulates the real runtime, where
    `root_span.inputs` comes back as a JSON STRING, not a dict."""
    inputs = {"messages": messages, "guidelines_context": "SHOULD-BE-DROPPED"}
    root = SimpleNamespace(inputs=json.dumps(inputs) if inputs_as_str else inputs)
    spans = [SimpleNamespace(span_type="TOOL", name=tool_name)] if tool_name else []
    data = SimpleNamespace(spans=spans)
    data._get_root_span = lambda: root
    t = SimpleNamespace(data=data, info=SimpleNamespace(trace_id=trace_id))
    t.search_assessments = lambda type=None: [
        SimpleNamespace(name=k, value=v) for k, v in (expectations or {}).items()
    ]
    return t


def test_trace_to_record_shape_and_only_messages():
    msgs = [{"role": "user", "content": "Show me pending documents"}]
    rec = _trace_to_record(_fake_trace(msgs, tool_name="search_documents"))
    # inputs carries ONLY messages — guidelines_context must be dropped (else
    # mlflow.genai.evaluate would call predict_fn with an unexpected kwarg).
    assert rec["inputs"] == {"messages": msgs}
    assert "guidelines_context" not in rec["inputs"]
    assert rec["tags"]["stratification_key"] == "search_documents"
    assert rec["tags"]["source"] == "curated_traces"
    assert rec["tags"]["trace_id"] == "tr-1"
    assert rec["expectations"] == {}


def test_trace_to_record_handles_json_string_inputs():
    # Real runtime: root_span.inputs is a JSON STRING, not a dict. (This is the
    # case that crashed prod with 'str' object has no attribute 'get'.)
    msgs = [{"role": "user", "content": "How many failed?"}]
    rec = _trace_to_record(
        _fake_trace(msgs, tool_name="search_documents", inputs_as_str=True)
    )
    assert rec is not None
    assert rec["inputs"] == {"messages": msgs}


def test_user_messages_from_inputs_coercions():
    msgs = [{"role": "user", "content": "q"}]
    # dict inputs
    assert _user_messages_from_inputs({"messages": msgs}) == msgs
    # JSON-string inputs
    assert _user_messages_from_inputs(json.dumps({"messages": msgs})) == msgs
    # messages itself a JSON string
    assert _user_messages_from_inputs({"messages": json.dumps(msgs)}) == msgs
    # role defaults to 'user'; content-less messages dropped
    assert _user_messages_from_inputs(
        {"messages": [{"content": "x"}, {"role": "u"}]}
    ) == [{"role": "user", "content": "x"}]
    # garbage → empty
    assert _user_messages_from_inputs("not json") == []
    assert _user_messages_from_inputs(None) == []


def test_trace_to_record_lifts_expectation_assessments():
    rec = _trace_to_record(
        _fake_trace(
            [{"role": "user", "content": "q"}],
            expectations={"expected_facts": ["a", "b"]},
        )
    )
    assert rec["expectations"] == {"expected_facts": ["a", "b"]}


def test_trace_to_record_none_without_user_message():
    assert _trace_to_record(_fake_trace([])) is None


def test_curated_stratification_key_refusal_when_no_tools():
    assert _curated_stratification_key(
        _fake_trace([{"role": "user", "content": "x"}])
    ) == ("refusal")


def test_records_from_traces_drops_empty():
    traces = [
        _fake_trace(
            [{"role": "user", "content": "keep"}], tool_name="get_review_statistics"
        ),
        _fake_trace([]),  # dropped
    ]
    recs = _records_from_traces(traces)
    assert len(recs) == 1
    assert recs[0]["tags"]["stratification_key"] == "get_review_statistics"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
