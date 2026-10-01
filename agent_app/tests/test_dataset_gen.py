"""Unit tests for eval.dataset_gen — the authored fixture-grounded eval set.

Offline only (no mlflow / no Lakebase): dataset_gen's mlflow/config imports are
lazy, so building records + generating facts runs with stdlib alone.

Run from agent_app/:
  python3 -m pytest tests/test_dataset_gen.py
"""

from __future__ import annotations

import json
import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.dataset_gen import (  # noqa: E402
    V2_RECORDS,
    V2_SPECS,
    _facts_from_fixture,
)


def _is_refusal(rec):
    return not rec["expectations"]["expected_tools"]


def test_dataset_size_and_diversity():
    # ≥66 so a stratified 30% holdout clears min_eval_samples (20) for the gate.
    assert len(V2_RECORDS) >= 66, "want enough rows that the 30% holdout ≥ 20"
    cats = {r["tags"]["category"] for r in V2_RECORDS}
    assert len(cats) >= 9, f"want diverse categories, got {sorted(cats)}"
    diffs = {r["tags"]["difficulty"] for r in V2_RECORDS}
    assert {"easy", "med", "hard"} <= diffs
    # Refusal + safety coverage must survive (guardrail signal for the gate).
    assert sum(_is_refusal(r) for r in V2_RECORDS) >= 4


def test_record_schema():
    for r in V2_RECORDS:
        inp = r["inputs"]
        msgs = inp["messages"]
        assert msgs and msgs[0]["role"] == "user" and msgs[0]["content"].strip()
        assert "_tool_fixtures" in inp and isinstance(inp["_tool_fixtures"], dict)
        exp = r["expectations"]
        assert isinstance(exp["expected_tools"], list)
        assert isinstance(exp["expected_facts"], list)
        tags = r["tags"]
        assert tags["source"] == "authored-grounded"
        assert tags["stratification_key"]
        # JSON-serializable end to end (UC dataset round-trips through JSON).
        json.dumps(r, default=str)


def test_non_refusal_records_have_fixture_per_tool_and_facts():
    for r in V2_RECORDS:
        if _is_refusal(r):
            continue
        tools = r["expectations"]["expected_tools"]
        fixtures = r["inputs"]["_tool_fixtures"]
        for t in tools:
            assert t in fixtures, f"{t} expected but no fixture in {r['tags']}"
        assert len(r["expectations"]["expected_facts"]) >= 1


def test_facts_are_derived_from_fixtures_no_drift():
    """Every fixture-derived fact must be present in expected_facts — proving the
    reference the judge scores against is generated FROM the served output."""
    for r in V2_RECORDS:
        facts = r["expectations"]["expected_facts"]
        fixtures = r["inputs"]["_tool_fixtures"]
        for tool in r["expectations"]["expected_tools"]:
            derived = _facts_from_fixture(tool, fixtures[tool])
            assert derived, f"no facts derivable for {tool} in {r['tags']}"
            for fact in derived:
                assert fact in facts, f"drift: {fact!r} not in expected_facts"


def test_refusal_records_have_no_fixtures_and_behavioral_facts():
    refusals = [r for r in V2_RECORDS if _is_refusal(r)]
    assert refusals
    for r in refusals:
        assert r["inputs"]["_tool_fixtures"] == {}
        facts = " ".join(r["expectations"]["expected_facts"]).lower()
        assert any(
            w in facts for w in ("decline", "refus", "does not", "cannot", "only")
        )


def test_facts_quote_fixture_values():
    """Numeric/identifier anchors in facts must actually appear in the fixture
    JSON — so a correct answer can be grounded in the served data."""
    for spec in V2_SPECS:
        if not spec.get("tools"):
            continue
        for tool in spec["tools"]:
            fx = spec["fixtures"][tool]
            blob = json.dumps(fx, default=str)
            for fact in _facts_from_fixture(tool, fx):
                # the trailing token of each generated fact is a fixture value
                # (count / name / pct / id); spot-check the strongest anchor.
                if tool == "get_review_statistics":
                    assert str(fx["accuracy_pct"]) in " ".join(
                        _facts_from_fixture(tool, fx)
                    )
                # document-name facts must be real fixture rows
                if fact.endswith(".png"):
                    assert fact in blob


def test_records_have_fixtures_key_presence_and_json_string():
    import json

    from eval.trace_replay import records_have_fixtures

    # Every V2 record (incl. refusals with empty {}) is self-contained.
    assert records_have_fixtures(V2_RECORDS) is True
    # Empty inputs / missing key → not self-contained (needs global replay).
    assert records_have_fixtures([{"inputs": {"messages": []}}]) is False
    assert records_have_fixtures([]) is False
    # Refusal-style empty fixture still counts (key present).
    assert records_have_fixtures([{"inputs": {"_tool_fixtures": {}}}]) is True
    # inputs serialized as a JSON string (UC backing column is string) is handled.
    s = json.dumps({"messages": [], "_tool_fixtures": {"x": {}}})
    assert records_have_fixtures([{"inputs": s}]) is True
    assert records_have_fixtures([{"inputs": json.dumps({"messages": []})}]) is False


def test_replay_tools_from_fixtures_serve_and_stub():
    from langchain_core.tools import StructuredTool

    from eval.trace_replay import replay_tools_from_fixtures

    def _rev() -> str:
        """stats"""
        return "{}"

    def _search() -> str:
        """search"""
        return "{}"

    base = [
        StructuredTool.from_function(
            func=_rev, name="get_review_statistics", description="stats"
        ),
        StructuredTool.from_function(
            func=_search, name="search_documents", description="search"
        ),
    ]
    fixtures = {"get_review_statistics": {"total_reviews": 39, "accuracy_pct": 89.7}}
    tools = replay_tools_from_fixtures(fixtures, base_tools=base)
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"get_review_statistics", "search_documents"}

    served = by_name["get_review_statistics"].func()
    assert json.loads(served)["total_reviews"] == 39
    # tool with no fixture → deterministic stub, not the served fixture
    stub = json.loads(by_name["search_documents"].func())
    assert stub.get("results") == [] and "total_reviews" not in stub


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
