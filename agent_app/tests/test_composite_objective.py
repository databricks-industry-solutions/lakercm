"""Regression tests for the GEPA aggregation objective.

The live GEPA run scored a flat 0.0 objective ("Base program full valset score:
0.0") even though per-scorer means were healthy (correctness 0.94). Cause: the
builtin judges return Feedback whose `.value` is the CATEGORICAL string/enum
"yes"/"no" — a bare float() dropped them, leaving compute_composite with no
weighted key. These tests lock in the categorical→float coercion.

Offline (composite.py has no heavy deps). Run from agent_app/:
  python3 -m pytest tests/test_composite_objective.py
"""

from __future__ import annotations

import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.composite import coerce_score, composite_objective  # noqa: E402


class _Feedback:
    """Shaped like mlflow Feedback (has a `.value`)."""

    def __init__(self, value):
        self.value = value


class _Rating:
    """Shaped like CategoricalRating (enum whose str() is 'Rating.YES')."""

    def __init__(self, v):
        self.value = v

    def __str__(self):
        return f"CategoricalRating.{self.value.upper()}"


def test_coerce_score_categorical_and_numeric():
    assert coerce_score("yes") == 1.0
    assert coerce_score("no") == 0.0
    assert coerce_score("PASS") == 1.0
    assert coerce_score(_Rating("yes")) == 1.0
    assert coerce_score(_Rating("no")) == 0.0
    assert coerce_score(0.88) == 0.88
    assert coerce_score(True) == 1.0 and coerce_score(False) == 0.0


def test_objective_nonzero_with_all_categorical_judges():
    """The exact shape that produced 0.0 in prod: every weighted judge is a
    categorical Feedback; only the (unweighted) code scorers are floats."""
    scores = {
        "correctness": _Feedback("yes"),
        "safety": _Feedback("yes"),
        "relevance_to_query": _Feedback(_Rating("yes")),
        "tool_call_correctness": _Feedback("yes"),
        "retrieval_groundedness": None,  # conditional skip
        "latency_under_slo": 1.0,
        "no_sql_warehouse_regression": 1.0,
        "tool_call_budget": 1.0,
    }
    assert composite_objective(scores) == 1.0


def test_objective_zero_when_all_judges_fail():
    scores = {
        "correctness": _Feedback("no"),
        "safety": _Feedback("no"),
        "relevance_to_query": _Feedback("no"),
        "tool_call_correctness": _Feedback("no"),
    }
    assert composite_objective(scores) == 0.0


def test_objective_partial_and_renormalizes_over_present_keys():
    # correctness good (0.40), safety bad (0.20); groundedness/relevance/tcc absent.
    # weighted over present keys: (0.40*1 + 0.20*0) / (0.40+0.20) = 0.6667
    scores = {"correctness": _Feedback("yes"), "safety": _Feedback("no")}
    assert abs(composite_objective(scores) - (0.40 / 0.60)) < 1e-9


def test_none_is_dropped_not_zeroed():
    # Only correctness present (all others None) → composite == correctness value.
    scores = {
        "correctness": _Feedback("yes"),
        "safety": None,
        "relevance_to_query": None,
    }
    assert composite_objective(scores) == 1.0


def test_list_of_feedback_takes_first():
    scores = {"correctness": [_Feedback("yes")], "safety": [_Feedback("yes")]}
    assert composite_objective(scores) == 1.0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
