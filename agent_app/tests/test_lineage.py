"""Unit tests for eval.lineage — content + scorer-set hashing.

Run from agent_app/:
  python3 -m pytest tests/test_lineage.py
"""

from __future__ import annotations

import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.lineage import content_hash, scorer_set_hash  # noqa: E402


def test_content_hash_order_independent():
    a = {"inputs": {"q": "one"}, "tags": {"x": 1}}
    b = {"inputs": {"q": "two"}, "tags": {"x": 2}}
    assert content_hash([a, b]) == content_hash([b, a])


def test_content_hash_changes_with_content():
    a = {"inputs": {"q": "one"}}
    b = {"inputs": {"q": "ONE"}}
    assert content_hash([a]) != content_hash([b])


class _FakeGuideline:
    def __init__(self, name, guidelines):
        self.name = name
        self.guidelines = guidelines


def test_scorer_hash_changes_when_guideline_reworded():
    s1 = _FakeGuideline("audit", "Answer must be grounded in evidence.")
    s2 = _FakeGuideline("audit", "Answer must be grounded in the retrieved evidence.")
    assert scorer_set_hash([s1]) != scorer_set_hash([s2])


def test_scorer_hash_stable_and_set_order_independent():
    s1 = _FakeGuideline("a", "g1")
    s2 = _FakeGuideline("b", "g2")
    assert scorer_set_hash([s1, s2]) == scorer_set_hash([s2, s1])
