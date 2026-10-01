"""Unit tests for eval.composite.evaluate_promotion_gate — CI-mode gate.

Run from agent_app/:
  python3 -m pytest tests/test_gate_ci.py
"""

from __future__ import annotations

import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.composite import evaluate_promotion_gate  # noqa: E402

# A balanced champion/candidate baseline used across tests.
CHAMP = {
    "correctness": 0.80,
    "retrieval_groundedness": 0.85,
    "safety": 1.0,
    "relevance_to_query": 0.80,
    "tool_call_correctness": 0.90,
}


def _cand(**overrides):
    c = dict(CHAMP)
    c.update(overrides)
    return c


def test_clean_win_promotes():
    cand = _cand(correctness=0.85, retrieval_groundedness=0.88)
    gate = evaluate_promotion_gate(CHAMP, cand, composite_ci=(0.01, 0.05), n_samples=50)
    assert gate.promote, gate.summary()


def test_noisy_ci_straddling_zero_blocks():
    cand = _cand(correctness=0.81)
    gate = evaluate_promotion_gate(
        CHAMP, cand, composite_ci=(-0.01, 0.05), n_samples=50
    )
    assert not gate.promote
    assert any("not > 0" in r for r in gate.reasons)


def test_too_few_samples_blocks():
    cand = _cand(correctness=0.90)
    gate = evaluate_promotion_gate(
        CHAMP, cand, composite_ci=(0.02, 0.06), n_samples=5, min_samples=20
    )
    assert not gate.promote
    assert any("paired samples" in r for r in gate.reasons)


def test_audit_regression_blocks_even_with_ci_win():
    cand = _cand(correctness=0.90)
    gate = evaluate_promotion_gate(
        CHAMP,
        cand,
        composite_ci=(0.02, 0.06),
        n_samples=50,
        audit_deltas={"audit_answer_supported": -0.05},
        audit_floors={"audit_answer_supported": 0.0},
    )
    assert not gate.promote
    assert any("Audit" in r and "audit_answer_supported" in r for r in gate.reasons)


def test_safety_floor_blocks():
    cand = _cand(safety=0.95, correctness=0.95)
    gate = evaluate_promotion_gate(CHAMP, cand, composite_ci=(0.02, 0.06), n_samples=50)
    assert not gate.promote
    assert any("Safety regressed" in r for r in gate.reasons)


def test_correctness_floor_blocks():
    cand = _cand(correctness=0.70)  # drop of 0.10, well past the -0.01 floor
    gate = evaluate_promotion_gate(CHAMP, cand, composite_ci=(0.02, 0.06), n_samples=50)
    assert not gate.promote
    assert any("Correctness regressed" in r for r in gate.reasons)


def test_high_degraded_rate_blocks_even_with_ci_win():
    # A clean CI win, but the candidate broke the agent on half the rows.
    cand = _cand(correctness=0.90)
    gate = evaluate_promotion_gate(
        CHAMP,
        cand,
        composite_ci=(0.02, 0.06),
        n_samples=50,
        degraded_rate=0.5,
        max_degraded_rate=0.10,
    )
    assert not gate.promote
    assert any("degraded-rollout rate" in r for r in gate.reasons)


def test_low_degraded_rate_still_promotes():
    cand = _cand(correctness=0.85)
    gate = evaluate_promotion_gate(
        CHAMP,
        cand,
        composite_ci=(0.02, 0.06),
        n_samples=50,
        degraded_rate=0.04,
        max_degraded_rate=0.10,
    )
    assert gate.promote, gate.summary()


def test_degraded_rate_ignored_when_thresholds_absent():
    # Backward compatible: no degraded args → no degraded gate.
    cand = _cand(correctness=0.85)
    gate = evaluate_promotion_gate(CHAMP, cand, composite_ci=(0.02, 0.06), n_samples=50)
    assert gate.promote, gate.summary()


def test_legacy_point_estimate_mode_still_works():
    # No composite_ci → falls back to the point-estimate threshold.
    cand = _cand(correctness=0.95, retrieval_groundedness=0.95)
    gate = evaluate_promotion_gate(CHAMP, cand)
    assert gate.promote, gate.summary()
    assert gate.composite_ci is None
