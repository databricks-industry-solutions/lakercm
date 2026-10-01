"""Unit tests for eval.aca — pure decision logic + scipy-free stats.

Run from agent_app/:
  python3 -m pytest tests/test_aca_decision.py
"""

from __future__ import annotations

import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.aca import (  # noqa: E402
    CanaryScore,
    Decision,
    MetricClass,
    MetricResult,
    MetricRole,
    aggregate_score,
    classify_metric,
    decide,
    mann_whitney_u,
    srm_chi_square,
)

# --- Mann-Whitney ---------------------------------------------------------


def test_mann_whitney_clear_separation_significant():
    champ = [0.1, 0.2, 0.15, 0.18, 0.12] * 4
    cand = [0.8, 0.9, 0.85, 0.88, 0.82] * 4
    _, _, p = mann_whitney_u(champ, cand)
    assert p < 0.01


def test_mann_whitney_identical_not_significant():
    x = [0.5, 0.6, 0.4, 0.55, 0.45] * 4
    _, _, p = mann_whitney_u(x, list(x))
    assert p > 0.5


# --- SRM ------------------------------------------------------------------


def test_srm_balanced_passes():
    # 10% candidate split, observed ~10%.
    chi2, p = srm_chi_square(900, 100, 0.10)
    assert p > 0.05


def test_srm_skew_flags():
    # Expected 10% candidate but candidate got almost nothing → SRM.
    chi2, p = srm_chi_square(995, 5, 0.10)
    assert p < 0.001


# --- classify_metric ------------------------------------------------------


def test_guardrail_regression_fails():
    champ = [1.0] * 30
    cand = [0.0] * 30  # safety collapsed
    r = classify_metric(
        "safety", MetricRole.GUARDRAIL, champ, cand, higher_is_better=True
    )
    assert r.classification is MetricClass.FAIL


def test_success_no_regression_passes():
    champ = [0.7] * 30
    cand = [0.72] * 30
    r = classify_metric(
        "composite", MetricRole.SUCCESS, champ, cand, higher_is_better=True
    )
    assert r.classification is MetricClass.PASS


def test_nodata_classifies_nodata():
    r = classify_metric("x", MetricRole.GUARDRAIL, [], [])
    assert r.classification is MetricClass.NODATA


# --- aggregate + decide ---------------------------------------------------


def _r(name, role, cls, critical=False):
    return MetricResult(name, role, cls, 0.0, 0.0, 0.5, critical)


def test_aggregate_all_pass_is_100():
    results = [
        _r("a", MetricRole.SUCCESS, MetricClass.PASS),
        _r("b", MetricRole.GUARDRAIL, MetricClass.PASS),
    ]
    assert aggregate_score(results).score == 100.0


def test_critical_breach_hard_zeros():
    results = [
        _r("a", MetricRole.SUCCESS, MetricClass.PASS),
        _r("safety", MetricRole.GUARDRAIL, MetricClass.FAIL, critical=True),
    ]
    score = aggregate_score(results)
    assert score.score == 0.0 and score.critical_breach


def test_decide_promote_on_clean_pass():
    canary = CanaryScore(
        score=100.0,
        results=[_r("a", MetricRole.SUCCESS, MetricClass.PASS)],
    )
    decision, _ = decide(
        canary, pass_threshold=95, marginal_threshold=75, offline_ci_lower=0.02
    )
    assert decision is Decision.PROMOTE


def test_decide_hold_in_marginal_zone():
    canary = CanaryScore(
        score=80.0,
        results=[
            _r("a", MetricRole.SUCCESS, MetricClass.PASS),
            _r("b", MetricRole.SUCCESS, MetricClass.FAIL),
            _r("c", MetricRole.SUCCESS, MetricClass.PASS),
            _r("d", MetricRole.SUCCESS, MetricClass.PASS),
            _r("e", MetricRole.SUCCESS, MetricClass.PASS),
        ],
    )
    decision, _ = decide(canary, pass_threshold=95, marginal_threshold=75)
    assert decision is Decision.HOLD


def test_decide_rollback_below_marginal():
    canary = CanaryScore(
        score=50.0, results=[_r("a", MetricRole.SUCCESS, MetricClass.FAIL)]
    )
    decision, _ = decide(canary, pass_threshold=95, marginal_threshold=75)
    assert decision is Decision.ROLLBACK


def test_decide_rollback_on_guardrail_even_if_score_high():
    canary = CanaryScore(
        score=96.0,
        results=[_r("safety", MetricRole.GUARDRAIL, MetricClass.FAIL)],
    )
    decision, _ = decide(canary, pass_threshold=95, marginal_threshold=75)
    assert decision is Decision.ROLLBACK


def test_decide_hold_when_pass_but_offline_ci_straddles_zero():
    canary = CanaryScore(
        score=100.0, results=[_r("a", MetricRole.SUCCESS, MetricClass.PASS)]
    )
    decision, _ = decide(
        canary, pass_threshold=95, marginal_threshold=75, offline_ci_lower=-0.01
    )
    assert decision is Decision.HOLD


def test_decide_hold_when_bake_not_elapsed():
    canary = CanaryScore(
        score=100.0, results=[_r("a", MetricRole.SUCCESS, MetricClass.PASS)]
    )
    decision, _ = decide(
        canary,
        pass_threshold=95,
        marginal_threshold=75,
        offline_ci_lower=0.05,
        bake_elapsed=False,
    )
    assert decision is Decision.HOLD
