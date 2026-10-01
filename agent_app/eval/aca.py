"""
Automated Canary Analysis (ACA) for prompt promotion.

Models the Netflix Kayenta judge: each metric is compared champion-vs-candidate,
classified pass/fail, aggregated to a 0-100 score, and mapped to a three-way
decision — promote / hold-for-human / rollback. A `critical` guardrail breach
hard-zeros the score (Kayenta's "NodataFailMetric"/critical convention).

Two metric roles (Google SRE guardrail-vs-success framing):
  - SUCCESS  metrics must *improve* (or at least not regress) — e.g. composite
    quality. The offline paired-bootstrap CI from stats.py is the primary
    success signal; ACA's online score is corroborating.
  - GUARDRAIL metrics must *not regress* past tolerance — Safety, raw-text
    leak, latency, cost, error rate. These are the fast surrogates that gate
    immediate auto-rollback (true accuracy labels lag — see the plan's
    lagging-label handling).

The decision logic and statistics here are pure (numpy + stdlib) and unit
tested in tests/test_aca_decision.py. `score_canary` — which pulls the actual
production traces — is the thin live-data shell on top.

Stats are scipy-free: Mann-Whitney U via the normal approximation with tie
correction; the SRM chi-square p-value via the closed form for df=1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = [
    "MetricRole",
    "MetricClass",
    "MetricResult",
    "CanaryScore",
    "Decision",
    "mann_whitney_u",
    "srm_chi_square",
    "classify_metric",
    "aggregate_score",
    "decide",
]


class MetricRole(str, Enum):
    SUCCESS = "success"
    GUARDRAIL = "guardrail"


class MetricClass(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NODATA = "nodata"


class Decision(str, Enum):
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


@dataclass
class MetricResult:
    name: str
    role: MetricRole
    classification: MetricClass
    champion_mean: float
    candidate_mean: float
    p_value: float
    critical: bool = False
    note: str = ""


@dataclass
class CanaryScore:
    score: float
    results: list[MetricResult] = field(default_factory=list)
    critical_breach: bool = False

    def guardrail_breached(self) -> bool:
        return any(
            r.role is MetricRole.GUARDRAIL and r.classification is MetricClass.FAIL
            for r in self.results
        )

    def summary(self) -> str:
        lines = [f"ACA score: {self.score:.1f}/100"]
        for r in self.results:
            lines.append(
                f"  [{r.role.value}/{r.classification.value}] {r.name}: "
                f"{r.champion_mean:.3f} → {r.candidate_mean:.3f} (p={r.p_value:.3f})"
                + ("  CRITICAL" if r.critical else "")
            )
        return "\n".join(lines)


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Mann-Whitney U for samples a (champion) vs b (candidate).

    Returns (U_b, z, two-sided p) via the normal approximation with tie
    correction. U_b is the statistic oriented to b; large U_b means b tends to
    rank above a. scipy-free.
    """
    a = list(a)
    b = list(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 1.0

    combined = np.array(a + b, dtype=float)
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=float)
    # Average ranks for ties.
    i = 0
    sorted_vals = combined[order]
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    rank_b = ranks[n1:].sum()
    u_b = rank_b - n2 * (n2 + 1) / 2.0

    mu = n1 * n2 / 2.0
    # Tie-corrected variance.
    _, counts = np.unique(combined, return_counts=True)
    tie_term = (counts**3 - counts).sum()
    n = n1 + n2
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return float(u_b), 0.0, 1.0
    z = (u_b - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided normal tail
    return float(u_b), float(z), float(p)


def srm_chi_square(
    observed_champion: int, observed_candidate: int, expected_candidate_ratio: float
) -> tuple[float, float]:
    """Sample-ratio-mismatch test. Returns (chi2, p) for df=1.

    Compares observed champion/candidate trace counts to the configured split
    (expected_candidate_ratio = CANDIDATE_TRAFFIC_PCT / 100). A small p means
    traffic isn't splitting as configured — e.g. candidate invocations are
    silently failing back to champion — and the comparison is untrustworthy.
    p computed in closed form: P(chi2_1 > x) = erfc(sqrt(x/2)).
    """
    total = observed_champion + observed_candidate
    if total == 0 or not (0.0 < expected_candidate_ratio < 1.0):
        return 0.0, 1.0
    exp_cand = total * expected_candidate_ratio
    exp_champ = total * (1.0 - expected_candidate_ratio)
    chi2 = (observed_candidate - exp_cand) ** 2 / exp_cand + (
        observed_champion - exp_champ
    ) ** 2 / exp_champ
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p)


def classify_metric(
    name: str,
    role: MetricRole,
    champion: list[float],
    candidate: list[float],
    higher_is_better: bool = True,
    allowed_abs_regression: float = 0.0,
    alpha: float = 0.05,
    critical: bool = False,
) -> MetricResult:
    """Classify one metric pass/fail/nodata via mean delta + Mann-Whitney.

    GUARDRAIL: fails if the candidate mean regresses beyond
    `allowed_abs_regression` AND the difference is significant (p < alpha).
    SUCCESS: passes if the candidate does not significantly regress (it need
    not significantly improve — the offline CI is the primary improvement
    signal; ACA online just guards against an online regression).
    """
    if not champion or not candidate:
        return MetricResult(
            name, role, MetricClass.NODATA, 0.0, 0.0, 1.0, critical, "no data"
        )

    champ_mean = float(np.mean(champion))
    cand_mean = float(np.mean(candidate))
    _, _, p = mann_whitney_u(champion, candidate)

    # Signed regression in the "bad" direction.
    delta = cand_mean - champ_mean
    regression = -delta if higher_is_better else delta  # positive = worse

    significant = p < alpha
    failed = significant and regression > allowed_abs_regression
    classification = MetricClass.FAIL if failed else MetricClass.PASS
    note = (
        f"regression {regression:+.4f} (allowed {allowed_abs_regression:.4f}), "
        f"{'sig' if significant else 'ns'}"
    )
    return MetricResult(
        name, role, classification, champ_mean, cand_mean, p, critical, note
    )


def aggregate_score(results: list[MetricResult]) -> CanaryScore:
    """Kayenta-style aggregate: passed/total × 100 over metrics with data.

    A FAIL on any `critical` metric hard-zeros the score.
    """
    scored = [r for r in results if r.classification is not MetricClass.NODATA]
    critical_breach = any(
        r.critical and r.classification is MetricClass.FAIL for r in results
    )
    if not scored:
        score = 100.0  # nothing to judge → don't block on absence of data alone
    else:
        passed = sum(1 for r in scored if r.classification is MetricClass.PASS)
        score = passed / len(scored) * 100.0
    if critical_breach:
        score = 0.0
    return CanaryScore(score=score, results=results, critical_breach=critical_breach)


def decide(
    canary: CanaryScore,
    *,
    pass_threshold: float,
    marginal_threshold: float,
    offline_ci_lower: float | None = None,
    bake_elapsed: bool = True,
) -> tuple[Decision, list[str]]:
    """Map an ACA score + offline CI + bake state to promote/hold/rollback.

    - rollback: score < marginal, or any critical/guardrail breach.
    - promote:  score >= pass AND guardrails clean AND (if provided) the
                offline paired-CI lower bound > 0 AND bake window elapsed.
    - hold:     everything in between (the human-review zone).
    """
    reasons: list[str] = []

    if canary.critical_breach:
        return Decision.ROLLBACK, ["critical guardrail breach → rollback"]
    if canary.guardrail_breached():
        return Decision.ROLLBACK, ["guardrail metric regressed → rollback"]
    if canary.score < marginal_threshold:
        return Decision.ROLLBACK, [
            f"score {canary.score:.1f} < marginal {marginal_threshold} → rollback"
        ]

    if canary.score < pass_threshold:
        return Decision.HOLD, [
            f"score {canary.score:.1f} in [{marginal_threshold}, {pass_threshold}) "
            f"→ hold for human"
        ]

    # score >= pass: check the remaining promote preconditions.
    if not bake_elapsed:
        return Decision.HOLD, ["score passes but bake window not elapsed → hold"]
    if offline_ci_lower is not None and offline_ci_lower <= 0.0:
        return Decision.HOLD, [
            f"score passes but offline CI lower bound {offline_ci_lower:+.4f} "
            f"not > 0 → hold"
        ]

    reasons.append(
        f"score {canary.score:.1f} >= pass {pass_threshold}, guardrails clean"
    )
    if offline_ci_lower is not None:
        reasons.append(f"offline CI lower bound {offline_ci_lower:+.4f} > 0")
    return Decision.PROMOTE, reasons
