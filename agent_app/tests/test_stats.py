"""Unit tests for eval.stats — paired bootstrap + always-valid CIs.

Run from agent_app/:
  python3 -m pytest tests/test_stats.py
"""

from __future__ import annotations

import os
import sys

# Make agent_app/ importable from tests/.
_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.stats import always_valid_ci, paired_bootstrap_ci  # noqa: E402


def test_bootstrap_clear_win_excludes_zero():
    champion = [0.5] * 60
    candidate = [0.7] * 60
    res = paired_bootstrap_ci(champion, candidate, seed=7)
    assert abs(res.delta_mean - 0.2) < 1e-9
    # Constant diff → CI collapses to the point; strictly above zero.
    assert res.lo > 0.0
    assert res.excludes_zero_above


def test_bootstrap_noise_straddles_zero():
    import numpy as np

    rng = np.random.default_rng(0)
    # Same underlying mean → the paired difference is centered on zero.
    champion = list(rng.normal(0.6, 0.1, size=80))
    candidate = list(rng.normal(0.6, 0.1, size=80))
    res = paired_bootstrap_ci(champion, candidate, seed=3)
    assert res.lo <= 0.0 <= res.hi
    assert not res.excludes_zero_above


def test_bootstrap_length_mismatch_raises():
    try:
        paired_bootstrap_ci([0.1, 0.2], [0.1])
        raise AssertionError("expected ValueError on mismatched lengths")
    except ValueError:
        pass


def test_always_valid_is_wider_than_fixed_horizon():
    import numpy as np

    rng = np.random.default_rng(1)
    champion = list(rng.normal(0.5, 0.1, size=50))
    candidate = list(rng.normal(0.55, 0.1, size=50))
    boot = paired_bootstrap_ci(champion, candidate, seed=1)
    seq = always_valid_ci(champion, candidate)
    # Anytime-valid sequence pays for continuous peeking → wider interval.
    assert (seq.hi - seq.lo) >= (boot.hi - boot.lo)
    # Both centered on the same mean diff.
    assert abs(seq.delta_mean - boot.delta_mean) < 1e-9


def test_always_valid_bounded_does_not_collapse_on_constant_diff():
    # A sub-Gaussian bounded CS uses an a-priori scale, so it does NOT collapse
    # to a point on degenerate (constant) data — it can't know the support is
    # degenerate. Center is correct; the interval is symmetric with width > 0.
    seq = always_valid_ci([0.4] * 30, [0.6] * 30)
    assert abs(seq.delta_mean - 0.2) < 1e-9
    assert seq.hi - seq.lo > 0.0
    assert abs((seq.hi + seq.lo) / 2 - 0.2) < 1e-9


def test_empty_inputs_safe():
    res = paired_bootstrap_ci([], [])
    assert res.n == 0 and res.delta_mean == 0.0


def test_always_valid_width_shrinks_with_n():
    """H1 regression: with a FIXED rho the half-width must tighten as n grows
    (~sigma*sqrt(ln n / n)). The old rho=n/10 + *sqrt(n) froze it ~constant."""
    import numpy as np

    rng = np.random.default_rng(7)
    widths = {}
    for n in (30, 120, 480):
        champ = list(rng.normal(0.50, 0.10, size=n))
        cand = list(rng.normal(0.55, 0.10, size=n))
        seq = always_valid_ci(champ, cand)
        widths[n] = seq.hi - seq.lo
    assert widths[120] < widths[30]
    assert widths[480] < widths[120]
    # 16x more data should roughly halve the width (sqrt-ish shrinkage).
    assert widths[480] < 0.6 * widths[30]


def test_always_valid_null_coverage_under_peeking():
    """Family-wise: under a true null (Δ=0), the probability the sequence EVER
    excludes 0 across many sequential peeks stays ≤ alpha. This is the anytime-
    valid guarantee that rho∝n had broken."""
    import numpy as np

    rng = np.random.default_rng(11)
    alpha = 0.05
    n_streams, n_max = 300, 200
    false_alarms = 0
    for _ in range(n_streams):
        diffs = rng.normal(0.0, 0.1, size=n_max)  # true mean diff = 0
        champ = [0.0] * n_max
        cand = list(diffs)
        ever = False
        for n in range(2, n_max + 1, 10):  # peek repeatedly as data accrues
            seq = always_valid_ci(champ[:n], cand[:n], alpha=alpha)
            if seq.lo > 0.0 or seq.hi < 0.0:
                ever = True
                break
        false_alarms += ever
    # Anytime-valid → far below alpha; allow Monte-Carlo slack.
    assert false_alarms / n_streams <= 0.10
