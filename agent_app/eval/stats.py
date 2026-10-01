"""
Statistical primitives for the promotion gate.

Two estimators, both on the *paired* per-example difference (candidate minus
champion scored on the same eval rows). Pairing cancels per-example difficulty,
so the difference has far lower variance than two independent means — the same
power argument behind the paired t-test / McNemar (NCSS PASS: ~30-60% fewer
samples for the same MDE).

  - `paired_bootstrap_ci` — fixed-horizon. Resample the paired diffs with
    replacement to get a percentile CI on the mean diff. Use when the gate is
    evaluated ONCE against a frozen eval set (eval/promote.py).

  - `always_valid_ci` — anytime-valid confidence sequence (normal-mixture, per
    Howard et al. 2021 "Time-uniform Chernoff bounds" / Johari et al. "Always
    Valid Inference", arXiv:1512.04922). Valid at EVERY peek, so it stays sound
    when the auto-promote job re-evaluates as labels trickle in. Fixed-horizon
    p-values are invalid under that repeated peeking; this is the fix.

Pure-numpy: scipy is not a guaranteed dependency in the Apps runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BootstrapResult",
    "SequenceResult",
    "paired_bootstrap_ci",
    "always_valid_ci",
]


@dataclass
class BootstrapResult:
    """A bootstrap CI on the paired mean difference (candidate − champion)."""

    delta_mean: float
    lo: float
    hi: float
    n: int
    ci: float

    @property
    def excludes_zero_above(self) -> bool:
        """True iff the whole CI is above zero — candidate genuinely better."""
        return self.lo > 0.0

    def summary(self) -> str:
        return (
            f"paired Δ={self.delta_mean:+.4f} "
            f"[{self.lo:+.4f}, {self.hi:+.4f}] "
            f"({int(self.ci * 100)}% bootstrap, n={self.n})"
        )


@dataclass
class SequenceResult:
    """An anytime-valid confidence sequence on the running paired mean diff."""

    delta_mean: float
    lo: float
    hi: float
    n: int
    alpha: float

    @property
    def excludes_zero_above(self) -> bool:
        return self.lo > 0.0

    def summary(self) -> str:
        return (
            f"paired Δ={self.delta_mean:+.4f} "
            f"[{self.lo:+.4f}, {self.hi:+.4f}] "
            f"(always-valid α={self.alpha}, n={self.n})"
        )


def _paired_diffs(champion: list[float], candidate: list[float]) -> np.ndarray:
    """Element-wise candidate − champion. Requires equal length / aligned rows."""
    if len(champion) != len(candidate):
        raise ValueError(
            f"paired stats need aligned rows: got {len(champion)} champion "
            f"vs {len(candidate)} candidate scores"
        )
    return np.asarray(candidate, dtype=float) - np.asarray(champion, dtype=float)


def paired_bootstrap_ci(
    champion: list[float],
    candidate: list[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Percentile bootstrap CI on the mean paired difference.

    `champion`/`candidate` are per-example scores in the SAME row order. The
    gate promotes only when the returned CI lies entirely above zero.
    """
    diffs = _paired_diffs(champion, candidate)
    n = len(diffs)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0, ci)

    rng = np.random.default_rng(seed)
    # Resample row indices with replacement, n_boot times, vectorized.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)

    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    lo, hi = np.percentile(boot_means, [lo_pct, hi_pct])
    return BootstrapResult(
        delta_mean=float(diffs.mean()),
        lo=float(lo),
        hi=float(hi),
        n=n,
        ci=ci,
    )


def always_valid_ci(
    champion: list[float],
    candidate: list[float],
    alpha: float = 0.05,
    rho: float = 1.0,
    scale: float = 1.0,
) -> SequenceResult:
    """Sub-Gaussian normal-mixture anytime-valid confidence sequence on the
    paired mean diff (Robbins 1970 / Howard et al. 2021).

    Half-width on the MEAN:

        r_n = scale * sqrt( ((n + rho) / n^2) * ( ln((n + rho)/rho) + 2 ln(1/alpha) ) )

    Two properties make this *valid at every peek* (the whole point — the
    auto-promote job re-evaluates as labels trickle in, and fixed-horizon CIs
    are invalid under that repeated peeking):

      - **Fixed `rho`** (a tuning constant, NOT a function of n). The ln term
        then grows ~½·ln(n), giving the time-uniform boundary; r_n shrinks like
        scale·sqrt(ln n / n).
      - **A KNOWN sub-Gaussian `scale` bound**, not the plug-in sample std.
        Paired composite diffs lie in [-1, 1] (each composite ∈ [0,1]), so the
        diffs are sub-Gaussian with parameter ≤ 1 ⇒ `scale=1.0` is valid a
        priori. Using the running sample std instead under-estimates the scale
        at small n and *breaks* coverage (empirically ~0.23 vs target 0.05).

    Two bugs in the prior version (both verified): `rho` defaulted to n/10
    (froze the ln term → not time-uniform) and the radius used the plug-in std
    × an extra sqrt(n). This conservative bounded form has empirical family-wise
    error ~0 under the null with repeated peeking.

    Tightening follow-up (not now): an empirical-Bernstein confidence sequence
    (Howard et al. 2021) adapts `scale` to the observed variance while staying
    valid — tighter for low-variance score diffs. Worth it before relying on
    this for fully-automated promotion.
    """
    diffs = _paired_diffs(champion, candidate)
    n = len(diffs)
    if n < 1:
        return SequenceResult(0.0, -math.inf, math.inf, n, alpha)

    mean = float(diffs.mean())
    rho_eff = rho if rho and rho > 0 else 1.0
    radius = scale * math.sqrt(
        ((n + rho_eff) / (n * n))
        * (math.log((n + rho_eff) / rho_eff) + 2.0 * math.log(1.0 / alpha))
    )
    return SequenceResult(
        delta_mean=mean,
        lo=mean - radius,
        hi=mean + radius,
        n=n,
        alpha=alpha,
    )
