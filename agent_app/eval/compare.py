"""
Champion-vs-candidate comparator.

Runs `run_eval` for two prompt versions back-to-back and prints a per-scorer
delta table. Exits non-zero if the gate fails (so it's pipeline-friendly).

CLI:

    python -m eval.compare --baseline-version 6 --candidate-version 7
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import settings
from eval.composite import evaluate_promotion_gate
from eval.run_eval import run_eval
from eval.eval_scorers import AUDIT_JUDGES, audit_floor_deltas
from eval.stats import paired_bootstrap_ci


def _paired_composites(
    baseline: dict, candidate: dict
) -> tuple[list[float], list[float]]:
    """Intersect the two per-example composite maps on shared row keys so the
    bootstrap operates on genuinely paired observations."""
    base_rows = baseline.get("per_row_composite", {})
    cand_rows = candidate.get("per_row_composite", {})
    shared = sorted(set(base_rows) & set(cand_rows))
    return [base_rows[k] for k in shared], [cand_rows[k] for k in shared]


def _holdout_records(
    dataset_name: str | None,
    candidate_version: int | None = None,
    baseline_version: int | None = None,
) -> list:
    """The deterministic holdout split, minus every row either prompt was tuned on.

    `_split_records` (eval/splitting.py) re-derives the split from the dataset
    as it is NOW. That matches what `optimize.optimize()` held out only if the
    dataset has not changed since: its per-stratum balancing makes a flipped
    key's side depend on the other rows, so rows curated in between can move a
    key GEPA trained on into this holdout (eighth review). Each optimize run
    records the keys it trained on, and `gate_holdout` drops the candidate's
    and the champion's. A candidate with no record (hand-authored, or optimized
    before the record existed) is gated on the re-derived holdout, with a
    warning.
    """
    import mlflow

    from agent.graph import PROMPT_NAME
    from config import settings
    from eval.optimize import _dataset_name, _split_records
    from eval.split_record import gate_holdout

    name = _dataset_name(dataset_name)
    df = mlflow.genai.datasets.get_dataset(name=name).to_df()
    _, holdout = _split_records(
        df.to_dict(orient="records"),
        train_pct=settings.eval_train_pct,
        stratify_by="stratification_key",
    )
    return gate_holdout(holdout, candidate_version, PROMPT_NAME, baseline_version)


def compare(
    baseline_version: int,
    candidate_version: int,
    dataset_name: str | None = None,
    num_runs: int | None = None,
    seed: int | None = None,
):
    # Gate on the held-out split only — the rows GEPA never reflected on.
    holdout = _holdout_records(dataset_name, candidate_version, baseline_version)
    print(f"Gate evaluating on {len(holdout)} held-out rows.")
    print(f"Evaluating baseline v{baseline_version}...")
    baseline = run_eval(
        prompt_version=baseline_version,
        records=holdout,
        num_runs=num_runs,
        seed=seed,
    )
    print(f"Evaluating candidate v{candidate_version}...")
    candidate = run_eval(
        prompt_version=candidate_version,
        records=holdout,
        num_runs=num_runs,
        seed=seed,
    )

    # Paired bootstrap CI on the per-example composite difference.
    champ_scores, cand_scores = _paired_composites(baseline, candidate)
    composite_ci = None
    n_samples = len(champ_scores)
    if n_samples > 0:
        boot = paired_bootstrap_ci(champ_scores, cand_scores, seed=seed)
        composite_ci = (boot.lo, boot.hi)

    # Audit-scorer deltas (candidate − champion) for the held-back judges.
    audit_names = {s.name for s in AUDIT_JUDGES}
    audit_deltas = {
        name: float(candidate["per_scorer"].get(name, 0.0))
        - float(baseline["per_scorer"].get(name, 0.0))
        for name in audit_names
    }

    gate = evaluate_promotion_gate(
        champion=baseline["per_scorer"],
        candidate=candidate["per_scorer"],
        composite_ci=composite_ci,
        n_samples=n_samples,
        min_samples=settings.min_eval_samples,
        audit_deltas=audit_deltas,
        audit_floors=audit_floor_deltas(),
        degraded_rate=candidate.get("degraded_rate", 0.0),
        max_degraded_rate=settings.max_degraded_rate,
    )
    print()
    print(f"baseline run_id:  {baseline['run_id']}")
    print(f"candidate run_id: {candidate['run_id']}")
    print()
    print(gate.summary())
    return baseline, candidate, gate


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare two prompt versions")
    parser.add_argument("--baseline-version", type=int, required=True)
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _, _, gate = compare(
        args.baseline_version, args.candidate_version, dataset_name=args.dataset
    )
    return 0 if gate.promote else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
