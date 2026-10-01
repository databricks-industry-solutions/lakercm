"""The record of what GEPA trained a candidate prompt on.

`optimize.optimize()` writes it on its MLflow run; the promotion gate
(`compare._holdout_records`) reads it and drops those rows from the holdout.

Why re-deriving the split is not enough: `splitting._split_records` balances
each stratum, so a flipped key's side depends on the rest of the dataset, and
the optimize and promotion jobs run separately with the weekly curation job free
to add rows in between. A stratum that was all-holdout has a key flipped into
train, GEPA trains on it, curation then adds a train-side key to that stratum,
the flip is no longer needed, and the gate scores the candidate on a row it was
optimized against (eighth review). With the record, the gate excludes exactly
what was trained on, whatever the dataset looks like by then.

Only SHA-256 digests of the split keys are stored, never question text, which
can quote claim details.
"""

from __future__ import annotations

import logging
from typing import Any

from eval.splitting import _split_key, key_digest

TRAIN_KEYS_ARTIFACT = "split/train_keys.json"

logger = logging.getLogger(__name__)


def train_key_digests(train_records: list[Any]) -> list[str]:
    """Sorted, de-duplicated digests of the split keys in `train_records`."""
    return sorted({key_digest(_split_key(r)) for r in train_records})


def log_train_keys(train_records: list[Any]) -> None:
    """Record the train split on the ACTIVE MLflow run (optimize's run)."""
    import mlflow

    mlflow.log_dict(
        {"train_key_sha256": train_key_digests(train_records)}, TRAIN_KEYS_ARTIFACT
    )


def trained_key_digests(candidate_version: int, prompt_name: str) -> set[str] | None:
    """Digests of the keys GEPA trained `candidate_version` on, or None.

    Found through the optimize run that produced the version: it logs
    `prompt_name` and `optimized_prompt_version` params and the artifact above,
    in the eval experiment the promotion job also runs in. None when there is
    no such run or it predates the record (a hand-authored candidate, or one
    optimized before this existed); the caller falls back and says so.
    """
    import mlflow

    try:
        runs = mlflow.search_runs(
            filter_string=(
                f"params.optimized_prompt_version = '{int(candidate_version)}' "
                f"and params.prompt_name = '{prompt_name}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        )
        if not runs:
            return None
        record = mlflow.artifacts.load_dict(
            f"runs:/{runs[0].info.run_id}/{TRAIN_KEYS_ARTIFACT}"
        )
        return set(record["train_key_sha256"])
    except Exception as e:  # noqa: BLE001 — the caller falls back, loudly
        logger.warning(
            "Could not read the train split recorded for v%s: %s",
            candidate_version,
            e,
        )
        return None


def drop_trained(records: list[Any], trained: set[str]) -> list[Any]:
    """`records` minus every record whose split key is in `trained`."""
    return [r for r in records if key_digest(_split_key(r)) not in trained]


def gate_holdout(
    holdout: list[Any],
    candidate_version: int | None,
    prompt_name: str,
    baseline_version: int | None = None,
) -> list[Any]:
    """The re-derived `holdout` minus every row EITHER prompt was tuned on.

    The gate scores the champion and the candidate on the same rows, so a row
    either one was optimized against biases the comparison. Dropping only the
    candidate's rows left the champion's (usually an earlier GEPA output) in,
    tilting the gate against promotion (ninth review).

    A candidate with no record (hand-authored, or optimized before the record
    existed) cannot rule out its own training rows, and says so. A champion with
    no record is the normal case for a hand-written first prompt, so it only
    has nothing to drop.
    """
    trained: set[str] = set()
    if candidate_version is not None:
        digests = trained_key_digests(candidate_version, prompt_name)
        if digests is None:
            logger.warning(
                "No recorded GEPA train split for v%s; gating on a holdout that "
                "cannot rule out rows it was optimized on.",
                candidate_version,
            )
        else:
            trained |= digests
    if baseline_version is not None:
        trained |= trained_key_digests(baseline_version, prompt_name) or set()
    if not trained:
        return holdout
    kept = drop_trained(holdout, trained)
    if len(kept) < len(holdout):
        logger.warning(
            "Dropped %d holdout row(s) that v%s or v%s was optimized on (the "
            "dataset changed after optimization).",
            len(holdout) - len(kept),
            candidate_version,
            baseline_version,
        )
    return kept
