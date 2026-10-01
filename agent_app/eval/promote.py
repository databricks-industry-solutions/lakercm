"""
Champion-promotion gate.

Re-runs the comparator on the latest dataset; if the candidate clears the
hard Safety gate, the Correctness floor, AND the composite delta threshold,
swaps the champion alias to the candidate version and clears the candidate
alias. Otherwise: prints why and exits non-zero with the alias untouched.

CLI:

    python -m eval.promote --candidate-version 7
    python -m eval.promote --candidate-version 7 --dry-run    # eval only, no alias change
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow

from agent.graph import CANDIDATE_ALIAS, CHAMPION_ALIAS, PROMPT_NAME
from eval.compare import compare
from eval.lineage import content_hash, scorer_set_hash
from eval.eval_scorers import offline_scorer_set

logger = logging.getLogger(__name__)

# Fixed seed for the promotion gate. Threaded into compare() → run_eval (RNG)
# and the paired-bootstrap CI so the gate verdict is DETERMINISTIC: re-running
# the same candidate can't flip promote/block on bootstrap resampling noise.
# Logged on the promote run for auditability.
_GATE_SEED = 1729


def _lineage_tags(dataset_name: str) -> dict[str, str]:
    """Content hashes that pin this promotion to exact data + scorer defs."""
    tags = {"scorer_set_hash": scorer_set_hash(offline_scorer_set())}
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=dataset_name)
        tags["dataset_content_hash"] = content_hash(dataset.to_df())
    except Exception as e:  # noqa: BLE001 — lineage is best-effort, never blocks
        logger.warning("Could not hash dataset for lineage: %s", e)
    return tags


def _resolve_champion_version() -> int:
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{CHAMPION_ALIAS}")
    return int(prompt.version)


def promote(
    candidate_version: int, dataset_name: str | None = None, dry_run: bool = False
) -> int:
    """Returns 0 on promote, 1 on block, 2 on usage error."""
    champion_version = _resolve_champion_version()
    if champion_version == candidate_version:
        print(
            f"Candidate v{candidate_version} already pointed at by '{CHAMPION_ALIAS}'. Nothing to do."
        )
        return 0

    baseline, candidate, gate = compare(
        baseline_version=champion_version,
        candidate_version=candidate_version,
        dataset_name=dataset_name,
        seed=_GATE_SEED,
    )

    # Log a single auditable run that records the gate decision.
    with mlflow.start_run(run_name=f"promote-eval-v{candidate_version}") as run:
        mlflow.log_params(
            {
                "prompt_name": PROMPT_NAME,
                "champion_version": champion_version,
                "candidate_version": candidate_version,
                "dataset_name": baseline["dataset_name"],
                "dataset_digest": baseline["dataset_digest"],
                "promote": gate.promote,
                "dry_run": dry_run,
                "gate_seed": _GATE_SEED,
            }
        )
        mlflow.log_metric("champion_composite", gate.champion_composite)
        mlflow.log_metric("candidate_composite", gate.candidate_composite)
        mlflow.log_metric(
            "composite_delta", gate.candidate_composite - gate.champion_composite
        )
        mlflow.log_metric(
            "candidate_degraded_rate", float(candidate.get("degraded_rate", 0.0))
        )
        for name, delta in gate.deltas.items():
            mlflow.log_metric(f"delta/{name}", delta)
        mlflow.set_tag("gate.summary", gate.summary())
        mlflow.set_tag(
            "gate.linked_runs",
            f"baseline={baseline['run_id']} candidate={candidate['run_id']}",
        )
        for tag_key, tag_val in _lineage_tags(baseline["dataset_name"]).items():
            mlflow.set_tag(tag_key, tag_val)

        if not gate.promote:
            print()
            print(f"BLOCKED — alias '{CHAMPION_ALIAS}' unchanged.")
            return 1

        if dry_run:
            print()
            print("Gate would PROMOTE — but --dry-run, no alias change.")
            return 0

        try:
            mlflow.genai.set_prompt_alias(
                name=PROMPT_NAME, alias=CHAMPION_ALIAS, version=candidate_version
            )
            logger.info(
                "Promoted: %s@%s = v%s (was v%s)",
                PROMPT_NAME,
                CHAMPION_ALIAS,
                candidate_version,
                champion_version,
            )
            try:
                mlflow.genai.delete_prompt_alias(
                    name=PROMPT_NAME, alias=CANDIDATE_ALIAS
                )
                logger.info("Cleared candidate alias %s", CANDIDATE_ALIAS)
            except Exception as e:
                logger.info("Could not clear candidate alias (probably unset): %s", e)
        except Exception as e:
            logger.error("Alias swap failed: %s", e)
            mlflow.set_tag("gate.alias_swap_error", str(e)[:500])
            return 1

        print()
        print(
            f"PROMOTED — {PROMPT_NAME}@{CHAMPION_ALIAS} now points to v{candidate_version} "
            f"(was v{champion_version}). Run id: {run.info.run_id}"
        )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Promote a candidate prompt version to champion"
    )
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    return promote(
        candidate_version=args.candidate_version,
        dataset_name=args.dataset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
