"""
Champion rollback.

MLflow prompt versions are immutable, so reverting is a single alias re-point —
no data is lost, the prior version is always recoverable. This module makes
that a one-command (and one-function) operation instead of a manual
`set_prompt_alias` someone has to remember the syntax for at 2am. The
auto-rollback monitor (Part B) calls `rollback()` directly.

CLI:

    python -m eval.rollback --to-version 6
    python -m eval.rollback --previous          # revert to the version before current
    python -m eval.rollback --history            # list recent versions, no change
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow

from agent.graph import CHAMPION_ALIAS, PROMPT_NAME

logger = logging.getLogger(__name__)


def current_champion_version() -> int:
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{CHAMPION_ALIAS}")
    return int(prompt.version)


def list_version_history(limit: int = 20) -> list[int]:
    """Best-effort descending list of registered versions for the prompt.

    The registry API surface varies across mlflow releases; try the known
    entry points and fall back to probing downward from the current champion.
    """
    client = mlflow.MlflowClient()
    for method in ("search_prompt_versions", "search_model_versions"):
        fn = getattr(client, method, None)
        if fn is None:
            continue
        try:
            results = fn(f"name='{PROMPT_NAME}'")
            versions = sorted(
                (int(getattr(v, "version", 0)) for v in results), reverse=True
            )
            if versions:
                return versions[:limit]
        except Exception as e:  # noqa: BLE001 — probe; fall through to next
            logger.debug("%s failed: %s", method, e)

    # Fallback: probe downward from the current champion.
    current = current_champion_version()
    found: list[int] = []
    for v in range(current, max(0, current - limit), -1):
        try:
            mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{v}")
            found.append(v)
        except Exception:  # noqa: BLE001
            continue
    return found


def _previous_version() -> int:
    versions = list_version_history()
    current = current_champion_version()
    older = [v for v in versions if v < current]
    if not older:
        raise RuntimeError(
            f"No version older than current champion v{current} to roll back to."
        )
    return max(older)


def rollback(prior_version: int, reason: str = "manual_revert") -> int:
    """Re-point the champion alias to `prior_version`. Returns that version.

    Logs an auditable MLflow run tagged with the reason so an automated
    rollback (guardrail breach) is distinguishable from a human one.
    """
    current = current_champion_version()
    if current == prior_version:
        logger.info("Champion already at v%s — nothing to roll back.", prior_version)
        return prior_version

    with mlflow.start_run(run_name=f"rollback-to-v{prior_version}") as run:
        mlflow.log_params(
            {
                "prompt_name": PROMPT_NAME,
                "from_version": current,
                "to_version": prior_version,
            }
        )
        mlflow.set_tag("rollback.reason", reason)
        mlflow.genai.set_prompt_alias(
            name=PROMPT_NAME, alias=CHAMPION_ALIAS, version=prior_version
        )
        logger.info(
            "ROLLED BACK: %s@%s = v%s (was v%s) reason=%s [run %s]",
            PROMPT_NAME,
            CHAMPION_ALIAS,
            prior_version,
            current,
            reason,
            run.info.run_id,
        )
    return prior_version


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Roll back the champion prompt")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--to-version", type=int, help="Revert champion to this version")
    group.add_argument(
        "--previous", action="store_true", help="Revert to the version before current"
    )
    group.add_argument(
        "--history", action="store_true", help="List recent versions, make no change"
    )
    parser.add_argument("--reason", type=str, default="manual_revert")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.history:
        versions = list_version_history()
        current = current_champion_version()
        print(f"Champion is v{current}. Recent versions: {versions}")
        return 0

    target = args.to_version if args.to_version is not None else _previous_version()
    rollback(target, reason=args.reason)
    print(f"Champion rolled back to v{target}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
