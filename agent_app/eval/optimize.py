"""
GEPA-driven prompt optimization.

Wraps `mlflow.genai.optimize_prompts(...)` with the LakeRCM-specific
substrate: our eval dataset, our predict_fn, our scorer set. Splits the
dataset (deterministic, by a stable hash of the user message; train fraction =
settings.eval_train_pct, default 60) so GEPA reflects on the train split and we
score the held-out split with the full scorer set.

The optimized prompt is auto-registered by GEPA as a new version. We
read it back, set the `candidate` alias to that version, and run
`run_eval.py` against the holdout to log an auditable MLflow run with
full coverage. Promotion (alias swap to champion) is a separate step
governed by `eval/promote.py`.

CLI:

    python -m eval.optimize
    python -m eval.optimize --reflection-model databricks-claude-sonnet-4-5
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import mlflow
from mlflow.genai.optimize import GepaPromptOptimizer

from agent.graph import (
    CANDIDATE_ALIAS,
    CHAMPION_ALIAS,
    PROMPT_NAME,
    load_prompt_version,
)
from config import settings
from eval.composite import composite_objective
from eval.dataset import _ensure_sql_warehouse_for_traces
from eval.gepa_reflection import (
    patch_gepa_reflection_callable,
    patch_harness_none_trace_guard,
    patch_harness_eval_result_df_from_memory,
)
from eval.lineage import content_hash, scorer_set_hash
from eval.run_eval import _build_predict_fn, run_eval
from eval.split_record import log_train_keys
from eval.splitting import (  # noqa: F401 — re-exported for eval.compare
    _record_field,
    _split_key,
    _split_records,
)
from eval.eval_scorers import gepa_scorer_set
from eval.trace_replay import (
    build_replay_map,
    build_replay_tools,
    load_recent_traces,
    records_have_fixtures,
)
from services.observability import init_tracing

logger = logging.getLogger(__name__)


# GEPA's per-example aggregation objective — lives in composite.py (the single
# source of truth for composite logic) so it's unit-testable without optimize's
# heavy imports. Re-exported here as the name optimize() passes to GEPA.
_composite_objective = composite_objective


def _extract_optimized_version(result: Any) -> int:
    """Resolve the optimized prompt's version from a GEPA result, across the
    mlflow schema variants we may run against.

    The result attribute is mlflow-version-dependent (verified against wheels
    3.2→3.14; none ever exposed `optimized_prompt_uris`/`result_uris`/
    `prompt_uris` as a result attribute — that earlier probe always failed):
      - mlflow >= 3.7:  result.optimized_prompts: list[PromptVersion]
      - mlflow <= 3.3:  result.prompt: PromptVersion  (singular)
    Take the first PromptVersion's `.version`. On an unrecognized schema, log
    the type MRO + attrs (so the job captures the real shape) and, as a last
    resort, try the legacy URI-style attributes before raising.
    """
    pv = None
    prompts = getattr(result, "optimized_prompts", None)  # plural (>=3.7)
    if prompts:
        pv = prompts[0]
    if pv is None:
        pv = getattr(result, "prompt", None)  # singular (<=3.3)
    version = getattr(pv, "version", None) if pv is not None else None
    if version is not None:
        return int(version)

    # Last-resort legacy probe (older builds / forks).
    for attr in ("optimized_prompt_uris", "result_uris", "prompt_uris"):
        value = getattr(result, attr, None)
        if value:
            uri = value[0] if isinstance(value, list) else value
            return int(str(uri).rsplit("/", 1)[-1])

    logger.error(
        "Unrecognized optimize_prompts result schema: mro=%s attrs=%s",
        getattr(type(result), "__mro__", type(result)),
        getattr(result, "__dict__", None),
    )
    raise RuntimeError(
        "GEPA finished but no optimized prompt version was found on the result "
        f"(type={type(result)!r}). See the logged schema diagnostic above."
    )


def _resolve_champion_version() -> int:
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{CHAMPION_ALIAS}")
    return int(prompt.version)


def _dataset_name(explicit: str | None) -> str:
    return (
        explicit
        or settings.eval_dataset_name
        or (f"{settings.catalog}.{settings.schema_name}.agent_eval_v2")
    )


def optimize(
    reflection_model: str | None = None,
    dataset_name: str | None = None,
) -> int:
    """Run GEPA, set candidate alias to the new version, run holdout eval. Returns the new prompt version."""
    init_tracing()
    name = _dataset_name(dataset_name)
    dataset = mlflow.genai.datasets.get_dataset(name=name)

    # Pull records via DataFrame so we can split them; reconstruct list-of-dicts
    # for GEPA. (mlflow.genai.evaluate accepts both Datasets and dict lists.)
    df = dataset.to_df()
    records = df.to_dict(orient="records")
    train_records, holdout_records = _split_records(
        records, train_pct=settings.eval_train_pct, stratify_by="stratification_key"
    )
    logger.info(
        "Dataset %s split: %d train / %d holdout",
        name,
        len(train_records),
        len(holdout_records),
    )
    if not train_records:
        raise RuntimeError(
            "Train split is empty — curate more traces before optimizing."
        )

    champion_version = _resolve_champion_version()
    champion_resolution = load_prompt_version(champion_version)
    # Per-record fixtures (agent_eval_v2) are the eval substrate — each train
    # record carries `_tool_fixtures`, so predict_fn builds per-record replay
    # tools. When EVERY row has fixtures, SKIP the global trace-replay map
    # entirely: `load_recent_traces()` cold-starts a STOPPED monitoring SQL
    # warehouse (waits up to 1200s) and scans the full OTel corpus — pure dead
    # weight here, and the dominant cause of the "hung" GEPA runs. Only build it
    # (best-effort) for fixture-less datasets that actually need the fallback.
    replay_tools = None
    if records_have_fixtures(train_records):
        logger.info(
            "All %d train rows carry _tool_fixtures — skipping global trace-replay "
            "(no SQL-warehouse cold-start / corpus scan).",
            len(train_records),
        )
    else:
        try:
            replay_tools = build_replay_tools(build_replay_map(load_recent_traces()))
            logger.info("GEPA trace-replay: %d replay tools built", len(replay_tools))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Global trace-replay unavailable (%s); relying on per-record "
                "fixtures.",
                e,
            )
    champion_uri = f"prompts:/{PROMPT_NAME}/{champion_version}"
    gepa_degraded: dict[str, int] = {}
    predict_fn = _build_predict_fn(
        champion_resolution,
        replay_tools=replay_tools,
        prompt_uri=champion_uri,
        degraded=gepa_degraded,
    )

    # GEPA's internal mlflow.genai.evaluate writes/reads each rollout's trace to
    # the UC trace store; reading needs MLFLOW_TRACING_SQL_WAREHOUSE_ID or
    # get_trace silently returns None (degrading trace-dependent scorers). Set it
    # from the experiment tag so GEPA scores against real traces, not None.
    _ensure_sql_warehouse_for_traces()
    # Replace GEPA's string reflection_lm with a bounded, litellm-free callable
    # (the real fix for the ~40-min reflection "hang"). See gepa_reflection.py.
    # Fail loud if the swap didn't take: a silent no-op (gepa unimportable, or a
    # future gepa upgrade that relocates gepa.optimize despite the requirements
    # pin) would let the ~40-min litellm reflection hang return with no error
    # signal. Refuse to run rather than burn a 40-min hung job.
    _patched = patch_gepa_reflection_callable(settings.gepa_reflection_timeout)
    import gepa as _gepa

    if not _patched or not getattr(
        _gepa.optimize, "_lakercm_callable_reflection", False
    ):
        raise RuntimeError(
            "GEPA reflection monkeypatch did not apply — refusing to run "
            "(the ~40-min litellm reflection hang would return silently). "
            "Check the gepa version pin in agent_app/requirements-eval.txt."
        )
    # Backstop the MLflow eval-harness None-trace bug for GEPA's internal evals too.
    patch_harness_none_trace_guard()
    # Build the harness per-row df from in-memory eval_results (the UC trace
    # read-back returns 0 rows for UC-bound experiments — dangling `_trace_unified`
    # tag — which otherwise empties GEPA's per-row scoring too).
    patch_harness_eval_result_df_from_memory()

    refl = reflection_model or settings.gepa_reflection_model
    # Cost cap: bound GEPA's metric calls (rollouts × per-call inference price).
    # Older GepaPromptOptimizer signatures may not accept the kwarg — degrade.
    max_calls = settings.gepa_max_metric_calls
    # gepa_kwargs — keys MLflow's GepaPromptOptimizer does NOT override, so they
    # pass straight to gepa.optimize:
    #   • reflection_minibatch_size: keep the reflection prompt within context.
    #   • raise_on_exception=False: a slow/failed (timed-out) reflection SKIPS the
    #     candidate instead of aborting the whole run.
    #   • stop_callbacks=[TimeoutStopCondition]: wall-clock ceiling so gepa.optimize
    #     RETURNS the best candidate (→ candidate registered → loop completes) even
    #     if reflections keep timing out — reflection failures don't increment
    #     max_metric_calls, so this is the only thing that guarantees termination.
    gepa_kwargs: dict[str, Any] = {
        "reflection_minibatch_size": settings.gepa_reflection_minibatch,
        "raise_on_exception": False,
    }
    try:
        from gepa.utils.stop_condition import TimeoutStopCondition

        gepa_kwargs["stop_callbacks"] = [
            TimeoutStopCondition(timeout_seconds=settings.gepa_wall_clock_seconds)
        ]
    except Exception as e:  # noqa: BLE001 — degrade to max_metric_calls + job timeout
        logger.warning("TimeoutStopCondition unavailable (%s); no wall-clock stop", e)
    try:
        optimizer = GepaPromptOptimizer(
            reflection_model=f"databricks:/{refl}",
            max_metric_calls=max_calls or None,
            gepa_kwargs=gepa_kwargs,
        )
    except TypeError:
        logger.warning(
            "GepaPromptOptimizer does not accept max_metric_calls/gepa_kwargs; "
            "falling back to reflection_model only.",
        )
        optimizer = GepaPromptOptimizer(reflection_model=f"databricks:/{refl}")

    gepa_scorers = gepa_scorer_set()
    dataset_hash = content_hash(train_records)
    scorers_hash = scorer_set_hash(gepa_scorers)

    with mlflow.start_run(run_name=f"gepa-optimize-from-v{champion_version}") as run:
        mlflow.log_params(
            {
                "prompt_name": PROMPT_NAME,
                "starting_champion_version": champion_version,
                "reflection_model": refl,
                "dataset_name": name,
                "train_pct": settings.eval_train_pct,
                "train_records": len(train_records),
                "holdout_records": len(holdout_records),
                "gepa_max_metric_calls": max_calls,
                "dataset_content_hash": dataset_hash,
                "scorer_set_hash": scorers_hash,
            }
        )
        # Exactly what GEPA trains on, so the promotion gate can exclude it even
        # if curation changes the dataset before promotion runs (see
        # eval/split_record.py). Recorded BEFORE the expensive optimization, and
        # allowed to fail the run: a candidate without it would be gated on a
        # re-derived holdout that can contain its own training rows.
        log_train_keys(train_records)

        result = mlflow.genai.optimize_prompts(
            predict_fn=predict_fn,
            train_data=train_records,
            prompt_uris=[f"prompts:/{PROMPT_NAME}/{champion_version}"],
            optimizer=optimizer,
            scorers=gepa_scorers,
            # Aggregate per-scorer scores into the single objective GEPA
            # maximizes. The composite skips None-valued scorers (conditional
            # correctness / groundedness on rows lacking the prerequisite) and
            # renormalizes over the rest — so GEPA never sees a non-numeric value.
            aggregation=_composite_objective,
        )

        # Degraded-rollout signals. A non-zero format_fallback means a candidate
        # prompt failed to load/.format() and the rollout silently ran the PINNED
        # CHAMPION template — so GEPA scored champion behavior under the
        # candidate's name, wasting the run. Surface it loudly rather than letting
        # it pass silently.
        _fallbacks = gepa_degraded.get("format_fallback", 0)
        mlflow.log_metric("gepa_format_fallback_rows", _fallbacks)
        mlflow.log_metric("degraded_rows", gepa_degraded.get("rollout_error", 0))
        if _fallbacks:
            logger.error(
                "GEPA: %d rollout(s) FELL BACK to the pinned champion template "
                "(candidate prompt failed to load/format) — the optimized result "
                "may reflect champion behavior, not the candidate. Investigate.",
                _fallbacks,
            )
            mlflow.set_tag("optimize.format_fallback_rows", str(_fallbacks))

        # GEPA's own base-vs-optimized objective — authoritative, independent of
        # our composite (and non-zero proof the run had signal).
        for _attr in ("initial_eval_score", "final_eval_score"):
            _val = getattr(result, _attr, None)
            if _val is not None:
                try:
                    mlflow.log_metric(f"gepa_{_attr}", float(_val))
                except (TypeError, ValueError):
                    pass
                logger.info("GEPA %s = %s", _attr, _val)

        new_version = _extract_optimized_version(result)
        mlflow.log_param("optimized_prompt_version", new_version)
        logger.info("GEPA produced %s/%s", PROMPT_NAME, new_version)

        # Update commit message and set candidate alias.
        try:
            mlflow.genai.set_prompt_alias(
                name=PROMPT_NAME, alias=CANDIDATE_ALIAS, version=new_version
            )
            logger.info("Set %s@%s = v%s", PROMPT_NAME, CANDIDATE_ALIAS, new_version)
        except Exception as e:
            logger.warning("Could not set candidate alias: %s", e)
            mlflow.set_tag("optimize.alias_error", str(e)[:500])

        mlflow.set_tag("optimize.gepa_run_id", run.info.run_id)

    # Eval the optimized prompt against the HELD-OUT split only — the rows GEPA
    # never reflected on — for an auditable, leakage-free MLflow run. Passing
    # `records=` (not `dataset_name=`) is essential: dataset_name re-fetches the
    # FULL set and would score GEPA's train rows too.
    if holdout_records:
        logger.info(
            "Running holdout eval for v%s on %d held-out rows...",
            new_version,
            len(holdout_records),
        )
        run_eval(prompt_version=new_version, records=holdout_records)
    else:
        logger.warning("Holdout split was empty; skipping holdout eval.")

    print()
    print(f"GEPA produced prompt v{new_version} from champion v{champion_version}.")
    print(f"Candidate alias '{CANDIDATE_ALIAS}' now points to v{new_version}.")
    print(f"Run `python -m eval.promote --candidate-version {new_version}` to promote.")
    return new_version


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Optimize the champion prompt with GEPA"
    )
    parser.add_argument("--reflection-model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    optimize(reflection_model=args.reflection_model, dataset_name=args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
