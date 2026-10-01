"""
Versioned offline evaluator.

Runs the full scorer set against a curated eval dataset for a *specific*
prompt version. Logs an MLflow run with `prompt_version`, `dataset.digest`,
and a `composite_score` metric so candidate versions are comparable in the
Experiment UI.

CLI:

    python -m eval.run_eval --prompt-version 7
    python -m eval.run_eval --prompt-version 7 --dataset my_catalog.my_schema.agent_eval_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import replace
from typing import Any

import mlflow
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import (
    PROMPT_NAME,
    PromptResolution,
    create_agent,
    load_prompt_version,
)
from agent.tools import set_authorized_user_email
from config import settings
from eval.composite import COMPOSITE_WEIGHTS, coerce_score, compute_composite
from eval.dataset import _ensure_sql_warehouse_for_traces, _warm_sql_warehouse
from eval.gepa_reflection import (
    patch_harness_none_trace_guard,
    patch_harness_eval_result_df_from_memory,
)
from eval.lineage import content_hash
from eval.eval_scorers import offline_scorer_set
from eval.trace_replay import (
    build_replay_map,
    build_replay_tools,
    load_recent_traces,
    records_have_fixtures,
)
from services.observability import init_tracing

logger = logging.getLogger(__name__)


def _row_key(value: Any) -> str:
    """Stable hash of a row's inputs, so champion and candidate per-example
    scores can be paired by content rather than by fragile row order."""
    canon = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def _dataset_name(explicit: str | None) -> str:
    if explicit:
        return explicit
    return (
        settings.eval_dataset_name
        or f"{settings.catalog}.{settings.schema_name}.agent_eval_v2"
    )


def _build_predict_fn(
    resolution: PromptResolution,
    replay_tools: list | None = None,
    prompt_uri: str | None = None,
    degraded: dict | None = None,
):
    """Return a predict_fn(messages=...) that runs the agent with a pinned prompt.

    When `replay_tools` is provided (trace-replay eval), the agent runs with tools
    that return RECORDED outputs instead of querying live Lakebase — so eval never
    touches Postgres. `replay_tools=None` keeps the live-tool behavior.

    When `prompt_uri` is provided (GEPA optimization), predict_fn loads that prompt
    from the registry and `.format()`s it per call — this is the MLflow GEPA
    contract: GEPA substitutes each candidate prompt at that URI, so the candidate
    actually reaches the agent (without this, GEPA optimizes nothing). When
    `prompt_uri` is None (plain run_eval), the pinned `resolution` is used as-is.
    """
    eval_user = {
        "user_email": "eval@lakercm.local",
        "first_name": "Eval",
        "role": "Reviewer",
    }
    set_authorized_user_email(eval_user["user_email"])

    # Degraded-rollout counters (mutated in place; the caller reads them after
    # evaluate to log/gate on the rate). `calls` = predict_fn invocations,
    # `rollout_error` = rows that raised (scored as an empty degraded row),
    # `format_fallback` = GEPA candidate prompts that failed to load/.format()
    # and silently ran the PINNED champion template instead.
    counters = degraded if degraded is not None else {}
    counters.setdefault("calls", 0)
    counters.setdefault("rollout_error", 0)
    counters.setdefault("format_fallback", 0)

    def _predict_inner(messages, _tool_fixtures):
        # Per-record fixtures (agent_eval_v2): build replay tools that serve THIS
        # record's authored tool outputs, from which its expected_facts were
        # derived. `is not None` (not truthiness) so a refusal record with an
        # empty `{}` fixture still gets STUB tools — never the real Lakebase
        # tools — which matters now that the global trace-replay map is skipped.
        # Only a record that omits `_tool_fixtures` entirely (legacy v1) falls
        # back to the global replay tools.
        record_tools = replay_tools
        if _tool_fixtures is not None:
            from eval.trace_replay import replay_tools_from_fixtures

            record_tools = replay_tools_from_fixtures(_tool_fixtures)
        res = resolution
        if prompt_uri is not None:
            # MUST call .format() on the GEPA-managed prompt so GEPA can track +
            # substitute candidate texts into the agent.
            try:
                gepa_prompt = mlflow.genai.load_prompt(prompt_uri)
                rendered = gepa_prompt.format(
                    first_name=eval_user["first_name"], role=eval_user["role"]
                )
                res = replace(resolution, template=rendered)
            except Exception as e:  # noqa: BLE001 — fall back to pinned template
                counters["format_fallback"] += 1
                logger.warning("GEPA prompt load/format failed (%s); using pinned", e)
        agent, _ = create_agent(
            user_context=eval_user,
            thread_id=None,
            prompt_resolution=res,
            tools=record_tools,
            stateless=True,
            # Keep eval/GEPA on the baseline /chat/completions path (no reasoning
            # effort) regardless of the deployed prod defaults — run-to-run
            # comparability matters more here than matching prod's reasoning
            # depth, and reasoning adds latency/cost/variance to every rollout.
            use_responses_api=False,
        )
        lc_messages = [
            HumanMessage(content=m["content"]) for m in messages if m["role"] == "user"
        ]
        result = agent.invoke({"messages": lc_messages})
        out_messages = result.get("messages", [])
        final = ""
        tool_calls: list[str] = []
        for msg in out_messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    content = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                    if content.strip():
                        final = content
                if getattr(msg, "tool_calls", None):
                    tool_calls.extend(tc.get("name", "") for tc in msg.tool_calls)
        return {"response": final, "tools_called": tool_calls}

    # `@mlflow.trace` (NOT a manual start_span) is MLflow's mechanism for a
    # per-call root span CORRELATED to the eval harness's eval_request_id — it's
    # exactly what `convert_predict_fn` does (`predict_fn = mlflow.trace(predict_fn)`)
    # when it decides to. That guarantees `mlflow.get_trace(eval_request_id)` is
    # non-None for EVERY row, so the harness's unguarded `eval_item.trace.info`
    # (_get_new_expectations) can't crash. The try/except turns a failed rollout
    # (e.g. an LLM-timeout) into a degraded low-scoring row instead of raising
    # (which would leave a None trace and abort the whole GEPA run / holdout eval).
    # A manual `mlflow.start_span` did NOT correlate to eval_request_id — hence the
    # earlier crash. Do NOT set MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION (it disables
    # the very tracing path that makes this correlation work).
    @mlflow.trace(name="agent_eval_predict")
    def predict_fn(
        messages: list[dict[str, str]],
        _tool_fixtures: dict | None = None,
    ) -> dict[str, Any]:
        counters["calls"] += 1
        try:
            return _predict_inner(messages, _tool_fixtures)
        except Exception as e:  # noqa: BLE001
            counters["rollout_error"] += 1
            logger.warning("predict_fn row failed (%s); returning degraded output", e)
            return {"response": "", "tools_called": []}

    return predict_fn


def _score_columns(df) -> dict[str, str]:
    """Map scorer name → its score column (`<name>/score` or `<name>/value`)."""
    cols: dict[str, str] = {}
    for col in df.columns:
        for suffix in ("/score", "/value"):
            if col.endswith(suffix):
                cols[col[: -len(suffix)]] = col
                break
    return cols


def _is_absent(v: Any) -> bool:
    """A scorer cell that should be DROPPED (not zeroed) — None or NaN.

    A conditional scorer that skipped a row (e.g. correctness on a no-reference
    row, groundedness with no RETRIEVER span) yields None/NaN. Dropping it lets
    `compute_composite` renormalize over the present scorers — matching exactly
    what the GEPA objective (`composite.composite_objective`) does, so the gate
    and the optimizer compute the same composite for the same scores.
    """
    if v is None:
        return True
    try:
        return bool(v != v)  # NaN is the only value not equal to itself
    except Exception:  # noqa: BLE001 — non-comparable object → present
        return False


def _means_from_eval_metrics(eval_results) -> dict[str, float]:
    """Fallback per-scorer means from mlflow's OWN aggregation metrics
    (`<name>/mean`), which are computed in-memory and do NOT depend on the UC
    trace read-back. Used when the `eval_results` table is absent (e.g. a cold
    SQL warehouse made `construct_eval_result_df` return None) so the composite
    stays correct instead of collapsing to 0.0."""
    metrics = getattr(eval_results, "metrics", None) or {}
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if k.endswith("/mean"):
            try:
                out[k[: -len("/mean")]] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def _extract_per_scorer_means(eval_results) -> dict[str, float]:
    """Pull per-scorer mean scores out of the mlflow.genai.evaluate result.

    Absent cells (None/NaN) are excluded from the mean; a scorer that is absent
    on every row is omitted entirely so the composite renormalizes over it.
    Coercion is `composite.coerce_score` — the SAME mapping the GEPA objective
    uses (categorical 'yes'/'correct'/enum → 1/0). If the per-row table is
    missing/empty (cold-warehouse trace read-back failure), fall back to mlflow's
    own `<name>/mean` aggregation so the composite isn't spuriously 0.0."""
    df = eval_results.tables.get("eval_results")
    if df is None:
        return _means_from_eval_metrics(eval_results)
    means: dict[str, float] = {}
    for name, col in _score_columns(df).items():
        vals = [coerce_score(v) for v in df[col] if not _is_absent(v)]
        if vals:
            means[name] = sum(vals) / len(vals)
    return means or _means_from_eval_metrics(eval_results)


def _extract_per_example(eval_results) -> dict[str, dict[str, float]]:
    """Per-row scorer scores keyed by a stable hash of the row inputs.

    Returns {row_key: {scorer_name: score}} so two prompt versions can be
    paired example-by-example for the bootstrap gate. Absent (None/NaN) cells
    are dropped from the row dict so the per-row composite renormalizes.
    """
    df = eval_results.tables.get("eval_results")
    if df is None:
        return {}
    score_cols = _score_columns(df)
    # Locate the inputs column MLflow echoes back (varies by version).
    input_col = next(
        (c for c in ("inputs", "request", "input") if c in df.columns), None
    )
    per_example: dict[str, dict[str, float]] = {}
    for i, (_, row) in enumerate(df.iterrows()):
        key = _row_key(row[input_col]) if input_col else f"row_{i}"
        per_example[key] = {
            name: coerce_score(row[col])
            for name, col in score_cols.items()
            if not _is_absent(row[col])
        }
    return per_example


def _avg_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Average aligned float dicts key-wise (keys present in any dict)."""
    keys = set().union(*dicts) if dicts else set()
    out: dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def run_eval(
    prompt_version: int,
    dataset_name: str | None = None,
    num_runs: int | None = None,
    seed: int | None = None,
    replay: bool = True,
    records: list | None = None,
) -> dict[str, Any]:
    """Evaluate a pinned prompt version against the eval dataset.

    Returns per-scorer means, the composite, AND per-example composites keyed
    by input hash (so a candidate can be paired against the champion for the
    bootstrap gate). `num_runs` > 1 repeats the evaluation and averages —
    reducing judge/retrieval variance (model temperature is not settable on
    the Databricks FMAPI, so this is the available variance control).

    `records` (optional): evaluate exactly these record dicts instead of the
    full registered dataset. This is how the post-GEPA holdout eval and the
    promotion gate score ONLY the held-out split (the rows GEPA never trained
    on) — without it, re-fetching the dataset by name scores the full set and
    leaks GEPA's train split into the gate. When given, the logged digest is
    `content_hash(records)`.

    `replay=True` (default) runs the agent with trace-replay tools (recorded
    tool outputs from the OTel traces) so eval never touches Lakebase. Set
    `replay=False` only if running where live Lakebase is available.
    """
    init_tracing()
    # mlflow.genai.evaluate WRITES each row's trace to the UC-backed trace store
    # and READS it back (mlflow.get_trace(eval_request_id)) to attach assessments
    # + link to the run. Reading UC traces requires MLFLOW_TRACING_SQL_WAREHOUSE_ID;
    # without it get_trace(silent=True) returns None (→ the None-trace harness
    # crashes) and the non-silent search path raises "SQL warehouse ID is required".
    # This is the ROOT cause of the None traces — set the warehouse id from the
    # experiment tag (cheap: just configures which warehouse reads the small set of
    # eval traces; it does NOT trigger the full-corpus search the fixture path skips).
    _ensure_sql_warehouse_for_traces()
    # Warm the warehouse to RUNNING BEFORE evaluate, so the very first eval's
    # trace read-back populates the per-row `eval_results` table (else the
    # champion side of the gate gets composite 0.0 + 0 paired samples while the
    # warehouse cold-starts). Bounded + best-effort.
    _warm_sql_warehouse()
    # Defense-in-depth: make a None trace non-fatal even if the read-back still
    # misses a row (backfills create_minimal_trace + filters None-trace links).
    patch_harness_none_trace_guard()
    # Build the per-row result table from in-memory eval_results instead of the UC
    # trace read-back, which returns 0 rows for UC-bound experiments (the dangling
    # `_trace_unified` tag) → otherwise the gate sees 0 paired samples and refuses.
    patch_harness_eval_result_df_from_memory()
    runs = num_runs if num_runs is not None else max(1, settings.eval_num_runs)
    name = _dataset_name(dataset_name)
    if records is not None:
        # Score exactly the provided rows (e.g. the deterministic holdout split).
        eval_data: Any = records
        digest = content_hash(records)
        have_fixtures = records_have_fixtures(records)
    else:
        dataset = mlflow.genai.datasets.get_dataset(name=name)
        eval_data = dataset
        digest = getattr(dataset, "digest", "") or ""
        # Cheap peek so a fixture-based dataset also skips the global pull.
        try:
            have_fixtures = records_have_fixtures(dataset.to_df().to_dict("records"))
        except Exception:  # noqa: BLE001
            have_fixtures = False

    resolution = load_prompt_version(prompt_version)

    # Global trace-replay is ONLY a fallback for rows without per-record
    # `_tool_fixtures`. When every row carries fixtures (holdout/gate path, or a
    # fixture-based dataset), SKIP the trace pull: `load_recent_traces()`
    # cold-starts a STOPPED SQL warehouse (waits up to 1200s) and scans the full
    # OTel corpus. Otherwise build it best-effort so a search_traces failure
    # never crashes an eval whose rows are self-contained.
    replay_tools = None
    if replay and not have_fixtures:
        try:
            traces = load_recent_traces()
            replay_tools = build_replay_tools(build_replay_map(traces))
            logger.info(
                "Trace-replay enabled: %d replay tools built", len(replay_tools)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Global trace-replay unavailable (%s); relying on per-record "
                "fixtures.",
                e,
            )
    degraded: dict[str, int] = {}
    predict_fn = _build_predict_fn(
        resolution, replay_tools=replay_tools, degraded=degraded
    )

    per_scorer_runs: list[dict[str, float]] = []
    per_example_runs: list[dict[str, dict[str, float]]] = []

    with mlflow.start_run(run_name=f"eval-prompt-v{prompt_version}") as run:
        mlflow.log_params(
            {
                "prompt_name": PROMPT_NAME,
                "prompt_version": prompt_version,
                "dataset_name": name,
                "dataset_digest": digest,
                "eval_run_count": runs,
                "eval_seed": "" if seed is None else seed,
            }
        )

        for r in range(runs):
            if seed is not None:
                random.seed(seed + r)
            eval_results = mlflow.genai.evaluate(
                data=eval_data,
                predict_fn=predict_fn,
                scorers=offline_scorer_set(),
            )
            per_scorer_runs.append(_extract_per_scorer_means(eval_results))
            per_example_runs.append(_extract_per_example(eval_results))

        per_scorer = _avg_dicts(per_scorer_runs)
        composite = compute_composite(per_scorer)

        # Per-example composite, averaged across runs by row key.
        per_row_composite: dict[str, float] = {}
        all_keys = set().union(*per_example_runs) if per_example_runs else set()
        for key in all_keys:
            comps = [
                compute_composite(run_map[key])
                for run_map in per_example_runs
                if key in run_map
            ]
            if comps:
                per_row_composite[key] = sum(comps) / len(comps)

        # Diagnostic: per-row pairing empty while per-scorer means exist means the
        # harness's per-row table came back empty (the UC trace read-back / df
        # build failed) — NOT "no improvement". The gate will refuse to decide
        # (n_samples=0); make the real cause loud so it isn't misread.
        if not per_row_composite and per_scorer:
            logger.error(
                "Per-row eval extraction returned 0 rows while per-scorer means "
                "exist — the per-row result df is empty (UC trace read-back / "
                "in-memory df build failed). The gate will see 0 paired samples. "
                "Check patch_harness_eval_result_df_from_memory and the eval "
                "experiment's trace binding."
            )
            mlflow.set_tag("eval.per_row_extraction_empty", "true")

        mlflow.log_metric("composite_score", composite)
        for name_, value in per_scorer.items():
            mlflow.log_metric(f"scorer/{name_}", value)
        mlflow.log_param("composite_weights", str(COMPOSITE_WEIGHTS))

        # Degraded-rollout rate: fraction of predict_fn calls that raised and
        # were scored as an empty row. A high rate means the prompt broke the
        # agent on many rows — the gate must fail-closed on this, not average
        # it into a "slightly worse" composite (see compute in compare.py).
        n_calls = degraded.get("calls", 0)
        rollout_errors = degraded.get("rollout_error", 0)
        degraded_rate = rollout_errors / n_calls if n_calls else 0.0
        mlflow.log_metric("degraded_rows", rollout_errors)
        mlflow.log_metric("degraded_rate", degraded_rate)

        logger.info(
            "Run %s — prompt v%d composite=%.4f (n_rows=%d, runs=%d)",
            run.info.run_id,
            prompt_version,
            composite,
            len(per_row_composite),
            runs,
        )

    return {
        "run_id": run.info.run_id,
        "prompt_version": prompt_version,
        "composite": composite,
        "per_scorer": per_scorer,
        "per_row_composite": per_row_composite,
        "n_samples": len(per_row_composite),
        "dataset_name": name,
        "dataset_digest": digest,
        "degraded_rate": degraded_rate,
        "degraded_rows": rollout_errors,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline eval for a prompt version"
    )
    parser.add_argument("--prompt-version", type=int, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = run_eval(prompt_version=args.prompt_version, dataset_name=args.dataset)
    print(f"\nrun_id: {result['run_id']}")
    print(f"prompt v{result['prompt_version']} composite={result['composite']:.4f}")
    for name_, value in sorted(result["per_scorer"].items()):
        print(f"  {name_}: {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
