"""
Litellm-free, hard-bounded reflection LM for GEPA prompt optimization.

Why this module exists (verified against installed gepa 0.1.1 + mlflow 3.10.1;
harness patches re-verified against the 3.11.0 -> 3.16.0 wheels, see
patch_harness_none_trace_guard):
GEPA only routes a *string* `reflection_lm` through litellm
(`gepa/api.py:256 if isinstance(reflection_lm, str): import litellm; litellm.completion(...)`)
with NO timeout → litellm's 600s default = the observed ~40-min reflection hang.
The released gepa rejects the `reflection_lm_kwargs` timeout knob (main-only).
A *callable* `reflection_lm` `(str|list[dict]) -> str` is gated OUT of the
litellm branch entirely, so we replace MLflow's string reflection_lm with a
callable that (a) calls the Databricks serving endpoint NATIVELY via
`mlflow.deployments` (no litellm; auto-auth in the job) and (b) enforces a HARD
per-call timeout via a worker thread.

MLflow's `GepaPromptOptimizer` forces `reflection_lm=f"{provider}/{model}"` and
calls `gepa.optimize(**kwargs)` (module-attr lookup), so `patch_gepa_reflection_callable()`
monkeypatches `gepa.optimize` to swap the string for our callable — no MLflow
fork, version-robust. Kept here (light imports) so it's unit-testable without
optimize.py's heavy agent/langchain imports.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "reflection_text_from_response",
    "make_bounded_reflection_lm",
    "patch_gepa_reflection_callable",
    "patch_harness_none_trace_guard",
    "patch_harness_eval_result_df_from_memory",
]


def patch_harness_none_trace_guard() -> bool:
    """Make the MLflow eval harness robust to a None ``eval_item.trace``.

    Root cause (verified against mlflow source, harness ``_run_single``): in the
    ``predict_fn`` branch the harness does
    ``eval_item.trace = mlflow.get_trace(eval_request_id, silent=True)`` with NO
    minimal-trace fallback — unlike the static-dataset branch, which builds
    ``create_minimal_trace(eval_item)``. So if a row's predict_fn emits no trace
    *correlated to* ``eval_request_id`` (a correlation miss — `@mlflow.trace` is
    best-effort, not guaranteed), ``eval_item.trace`` is None and EVERY
    downstream ``eval_item.trace.info...`` dereference crashes the whole run.
    There are ≥3 such unguarded sites (``_get_new_expectations``,
    ``batch_link_traces_to_run``, the eval-df builder), so guarding each consumer
    is whack-a-mole. Instead apply TWO idempotent patches:

      1. Wrap the predict step so a None trace is backfilled with
         ``create_minimal_trace(eval_item)`` — the SAME fallback the
         static-dataset branch already uses — guaranteeing a non-None trace for
         every row and fixing all downstream consumers at once.
      2. Guard ``harness._get_new_expectations`` to return ``[]`` on a None
         trace, because it runs inside the scoring step (before patch #1's
         post-hoc backfill on older mlflow) and would otherwise crash there.
         (The tag-logging loop right after it is already wrapped in try/except
         upstream.)

    TWO HARNESS SHAPES — verified by reading the wheels for 3.10.0/3.10.1 (old)
    and 3.11.0/3.13.0/3.15.2/3.16.0 (new):

      * mlflow <= 3.10.x: one ``_run_single(...) -> EvalResult`` per row,
        dispatched ``executor.submit(_run_single, ...)``. Backfill reads
        ``result.eval_item``.
      * mlflow >= 3.11.0: ``_run_single`` was REMOVED and the harness split into
        a predict/score pipeline. ``_run_predict(eval_item, ...)`` MUTATES the
        eval_item in place and returns None; the unguarded
        ``eval_item.trace = mlflow.get_trace(eval_request_id, silent=True)`` on
        the predict_fn path is still there, so the correlation miss still
        happens. Backfill reads ``args[0]`` / ``kwargs["eval_item"]``.

    Both are called as MODULE GLOBALS at call time (3.16:
    ``_PredictSubmitter._timed_predict`` does a bare ``_run_predict(*args)``), so
    replacing the module attribute takes effect either way. Handling only
    ``_run_single`` made this patch a SILENT NO-OP on every mlflow >= 3.11.

    What 3.16.0 fixed upstream (so patches #2/#3 are now belt-and-suspenders
    there, and still load-bearing on 3.11-3.14): ``_get_new_expectations`` grew
    its own None guard (~3.15), ``batch_link_traces_to_run`` now filters None
    traces, and ``construct_eval_result_df`` skips None-trace rows. What it did
    NOT fix: the missing minimal-trace fallback in ``_run_predict``. On 3.16 a
    correlation miss is therefore no longer a crash but a SILENTLY DROPPED row —
    no trace, no assessments logged, excluded from the result df and the link
    batch — i.e. fewer paired samples for the eval gate. Patch #1 turns that
    dropped row back into a scored, logged, counted one.

    The primary fix is still ``@mlflow.trace`` on predict_fn (real traces when
    correlation succeeds); these patches make a correlation miss NON-FATAL (a
    documented open MLflow harness bug — e.g. issue #20269).
    Idempotent. Returns True if both patches are in place.
    """
    try:
        from mlflow.genai.evaluation import harness
    except Exception as e:  # noqa: BLE001
        logger.warning("mlflow eval harness not importable; no trace guard (%s)", e)
        return False

    ok = True

    # Patch 1 — minimal-trace backfill on the predict step (fixes all downstream
    # `eval_item.trace...` consumers in one place). Version-tolerant across the
    # two harness shapes described in the docstring: prefer `_run_single`
    # (mlflow <= 3.10), else `_run_predict` (mlflow >= 3.11).
    def _backfill(item) -> None:
        try:
            if item is not None and getattr(item, "trace", None) is None:
                # Resolve create_minimal_trace at CALL time (module attr), so a
                # test's fake — and any upstream reshuffle — is honored.
                item.trace = harness.create_minimal_trace(item)
        except Exception as e:  # noqa: BLE001 — never let the backfill crash a row
            logger.warning("minimal-trace backfill failed (%s)", e)

    run_single = getattr(harness, "_run_single", None)
    run_predict = getattr(harness, "_run_predict", None)

    if run_single is not None:
        if not getattr(run_single, "_lakercm_minimal_trace_fallback", False):

            def _patched_run_single(*args, **kwargs):
                result = run_single(*args, **kwargs)
                _backfill(getattr(result, "eval_item", None))
                return result

            _patched_run_single._lakercm_minimal_trace_fallback = True
            harness._run_single = _patched_run_single
            logger.info("Patched mlflow harness _run_single minimal-trace backfill.")
    elif run_predict is not None:
        if not getattr(run_predict, "_lakercm_minimal_trace_fallback", False):

            def _patched_run_predict(*args, **kwargs):
                # Returns None and mutates eval_item IN PLACE, so read the arg,
                # not the return value.
                result = run_predict(*args, **kwargs)
                item = (
                    kwargs["eval_item"]
                    if "eval_item" in kwargs
                    else (args[0] if args else None)
                )
                _backfill(item)
                return result

            _patched_run_predict._lakercm_minimal_trace_fallback = True
            harness._run_predict = _patched_run_predict
            logger.info("Patched mlflow harness _run_predict minimal-trace backfill.")
    else:
        logger.warning(
            "neither harness._run_single (mlflow<=3.10) nor harness._run_predict "
            "(mlflow>=3.11) is present; no minimal-trace backfill"
        )
        ok = False

    # Patch 2 — None-trace guard on _get_new_expectations, which runs inside the
    # scoring step (<=3.10: _run_single; >=3.11: _run_score), i.e. BEFORE patch
    # #1's post-hoc backfill on the old shape. mlflow grew its own None guard
    # here around 3.15, so this is redundant on 3.15+ and still load-bearing on
    # 3.11-3.14.
    orig = getattr(harness, "_get_new_expectations", None)
    if orig is None:
        logger.warning("harness._get_new_expectations missing; no None-trace guard")
        ok = False
    elif not getattr(orig, "_lakercm_none_guarded", False):

        def _guarded(eval_item):
            if getattr(eval_item, "trace", None) is None:
                return []
            return orig(eval_item)

        _guarded._lakercm_none_guarded = True
        harness._get_new_expectations = _guarded
        logger.info("Patched mlflow harness _get_new_expectations None-trace guard.")

    # Patch 3 — the GUARANTEED stop: filter None-trace rows out of
    # `batch_link_traces_to_run`. This is the ONLY post-_run_single consumer that
    # was NOT already None-guarded by mlflow itself on the versions this was
    # written against (`construct_eval_result_df` is
    # try/except-wrapped; `_refresh_eval_result_traces` and the eval-df merge both
    # `continue` on a None trace). Its `[r.eval_item.trace.info.trace_id for r in
    # eval_results]` list-comp crashes the whole run on a single None trace.
    # 3.16 added its own `if ... trace is not None` filter inside
    # batch_link_traces_to_run, so this too is now belt-and-suspenders there and
    # still load-bearing on older versions.
    # `harness.run()` calls it via the name imported into the harness namespace,
    # so patch THAT attribute. Linking is a pure side-effect (associates traces
    # with the MLflow run for the UI) — dropping a None-trace row from the link
    # batch does NOT change any scorer means / the gate composite (those come from
    # the untouched eval_results that evaluate() returns). Belt-and-suspenders with
    # patch #1's backfill: if create_minimal_trace ever fails for a row, this still
    # prevents the crash.
    link = getattr(harness, "batch_link_traces_to_run", None)
    if link is None:
        logger.warning("harness.batch_link_traces_to_run missing; cannot guard linking")
    elif not getattr(link, "_lakercm_none_trace_filtered", False):

        def _has_trace(r) -> bool:
            return getattr(getattr(r, "eval_item", None), "trace", None) is not None

        def _patched_batch_link(*args, **kwargs):
            if "eval_results" in kwargs and kwargs["eval_results"] is not None:
                kwargs["eval_results"] = [
                    r for r in kwargs["eval_results"] if _has_trace(r)
                ]
            elif len(args) >= 2 and args[1] is not None:
                args = list(args)
                args[1] = [r for r in args[1] if _has_trace(r)]
                args = tuple(args)
            return link(*args, **kwargs)

        _patched_batch_link._lakercm_none_trace_filtered = True
        harness.batch_link_traces_to_run = _patched_batch_link
        logger.info(
            "Patched mlflow harness batch_link_traces_to_run None-trace filter."
        )

    return ok


def patch_harness_eval_result_df_from_memory() -> bool:
    """Build the harness's per-row eval-result DataFrame from the IN-MEMORY
    ``eval_results`` instead of a UC trace read-back.

    MLflow's harness constructs the per-row table via
    ``construct_eval_result_df(run_id, traces, eval_results)`` where ``traces`` is
    ``mlflow.search_traces(run_id=...)`` (harness.py). For a UC-bound experiment,
    search_traces resolves the span table from the experiment's
    ``mlflow.experiment.databricksTraceStorageTable`` tag, which points at the
    dangling ``..._trace_unified`` table (a platform quirk — the real spans live in
    ``<prefix>_otel_spans``). So the read-back returns 0 rows → ``construct_eval_
    result_df`` returns None → ``_extract_per_example`` returns {} → the gate sees
    0 paired samples and refuses to decide.

    The per-row scores are already in memory: ``construct_eval_result_df`` RECEIVES
    ``eval_results`` (list[EvalResult]); each carries ``.eval_item.inputs`` and
    ``.get_assessments_dict()`` → ``{<scorer>/value: ...}``. We rebuild the df from
    those, ignoring the broken ``traces`` arg. Idempotent; version-fenced (raises if
    the symbol is gone so a silent no-op can't quietly resurrect the 0-paired bug).
    """
    try:
        from mlflow.genai.evaluation import harness
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "mlflow eval harness not importable; cannot patch df build (%s)", e
        )
        return False
    if not hasattr(harness, "construct_eval_result_df"):
        raise RuntimeError(
            "mlflow harness.construct_eval_result_df is absent — the in-memory "
            "per-row df patch cannot apply (the 0-paired-samples gate bug would "
            "return). Check the installed mlflow version."
        )
    if getattr(harness.construct_eval_result_df, "_lakercm_in_memory_df", False):
        return True

    import pandas as pd

    def _from_memory(
        run_id, traces, eval_results
    ):  # noqa: ARG001 — traces ignored by design
        if not eval_results:
            return None
        rows = []
        for er in eval_results:
            item = getattr(er, "eval_item", None)
            row = {"inputs": getattr(item, "inputs", None)}
            try:
                row.update(er.get_assessments_dict())
            except Exception:  # noqa: BLE001 — a row with no assessments still counts
                pass
            rows.append(row)
        return pd.DataFrame(rows)

    _from_memory._lakercm_in_memory_df = True
    harness.construct_eval_result_df = _from_memory
    logger.info(
        "Patched mlflow harness construct_eval_result_df to build the per-row "
        "table from in-memory eval_results (bypasses the UC trace read-back)."
    )
    return True


def reflection_text_from_response(resp: Any) -> str:
    """Extract assistant text from a Databricks FMAPI chat response (dict OR
    object), defensively across response shapes."""
    choices = (
        resp.get("choices")
        if isinstance(resp, dict)
        else getattr(resp, "choices", None)
    )
    if not choices:
        raise RuntimeError(f"reflection response had no choices: {resp!r}")
    first = choices[0]
    msg = (
        first.get("message")
        if isinstance(first, dict)
        else getattr(first, "message", None)
    )
    if msg is None:
        txt = (
            first.get("text")
            if isinstance(first, dict)
            else getattr(first, "text", None)
        )
        if txt:
            return str(txt)
        raise RuntimeError(f"reflection choice had no message/text: {first!r}")
    content = (
        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    )
    if content is None:
        raise RuntimeError(f"reflection message had no content: {msg!r}")
    return str(content)


def make_bounded_reflection_lm(model_string: str, timeout_s: int, _caller=None):
    """Return a litellm-free, hard-bounded reflection callable `(prompt) -> str`.

    `model_string` is MLflow's "databricks/<endpoint>"; we call `<endpoint>`
    natively. `_caller` is an injection seam for tests (defaults to the real
    `mlflow.deployments` predict). On timeout the worker thread is abandoned
    (`shutdown(wait=False)`) and TimeoutError propagates — with gepa
    `raise_on_exception=False` that just skips the candidate.
    """
    endpoint = model_string.rsplit("/", 1)[-1]  # "databricks/<ep>" → "<ep>"

    def _default_caller(messages: list[dict]) -> str:
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        resp = client.predict(
            endpoint=endpoint,
            inputs={"messages": messages, "max_tokens": 8000},
        )
        return reflection_text_from_response(resp)

    caller = _caller or _default_caller

    def _reflect(prompt) -> str:
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(caller, messages).result(timeout=timeout_s)
        finally:
            ex.shutdown(wait=False)

    return _reflect


def patch_gepa_reflection_callable(timeout_s: int) -> bool:
    """Monkeypatch `gepa.optimize` so a STRING `reflection_lm` is swapped for our
    bounded litellm-free callable before gepa processes it. Idempotent. Returns
    True if the patch is in place (newly applied or already), False if gepa is
    unimportable."""
    try:
        import gepa
    except Exception as e:  # noqa: BLE001
        logger.warning("gepa not importable; cannot patch reflection (%s)", e)
        return False
    if getattr(gepa.optimize, "_lakercm_callable_reflection", False):
        return True
    _orig_optimize = gepa.optimize

    def _patched_optimize(*args, **kwargs):
        rl = kwargs.get("reflection_lm")
        if isinstance(rl, str):
            kwargs["reflection_lm"] = make_bounded_reflection_lm(rl, timeout_s)
            logger.info(
                "Swapped string reflection_lm '%s' for a bounded litellm-free "
                "callable (timeout=%ss).",
                rl,
                timeout_s,
            )
        return _orig_optimize(*args, **kwargs)

    _patched_optimize._lakercm_callable_reflection = True
    gepa.optimize = _patched_optimize
    return True
