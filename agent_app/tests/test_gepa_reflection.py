"""Unit tests for eval.gepa_reflection — the litellm-free bounded reflection LM.

Offline (light imports only). Run from agent_app/:
  python3 -m pytest tests/test_gepa_reflection.py
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

import pytest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from eval.gepa_reflection import (  # noqa: E402
    make_bounded_reflection_lm,
    patch_gepa_reflection_callable,
    patch_harness_none_trace_guard,
    reflection_text_from_response,
)


def test_reflection_text_extraction_dict_and_object():
    # OpenAI/FMAPI dict shape
    assert (
        reflection_text_from_response({"choices": [{"message": {"content": "hi"}}]})
        == "hi"
    )
    # {"text": ...} shape
    assert reflection_text_from_response({"choices": [{"text": "yo"}]}) == "yo"

    # object/attr shape
    class _Msg:
        content = "obj"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    assert reflection_text_from_response(_Resp()) == "obj"


def test_reflection_text_extraction_errors_on_empty():
    for bad in ({}, {"choices": []}, {"choices": [{}]}):
        try:
            reflection_text_from_response(bad)
            raise AssertionError(f"expected error on {bad}")
        except RuntimeError:
            pass


def test_bounded_callable_str_and_list_inputs_bypass_litellm():
    seen = {}

    def fake_caller(messages):
        seen["messages"] = messages
        return "PROPOSED PROMPT"

    fn = make_bounded_reflection_lm(
        "databricks/databricks-claude-sonnet-4-5", 5, _caller=fake_caller
    )
    # string prompt -> wrapped into a user message
    assert fn("reflect please") == "PROPOSED PROMPT"
    assert seen["messages"] == [{"role": "user", "content": "reflect please"}]
    # list[dict] prompt -> passed through
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    assert fn(msgs) == "PROPOSED PROMPT"
    assert seen["messages"] == msgs


def test_bounded_callable_hard_timeout():
    def slow_caller(messages):
        time.sleep(5)
        return "too slow"

    fn = make_bounded_reflection_lm("databricks/ep", 1, _caller=slow_caller)
    t0 = time.time()
    try:
        fn("hi")
        raise AssertionError("expected TimeoutError")
    except Exception as e:  # concurrent.futures.TimeoutError
        assert "TimeoutError" in type(e).__name__ or "Timeout" in type(e).__name__
    # returns within ~the timeout, not the 5s sleep
    assert time.time() - t0 < 3.5


def test_patch_gepa_swaps_string_reflection_lm_only():
    # Fake a `gepa` module with an optimize() that records its reflection_lm.
    import types

    captured = {}

    def fake_optimize(**kwargs):
        captured["reflection_lm"] = kwargs.get("reflection_lm")
        return "RESULT"

    fake_gepa = types.ModuleType("gepa")
    fake_gepa.optimize = fake_optimize
    sys.modules["gepa"] = fake_gepa
    try:
        assert patch_gepa_reflection_callable(30) is True
        # idempotent
        assert patch_gepa_reflection_callable(30) is True
        # string reflection_lm -> swapped to a callable
        fake_gepa.optimize(reflection_lm="databricks/databricks-claude-sonnet-4-5")
        assert callable(captured["reflection_lm"])
        # a non-string (already a callable) is left untouched
        sentinel = lambda p: "x"  # noqa: E731
        fake_gepa.optimize(reflection_lm=sentinel)
        assert captured["reflection_lm"] is sentinel
    finally:
        sys.modules.pop("gepa", None)


_ABSENT = object()


@contextlib.contextmanager
def _only_harness_shape(harness, keep: str):
    """Force the mlflow eval harness to expose exactly ONE predict-step shape.

    mlflow <= 3.10 dispatched ``_run_single``; 3.11.0 REMOVED it and split the
    harness into a ``_run_predict``/``_run_score`` pipeline. The guard has to
    work on both, and handling only ``_run_single`` made it a silent no-op on
    every mlflow >= 3.11 — so exercise both shapes here regardless of which
    mlflow happens to be installed in this test env. Restores both attributes
    (including "was absent") on exit.
    """
    names = ("_run_single", "_run_predict")
    saved = {n: getattr(harness, n, _ABSENT) for n in names}
    try:
        for name in names:
            if name != keep and hasattr(harness, name):
                delattr(harness, name)
        yield
    finally:
        for name, val in saved.items():
            if val is _ABSENT:
                if hasattr(harness, name):
                    delattr(harness, name)
            else:
                setattr(harness, name, val)


@pytest.mark.parametrize("shape", ["_run_single", "_run_predict"])
def test_patch_harness_makes_none_trace_non_fatal(shape):
    # Patches the REAL installed mlflow eval harness but swaps in controllable
    # fakes for the predict step + create_minimal_trace so both patch behaviors
    # are exercised offline, once per harness shape. Restores everything after.
    try:
        from mlflow.genai.evaluation import harness
    except Exception:  # pragma: no cover — mlflow always present in test env
        pytest.skip("mlflow eval harness not importable")
    if not hasattr(harness, "_run_single"):
        # The module-level pytest. A function-local `import pytest` here made
        # `pytest` local to the WHOLE function, so the skip above raised
        # UnboundLocalError instead of skipping (ruff F823).
        pytest.skip("mlflow eval harness no longer exposes private _run_single")
    orig_gne = harness._get_new_expectations
    orig_cmt = harness.create_minimal_trace
    orig_link = harness.batch_link_traces_to_run
    try:
        with _only_harness_shape(harness, shape):

            class _Item:
                def __init__(self, trace):
                    self.trace = trace

            class _Result:
                def __init__(self, trace=None):
                    self.eval_item = _Item(trace)

            sentinel_trace = object()
            captured = {}
            link_seen = {}

            # <=3.10 shape: returns an EvalResult carrying a None-trace item.
            def fake_run_single(*a, **k):
                return _Result(trace=None)

            # >=3.11 shape: mutates eval_item IN PLACE and returns None. Mimic
            # the real _run_predict, whose predict_fn branch assigns
            # `eval_item.trace = mlflow.get_trace(..., silent=True)` -> None on a
            # correlation miss, with no minimal-trace fallback.
            def fake_run_predict(eval_item, *a, **k):
                eval_item.trace = None
                return None

            def fake_create_minimal_trace(item):
                captured["item"] = item
                return sentinel_trace

            def fake_batch_link(run_id, eval_results, max_batch_size=100):
                # Pre-3.15 original would crash on a None-trace row; record what
                # it receives.
                link_seen["results"] = eval_results
                _ = [r.eval_item.trace for r in eval_results]  # mimics .info deref

            # Install fakes BEFORE patching so the patch wraps our controllable origs.
            setattr(
                harness,
                shape,
                fake_run_single if shape == "_run_single" else fake_run_predict,
            )
            harness.create_minimal_trace = fake_create_minimal_trace
            harness.batch_link_traces_to_run = fake_batch_link

            assert patch_harness_none_trace_guard() is True
            # idempotent — re-patching keeps a single layer of each patch
            assert patch_harness_none_trace_guard() is True
            assert getattr(
                harness._get_new_expectations, "_lakercm_none_guarded", False
            )
            assert getattr(
                getattr(harness, shape), "_lakercm_minimal_trace_fallback", False
            )
            # the OTHER shape must not have been resurrected
            other = "_run_predict" if shape == "_run_single" else "_run_single"
            assert not hasattr(harness, other)
            assert getattr(
                harness.batch_link_traces_to_run, "_lakercm_none_trace_filtered", False
            )

            # Patch 2: _get_new_expectations returns [] on a None trace (no crash).
            class _ItemNoneTrace:
                trace = None

            assert harness._get_new_expectations(_ItemNoneTrace()) == []

            # Patch 1: the predict step backfills a None trace via
            # create_minimal_trace, so every downstream `eval_item.trace`
            # consumer sees a real trace instead of silently dropping the row.
            if shape == "_run_single":
                res = harness._run_single()
                item = res.eval_item
            else:
                item = _Item(trace=object())  # overwritten to None by the fake
                assert harness._run_predict(item) is None
            assert item.trace is sentinel_trace
            assert captured.get("item") is item

            # Patch 3: batch_link filters None-trace rows before the original
            # runs, via BOTH kwargs and positional call styles (harness.run
            # uses kwargs).
            none_row, real_row = _Result(trace=None), _Result(trace=sentinel_trace)
            harness.batch_link_traces_to_run(
                run_id="r", eval_results=[none_row, real_row]
            )
            assert link_seen["results"] == [real_row]
            harness.batch_link_traces_to_run("r", [none_row, real_row])
            assert link_seen["results"] == [real_row]
    finally:
        harness._get_new_expectations = orig_gne
        harness.create_minimal_trace = orig_cmt
        harness.batch_link_traces_to_run = orig_link


def test_patch_harness_none_trace_guard_reports_failure_with_no_predict_step():
    """If BOTH predict-step names vanish in a future mlflow, the guard must
    return False (loudly) rather than silently applying nothing — that silent
    no-op is exactly how the 3.11 removal of _run_single went unnoticed."""
    try:
        from mlflow.genai.evaluation import harness
    except Exception:  # pragma: no cover
        pytest.skip("mlflow eval harness not importable")
    names = ("_run_single", "_run_predict")
    saved = {n: getattr(harness, n, _ABSENT) for n in names}
    orig_gne = harness._get_new_expectations
    orig_link = harness.batch_link_traces_to_run
    try:
        for n in names:
            if hasattr(harness, n):
                delattr(harness, n)
        assert patch_harness_none_trace_guard() is False
    finally:
        for n, v in saved.items():
            if v is not _ABSENT:
                setattr(harness, n, v)
        harness._get_new_expectations = orig_gne
        harness.batch_link_traces_to_run = orig_link


def test_eval_result_df_built_from_memory_not_trace_readback():
    # The patch must build the per-row df from in-memory eval_results (inputs +
    # <scorer>/value columns), ignoring the broken UC trace read-back `traces` arg.
    try:
        from mlflow.genai.evaluation import harness
    except Exception:  # pragma: no cover
        import pytest

        pytest.skip("mlflow eval harness not importable")
    from eval.gepa_reflection import patch_harness_eval_result_df_from_memory

    orig = harness.construct_eval_result_df
    try:
        assert patch_harness_eval_result_df_from_memory() is True
        # idempotent
        assert patch_harness_eval_result_df_from_memory() is True
        assert getattr(harness.construct_eval_result_df, "_lakercm_in_memory_df", False)

        class _Item:
            inputs = {"messages": [{"role": "user", "content": "q1"}]}

        class _ER:
            eval_item = _Item()

            def get_assessments_dict(self):
                return {"correctness/value": 1.0, "safety/value": 1.0}

        # `traces` arg is deliberately empty (simulates the broken read-back).
        df = harness.construct_eval_result_df("run1", [], [_ER(), _ER()])
        assert df is not None and len(df) == 2
        assert "inputs" in df.columns
        assert "correctness/value" in df.columns and "safety/value" in df.columns
        # empty eval_results → None (gate then refuses, which is safe)
        assert harness.construct_eval_result_df("run1", [], []) is None
    finally:
        harness.construct_eval_result_df = orig


def test_eval_result_df_patch_version_fenced():
    # If mlflow ever removes construct_eval_result_df, the patch must FAIL LOUD
    # (not silently no-op and resurrect the 0-paired-samples bug).
    try:
        from mlflow.genai.evaluation import harness
    except Exception:  # pragma: no cover
        import pytest

        pytest.skip("mlflow eval harness not importable")
    from eval.gepa_reflection import patch_harness_eval_result_df_from_memory

    orig = harness.construct_eval_result_df
    try:
        del harness.construct_eval_result_df
        try:
            patch_harness_eval_result_df_from_memory()
            raise AssertionError("expected RuntimeError when symbol is absent")
        except RuntimeError:
            pass
    finally:
        harness.construct_eval_result_df = orig


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
