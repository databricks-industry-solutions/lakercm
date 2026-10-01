"""The train-split record, round-tripped through a real MLflow tracking store.

optimize() writes it on its run; the promotion gate finds that run by the
params optimize() logs and reads it back (eval/split_record.py). A pure unit
test would take on faith that the filter string, the artifact path and the
search API agree; this runs them against real mlflow on a throwaway sqlite
store. Skips cleanly without mlflow.

Run from agent_app/:
  python3 -m pytest tests/test_split_record.py
"""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)


def _real_mlflow() -> bool:
    try:
        if importlib.util.find_spec("mlflow") is None:
            return False
    except (ImportError, ValueError):
        return False
    mod = sys.modules.get("mlflow")
    return mod is None or bool(getattr(mod, "__file__", None))


def _rec(question: str) -> dict:
    return {"inputs": {"messages": [{"role": "user", "content": question}]}}


@unittest.skipUnless(_real_mlflow(), "real mlflow not importable (absent or stubbed)")
class TestTrainSplitRecordRoundTrip(unittest.TestCase):
    PROMPT = "cat.sch.lakercm_agent"

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        from eval import split_record

        cls.mlflow, cls.sr = mlflow, split_record
        cls._prev_uri = mlflow.get_tracking_uri()
        cls._tmp = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"sqlite:///{cls._tmp}/record.db")
        exp = mlflow.create_experiment(
            "split-record", artifact_location=Path(cls._tmp, "artifacts").as_uri()
        )
        mlflow.set_experiment(experiment_id=exp)

        cls.train = [_rec("How many denials last week?"), _rec("Show claim CLM-1.")]
        cls.champion_train = [_rec("Which payer denies the most MRIs?")]
        # What optimize() logs, in its order: params first, the record before
        # GEPA runs, the produced version after. v6 is the champion, an earlier
        # GEPA output; v7 is the candidate.
        for version, train in ((6, cls.champion_train), (7, cls.train)):
            with mlflow.start_run():
                mlflow.log_params(
                    {"prompt_name": cls.PROMPT, "train_records": len(train)}
                )
                split_record.log_train_keys(train)
                mlflow.log_param("optimized_prompt_version", version)
        # An optimize run from before the record existed.
        with mlflow.start_run():
            mlflow.log_params(
                {"prompt_name": cls.PROMPT, "optimized_prompt_version": 5}
            )

    @classmethod
    def tearDownClass(cls):
        cls.mlflow.set_tracking_uri(cls._prev_uri)
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_the_gate_reads_back_exactly_what_optimize_recorded(self):
        self.assertEqual(
            self.sr.trained_key_digests(7, self.PROMPT),
            set(self.sr.train_key_digests(self.train)),
        )

    def test_another_version_or_prompt_has_no_record(self):
        self.assertIsNone(self.sr.trained_key_digests(8, self.PROMPT))
        self.assertIsNone(self.sr.trained_key_digests(7, "cat.sch.other_prompt"))

    def test_a_run_without_the_record_falls_back_loudly(self):
        with self.assertLogs("eval.split_record", level="WARNING"):
            self.assertIsNone(self.sr.trained_key_digests(5, self.PROMPT))

    def test_the_recorded_rows_are_dropped(self):
        trained = self.sr.trained_key_digests(7, self.PROMPT)
        other = _rec("What is the clean-claim rate?")
        self.assertEqual(self.sr.drop_trained(self.train + [other], trained), [other])

    def test_the_gate_holdout_drops_the_candidates_training_rows(self):
        other = _rec("What is the clean-claim rate?")
        with self.assertLogs("eval.split_record", level="WARNING") as logs:
            gate = self.sr.gate_holdout(self.train + [other], 7, self.PROMPT)
        self.assertEqual(gate, [other])
        self.assertIn("Dropped 2 holdout row(s)", "\n".join(logs.output))

    def test_an_unrecorded_candidate_keeps_the_holdout_and_says_so(self):
        holdout = self.train + [_rec("What is the clean-claim rate?")]
        with self.assertLogs("eval.split_record", level="WARNING") as logs:
            self.assertEqual(self.sr.gate_holdout(holdout, 8, self.PROMPT), holdout)
        self.assertIn("No recorded GEPA train split", "\n".join(logs.output))
        self.assertEqual(self.sr.gate_holdout(holdout, None, self.PROMPT), holdout)

    def test_the_champions_training_rows_are_dropped_too(self):
        # Ninth review: the champion is usually an earlier GEPA output, and
        # leaving its training rows in tilted the paired comparison its way.
        other = _rec("What is the clean-claim rate?")
        holdout = self.champion_train + self.train + [other]
        self.assertEqual(
            self.sr.gate_holdout(holdout, 7, self.PROMPT, baseline_version=6), [other]
        )
        # A hand-written champion has no record, and only has nothing to drop.
        kept = self.sr.gate_holdout(holdout, 7, self.PROMPT, baseline_version=2)
        self.assertEqual(kept, self.champion_train + [other])


class TestTheRecordIsWiredIn(unittest.TestCase):
    """optimize() and the gate import mlflow, gepa and the agent graph, so they
    cannot run here; pin the two call sites by reading the source instead."""

    @staticmethod
    def _function(module: str, name: str):
        path = Path(_AGENT_APP_DIR, "eval", f"{module}.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        (fn,) = [n for n in tree.body if getattr(n, "name", None) == name]
        return fn

    @staticmethod
    def _calls(node, callee: str) -> list[ast.Call]:
        return [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and callee in (getattr(n.func, "id", None), getattr(n.func, "attr", None))
        ]

    def test_optimize_records_the_split_inside_its_run_before_gepa(self):
        fn = self._function("optimize", "optimize")
        (run,) = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.With)
            and "start_run" in ast.unparse(n.items[0].context_expr)
        ]
        logged = self._calls(run, "log_train_keys")
        gepa = self._calls(run, "optimize_prompts")
        self.assertEqual((len(logged), len(gepa)), (1, 1), "record the split once")
        self.assertLess(logged[0].lineno, gepa[0].lineno, "record it before GEPA")

    def test_the_gate_filters_its_holdout_through_the_record(self):
        gated = self._calls(
            self._function("compare", "_holdout_records"), "gate_holdout"
        )
        self.assertEqual(len(gated), 1, "the gate must filter through the record")
        self.assertEqual(
            [ast.unparse(a) for a in gated[0].args[1:]],
            ["candidate_version", "PROMPT_NAME", "baseline_version"],
        )
        (call,) = self._calls(self._function("compare", "compare"), "_holdout_records")
        self.assertEqual(
            [ast.unparse(a) for a in call.args],
            ["dataset_name", "candidate_version", "baseline_version"],
        )


if __name__ == "__main__":
    unittest.main()
