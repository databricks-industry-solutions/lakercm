"""Unit tests for eval.review_queues (MLflow review-queue orchestration).

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_review_queues

The mlflow labeling API is stubbed in sys.modules before import (no mlflow
install needed). Covers schema creation, queue creation (session + trace add +
url passthrough), sync, listing, and the version guard.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

# --- record buffers ---------------------------------------------------------
_CALLS: dict = {"schemas": [], "sessions": [], "searched": []}


class _FakeSession:
    def __init__(self, name, assigned_users, label_schemas):
        self.name = name
        self.assigned_users = assigned_users
        self.label_schemas = label_schemas
        self.labeling_session_id = "sess-123"
        self.url = "https://workspace/ml/review-app/sess-123"
        self.added = []
        self.synced_to = None

    def add_traces(self, traces):
        self.added.extend(list(traces))

    def sync(self, to_dataset):
        self.synced_to = to_dataset


def _install_stubs() -> None:
    mlflow = types.ModuleType("mlflow")

    def _search_traces(**kwargs):
        _CALLS["searched"].append(kwargs)
        return ["trace-a", "trace-b", "trace-c"]

    def _get_experiment(exp_id):
        return types.SimpleNamespace(experiment_id=exp_id)

    def _get_experiment_by_name(path):
        return types.SimpleNamespace(experiment_id="by-name")

    mlflow.search_traces = _search_traces
    mlflow.get_experiment = _get_experiment
    mlflow.get_experiment_by_name = _get_experiment_by_name

    genai = types.ModuleType("mlflow.genai")

    label_schemas = types.ModuleType("mlflow.genai.label_schemas")

    def _create_label_schema(**kwargs):
        _CALLS["schemas"].append(kwargs)
        return types.SimpleNamespace(name=kwargs["name"])

    label_schemas.create_label_schema = _create_label_schema
    label_schemas.InputCategorical = lambda options: {"categorical": options}
    label_schemas.InputText = lambda: {"text": True}
    label_schemas.InputTextList = lambda: {"text_list": True}

    labeling = types.ModuleType("mlflow.genai.labeling")

    def _create_labeling_session(name, assigned_users, label_schemas):
        s = _FakeSession(name, assigned_users, label_schemas)
        _CALLS["sessions"].append(s)
        return s

    labeling.create_labeling_session = _create_labeling_session
    labeling.get_labeling_sessions = lambda: list(_CALLS["sessions"])

    genai.label_schemas = label_schemas
    genai.labeling = labeling
    mlflow.genai = genai

    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.genai"] = genai
    sys.modules["mlflow.genai.label_schemas"] = label_schemas
    sys.modules["mlflow.genai.labeling"] = labeling


from tests._isolation import IsolatedModules  # noqa: E402

# The mlflow stubs are live only while this module's tests run (setUpModule ..
# tearDownModule); installed at import time they replaced mlflow for every
# other module of a single pytest run. `review_queues` is imported fresh.
_ISOLATION = IsolatedModules()
review_queues = None


def setUpModule():
    global review_queues
    _ISOLATION.start(fresh=("eval.review_queues",))
    _install_stubs()
    from eval import review_queues as stubbed_review_queues

    review_queues = stubbed_review_queues


def tearDownModule():
    _ISOLATION.stop()


class ReviewQueueTests(unittest.TestCase):
    def setUp(self):
        _CALLS["schemas"].clear()
        _CALLS["sessions"].clear()
        _CALLS["searched"].clear()
        os.environ["MLFLOW_EXPERIMENT_ID"] = "42"

    def test_ensure_label_schemas_creates_all(self):
        names = review_queues.ensure_label_schemas()
        self.assertEqual(len(names), 5)
        # every schema was created with overwrite (idempotent) + a comment box
        types_seen = {c["type"] for c in _CALLS["schemas"]}
        self.assertEqual(types_seen, {"feedback", "expectation"})
        self.assertTrue(all(c["overwrite"] for c in _CALLS["schemas"]))
        # expected_facts is the expectation schema (seeds Correctness)
        exp = [c for c in _CALLS["schemas"] if c["type"] == "expectation"]
        self.assertEqual(len(exp), 1)

    def test_create_review_queue_opens_session_and_adds_traces(self):
        summary = review_queues.create_review_queue(
            name="q1",
            assigned_users=["a@x.com", "b@x.com"],
            max_traces=10,
        )
        self.assertEqual(summary["session_name"], "q1")
        self.assertEqual(summary["trace_count"], 3)
        self.assertTrue(summary["url"].startswith("https://"))
        # schemas were ensured (5) since none were passed in
        self.assertEqual(len(summary["label_schemas"]), 5)
        # the session received the searched traces
        self.assertEqual(_CALLS["sessions"][0].added, ["trace-a", "trace-b", "trace-c"])
        # experiment id from env flowed into the search
        self.assertEqual(_CALLS["searched"][0]["experiment_ids"], ["42"])

    def test_create_review_queue_respects_explicit_schemas(self):
        review_queues.create_review_queue(
            name="q2",
            assigned_users=["a@x.com"],
            label_schemas=["only_one"],
        )
        # no schema creation when caller supplies schema names
        self.assertEqual(_CALLS["schemas"], [])
        self.assertEqual(_CALLS["sessions"][0].label_schemas, ["only_one"])

    def test_sync_review_queue(self):
        review_queues.create_review_queue(name="q3", assigned_users=["a@x.com"])
        result = review_queues.sync_review_queue("q3", to_dataset="cat.sch.ds")
        self.assertEqual(result["to_dataset"], "cat.sch.ds")
        self.assertEqual(_CALLS["sessions"][0].synced_to, "cat.sch.ds")

    def test_sync_missing_session_errors(self):
        with self.assertRaises(RuntimeError):
            review_queues.sync_review_queue("nope", to_dataset="cat.sch.ds")

    def test_list_review_queues(self):
        review_queues.create_review_queue(name="q4", assigned_users=["a@x.com"])
        queues = review_queues.list_review_queues()
        self.assertEqual(len(queues), 1)
        self.assertEqual(queues[0]["session_name"], "q4")

    def test_version_guard(self):
        saved = review_queues._labeling
        review_queues._labeling = None
        try:
            with self.assertRaises(RuntimeError):
                review_queues.ensure_label_schemas()
        finally:
            review_queues._labeling = saved


if __name__ == "__main__":
    unittest.main()
