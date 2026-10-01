"""Unit tests for the reviewer-pane action tools in agent.tools.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_reviewer_action_tools

No heavy deps: langchain_core / databricks.sdk / config / services.lakehouse_db
are stubbed in sys.modules before import (mirrors tests.test_delete_thread), and
the @tool decorator is stubbed to identity so each tool is exercised as a plain
function. Covers:
  - document-scoping: the tools decline when no document is open
  - propose_review_verdict validation (verdict enum + reasoning-required)
  - propose_extraction_edit correction-key guard + staged payload shape
  - add_review_note guard + staged payload shape
  - get_active_review_context field mapping (id:<idx> keys, corrections overlay,
    verdict + notepad passthrough)
"""

from __future__ import annotations

import json
import os
import sys
import types
import unittest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

# Holder so the stubbed get_db can be swapped per test without re-importing.
_DB_HOLDER: dict = {"db": None}


def _install_stubs() -> None:
    lc = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.tool = lambda f: f  # identity — tools become plain callables
    lc.tools = lc_tools
    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.tools"] = lc_tools

    dbx = types.ModuleType("databricks")
    dbx_sdk = types.ModuleType("databricks.sdk")
    dbx_sdk.WorkspaceClient = object
    dbx.sdk = dbx_sdk
    sys.modules["databricks"] = dbx
    sys.modules["databricks.sdk"] = dbx_sdk

    cfg = types.ModuleType("config")
    cfg.settings = types.SimpleNamespace(
        auto_verdict_threshold=0.92, pipeline_id="", schema_name="lakercm"
    )
    sys.modules["config"] = cfg

    svc = types.ModuleType("services")
    svc_db = types.ModuleType("services.lakehouse_db")
    svc_db.get_db = lambda: _DB_HOLDER["db"]
    svc.lakehouse_db = svc_db
    sys.modules["services"] = svc
    sys.modules["services.lakehouse_db"] = svc_db


from tests._isolation import IsolatedModules  # noqa: E402

# The stubs are live only while this module's tests run (setUpModule ..
# tearDownModule); installed at import time they leaked into every other module
# of a single pytest run. `tools` is imported fresh against them.
_ISOLATION = IsolatedModules()
tools = None


def setUpModule():
    global tools
    _ISOLATION.start(fresh=("agent.tools",))
    _install_stubs()
    from agent import tools as stubbed_tools

    tools = stubbed_tools


def tearDownModule():
    _ISOLATION.stop()


class FakeDB:
    """Minimal Lakebase stand-in routed by SQL substring."""

    def __init__(self, doc_row=None, review_rows=None, note_rows=None, gold=True):
        self._doc = doc_row
        self._reviews = review_rows or []
        self._notes = note_rows or []
        self._gold = gold

    def gold_sync_available(self) -> bool:
        return self._gold

    def execute_query(self, query, params=None):
        if "document_extraction_reviews" in query:
            return list(self._reviews)
        if "document_notes" in query:
            return list(self._notes)
        if "medical_documents" in query:
            return [self._doc] if self._doc else []
        return []


class ReviewerActionToolTests(unittest.TestCase):
    def setUp(self):
        tools.set_active_document_id(None)
        tools.set_authorized_user_email("reviewer@example.com")
        _DB_HOLDER["db"] = None

    def tearDown(self):
        tools.set_active_document_id(None)

    # -- document scoping ---------------------------------------------------
    def test_tools_decline_without_open_document(self):
        for out in (
            tools.get_active_review_context(),
            tools.propose_extraction_edit("id:0", "x", "why"),
            tools.propose_review_verdict("correct", ""),
            tools.add_review_note("hi"),
        ):
            self.assertIn("error", json.loads(out))

    # -- propose_review_verdict --------------------------------------------
    def test_verdict_valid(self):
        tools.set_active_document_id("doc-1")
        out = json.loads(tools.propose_review_verdict("incorrect", "wrong code"))
        self.assertTrue(out["staged"])
        act = out["_frontend_action"]
        self.assertEqual(act["type"], "verdict")
        self.assertEqual(act["verdict"], "incorrect")
        self.assertEqual(act["document_id"], "doc-1")

    def test_verdict_rejects_bad_value(self):
        tools.set_active_document_id("doc-1")
        out = json.loads(tools.propose_review_verdict("bogus", "x"))
        self.assertIn("error", out)

    def test_verdict_requires_reasoning_when_not_correct(self):
        tools.set_active_document_id("doc-1")
        out = json.loads(tools.propose_review_verdict("partially_correct", "  "))
        self.assertIn("error", out)
        # 'correct' needs no reasoning
        ok = json.loads(tools.propose_review_verdict("correct", ""))
        self.assertTrue(ok["staged"])

    # -- propose_extraction_edit -------------------------------------------
    def test_edit_requires_id_key(self):
        tools.set_active_document_id("doc-1")
        self.assertIn("error", json.loads(tools.propose_extraction_edit("x", "v", "r")))
        out = json.loads(tools.propose_extraction_edit("id:3", "NEW", "typo"))
        self.assertEqual(out["_frontend_action"]["correction_key"], "id:3")
        self.assertEqual(out["_frontend_action"]["proposed_value"], "NEW")

    def test_the_review_reason_rides_the_staged_payload(self):
        """The pane reports acceptance per reason. A proposal that arrives
        without one cannot be attributed to the finding it was meant to fix."""
        tools.set_active_document_id("doc-1")
        out = json.loads(
            tools.propose_extraction_edit(
                "id:3", "M54.50", "parent code", "non_billable_code"
            )
        )
        self.assertEqual(out["_frontend_action"]["review_reason"], "non_billable_code")

    def test_an_absent_review_reason_is_empty_not_missing(self):
        """An unprompted edit is legitimate; the pane distinguishes it from a
        pipeline finding rather than the key being absent."""
        tools.set_active_document_id("doc-1")
        out = json.loads(tools.propose_extraction_edit("id:3", "v", "r"))
        self.assertEqual(out["_frontend_action"]["review_reason"], "")

    # -- add_review_note ----------------------------------------------------
    def test_note_staged_and_empty_rejected(self):
        tools.set_active_document_id("doc-1")
        self.assertIn("error", json.loads(tools.add_review_note("   ")))
        out = json.loads(tools.add_review_note("check page 2"))
        self.assertTrue(out["saved"])
        self.assertEqual(out["_frontend_action"]["type"], "note_append")
        self.assertEqual(out["_frontend_action"]["note_text"], "check page 2")

    # -- get_active_review_context -----------------------------------------
    def test_context_maps_fields_and_overlays_corrections(self):
        tools.set_active_document_id("doc-1")
        _DB_HOLDER["db"] = FakeDB(
            doc_row={
                "document_name": "Claim A",
                "document_type": "denial_management",
                "processing_status": "pending",
                "document_path": "/v/claim_a.pdf",
                "label": "Denial Management",
                "identifiers": [
                    {"name": "patient_name", "value": "Jane Doe", "confidence": 0.99},
                    {"name": "diagnosis_code", "value": "E11.9", "confidence": 0.4},
                ],
                "confidence_score": 0.71,
            },
            review_rows=[
                {
                    "verdict": "partially_correct",
                    "reasoning": "code off",
                    "corrections": {"id:1": "E11.8"},
                    "is_automated": False,
                    "reviewer_email": "r@x.com",
                    "updated_at": "2026-09-13T00:00:00Z",
                }
            ],
            note_rows=[{"note_text": "verify dx"}],
        )
        out = json.loads(tools.get_active_review_context())
        self.assertEqual(out["document_name"], "Claim A")
        self.assertEqual(len(out["fields"]), 2)
        f0, f1 = out["fields"]
        self.assertEqual(f0["correction_key"], "id:0")
        self.assertEqual(f1["correction_key"], "id:1")
        # id:1 has a saved correction → current_value reflects it
        self.assertEqual(f1["extracted_value"], "E11.9")
        self.assertEqual(f1["current_value"], "E11.8")
        self.assertEqual(out["current_review"]["verdict"], "partially_correct")
        self.assertEqual(out["notepad"], "verify dx")

    def test_context_returns_the_document_path_chunk_search_scopes_on(self):
        """The open document's PATH, not just its display name.

        search_document_chunks scopes on document_path, and this dict was the
        model's only source for it on the "chat with this document" path. Omitting
        it meant the model passed the file NAME instead, which matched no rows in
        public.document_chunks (keyed on the full dbfs:/Volumes/... path), so
        every scoped question came back empty and fell through to the extracted
        fields. The query already selected the column; only this dict dropped it.
        """
        tools.set_active_document_id("doc-1")
        _DB_HOLDER["db"] = FakeDB(
            doc_row={
                "document_name": "Claim A",
                "document_path": "dbfs:/Volumes/cat/schema/documents_input/a.pdf",
                "processing_status": "pending",
                "identifiers": [],
            }
        )
        out = json.loads(tools.get_active_review_context())
        self.assertEqual(
            out["document_path"],
            "dbfs:/Volumes/cat/schema/documents_input/a.pdf",
        )
        # And it is the full path, not the basename — the distinction is the bug.
        self.assertNotEqual(out["document_path"], "a.pdf")

    def test_context_survives_missing_notepad_table(self):
        tools.set_active_document_id("doc-1")

        class NoNotesDB(FakeDB):
            def execute_query(self, query, params=None):
                if "document_notes" in query:
                    raise RuntimeError("relation does not exist")
                return super().execute_query(query, params)

        _DB_HOLDER["db"] = NoNotesDB(
            doc_row={
                "document_name": "Claim B",
                "document_type": None,
                "processing_status": "pending",
                "document_path": "/v/b.pdf",
                "label": None,
                "identifiers": [],
                "confidence_score": None,
            }
        )
        out = json.loads(tools.get_active_review_context())
        self.assertIsNone(out["notepad"])
        self.assertEqual(out["fields"], [])


if __name__ == "__main__":
    unittest.main()
