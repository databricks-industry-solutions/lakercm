"""Unit tests for the get_review_remediation tool in agent.tools.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_review_remediation_tool

Same stubbing approach as tests.test_reviewer_action_tools: langchain_core /
databricks.sdk / config / services.lakehouse_db are replaced in sys.modules and
the @tool decorator becomes identity, so the tool runs as a plain function.

The tool is the agent's only view of WHY a document was held and which codes it
is allowed to propose, so these tests concentrate on the cases where being
wrong would be actively harmful: a withheld (control-group) proposal leaking
into the agent's context, and a `not_resolvable` item arriving without the
instruction not to fill it in.
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

_DB_HOLDER: dict = {"db": None}


def _install_stubs() -> None:
    lc = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.tool = lambda f: f
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
    # A package, not a plain module: get_all_tools() pulls in agent.memory_tools,
    # which imports services.store, and a non-package stub makes that a
    # ModuleNotFoundError rather than resolving to the stub below.
    svc.__path__ = []  # type: ignore[attr-defined]
    svc_db = types.ModuleType("services.lakehouse_db")
    svc_db.get_db = lambda: _DB_HOLDER["db"]
    svc.lakehouse_db = svc_db
    svc_store = types.ModuleType("services.store")
    svc_store.get_store = lambda: None
    svc.store = svc_store
    sys.modules["services"] = svc
    sys.modules["services.lakehouse_db"] = svc_db
    sys.modules["services.store"] = svc_store


from tests._isolation import IsolatedModules  # noqa: E402

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
    """Lakebase stand-in routed by SQL substring, recording every query."""

    def __init__(
        self,
        gold_row=None,
        proposal_rows=None,
        gold=True,
        has_reasons_column=True,
        proposals_raise=False,
    ):
        self._gold_row = gold_row
        self._proposals = proposal_rows or []
        self._gold = gold
        self._has_reasons = has_reasons_column
        self._proposals_raise = proposals_raise
        self.queries: list = []

    def gold_sync_available(self) -> bool:
        return self._gold

    def gold_sync_has_column(self, column: str) -> bool:
        return self._has_reasons

    def execute_query(self, query, params=None):
        self.queries.append(query)
        if "document_review_proposals" in query:
            if self._proposals_raise:
                raise RuntimeError("relation does not exist")
            return list(self._proposals)
        if "medical_documents" in query:
            return [self._gold_row] if self._gold_row else []
        return []


def _proposal(**over):
    base = {
        "review_reason": "non_billable_code",
        "resolution": "needs_judgment",
        "field_name": "primary_diagnosis",
        "correction_key": "id:3",
        "observed_value": "M25.56",
        "proposed_value": None,
        "rationale": "parent code, pick a child",
        "candidates": [
            {"code": "M25.561", "description": "Pain in right knee"},
            {"code": "M25.562", "description": "Pain in left knee"},
        ],
        "disposition": "pending",
    }
    base.update(over)
    return base


class GetReviewRemediationTests(unittest.TestCase):
    def setUp(self):
        tools.set_active_document_id(None)
        tools.set_authorized_user_email("reviewer@example.com")
        _DB_HOLDER["db"] = None

    def tearDown(self):
        tools.set_active_document_id(None)

    def _call(self, db):
        _DB_HOLDER["db"] = db
        tools.set_active_document_id("doc-1")
        return json.loads(tools.get_review_remediation())

    # -- scoping ------------------------------------------------------------

    def test_declines_without_an_open_document(self):
        _DB_HOLDER["db"] = FakeDB()
        self.assertIn("error", json.loads(tools.get_review_remediation()))

    # -- reading the hold reason -------------------------------------------

    def test_review_reasons_pass_through_as_a_list(self):
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["non_billable_code"]},
            proposal_rows=[_proposal()],
        )
        out = self._call(db)
        self.assertEqual(out["review_reasons"], ["non_billable_code"])

    def test_review_reasons_arriving_as_json_text_are_parsed(self):
        """The synced column can come back as JSONB text rather than a list."""
        db = FakeDB(
            gold_row={
                "is_automated": False,
                "review_reasons": '["invalid_code"]',
            }
        )
        out = self._call(db)
        self.assertEqual(out["review_reasons"], ["invalid_code"])

    def test_malformed_reasons_json_is_empty_not_an_exception(self):
        db = FakeDB(gold_row={"is_automated": False, "review_reasons": "{not json"})
        out = self._call(db)
        self.assertEqual(out["review_reasons"], [])

    def test_an_auto_verified_document_says_there_is_nothing_to_do(self):
        db = FakeDB(gold_row={"is_automated": True, "review_reasons": []})
        out = self._call(db)
        self.assertIn("auto-verified", out["note"])

    def test_a_document_with_no_gold_row_does_not_invent_a_problem(self):
        db = FakeDB(gold_row=None)
        out = self._call(db)
        self.assertEqual(out["review_reasons"], [])
        self.assertIn("rather than inferring", out["note"])

    def test_a_missing_reasons_column_degrades_instead_of_failing(self):
        """_review_reasons_column substitutes NULL until the sync catches up."""
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": None},
            has_reasons_column=False,
        )
        out = self._call(db)
        self.assertEqual(out["review_reasons"], [])

    # -- the candidate shortlist -------------------------------------------

    def test_candidates_reach_the_agent(self):
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["non_billable_code"]},
            proposal_rows=[_proposal()],
        )
        out = self._call(db)
        self.assertEqual(len(out["items"]), 1)
        self.assertEqual(
            [c["code"] for c in out["items"][0]["candidates"]],
            ["M25.561", "M25.562"],
        )

    def test_candidates_arriving_as_json_text_are_parsed(self):
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["non_billable_code"]},
            proposal_rows=[
                _proposal(candidates='[{"code": "M54.50", "description": "d"}]')
            ],
        )
        out = self._call(db)
        self.assertEqual(out["items"][0]["candidates"][0]["code"], "M54.50")

    def test_malformed_candidates_json_becomes_an_empty_list(self):
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["invalid_code"]},
            proposal_rows=[_proposal(candidates="{not json")],
        )
        out = self._call(db)
        self.assertEqual(out["items"][0]["candidates"], [])

    def test_withheld_proposals_are_excluded_at_the_query(self):
        """A control-slice proposal reaching the agent would put its document in
        both arms of the comparison the holdout exists to support. Excluded in
        SQL, not by filtering afterwards, so there is no path that forgets."""
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["invalid_code"]},
            proposal_rows=[_proposal()],
        )
        self._call(db)
        proposal_query = next(q for q in db.queries if "document_review_proposals" in q)
        self.assertIn("withheld = FALSE", proposal_query)

    def test_a_held_document_with_no_shortlist_says_so(self):
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["invalid_code"]},
            proposal_rows=[],
        )
        out = self._call(db)
        self.assertIn("no candidate shortlist", out["note"])

    def test_an_unavailable_proposal_store_degrades(self):
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["invalid_code"]},
            proposals_raise=True,
        )
        out = self._call(db)
        self.assertIn("unavailable", out["note"])
        self.assertEqual(out["items"], [])

    def test_the_note_forbids_proposing_off_the_shortlist(self):
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["non_billable_code"]},
            proposal_rows=[_proposal()],
        )
        out = self._call(db)
        self.assertIn("ONLY codes that appear in a candidate list", out["note"])

    def test_a_not_resolvable_item_carries_its_refusal_forward(self):
        """missing_member_id: the agent must see both that there is nothing to
        propose and the guidance saying not to."""
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["missing_member_id"]},
            proposal_rows=[
                _proposal(
                    review_reason="missing_member_id",
                    resolution="not_resolvable",
                    observed_value=None,
                    candidates=[],
                    disposition="declined",
                    rationale="... do not propose a value ...",
                )
            ],
        )
        out = self._call(db)
        item = out["items"][0]
        self.assertEqual(item["resolution"], "not_resolvable")
        self.assertIsNone(item["already_proposed"])
        self.assertIn("do not propose", item["guidance"])
        self.assertIn("must not receive a proposed value", out["note"])

    def test_an_already_dispositioned_proposal_is_visible(self):
        """So the agent does not re-propose something the reviewer rejected."""
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["invalid_code"]},
            proposal_rows=[_proposal(disposition="rejected", proposed_value="99213")],
        )
        out = self._call(db)
        self.assertEqual(out["items"][0]["disposition"], "rejected")
        self.assertEqual(out["items"][0]["already_proposed"], "99213")

    def test_the_result_is_json_serialisable(self):
        db = FakeDB(
            gold_row={"is_automated": False, "review_reasons": ["invalid_code"]},
            proposal_rows=[_proposal()],
        )
        json.dumps(self._call(db))


class ToolRegistrationTests(unittest.TestCase):
    def test_the_tool_is_registered(self):
        names = {
            getattr(t, "name", None) or getattr(t, "__name__", "")
            for t in tools.get_all_tools()
        }
        self.assertIn("get_review_remediation", names)

    def test_it_is_guarded_unlike_the_staging_tools(self):
        """Its output is document-derived text the model reads as instructions,
        so it goes through the injection guard. The propose_* tools are exempt
        only because the pane JSON.parses their payload."""
        self.assertNotIn("get_review_remediation", tools._UNGUARDED_TOOLS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
