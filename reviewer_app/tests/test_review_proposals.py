"""Unit tests for the agent-assisted review surface.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_review_proposals

Two halves:
  - routes.proposals — the remediation read and the proposal lifecycle, with
    fastapi / databricks.sdk / schemas / dependencies / config stubbed the same
    way tests.test_document_notes does it.
  - services.review_proposals — the holdout split, warehouse column coercion,
    and what record_remediations writes.

The emphasis is on the invariants the analytics depend on: a proposal is
counted when it is SHOWN (not when approved), a refusal can never carry a
value, and a document never changes sides of the control split.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import types
import unittest

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


class _HTTPException(Exception):
    def __init__(self, status_code=500, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _identity_decorator(*_a, **_k):
    def wrap(fn):
        return fn

    return wrap


def _install_stubs() -> None:
    fastapi = types.ModuleType("fastapi")

    class _Router:
        def __init__(self, *a, **k):
            pass

        get = post = patch = delete = put = api_route = staticmethod(
            _identity_decorator
        )

    fastapi.APIRouter = _Router
    fastapi.HTTPException = _HTTPException
    fastapi.Depends = lambda *a, **k: None
    fastapi.Request = object

    class _BackgroundTasks:
        def __init__(self):
            self.tasks = []

        def add_task(self, fn, *a, **k):
            self.tasks.append((fn, a, k))

    fastapi.BackgroundTasks = _BackgroundTasks
    sys.modules["fastapi"] = fastapi

    dbx = types.ModuleType("databricks")
    dbx_sdk = types.ModuleType("databricks.sdk")
    dbx_sdk.WorkspaceClient = object
    dbx.sdk = dbx_sdk
    sdk_sql = types.ModuleType("databricks.sdk.service.sql")

    class _Enum:
        def __init__(self, value):
            self.value = value

    sdk_sql.Disposition = types.SimpleNamespace(INLINE=_Enum("INLINE"))
    sdk_sql.Format = types.SimpleNamespace(JSON_ARRAY=_Enum("JSON_ARRAY"))
    sdk_sql.StatementParameterListItem = lambda **kw: types.SimpleNamespace(**kw)
    sdk_sql.StatementState = types.SimpleNamespace(
        PENDING="PENDING", RUNNING="RUNNING", SUCCEEDED="SUCCEEDED"
    )
    sdk_service = types.ModuleType("databricks.sdk.service")
    sdk_service.sql = sdk_sql
    dbx_sdk.service = sdk_service
    sys.modules["databricks"] = dbx
    sys.modules["databricks.sdk"] = dbx_sdk
    sys.modules["databricks.sdk.service"] = sdk_service
    sys.modules["databricks.sdk.service.sql"] = sdk_sql

    schemas = types.ModuleType("schemas")

    def _ns(**kw):
        return types.SimpleNamespace(**kw)

    # Must list every name routes.proposals imports from schemas. This stub
    # replaces the real module, so a name added to that import list and not added
    # here fails collection for the whole file with "cannot import name ... from
    # 'schemas' (unknown location)" — which reads like a packaging problem rather
    # than the one-line omission it is.
    for name in (
        "DocumentFidelityResponse",
        "DocumentHoldReasonsResponse",
        "DocumentRemediationResponse",
        "HoldReasonItem",
        "ProposalDispositionSubmission",
        "RemediationCandidate",
        "RemediationItem",
        "ReviewProposal",
        "ReviewProposalListResponse",
        "ReviewProposalSubmission",
        "RewrittenCode",
    ):
        setattr(schemas, name, _ns)
    sys.modules["schemas"] = schemas

    deps = types.ModuleType("dependencies")
    deps.get_workspace_client = lambda: None
    deps.get_lakercm_db = lambda: None
    deps.get_current_user_email = lambda request: "reviewer@example.com"
    sys.modules["dependencies"] = deps

    cfg = types.ModuleType("config")
    cfg.settings = types.SimpleNamespace(
        catalog="cat",
        lakercm_schema="lakercm",
        get_warehouse_id=lambda: "wh-1",
        # Same default as config.py. The hold-reasons payload carries the
        # threshold it actually compared against, so the banner can never quote
        # a different number from the one that made the decision.
        auto_verdict_threshold=0.92,
    )
    sys.modules["config"] = cfg


from tests._isolation import IsolatedModules  # noqa: E402

_ISOLATION = IsolatedModules()
proposals = None
rp = None
rem = None


def setUpModule():
    global proposals, rp, rem
    _ISOLATION.start(purge_first_party=True)
    _install_stubs()
    from routes import proposals as stubbed_proposals
    from services import review_proposals as stubbed_rp
    from services import remediation as stubbed_rem

    proposals = stubbed_proposals
    rp = stubbed_rp
    rem = stubbed_rem


def tearDownModule():
    _ISOLATION.stop()


DOC_ID = "11111111-1111-1111-1111-111111111111"


class FakeDB:
    def __init__(self, doc=None, proposals_rows=None, disposition_row=None):
        self._doc = doc
        self._rows = proposals_rows or []
        self._disposition_row = disposition_row
        self.inserted = []
        self.superseded = []
        self.dispositioned = []

    def get_document_by_id(self, document_id):
        return self._doc

    def list_review_proposals(self, document_id, include_withheld=False, **_k):
        self.last_include_withheld = include_withheld
        return self._rows

    def insert_review_proposal(self, **kwargs):
        self.inserted.append(kwargs)
        row = dict(kwargs)
        row.setdefault("id", "p-1")
        row.setdefault("disposition", "pending")
        return row

    def supersede_pending_proposals(self, document_id, correction_key):
        self.superseded.append((document_id, correction_key))
        return 0

    def set_proposal_disposition(self, **kwargs):
        self.dispositioned.append(kwargs)
        return self._disposition_row


def _run(coro):
    return asyncio.run(coro)


@contextlib.contextmanager
def _patch(module, name, value):
    """Swap a module attribute for the duration of a block.

    The save/try/finally/restore dance appears often enough in this file that
    doing it by hand is where a leaked stub comes from — and a stub that leaks
    into the next test module is exactly how one missing schema name failed an
    unrelated file's collection.
    """
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def _bg():
    """A stand-in BackgroundTasks: records add_task calls, never runs them —
    matching FastAPI, which runs background tasks after the response (so the
    analytics trigger never fires during a direct handler call)."""
    return types.SimpleNamespace(add_task=lambda *a, **k: None)


def _submission(**over):
    base = {
        "review_reason": "non_billable_code",
        "resolution": "deterministic",
        "field_name": "primary_diagnosis",
        "correction_key": "id:3",
        "observed_value": "M54.5",
        "proposed_value": "M54.50",
        "rationale": "parent code",
        "candidates": [],
        "model": "test-model",
    }
    base.update(over)
    return types.SimpleNamespace(**base)


class TestRemediationRoute(unittest.TestCase):
    def test_unknown_document_is_404(self):
        db = FakeDB(doc=None)
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                proposals.get_document_remediation(DOC_ID, db=db, workspace_client=None)
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_a_document_with_no_path_is_reported_not_held(self):
        db = FakeDB(doc={"file_path": ""})
        out = _run(
            proposals.get_document_remediation(DOC_ID, db=db, workspace_client=None)
        )
        self.assertFalse(out.is_held)

    def test_a_document_with_no_reasons_is_not_held(self):
        db = FakeDB(doc={"file_path": "dbfs:/x/a.pdf"})
        orig = rp.fetch_flagged_items
        rp.fetch_flagged_items = lambda *a, **k: ([], [])
        try:
            out = _run(
                proposals.get_document_remediation(DOC_ID, db=db, workspace_client=None)
            )
        finally:
            rp.fetch_flagged_items = orig
        self.assertFalse(out.is_held)

    def test_a_held_document_returns_its_reasons_and_items(self):
        db = FakeDB(doc={"file_path": "dbfs:/x/a.pdf"})
        orig_fetch = rp.fetch_flagged_items
        orig_load = rp.load_terminology
        rp.fetch_flagged_items = lambda *a, **k: (
            ["missing_member_id"],
            [],
        )
        rp.load_terminology = lambda *a, **k: rem.Terminology([])
        try:
            out = _run(
                proposals.get_document_remediation(DOC_ID, db=db, workspace_client=None)
            )
        finally:
            rp.fetch_flagged_items = orig_fetch
            rp.load_terminology = orig_load
        self.assertTrue(out.is_held)
        self.assertEqual(out.review_reasons, ["missing_member_id"])
        self.assertEqual(len(out.items), 1)

    def test_a_warehouse_failure_is_503_not_an_empty_shortlist(self):
        """An empty list reads as 'no fix exists'. That is not the same claim
        as 'we could not look', and a reviewer would act on it differently."""
        db = FakeDB(doc={"file_path": "dbfs:/x/a.pdf"})
        orig = rp.fetch_flagged_items

        def _boom(*a, **k):
            raise proposals.WarehouseReadError("RT down")

        rp.fetch_flagged_items = _boom
        try:
            with self.assertRaises(_HTTPException) as ctx:
                _run(
                    proposals.get_document_remediation(
                        DOC_ID, db=db, workspace_client=None
                    )
                )
        finally:
            rp.fetch_flagged_items = orig
        self.assertEqual(ctx.exception.status_code, 503)


class TestHoldReasonsRoute(unittest.TestCase):
    """The endpoint that replaced a banner computed in the browser.

    The old surface decided held-vs-auto-verified from React props the only
    routed page never passed, so every document rendered as held with an
    invented explanation. These pin the two things that made that possible: a
    verdict assembled from something other than the pipeline's own column, and
    an empty explanation being filled in with a plausible guess.
    """

    DOC = {"file_path": "dbfs:/Volumes/c/s/documents_input/a.pdf"}

    def _routing(self, **over):
        base = {
            "is_automated": False,
            "confidence_score": 0.84,
            "review_reasons": ["low_confidence"],
            "validated_codes": [],
        }
        base.update(over)
        return base

    def test_unknown_document_is_404(self):
        db = FakeDB(doc=None)
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                proposals.get_document_hold_reasons(
                    DOC_ID, db=db, workspace_client=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_a_document_with_no_gold_row_reports_unknown_not_held(self):
        """No gold row means the pipeline has not decided. Reporting that as
        "held" is what put a hold banner on documents that had none."""
        db = FakeDB(doc=self.DOC)
        with _patch(rp, "fetch_routing_decision", lambda *a, **k: None):
            out = _run(
                proposals.get_document_hold_reasons(
                    DOC_ID, db=db, workspace_client=None
                )
            )
        self.assertEqual(out.state, "unknown")
        self.assertEqual(out.blocking, [])

    def test_a_low_confidence_hold_states_both_numbers(self):
        db = FakeDB(doc=self.DOC)
        with _patch(rp, "fetch_routing_decision", lambda *a, **k: self._routing()):
            with _patch(rp, "fetch_uncaptured_codes", lambda *a, **k: None):
                with _patch(rp, "fetch_extraction_fidelity", lambda *a, **k: None):
                    out = _run(
                        proposals.get_document_hold_reasons(
                            DOC_ID, db=db, workspace_client=None
                        )
                    )
        self.assertEqual(out.state, "held")
        self.assertEqual([i.code for i in out.blocking], ["low_confidence"])
        self.assertIn("84%", out.blocking[0].detail)
        self.assertIn("92%", out.blocking[0].detail)
        self.assertFalse(out.unexplained)

    def test_the_document_that_prompted_this_is_not_given_a_guess(self):
        """Held, no recorded reason, confidence 0.9996. The old banner said
        "confidence was 100%, below the 92% threshold"."""
        db = FakeDB(doc=self.DOC)
        routing = self._routing(confidence_score=0.9996, review_reasons=[])
        with _patch(rp, "fetch_routing_decision", lambda *a, **k: routing):
            with _patch(rp, "fetch_uncaptured_codes", lambda *a, **k: None):
                with _patch(rp, "fetch_extraction_fidelity", lambda *a, **k: None):
                    out = _run(
                        proposals.get_document_hold_reasons(
                            DOC_ID, db=db, workspace_client=None
                        )
                    )
        self.assertTrue(out.unexplained)
        self.assertEqual([i.code for i in out.blocking], ["reason_not_recorded"])
        self.assertNotIn("below", out.blocking[0].detail)
        self.assertNotIn("100%", out.blocking[0].detail)

    def test_a_stringified_boolean_is_read_as_auto_verified(self):
        """Statement Execution returns booleans as strings; `=== true` in the
        browser read 'true' as "held"."""
        db = FakeDB(doc=self.DOC)
        routing = self._routing(
            is_automated="true", confidence_score=0.9996, review_reasons=[]
        )
        with _patch(rp, "fetch_routing_decision", lambda *a, **k: routing):
            with _patch(rp, "fetch_uncaptured_codes", lambda *a, **k: None):
                with _patch(rp, "fetch_extraction_fidelity", lambda *a, **k: None):
                    out = _run(
                        proposals.get_document_hold_reasons(
                            DOC_ID, db=db, workspace_client=None
                        )
                    )
        self.assertEqual(out.state, "auto_verified")
        self.assertEqual(out.blocking, [])

    def test_a_blocking_read_failure_is_503_not_an_empty_answer(self):
        db = FakeDB(doc=self.DOC)

        def boom(*a, **k):
            raise proposals.WarehouseReadError("warehouse down")

        with _patch(rp, "fetch_routing_decision", boom):
            with self.assertRaises(_HTTPException) as ctx:
                _run(
                    proposals.get_document_hold_reasons(
                        DOC_ID, db=db, workspace_client=None
                    )
                )
        self.assertEqual(ctx.exception.status_code, 503)

    def test_an_advisory_failure_degrades_instead_of_failing(self):
        """ "Could not check" and "nothing to report" look identical for an
        advisory, so only one of them may be silent."""
        db = FakeDB(doc=self.DOC)

        def boom(*a, **k):
            raise RuntimeError("analytics pipeline lagging")

        with _patch(rp, "fetch_routing_decision", lambda *a, **k: self._routing()):
            with _patch(rp, "fetch_uncaptured_codes", boom):
                out = _run(
                    proposals.get_document_hold_reasons(
                        DOC_ID, db=db, workspace_client=None
                    )
                )
        self.assertEqual(out.state, "held")
        self.assertEqual(out.degraded, ["advisory_unavailable"])
        self.assertTrue(out.blocking)

    def test_advisories_are_separated_from_blocking(self):
        db = FakeDB(doc=self.DOC)
        with _patch(rp, "fetch_routing_decision", lambda *a, **k: self._routing()):
            with _patch(
                rp,
                "fetch_uncaptured_codes",
                lambda *a, **k: {"uncaptured_codes": ["E11.9"]},
            ):
                with _patch(
                    rp,
                    "fetch_extraction_fidelity",
                    lambda *a, **k: {
                        "fidelity_status": "rewritten",
                        "codes_rewritten": [{"source_code": "I1O", "stored_as": "I10"}],
                        "rewritten_and_auto_verified": False,
                    },
                ):
                    out = _run(
                        proposals.get_document_hold_reasons(
                            DOC_ID, db=db, workspace_client=None
                        )
                    )
        self.assertEqual([i.code for i in out.blocking], ["low_confidence"])
        self.assertEqual(
            {i.code for i in out.advisory},
            {"uncaptured_code", "code_rewritten"},
        )
        for item in out.advisory:
            self.assertEqual(item.severity, "advisory")


class TestListProposals(unittest.TestCase):
    def test_withheld_proposals_are_never_listed(self):
        """Showing a control-slice proposal puts its document in both arms."""
        db = FakeDB(doc={"file_path": "x"}, proposals_rows=[])
        _run(proposals.list_document_proposals(DOC_ID, db=db))
        self.assertFalse(db.last_include_withheld)

    def test_unknown_document_is_404(self):
        db = FakeDB(doc=None)
        with self.assertRaises(_HTTPException) as ctx:
            _run(proposals.list_document_proposals(DOC_ID, db=db))
        self.assertEqual(ctx.exception.status_code, 404)


class TestCreateProposal(unittest.TestCase):
    def test_an_unknown_resolution_is_rejected(self):
        db = FakeDB(doc={"file_path": "x"})
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                proposals.create_document_proposal(
                    DOC_ID, _submission(resolution="whatever"), db=db
                )
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_a_value_is_required_unless_the_agent_declined(self):
        db = FakeDB(doc={"file_path": "x"})
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                proposals.create_document_proposal(
                    DOC_ID, _submission(proposed_value=""), db=db
                )
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_a_refusal_is_stored_without_a_value(self):
        """The missing_member_id guarantee: no path stores a guessed ID."""
        db = FakeDB(doc={"file_path": "x"})
        _run(
            proposals.create_document_proposal(
                DOC_ID,
                _submission(
                    review_reason="missing_member_id",
                    resolution="not_resolvable",
                    proposed_value="W123456789",
                ),
                db=db,
            )
        )
        self.assertIsNone(db.inserted[0]["proposed_value"])

    def test_a_new_proposal_supersedes_the_pending_one_for_the_same_field(self):
        db = FakeDB(doc={"file_path": "x"})
        _run(proposals.create_document_proposal(DOC_ID, _submission(), db=db))
        self.assertEqual(db.superseded, [(DOC_ID, "id:3")])

    def test_a_pane_proposal_is_never_withheld(self):
        """The reviewer is already looking at it; withholding is a triage-time
        decision, not something to apply to a card already on screen."""
        db = FakeDB(doc={"file_path": "x"})
        _run(proposals.create_document_proposal(DOC_ID, _submission(), db=db))
        self.assertFalse(db.inserted[0]["withheld"])

    def test_the_source_records_that_it_came_from_the_pane(self):
        db = FakeDB(doc={"file_path": "x"})
        _run(proposals.create_document_proposal(DOC_ID, _submission(), db=db))
        self.assertEqual(db.inserted[0]["source"], rp.SOURCE_PANE)


class TestDisposition(unittest.TestCase):
    def _request(self):
        return types.SimpleNamespace()

    def test_a_system_transition_cannot_be_set_by_a_reviewer(self):
        """'declined' and 'superseded' are how the system marks rows; letting a
        client set them would corrupt the acceptance denominator."""
        db = FakeDB()
        for bad in ("declined", "superseded", "pending", "nonsense"):
            with self.assertRaises(_HTTPException) as ctx:
                _run(
                    proposals.set_proposal_disposition(
                        self._request(),
                        "p-1",
                        types.SimpleNamespace(disposition=bad, human_value=None),
                        _bg(),
                        db=db,
                    )
                )
            self.assertEqual(ctx.exception.status_code, 422, bad)

    def test_modified_requires_the_value_the_reviewer_used(self):
        db = FakeDB()
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                proposals.set_proposal_disposition(
                    self._request(),
                    "p-1",
                    types.SimpleNamespace(disposition="modified", human_value=" "),
                    _bg(),
                    db=db,
                )
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_a_second_click_is_409_not_a_restamp(self):
        """set_proposal_disposition only matches a pending row, so the repeat
        returns nothing — it must not overwrite the first decision or move the
        timestamp the turnaround metric is built on."""
        db = FakeDB(disposition_row=None)
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                proposals.set_proposal_disposition(
                    self._request(),
                    "p-1",
                    types.SimpleNamespace(disposition="accepted", human_value=None),
                    _bg(),
                    db=db,
                )
            )
        self.assertEqual(ctx.exception.status_code, 409)

    def test_an_accepted_proposal_records_who_accepted_it(self):
        db = FakeDB(disposition_row={"id": "p-1", "disposition": "accepted"})
        _run(
            proposals.set_proposal_disposition(
                self._request(),
                "p-1",
                types.SimpleNamespace(disposition="accepted", human_value=None),
                _bg(),
                db=db,
            )
        )
        self.assertEqual(db.dispositioned[0]["disposition_by"], "reviewer@example.com")

    def test_a_successful_disposition_schedules_an_analytics_refresh(self):
        """A real disposition change must kick the analytics pipeline so review
        analytics refresh event-driven (no cron)."""
        db = FakeDB(disposition_row={"id": "p-1", "disposition": "accepted"})
        scheduled = []
        bg = types.SimpleNamespace(add_task=lambda fn, *a, **k: scheduled.append(fn))
        _run(
            proposals.set_proposal_disposition(
                self._request(),
                "p-1",
                types.SimpleNamespace(disposition="accepted", human_value=None),
                bg,
                db=db,
            )
        )
        self.assertEqual(
            [fn.__name__ for fn in scheduled], ["trigger_analytics_pipeline"]
        )

    def test_a_409_does_not_schedule_an_analytics_refresh(self):
        """No transition happened, so nothing should be recomputed."""
        db = FakeDB(disposition_row=None)
        scheduled = []
        bg = types.SimpleNamespace(add_task=lambda fn, *a, **k: scheduled.append(fn))
        with self.assertRaises(_HTTPException):
            _run(
                proposals.set_proposal_disposition(
                    self._request(),
                    "p-1",
                    types.SimpleNamespace(disposition="accepted", human_value=None),
                    bg,
                    db=db,
                )
            )
        self.assertEqual(scheduled, [])


class TestHoldout(unittest.TestCase):
    def test_a_document_never_changes_sides(self):
        """Re-running triage must not move a document between arms."""
        for doc in (DOC_ID, "abc", "x" * 40):
            first = rp.should_withhold(doc, 0.2)
            for _ in range(5):
                self.assertEqual(rp.should_withhold(doc, 0.2), first, doc)

    def test_a_zero_rate_withholds_nothing(self):
        self.assertFalse(rp.should_withhold(DOC_ID, 0.0))

    def test_a_full_rate_withholds_everything(self):
        self.assertTrue(rp.should_withhold(DOC_ID, 1.0))

    def test_the_split_is_roughly_the_requested_rate(self):
        ids = [f"doc-{i:05d}" for i in range(4000)]
        held = sum(1 for d in ids if rp.should_withhold(d, 0.2))
        self.assertGreater(held / len(ids), 0.17)
        self.assertLess(held / len(ids), 0.23)

    def test_the_split_does_not_depend_on_process_state(self):
        """Hash-derived, not random: no seeding, no ordering effect."""
        forward = [rp.should_withhold(f"d{i}", 0.3) for i in range(50)]
        backward = [rp.should_withhold(f"d{i}", 0.3) for i in reversed(range(50))]
        self.assertEqual(forward, list(reversed(backward)))


class TestColumnCoercion(unittest.TestCase):
    def test_json_text_from_the_warehouse_is_parsed(self):
        self.assertEqual(rp._as_list('["a","b"]'), ["a", "b"])

    def test_a_parsed_list_from_lakebase_passes_through(self):
        self.assertEqual(rp._as_list(["a"]), ["a"])

    def test_null_and_blank_are_empty(self):
        self.assertEqual(rp._as_list(None), [])
        self.assertEqual(rp._as_list("   "), [])

    def test_malformed_json_is_empty_not_an_exception(self):
        self.assertEqual(rp._as_list("{not json"), [])

    def test_a_json_object_is_not_treated_as_a_list(self):
        self.assertEqual(rp._as_list('{"a":1}'), [])


class TestRecordRemediations(unittest.TestCase):
    def _rem(self, resolution, reason="non_billable_code"):
        return rem.Remediation(
            review_reason=reason,
            resolution=resolution,
            guidance="g",
            observed_code="M54.5",
        )

    def test_candidates_are_recorded_without_a_proposed_value(self):
        """Triage records the shortlist. Choosing from it is the agent's job,
        and it happens later through the pane."""
        db = FakeDB()
        rp.record_remediations(
            db, DOC_ID, [self._rem("needs_judgment")], rp.SOURCE_TRIAGE
        )
        self.assertIsNone(db.inserted[0].get("proposed_value"))

    def test_a_refusal_is_never_withheld(self):
        """There is nothing to withhold, and counting it in the control arm
        would dilute the comparison with documents the agent could not help."""
        db = FakeDB()
        rp.record_remediations(
            db,
            DOC_ID,
            [self._rem("not_resolvable", "missing_member_id")],
            rp.SOURCE_TRIAGE,
            holdout_rate=1.0,
        )
        self.assertFalse(db.inserted[0]["withheld"])

    def test_a_resolvable_proposal_follows_the_holdout_split(self):
        db = FakeDB()
        rp.record_remediations(
            db,
            DOC_ID,
            [self._rem("deterministic")],
            rp.SOURCE_TRIAGE,
            holdout_rate=1.0,
        )
        self.assertTrue(db.inserted[0]["withheld"])

    def test_the_source_is_recorded_as_triage(self):
        db = FakeDB()
        rp.record_remediations(
            db, DOC_ID, [self._rem("deterministic")], rp.SOURCE_TRIAGE
        )
        self.assertEqual(db.inserted[0]["source"], rp.SOURCE_TRIAGE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
