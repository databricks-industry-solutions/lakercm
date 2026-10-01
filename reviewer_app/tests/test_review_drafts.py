"""Unit tests for the review-draft autosave routes in routes.documents.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_review_drafts

Same stubbing philosophy as tests.test_document_notes (whose harness this
reuses): heavy collaborators are replaced in sys.modules before import, so this
runs with no app deps installed.

The behaviours that matter here are the ones that separate a DRAFT from a
REVIEW:
  - a draft may be partial (no verdict yet) and must still save
  - GET returns an empty draft, not a 404, so the form renders unconditionally
  - writing a draft must NOT submit anything
  - submitting must DELETE the draft, or the client (which prefers a draft over
    a submitted review on load) would resurrect the pre-submit state
  - a failing draft cleanup must never fail a review that is already written
"""

from __future__ import annotations

import asyncio
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

        get = post = delete = put = api_route = staticmethod(_identity_decorator)

    fastapi.APIRouter = _Router
    fastapi.HTTPException = _HTTPException
    fastapi.Depends = lambda *a, **k: None
    fastapi.File = lambda *a, **k: None
    fastapi.Form = lambda *a, **k: None
    fastapi.UploadFile = object
    fastapi.Request = object
    fastapi.BackgroundTasks = object
    responses = types.ModuleType("fastapi.responses")
    responses.Response = object
    fastapi.responses = responses
    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.responses"] = responses

    dbx = types.ModuleType("databricks")
    dbx_sdk = types.ModuleType("databricks.sdk")
    dbx_sdk.WorkspaceClient = object
    dbx.sdk = dbx_sdk
    sys.modules["databricks"] = dbx
    sys.modules["databricks.sdk"] = dbx_sdk

    schemas = types.ModuleType("schemas")

    def _ns(**kw):
        return types.SimpleNamespace(**kw)

    for name in (
        "DocumentUploadResponse",
        "DocumentListResponse",
        "DocumentStatusCountsResponse",
        "DocumentDetailResponse",
        "DocumentExtraction",
        "ExtractionComparisonItem",
        "ExtractionComparisonResponse",
        "HumanReviewSubmission",
        "HumanReviewResponse",
        "DocumentNoteSubmission",
        "DocumentNoteResponse",
        "ReviewDraftSubmission",
        "ReviewDraftResponse",
        "ProcessingStatus",
    ):
        setattr(schemas, name, _ns)
    sys.modules["schemas"] = schemas

    deps = types.ModuleType("dependencies")
    deps.get_workspace_client = lambda: None
    deps.get_lakercm_db = lambda: None
    deps.get_current_user_email = lambda request: "reviewer@example.com"
    deps.resolve_user_identity = lambda *a, **k: None
    sys.modules["dependencies"] = deps

    cfg = types.ModuleType("config")
    cfg.settings = types.SimpleNamespace(pipeline_id="", lakercm_schema="lakercm")
    sys.modules["config"] = cfg


from tests._isolation import IsolatedModules  # noqa: E402

_ISOLATION = IsolatedModules()
documents = None


def setUpModule():
    global documents
    _ISOLATION.start(purge_first_party=True)
    _install_stubs()
    from routes import documents as stubbed_documents

    documents = stubbed_documents


def tearDownModule():
    _ISOLATION.stop()


def _verdict(value):
    """Stand-in for the ExtractionVerdict enum: the route reads `.value`."""
    return types.SimpleNamespace(value=value)


class FakeDB:
    def __init__(self, doc=None, draft=None, delete_raises=False):
        self._doc = doc
        self._draft = draft
        self._delete_raises = delete_raises
        self.upserted = None
        self.deleted = []
        self.review_upserted = None

    def get_document_by_id(self, document_id):
        return self._doc

    def get_review_draft(self, document_id, user_email):
        return self._draft

    def upsert_review_draft(
        self, document_id, user_email, verdict=None, reasoning=None, corrections=None
    ):
        self.upserted = (document_id, user_email, verdict, reasoning, corrections)
        return {
            "document_id": document_id,
            "user_email": user_email,
            "verdict": verdict,
            "reasoning": reasoning,
            "corrections": corrections or {},
            "updated_at": None,
        }

    def delete_review_draft(self, document_id, user_email):
        if self._delete_raises:
            raise RuntimeError("lakebase unreachable")
        self.deleted.append((document_id, user_email))
        return True

    # --- collaborators the submit path needs ------------------------------
    def upsert_extraction_review(self, **kw):
        self.review_upserted = kw
        return {
            "id": "r1",
            "document_id": kw["document_id"],
            "reviewer_email": kw["reviewer_email"],
            "verdict": kw["verdict"],
            "reasoning": kw.get("reasoning"),
            "is_automated": False,
            "corrections": kw.get("corrections"),
            "created_at": None,
            "updated_at": None,
        }

    def reconcile_accepted_proposals(self, document_id, corrections):
        return None


class _FakeBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *a, **k):
        self.tasks.append(fn)


def _run(coro):
    return asyncio.run(coro)


class GetDraftTests(unittest.TestCase):
    def test_missing_document_404(self):
        db = FakeDB(doc=None)
        with self.assertRaises(_HTTPException) as ctx:
            _run(documents.get_review_draft(request=object(), document_id="d1", db=db))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_no_draft_returns_200_not_404(self):
        # The form must render without special-casing a missing draft, and the
        # client must never treat "no draft yet" as an error.
        db = FakeDB(doc={"id": "d1"}, draft=None)
        resp = _run(
            documents.get_review_draft(request=object(), document_id="d1", db=db)
        )
        self.assertIs(resp.exists, False)
        self.assertEqual(resp.document_id, "d1")

    def test_existing_draft_is_echoed_with_exists_true(self):
        db = FakeDB(
            doc={"id": "d1"},
            draft={
                "document_id": "d1",
                "user_email": "reviewer@example.com",
                "verdict": "partially_correct",
                "reasoning": "dx code looks wrong",
                "corrections": {"id:3": "E11.9"},
                "updated_at": None,
            },
        )
        resp = _run(
            documents.get_review_draft(request=object(), document_id="d1", db=db)
        )
        self.assertIs(resp.exists, True)
        self.assertEqual(resp.verdict, "partially_correct")
        self.assertEqual(resp.corrections, {"id:3": "E11.9"})


class SaveDraftTests(unittest.TestCase):
    def test_missing_document_404(self):
        db = FakeDB(doc=None)
        body = types.SimpleNamespace(verdict=None, reasoning=None, corrections=None)
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                documents.save_review_draft(
                    request=object(), document_id="d1", body=body, db=db
                )
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_saves_a_full_draft(self):
        db = FakeDB(doc={"id": "d1"})
        body = types.SimpleNamespace(
            verdict=_verdict("incorrect"),
            reasoning="wrong payer",
            corrections={"id:1": "Veridane"},
        )
        resp = _run(
            documents.save_review_draft(
                request=object(), document_id="d1", body=body, db=db
            )
        )
        self.assertEqual(db.upserted[2], "incorrect")
        self.assertEqual(db.upserted[3], "wrong payer")
        self.assertEqual(db.upserted[4], {"id:1": "Veridane"})
        self.assertIs(resp.exists, True)

    def test_saves_a_PARTIAL_draft_with_no_verdict(self):
        # The whole reason a draft cannot be a row in document_extraction_reviews:
        # a reviewer corrects a field before choosing a verdict, and that has to
        # persist. A verdict-required path here would drop exactly that work.
        db = FakeDB(doc={"id": "d1"})
        body = types.SimpleNamespace(
            verdict=None, reasoning=None, corrections={"id:2": "99213"}
        )
        resp = _run(
            documents.save_review_draft(
                request=object(), document_id="d1", body=body, db=db
            )
        )
        self.assertIsNone(db.upserted[2])
        self.assertEqual(db.upserted[4], {"id:2": "99213"})
        self.assertIs(resp.exists, True)

    def test_saving_a_draft_does_not_submit_a_review(self):
        # A draft is explicitly not a review: no verdict validation, no analytics
        # trigger, no proposal reconciliation.
        db = FakeDB(doc={"id": "d1"})
        body = types.SimpleNamespace(
            verdict=_verdict("incorrect"), reasoning=None, corrections=None
        )
        _run(
            documents.save_review_draft(
                request=object(), document_id="d1", body=body, db=db
            )
        )
        self.assertIsNone(db.review_upserted)


class DiscardDraftTests(unittest.TestCase):
    def test_delete_discards(self):
        db = FakeDB(doc={"id": "d1"})
        resp = _run(
            documents.discard_review_draft(request=object(), document_id="d1", db=db)
        )
        self.assertIs(resp["deleted"], True)
        self.assertEqual(db.deleted, [("d1", "reviewer@example.com")])


class SubmitClearsDraftTests(unittest.TestCase):
    def _submit(self, db):
        body = types.SimpleNamespace(
            verdict=_verdict("correct"), reasoning="looks right", corrections=None
        )
        return _run(
            documents.submit_extraction_review(
                request=object(),
                document_id="d1",
                body=body,
                background_tasks=_FakeBackgroundTasks(),
                db=db,
                workspace_client=object(),
            )
        )

    def test_submitting_deletes_the_draft(self):
        # Without this the client, which prefers a draft over a submitted review
        # on load, would show the pre-submit state forever.
        db = FakeDB(doc={"id": "d1"})
        self._submit(db)
        self.assertEqual(db.deleted, [("d1", "reviewer@example.com")])

    def test_a_failing_draft_cleanup_does_not_fail_the_review(self):
        # The review row is already written by this point. Losing the response to
        # a cleanup error would tell the reviewer their submit failed when it did
        # not, and they would submit again.
        db = FakeDB(doc={"id": "d1"}, delete_raises=True)
        resp = self._submit(db)
        self.assertIsNotNone(db.review_upserted)
        self.assertEqual(resp.verdict, "correct")


if __name__ == "__main__":
    unittest.main()
