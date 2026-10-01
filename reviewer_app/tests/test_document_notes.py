"""Unit tests for the reviewer notepad routes in routes.documents.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_document_notes

Heavy collaborators (fastapi, databricks.sdk, schemas, dependencies, config)
are stubbed in sys.modules before import — same philosophy as
tests.test_transcribe — so this runs with no app deps installed. Covers:
  - GET returns an empty notepad (200, note_text="") when no row exists,
    rather than 404 (deliberate: the editor renders without special-casing)
  - GET / POST raise 404 when the document does not exist
  - POST upserts and echoes the saved text
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

    # schemas: lightweight stand-ins that echo their kwargs.
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

# The stubs are live only while this module's tests run (setUpModule ..
# tearDownModule); installed at import time they leaked into every other module
# of a single pytest run. `documents` is imported fresh against them.
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


class FakeDB:
    def __init__(self, doc=None, note=None):
        self._doc = doc
        self._note = note
        self.upserted = None

    def get_document_by_id(self, document_id):
        return self._doc

    def get_document_notes(self, document_id, user_email):
        return self._note

    def upsert_document_notes(self, document_id, user_email, note_text):
        self.upserted = (document_id, user_email, note_text)
        return {
            "document_id": document_id,
            "user_email": user_email,
            "note_text": note_text,
            "updated_at": None,
        }


def _run(coro):
    # asyncio.run, not get_event_loop(): after any earlier asyncio.run() in the
    # same process there is no current loop, and get_event_loop() raises.
    return asyncio.run(coro)


class DocumentNotesRouteTests(unittest.TestCase):
    def test_get_missing_document_404(self):
        db = FakeDB(doc=None)
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                documents.get_document_notes(request=object(), document_id="d1", db=db)
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_get_empty_returns_200_empty_text(self):
        db = FakeDB(doc={"id": "d1"}, note=None)
        resp = _run(
            documents.get_document_notes(request=object(), document_id="d1", db=db)
        )
        self.assertEqual(resp.note_text, "")
        self.assertEqual(resp.document_id, "d1")

    def test_get_existing_note(self):
        db = FakeDB(
            doc={"id": "d1"},
            note={
                "document_id": "d1",
                "user_email": "reviewer@example.com",
                "note_text": "look at page 3",
                "updated_at": None,
            },
        )
        resp = _run(
            documents.get_document_notes(request=object(), document_id="d1", db=db)
        )
        self.assertEqual(resp.note_text, "look at page 3")

    def test_save_missing_document_404(self):
        db = FakeDB(doc=None)
        body = types.SimpleNamespace(note_text="hi")
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                documents.save_document_notes(
                    request=object(), document_id="d1", body=body, db=db
                )
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_save_upserts_and_echoes(self):
        db = FakeDB(doc={"id": "d1"})
        body = types.SimpleNamespace(note_text="remember to check dx code")
        resp = _run(
            documents.save_document_notes(
                request=object(), document_id="d1", body=body, db=db
            )
        )
        self.assertEqual(db.upserted[0], "d1")
        self.assertEqual(db.upserted[2], "remember to check dx code")
        self.assertEqual(resp.note_text, "remember to check dx code")


if __name__ == "__main__":
    unittest.main()
