"""Unit tests for the document upload route's stored path scheme.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_documents_upload

Pins the path-scheme contract that caused a prod incident: the Lakebase
row's file_path must be the dbfs: URI form (matching Auto Loader's path
column in gold_extraction_labels_sync.document_path, which status sync and
the streamed-doc backfill join on by exact string equality), while the
Files API upload must receive the bare /Volumes/... POSIX form. A bare
/Volumes/... row never matches gold: it stays 'processing' forever and the
backfill inserts a duplicate row for the same file.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock

_REVIEWER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REVIEWER_DIR not in sys.path:
    sys.path.insert(0, _REVIEWER_DIR)

from fastapi import BackgroundTasks  # noqa: E402

from tests._isolation import IsolatedModules  # noqa: E402


def _install_config_stub() -> None:
    """Stub config before routes.documents imports `settings` (the real module
    reads deployment env at import time)."""
    cfg = types.ModuleType("config")

    class _Settings:
        lakercm_schema = "lakercm"
        catalog = "test_catalog"
        lakercm_schema = "test_schema"
        pipeline_id = None  # skip the pipeline-trigger background task
        # Required by services.lakehouse_db at import time (routes.documents →
        # dependencies → lakehouse_db); this test never reads them.
        auto_verdict_threshold = 0.85
        automated_reviewer_email = "<automated>"

        def get_warehouse_id(self):
            return "wh-test"

    cfg.settings = _Settings()  # type: ignore[attr-defined]
    sys.modules["config"] = cfg


# The stub is live only while this module's tests run; installed at import time
# (and only "if not already there") it leaked, or lost to another module's.
_ISOLATION = IsolatedModules()
documents = None


def setUpModule():
    global documents
    _ISOLATION.start(purge_first_party=True)
    _install_config_stub()
    import routes.documents as stubbed_documents

    documents = stubbed_documents


def tearDownModule():
    _ISOLATION.stop()


def _fake_upload_file(name="referral_test.png", content=b"\x89PNG fake bytes"):
    f = MagicMock()
    f.filename = name
    f.content_type = "image/png"
    f.read = AsyncMock(return_value=content)
    f.close = AsyncMock()
    return f


class UploadPathSchemeTest(unittest.TestCase):
    def setUp(self):
        self._orig_get_email = documents.get_current_user_email
        documents.get_current_user_email = lambda request: "tester@x.com"

    def tearDown(self):
        documents.get_current_user_email = self._orig_get_email

    def test_stored_file_path_is_dbfs_uri_and_upload_is_posix(self):
        db = MagicMock()
        db.find_document_by_content_hash.return_value = None
        db.create_document_record.return_value = "new-doc-id"
        client = MagicMock()

        response = asyncio.run(
            documents.upload_document(
                request=MagicMock(),
                background_tasks=BackgroundTasks(),
                file=_fake_upload_file(),
                document_type=None,
                notes=None,
                workspace_client=client,
                db=db,
            )
        )

        # Files API gets the bare POSIX volume path.
        upload_kwargs = client.files.upload.call_args.kwargs
        posix_path = upload_kwargs["file_path"]
        self.assertTrue(
            posix_path.startswith(
                "/Volumes/test_catalog/test_schema/documents_input/tester_"
            ),
            posix_path,
        )
        self.assertTrue(posix_path.endswith("_referral_test.png"), posix_path)

        # Lakebase row gets the dbfs: URI form of the SAME path — the exact
        # string Auto Loader records as document_path.
        record = db.create_document_record.call_args.args[0]
        self.assertEqual(record["file_path"], f"dbfs:{posix_path}")
        self.assertEqual(record["document_name"], "referral_test.png")
        self.assertEqual(record["processing_status"], "processing")

        # Response keeps the POSIX form and reports processing.
        self.assertEqual(response.volume_path, posix_path)
        self.assertEqual(response.status, "processing")
        self.assertEqual(response.id, "new-doc-id")


if __name__ == "__main__":
    unittest.main()
