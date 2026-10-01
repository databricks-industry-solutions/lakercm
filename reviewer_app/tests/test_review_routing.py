"""The reviewer app routes documents by the pipeline's decision, not confidence.

The app used to re-derive auto-verification from confidence alone
(`confidence_score >= AUTO_VERDICT_THRESHOLD`), so a confident document with an
invalid code, a non-billable code or no member ID skipped the review queue even
after the pipeline flagged it. It now reads `is_automated` from the Lakebase
copy of gold_extraction_labels, which the pipeline sets.

These tests run the app's REAL status SQL on DuckDB, against stand-ins for
public.medical_documents, public.document_extraction_reviews and the synced
gold table. A placeholder left over from the old threshold parameter would
misalign every query's parameters, so executing the SQL catches that too.

Run from reviewer_app/:
    python3 -m pytest tests/test_review_routing.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import unittest

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

_HAVE_DEPS = all(
    importlib.util.find_spec(m) is not None for m in ("duckdb", "psycopg", "fastapi")
)

# (file_path, confidence, is_automated, review_reasons, reviewed)
_DOCS = [
    # Confident, but the pipeline found an invalid code: review.
    ("/Volumes/c/s/documents_input/note.pdf", 1.0, False, ["invalid_code"], False),
    # Confident, but no member ID: review.
    ("/Volumes/c/s/documents_input/eob.pdf", 1.0, False, ["missing_member_id"], False),
    # Clean and confident: auto-verified.
    ("/Volumes/c/s/documents_input/referral.pdf", 0.97, True, [], False),
    # Clean but unsure: review.
    ("/Volumes/c/s/documents_input/unsure.pdf", 0.5, False, [], False),
    # Auto-verified AND human-reviewed: the reviewed bucket wins.
    ("/Volumes/c/s/documents_input/reviewed.pdf", 0.99, True, [], True),
]


def _database(schema: str = "lakercm"):
    """A LakeRCMDatabase whose queries run on a seeded DuckDB."""
    import duckdb

    from services.lakehouse_db import LakeRCMDatabase

    con = duckdb.connect()
    con.execute("CREATE SCHEMA public")
    con.execute(f"CREATE SCHEMA {schema}")
    con.execute("CREATE SEQUENCE doc_ids START 1")
    con.execute(
        "CREATE TABLE public.medical_documents ("
        " id INTEGER DEFAULT nextval('doc_ids'), user_email VARCHAR,"
        " document_name VARCHAR, file_path VARCHAR UNIQUE, file_size BIGINT,"
        " document_type VARCHAR, notes VARCHAR, processing_status VARCHAR,"
        " upload_timestamp TIMESTAMP, processing_timestamp TIMESTAMP,"
        " created_at TIMESTAMP DEFAULT now(), deleted_at TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE public.document_extraction_reviews ("
        " id INTEGER, document_id INTEGER, verdict VARCHAR)"
    )
    con.execute(
        f"CREATE TABLE {schema}.gold_extraction_labels_sync ("
        " document_path VARCHAR, document_name VARCHAR, user_email VARCHAR,"
        " label VARCHAR, confidence_score DOUBLE, is_automated BOOLEAN,"
        " review_reasons VARCHAR[], extracted_at TIMESTAMP)"
    )
    for i, (path, conf, auto, reasons, reviewed) in enumerate(_DOCS, start=1):
        con.execute(
            f"INSERT INTO {schema}.gold_extraction_labels_sync VALUES"
            " (?, ?, 'u@x.org', 'referral_workqueue', ?, ?, ?,"
            " TIMESTAMP '2026-09-25 05:00:00')",
            [path, path.rsplit("/", 1)[-1], conf, auto, reasons],
        )
        if reviewed:
            # A reviewed doc has a medical_documents row with a review on it;
            # the others come in through the streamed-document backfill.
            con.execute(
                "INSERT INTO public.medical_documents (id, user_email, document_name,"
                " file_path, file_size, processing_status) VALUES"
                " (?, 'u@x.org', ?, ?, 0, 'pending')",
                [100 + i, path.rsplit("/", 1)[-1], path],
            )
            con.execute(
                "INSERT INTO public.document_extraction_reviews VALUES (1, ?, 'correct')",
                [100 + i],
            )

    def execute(query, params=None, fetch=True):
        cur = con.execute(query.replace("%s", "?"), list(params or []))
        if not fetch:
            return 0
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    db = LakeRCMDatabase.__new__(LakeRCMDatabase)
    db._connection_pool = None  # no pool: __del__ closes nothing
    db.GOLD_SYNC = f"{schema}.gold_extraction_labels_sync"
    db._execute_query = execute
    db._gold_sync_available = lambda: True
    return db, con


@unittest.skipUnless(_HAVE_DEPS, "duckdb / psycopg / fastapi not installed")
class TestReviewRouting(unittest.TestCase):
    def setUp(self):
        self.db, self.con = _database()

    def test_status_counts_follow_the_pipeline_not_confidence(self):
        counts = self.db.get_status_counts()
        self.assertEqual(
            {k: counts[k] for k in ("pending", "auto_verified", "reviewed")},
            {"pending": 3, "auto_verified": 1, "reviewed": 1},
            "a confident document the pipeline flagged skipped the review queue",
        )
        self.assertEqual(counts["total"], len(_DOCS))

    def test_the_review_queue_holds_every_flagged_document(self):
        pending = self.db.list_all_documents(status_filter="pending")
        self.assertEqual(
            sorted(d["document_name"] for d in pending),
            ["eob.pdf", "note.pdf", "unsure.pdf"],
        )

    def test_streamed_documents_are_backfilled_with_the_pipelines_status(self):
        self.db.list_all_documents()
        stored = dict(
            self.con.execute(
                "SELECT document_name, processing_status FROM public.medical_documents"
            ).fetchall()
        )
        self.assertEqual(stored["note.pdf"], "pending")
        self.assertEqual(stored["referral.pdf"], "auto_verified")

    def test_a_single_document_reports_the_pipelines_status(self):
        self.db.list_all_documents()  # backfill, so the documents have rows
        doc_id = self.con.execute(
            "SELECT id FROM public.medical_documents WHERE document_name = 'note.pdf'"
        ).fetchone()[0]
        self.assertEqual(
            self.db.get_document_by_id(doc_id)["processing_status"], "pending"
        )


class _SyncDB:
    """Just what the sync-status route reads and writes."""

    def __init__(self, gold_rows):
        self._gold = gold_rows
        self.writes = {}

    def get_pending_documents(self):
        return [
            {"id": i, "file_path": r["document_path"]} for i, r in enumerate(self._gold)
        ]

    def get_documents_needing_timestamp_resync(self):
        return []

    def get_unreviewed_ready_documents(self):
        return []

    def get_gold_labels_by_paths(self, paths):
        return [r for r in self._gold if r["document_path"] in paths]

    def update_document_status_with_timestamp(self, path, status, extracted_at):
        self.writes[path] = status
        return True


@unittest.skipUnless(_HAVE_DEPS, "duckdb / psycopg / fastapi not installed")
class TestSyncStatusRoute(unittest.TestCase):
    def test_sync_status_writes_the_pipelines_decision(self):
        from routes.documents import sync_document_status

        db = _SyncDB(
            [
                {
                    "document_path": "a.pdf",
                    "confidence_score": 1.0,
                    "is_automated": False,
                    "extracted_at": "2026-09-25 05:00:00",
                },
                {
                    "document_path": "b.pdf",
                    "confidence_score": 0.95,
                    "is_automated": True,
                    "extracted_at": "2026-09-25 05:00:00",
                },
            ]
        )
        result = asyncio.run(sync_document_status(db=db))
        self.assertEqual(db.writes, {"a.pdf": "pending", "b.pdf": "auto_verified"})
        self.assertEqual((result["pending"], result["auto_verified"]), (1, 1))


@unittest.skipUnless(_HAVE_DEPS, "duckdb / psycopg / fastapi not installed")
class TestReviewRoutingUnderTheDevSchema(TestReviewRouting):
    """The same routing against a dev-style Postgres schema.

    A synced table takes its Postgres schema from its Unity Catalog schema, so
    dev reads lakercm_dev.gold_extraction_labels_sync. The app follows
    LAKERCM_SCHEMA; a hardcoded schema showed dev an empty queue.
    """

    def setUp(self):
        self.db, self.con = _database(schema="lakercm_dev")


if __name__ == "__main__":
    unittest.main()
