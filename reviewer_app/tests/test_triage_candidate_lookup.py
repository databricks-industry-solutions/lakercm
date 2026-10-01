"""The triage job resolves held documents by PATH, not by a recency window.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_triage_candidate_lookup

Why this exists. `documents_awaiting_triage(limit)` used to return the N newest
documents with no proposal, and jobs/triage_review_queue.py passed
`max_docs * 4` and intersected that window with the held set by path. Nothing
guaranteed the two overlapped. Once the corpus outgrew the window they stopped
overlapping entirely: every held document fell through the "not found" branch,
which incremented `skipped_already_triaged`, and the run wrote zero proposals
while reporting SUCCESS. Measured on a 1,997-document dev corpus, `max_docs=50`
gave a 200-row window and triaged 0 of 50 held documents — the agent-assisted
review surface was silently inert, and the summary read like an idempotent
no-op.

So the invariants worth pinning are the ones whose violation is invisible:
the lookup is scoped to the caller's paths, it carries `has_proposal` so
"already triaged" can be told apart from "no Lakebase row", and it imposes no
row cap of its own.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from tests._isolation import IsolatedModules  # noqa: E402


def _install_config_stub() -> None:
    """Stub config before services.lakehouse_db imports `settings` (the real
    module reads deployment env at import time)."""
    cfg = types.ModuleType("config")

    class _Settings:
        catalog = "test_catalog"
        lakercm_schema = "test_schema"
        auto_verdict_threshold = 0.92
        automated_reviewer_email = "<automated>"

        def get_warehouse_id(self):
            return "wh-test"

    cfg.settings = _Settings()  # type: ignore[attr-defined]
    sys.modules["config"] = cfg


_ISOLATION = IsolatedModules()
lakehouse_db = None


def setUpModule():
    global lakehouse_db
    _ISOLATION.start(fresh=("config", "services.lakehouse_db"))
    _install_config_stub()
    import services.lakehouse_db as _mod

    lakehouse_db = _mod


def tearDownModule():
    _ISOLATION.stop()


class _Captor:
    """A LakeRCMDatabase with only _execute_query wired, built without __init__
    so no Postgres pool or workspace client is created."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def build(self):
        db = lakehouse_db.LakeRCMDatabase.__new__(lakehouse_db.LakeRCMDatabase)
        db._execute_query = self._execute_query  # type: ignore[attr-defined]
        # __init__ is skipped on purpose, but __del__ -> close() reads this and
        # would raise an (ignored) AttributeError that clutters the test log.
        db._connection_pool = None  # type: ignore[attr-defined]
        return db

    def _execute_query(self, query, params=None, fetch=True):
        self.calls.append((query, params))
        return self.rows


class TestLookupDocumentsForTriage(unittest.TestCase):
    def test_scopes_the_query_to_the_given_paths(self):
        cap = _Captor(rows=[])
        lakehouse_db.LakeRCMDatabase.lookup_documents_for_triage(
            cap.build(), ["dbfs:/a.pdf", "dbfs:/b.pdf", "dbfs:/c.pdf"]
        )
        self.assertEqual(len(cap.calls), 1)
        query, params = cap.calls[0]
        self.assertEqual(
            params,
            ("dbfs:/a.pdf", "dbfs:/b.pdf", "dbfs:/c.pdf"),
            "the held paths must be bound as parameters, one placeholder each",
        )
        self.assertIn(
            "file_path IN",
            query,
            "the lookup must filter by the caller's paths; a bare recency "
            "window silently stops overlapping the held set",
        )

    def test_imposes_no_row_cap(self):
        """A LIMIT here is what let the window miss the held set."""
        cap = _Captor(rows=[])
        lakehouse_db.LakeRCMDatabase.lookup_documents_for_triage(
            cap.build(), ["dbfs:/a.pdf"]
        )
        query, _ = cap.calls[0]
        self.assertNotIn(
            "LIMIT",
            query.upper(),
            "the result is already bounded by the number of paths asked for",
        )

    def test_reports_whether_a_proposal_already_exists(self):
        """Without this the job cannot tell 'already triaged' from 'no row',
        and both collapsed into skipped_already_triaged."""
        cap = _Captor(rows=[])
        lakehouse_db.LakeRCMDatabase.lookup_documents_for_triage(
            cap.build(), ["dbfs:/a.pdf"]
        )
        query, _ = cap.calls[0]
        self.assertIn("has_proposal", query)
        self.assertIn("document_review_proposals", query)

    def test_empty_paths_does_not_query(self):
        cap = _Captor(rows=[])
        out = lakehouse_db.LakeRCMDatabase.lookup_documents_for_triage(cap.build(), [])
        self.assertEqual(out, [])
        self.assertEqual(cap.calls, [], "no paths means nothing to resolve")

    def test_returns_rows_as_dicts(self):
        cap = _Captor(
            rows=[
                {
                    "document_id": "doc-1",
                    "document_name": "a.pdf",
                    "file_path": "dbfs:/a.pdf",
                    "has_proposal": False,
                }
            ]
        )
        out = lakehouse_db.LakeRCMDatabase.lookup_documents_for_triage(
            cap.build(), ["dbfs:/a.pdf"]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["document_id"], "doc-1")
        self.assertIs(out[0]["has_proposal"], False)


if __name__ == "__main__":
    unittest.main()
