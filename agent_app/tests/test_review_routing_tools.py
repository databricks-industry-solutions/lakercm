"""The agent's status tools follow the pipeline's routing and say why.

The tools used to re-derive auto-verification from confidence alone, so a
confident document the pipeline sent to review (an invalid or non-billable code,
no member ID) counted as auto-verified, and the agent could not say why a
document was waiting. They now read `is_automated`, and `review_reasons` once
the Lakebase copy of gold has that column.

The tool SQL runs for real on DuckDB, against stand-ins for the Lakebase tables.
langchain_core / databricks.sdk / config / services.lakehouse_db are stubbed
before import, as in tests.test_reviewer_action_tools.

Run from agent_app/:
    python3 -m pytest tests/test_review_routing_tools.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

_HAVE_DUCKDB = importlib.util.find_spec("duckdb") is not None
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


# (name, confidence, is_automated, review_reasons, reviewed)
_DOCS = [
    ("note.pdf", 1.0, False, ["invalid_code"], False),
    ("eob.pdf", 1.0, False, ["missing_member_id"], False),
    ("referral.pdf", 0.97, True, [], False),
    ("unsure.pdf", 0.5, False, [], False),
    ("reviewed.pdf", 0.99, True, [], True),
]


class DuckLakebase:
    """Lakebase stand-in that runs the tools' SQL on DuckDB."""

    def __init__(self, with_reasons: bool = True, schema: str = "lakercm"):
        import duckdb

        self.schema = schema
        self.con = con = duckdb.connect()
        con.execute("CREATE SCHEMA public")
        con.execute(f"CREATE SCHEMA {schema}")
        con.execute(
            "CREATE TABLE public.medical_documents (id INTEGER, document_name VARCHAR,"
            " file_path VARCHAR, file_size BIGINT, user_email VARCHAR,"
            " document_type VARCHAR, notes VARCHAR, processing_status VARCHAR,"
            " upload_timestamp TIMESTAMP, processing_timestamp TIMESTAMP,"
            " deleted_at TIMESTAMP)"
        )
        con.execute(
            "CREATE TABLE public.document_extraction_reviews (id INTEGER,"
            " document_id INTEGER, verdict VARCHAR, reviewer_email VARCHAR)"
        )
        reasons_col = ", review_reasons VARCHAR[]" if with_reasons else ""
        con.execute(
            f"CREATE TABLE {schema}.gold_extraction_labels_sync (document_path"
            " VARCHAR, document_name VARCHAR, label VARCHAR, identifiers VARCHAR,"
            " elements VARCHAR, confidence_score DOUBLE, is_automated BOOLEAN,"
            f" extracted_at TIMESTAMP{reasons_col})"
        )
        for i, (name, conf, auto, reasons, reviewed) in enumerate(_DOCS, start=1):
            path = f"/Volumes/c/s/documents_input/{name}"
            con.execute(
                "INSERT INTO public.medical_documents VALUES (?, ?, ?, 0, 'u@x.org',"
                " NULL, NULL, 'pending', TIMESTAMP '2026-09-25 05:00:00', NULL, NULL)",
                [i, name, path],
            )
            values = [path, name, conf, auto] + ([reasons] if with_reasons else [])
            con.execute(
                f"INSERT INTO {schema}.gold_extraction_labels_sync VALUES (?, ?,"
                " 'referral_workqueue', '[]', '[]', ?, ?,"
                " TIMESTAMP '2026-09-25 05:00:00'" + (", ?)" if with_reasons else ")"),
                values,
            )
            if reviewed:
                con.execute(
                    "INSERT INTO public.document_extraction_reviews"
                    " VALUES (1, ?, 'correct', 'r@x.org')",
                    [i],
                )

    def gold_sync_available(self) -> bool:
        return True

    def gold_sync_has_column(self, column: str) -> bool:
        # The same catalog lookup as services.lakehouse_db.gold_sync_has_column.
        return bool(
            self.execute_query(
                "SELECT 1 AS present FROM information_schema.columns"
                f" WHERE table_schema = '{self.schema}'"
                " AND table_name = 'gold_extraction_labels_sync'"
                " AND column_name = %s",
                (column,),
            )
        )

    def execute_query(self, query, params=None):
        # DuckDB has no JSONB; the NULL stand-in's type is all that differs.
        q = query.replace("%s", "?").replace("::JSONB", "::VARCHAR")
        cur = self.con.execute(q, list(params or []))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@unittest.skipUnless(_HAVE_DUCKDB, "duckdb not installed")
class ReviewRoutingToolTests(unittest.TestCase):
    def setUp(self):
        _DB_HOLDER["db"] = DuckLakebase()

    def test_counts_follow_the_pipeline_not_confidence(self):
        counts = json.loads(tools.get_documents_by_status())
        self.assertEqual(
            {k: counts[k] for k in ("pending", "auto_verified", "reviewed")},
            {"pending": 3, "auto_verified": 1, "reviewed": 1},
            "a confident document the pipeline flagged counted as auto-verified",
        )
        self.assertEqual(counts["total"], len(_DOCS))

    def test_pending_documents_carry_their_review_reasons(self):
        out = json.loads(tools.get_documents_by_status(status="pending"))
        reasons = {d["document_name"]: d["review_reasons"] for d in out["documents"]}
        self.assertEqual(
            reasons,
            {
                "note.pdf": ["invalid_code"],
                "eob.pdf": ["missing_member_id"],
                "unsure.pdf": [],
            },
        )
        self.assertIn("missing_member_id", out["review_reasons_note"])

    def test_document_details_say_why_it_waits(self):
        doc = json.loads(tools.get_document_details("note"))
        self.assertIs(doc["is_automated"], False)
        self.assertEqual(doc["review_reasons"], ["invalid_code"])
        self.assertIn("review_reasons_note", doc)

    def test_review_statistics_count_only_pipeline_auto_verified_documents(self):
        stats = json.loads(tools.get_review_statistics())
        self.assertEqual(stats["auto_verified_count"], 1)

    def test_tools_work_before_the_synced_table_has_review_reasons(self):
        _DB_HOLDER["db"] = DuckLakebase(with_reasons=False)
        out = json.loads(tools.get_documents_by_status(status="pending"))
        self.assertEqual(out["count"], 3)
        self.assertTrue(all(d["review_reasons"] is None for d in out["documents"]))
        doc = json.loads(tools.get_document_details("eob"))
        self.assertIsNone(doc["review_reasons"])


@unittest.skipUnless(_HAVE_DUCKDB, "duckdb not installed")
class ReviewRoutingUnderTheDevSchemaTests(ReviewRoutingToolTests):
    """Every one of those tools again, against a dev-style Postgres schema.

    A synced table takes its Postgres schema from its Unity Catalog schema, so
    dev's copy of gold is lakercm_dev.gold_extraction_labels_sync. The tools
    must follow LAKERCM_SCHEMA; when they hardcoded 'lakercm' the dev
    agent reported an empty environment while 12 documents sat in dev's gold.
    """

    SCHEMA = "lakercm_dev"

    def setUp(self):
        _DB_HOLDER["db"] = DuckLakebase(schema=self.SCHEMA)
        self._schema_before = tools.settings.schema_name
        tools.settings.schema_name = self.SCHEMA

    def tearDown(self):
        tools.settings.schema_name = self._schema_before

    def test_tools_work_before_the_synced_table_has_review_reasons(self):
        _DB_HOLDER["db"] = DuckLakebase(with_reasons=False, schema=self.SCHEMA)
        out = json.loads(tools.get_documents_by_status(status="pending"))
        self.assertEqual(out["count"], 3)
        self.assertTrue(all(d["review_reasons"] is None for d in out["documents"]))


if __name__ == "__main__":
    unittest.main()
