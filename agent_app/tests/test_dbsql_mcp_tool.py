"""Unit tests for the Reyden analytics tier in agent/tools.py.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_dbsql_mcp_tool

Covers the read-only guard (_qualify_read_only), the fail-closed pin, the
system.ai.dbsql JSON-RPC client (JSON and SSE bodies, polling, row cap), the two
aggregate tools re-backed onto the gold layer (and their Lakebase fallback), and
query_lakehouse. Heavy deps are stubbed in sys.modules before import, the same
way as tests.test_semantic_search, so this runs offline: the HTTP client and the
workspace client are fakes, and nothing reaches a warehouse.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
import unittest
from unittest import mock

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

_DB_HOLDER: dict = {"db": None}
_PIN = "00000000abcd1234"


def _install_stubs() -> None:
    lc = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.tool = lambda f: f  # identity → tools become plain callables
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
        auto_verdict_threshold=0.92,
        pipeline_id="",
        catalog="cat",
        schema_name="sch",
        dbsql_mcp_warehouse_id=_PIN,
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


def _content(state="SUCCEEDED", columns=(), rows=(), statement_id="s-1", error=None):
    """A system.ai.dbsql structuredContent payload, as the live service returns it."""
    status = {"state": state}
    if error:
        status["error"] = {"message": error}
    return {
        "statement_id": statement_id,
        "status": status,
        "manifest": {"schema": {"columns": [{"name": c} for c in columns]}},
        "result": {
            "data_array": [
                {"values": [{} if v is None else {"string_value": v} for v in row]}
                for row in rows
            ]
        },
    }


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class FakeHttp:
    """Serves queued structuredContent payloads; records every request body."""

    def __init__(self, *contents, sse=False):
        self.contents = list(contents)
        self.sse = sse
        self.bodies = []

    def post(self, url, headers=None, json=None):  # noqa: A002 — httpx's name
        self.bodies.append({"url": url, "headers": headers, "json": json})
        message = {
            "jsonrpc": "2.0",
            "id": json["id"],
            "result": {"isError": False, "structuredContent": self.contents.pop(0)},
        }
        body = __import__("json").dumps(message)
        return _Resp(f"event: message\ndata: {body}\n\n" if self.sse else body)


class _FakeConfig:
    host = "https://ws.example.com/"

    def authenticate(self):
        return {"Authorization": "Bearer t"}


class _TierTest(unittest.TestCase):
    def setUp(self):
        self.http = None
        self._patches = [
            mock.patch.object(
                tools,
                "_get_workspace_client",
                return_value=types.SimpleNamespace(config=_FakeConfig()),
            ),
            mock.patch.object(tools, "_REYDEN_POLL_SECONDS", 0.0),
            # _span imports the real mlflow; imported here under the
            # databricks.sdk stub, it would stay bound to it for later modules.
            mock.patch.object(tools, "_span", lambda name: contextlib.nullcontext()),
            mock.patch.dict(os.environ, {"MLFLOW_TRACING_SQL_WAREHOUSE_ID": "std-wh"}),
        ]
        for p in self._patches:
            p.start()
        tools.settings.dbsql_mcp_warehouse_id = _PIN
        _DB_HOLDER["db"] = None

    def tearDown(self):
        for p in self._patches:
            p.stop()
        tools._dbsql_http = None

    def serve(self, *contents, sse=False):
        self.http = FakeHttp(*contents, sse=sse)
        tools._dbsql_http = self.http
        return self.http


class ReadOnlyGuardTests(unittest.TestCase):
    def q(self, sql):
        return tools._qualify_read_only(sql)

    def test_bare_objects_are_qualified(self):
        out = self.q(
            "SELECT verdict, MEASURE(review_count) FROM review_metrics GROUP BY verdict"
        )
        self.assertIn("FROM `cat`.`sch`.`review_metrics`", out)

    def test_ctes_joins_and_function_from_pass(self):
        out = self.q(
            "WITH m AS (SELECT document_type, EXTRACT(YEAR FROM reviewed_at) AS y "
            "FROM fact_review) SELECT * FROM m JOIN dim_document d "
            "ON m.document_type = d.document_type"
        )
        self.assertIn("FROM `cat`.`sch`.`fact_review`", out)
        self.assertIn("JOIN `cat`.`sch`.`dim_document`", out)
        self.assertIn("FROM m ", out)  # the CTE stays a CTE

    def test_qualified_names_in_this_schema_pass(self):
        for ref in (
            "cat.sch.fact_review",
            "sch.fact_review",
            "`cat`.`sch`.`fact_review`",
        ):
            with self.subTest(ref=ref):
                self.assertIn(
                    "`cat`.`sch`.`fact_review`", self.q(f"SELECT 1 FROM {ref}")
                )

    def test_literals_neither_hide_nor_trip_a_check(self):
        out = self.q(
            "SELECT * FROM fact_review WHERE reasoning = 'drop table x; -- FROM secret'"
        )
        self.assertIn("'drop table x; -- FROM secret'", out)

    def test_writes_and_other_statements_are_refused(self):
        for sql in (
            "INSERT INTO fact_review SELECT * FROM fact_review",
            "CREATE TABLE t AS SELECT * FROM fact_review",
            "WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x",
            "SELECT 1 FROM fact_review; DROP TABLE fact_review",
            "DESCRIBE TABLE fact_review",
            "UPDATE fact_review SET verdict = 'correct'",
            "SELECT 1 FROM fact_review -- hide",
            "SELECT /* x */ 1 FROM fact_review",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(tools.ReadOnlyViolation):
                    self.q(sql)

    def test_objects_outside_the_allowlist_are_refused(self):
        for ref in (
            "gold_claim_codes",
            "bronze_doc_parsed",
            "agent_traces_otel_spans",
            "other.sch.fact_review",
            "cat.other.fact_review",
            "system.query.history",
        ):
            with self.subTest(ref=ref):
                with self.assertRaises(tools.ReadOnlyViolation):
                    self.q(f"SELECT * FROM {ref}")

    def test_comma_joins_are_refused(self):
        for sql in (
            "SELECT * FROM fact_review, other.sch.secret",
            "SELECT * FROM fact_review r, dim_document d",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(tools.ReadOnlyViolation):
                    self.q(sql)

    def test_side_effecting_functions_are_refused(self):
        for fn in ("http_request(", "secret(", "ai_query(", "read_files("):
            with self.subTest(fn=fn):
                with self.assertRaises(tools.ReadOnlyViolation):
                    self.q(f"SELECT {fn}'x') FROM fact_review")


class PinTests(_TierTest):
    def test_no_pin_fails_closed(self):
        tools.settings.dbsql_mcp_warehouse_id = ""
        with self.assertRaises(tools.ReydenUnavailable):
            tools._reyden_sql("SELECT 1 FROM fact_review")

    def test_pin_to_the_standard_warehouse_fails_closed(self):
        tools.settings.dbsql_mcp_warehouse_id = "std-wh"
        with self.assertRaises(tools.ReydenUnavailable):
            tools._reyden_sql("SELECT 1 FROM fact_review")

    def test_a_refused_statement_never_reaches_the_service(self):
        http = self.serve()
        with self.assertRaises(tools.ReadOnlyViolation):
            tools._reyden_sql("DROP TABLE fact_review")
        self.assertEqual(http.bodies, [])


class TransportTests(_TierTest):
    def test_every_call_is_pinned_and_qualified(self):
        http = self.serve(_content(columns=["n"], rows=[["3"]]))
        rows, truncated = tools._reyden_sql("SELECT COUNT(*) AS n FROM fact_review")
        self.assertEqual(rows, [{"n": "3"}])
        self.assertFalse(truncated)
        sent = http.bodies[0]
        self.assertEqual(
            sent["url"],
            "https://ws.example.com/ai-gateway/mcp-services/system.ai.dbsql",
        )
        self.assertEqual(sent["headers"]["Authorization"], "Bearer t")
        params = sent["json"]["params"]
        self.assertEqual(params["name"], "execute_sql")
        self.assertEqual(params["_meta"], {"warehouse_id": _PIN})
        self.assertIn("`cat`.`sch`.`fact_review`", params["arguments"]["query"])

    def test_sse_bodies_are_parsed(self):
        self.serve(_content(columns=["n"], rows=[["7"]]), sse=True)
        rows, _ = tools._reyden_sql("SELECT COUNT(*) AS n FROM fact_review")
        self.assertEqual(rows, [{"n": "7"}])

    def test_a_running_statement_is_polled(self):
        http = self.serve(
            _content(state="RUNNING", statement_id="s-9"),
            _content(columns=["n"], rows=[["1"]], statement_id="s-9"),
        )
        rows, _ = tools._reyden_sql("SELECT 1 AS n FROM fact_review")
        self.assertEqual(rows, [{"n": "1"}])
        poll = http.bodies[1]["json"]["params"]
        self.assertEqual(poll["name"], "poll_sql_result")
        self.assertEqual(poll["arguments"], {"statement_id": "s-9"})
        self.assertEqual(poll["_meta"], {"warehouse_id": _PIN})

    def test_a_failed_statement_raises(self):
        self.serve(_content(state="FAILED", error="[UNSUPPORTED_FEATURE] nope"))
        with self.assertRaises(RuntimeError):
            tools._reyden_sql("SELECT 1 FROM fact_review")

    def test_rows_are_capped(self):
        self.serve(_content(columns=["i"], rows=[[str(i)] for i in range(250)]))
        rows, truncated = tools._reyden_sql("SELECT i FROM fact_review")
        self.assertEqual(len(rows), tools._REYDEN_MAX_ROWS)
        self.assertTrue(truncated)


class ReviewStatisticsTests(_TierTest):
    def test_reads_the_metric_view_in_the_lakebase_shape(self):
        http = self.serve(
            _content(
                columns=["verdict", "is_automated", "cnt", "docs", "as_of"],
                rows=[
                    ["correct", "false", "8", "7", "2026-09-25T00:00:00Z"],
                    ["incorrect", "false", "2", "2", "2026-09-25T00:00:00Z"],
                    ["correct", "true", "936", "936", "2026-09-25T00:00:00Z"],
                ],
            )
        )
        out = json.loads(tools.get_review_statistics())
        self.assertEqual(out["total_reviews"], 10)
        self.assertEqual(out["total_documents_reviewed"], 9)
        self.assertEqual(out["correct_count"], 8)
        self.assertEqual(out["incorrect_count"], 2)
        self.assertEqual(out["accuracy_pct"], 80.0)
        self.assertEqual(out["auto_verified_count"], 936)
        self.assertEqual(out["as_of"], "2026-09-25T00:00:00Z")
        self.assertIn("review_metrics", out["source"])
        self.assertIn(
            "MEASURE(review_count)",
            http.bodies[0]["json"]["params"]["arguments"]["query"],
        )

    def test_no_reviews_still_reports_as_of(self):
        self.serve(
            _content(
                columns=["verdict", "is_automated", "cnt", "docs", "as_of"],
                rows=[[None, None, None, None, "2026-08-26T18:35:18Z"]],
            )
        )
        out = json.loads(tools.get_review_statistics())
        self.assertEqual(out["total_reviews"], 0)
        self.assertEqual(out["accuracy_pct"], 0.0)
        self.assertEqual(out["as_of"], "2026-08-26T18:35:18Z")

    def test_a_reviewer_filter_is_a_safe_literal(self):
        http = self.serve(
            _content(columns=["verdict", "is_automated", "cnt", "docs", "as_of"])
        )
        tools.get_review_statistics("o'brien@example.com")
        query = http.bodies[0]["json"]["params"]["arguments"]["query"]
        self.assertIn("reviewer_email = 'o''brien@example.com'", query)

    def test_falls_back_to_lakebase_without_a_pin(self):
        tools.settings.dbsql_mcp_warehouse_id = ""

        class FakeDB:
            def execute_query(self, query, params=None):
                return [{"verdict": "correct", "cnt": 3, "docs": 3}]

            def gold_sync_available(self):
                return False

        _DB_HOLDER["db"] = FakeDB()
        out = json.loads(tools.get_review_statistics())
        self.assertEqual(out["total_reviews"], 3)
        self.assertEqual(out["source"], "Lakebase (live)")


class PipelineLatencyTests(_TierTest):
    def test_reads_fact_document_processing_in_the_lakebase_shape(self):
        http = self.serve(
            _content(
                columns=[
                    "total",
                    "avg_seconds",
                    "min_seconds",
                    "max_seconds",
                    "median_seconds",
                    "as_of",
                ],
                rows=[
                    [
                        "1000",
                        "1329.2",
                        "46.8",
                        "1877.6",
                        "1346.8",
                        "2026-08-26T18:35:18Z",
                    ]
                ],
            )
        )
        out = json.loads(tools.get_pipeline_latency_stats(48))
        self.assertEqual(out["hours"], 48)
        self.assertEqual(out["count"], 1000)
        self.assertEqual(out["median_seconds"], 1346.8)
        self.assertIn(
            "INTERVAL 48 HOURS", http.bodies[0]["json"]["params"]["arguments"]["query"]
        )

    def test_an_empty_window_says_so(self):
        self.serve(
            _content(
                columns=[
                    "total",
                    "avg_seconds",
                    "min_seconds",
                    "max_seconds",
                    "median_seconds",
                    "as_of",
                ],
                rows=[["0", None, None, None, None, "2026-08-26T18:35:18Z"]],
            )
        )
        out = json.loads(tools.get_pipeline_latency_stats(24))
        self.assertEqual(out["count"], 0)
        self.assertEqual(out["message"], "No eligible documents in window.")
        self.assertEqual(out["as_of"], "2026-08-26T18:35:18Z")


class QueryLakehouseTests(_TierTest):
    def test_returns_columns_and_rows(self):
        self.serve(_content(columns=["document_type", "n"], rows=[["invoice", "4"]]))
        out = json.loads(
            tools.query_lakehouse(
                "SELECT document_type, MEASURE(review_count) AS n FROM review_metrics "
                "GROUP BY document_type"
            )
        )
        self.assertEqual(out["columns"], ["document_type", "n"])
        self.assertEqual(out["rows"], [["invoice", "4"]])

    def test_a_write_is_refused_in_plain_language(self):
        http = self.serve()
        out = tools.query_lakehouse("CREATE TABLE t AS SELECT * FROM fact_review")
        self.assertTrue(out.startswith("Query refused:"))
        self.assertEqual(http.bodies, [])

    def test_unavailable_points_at_the_other_tools(self):
        tools.settings.dbsql_mcp_warehouse_id = ""
        out = tools.query_lakehouse("SELECT 1 FROM fact_review")
        self.assertIn("not available", out)


class AntiWideningTests(unittest.TestCase):
    """Verify isolation between query_lakehouse and KG tools."""

    def test_triplestore_not_in_default_analytics_allowlist(self):
        """query_lakehouse cannot access the triplestore."""
        with self.assertRaises(tools.ReadOnlyViolation) as cm:
            tools._qualify_read_only("SELECT * FROM triplestore_lakercm_v1")
        self.assertIn("not readable", str(cm.exception))

    def test_kg_objects_not_in_analytics_allowlist(self):
        """KG-scoped allowlist does not permit analytics gold objects."""
        with self.assertRaises(tools.ReadOnlyViolation) as cm:
            tools._qualify_read_only(
                "SELECT * FROM review_metrics",
                schema="ontobricks_registry_dev",
                objects=frozenset({"triplestore_lakercm_v1"}),
            )
        self.assertIn("not readable", str(cm.exception))

    def test_kg_scope_does_admit_the_triplestore(self):
        # The negative cases above pass trivially if the kwargs are ignored, so
        # pin the positive direction too: the same call that refuses gold must
        # qualify the triplestore into the KG schema.
        out = tools._qualify_read_only(
            "SELECT * FROM triplestore_lakercm_v1",
            schema="ontobricks_registry_dev",
            objects=frozenset({"triplestore_lakercm_v1"}),
        )
        self.assertIn("`ontobricks_registry_dev`.`triplestore_lakercm_v1`", out)

    def test_multi_hop_ctes_are_not_mistaken_for_tables(self):
        # Regression: the duplicated guard lost the CTE check, and a generated
        # multi-hop query was rejected with "hop_1 is not readable". Comma
        # separators are what make hop_1+ visible to _CTE_NAME at all.
        sql = (
            "WITH hop_0 AS (SELECT object AS endpoint FROM triplestore_lakercm_v1),"
            " hop_1 AS (SELECT t.object AS endpoint FROM hop_0 p JOIN"
            " triplestore_lakercm_v1 t ON p.endpoint = t.subject)"
            " SELECT endpoint FROM hop_1"
        )
        out = tools._qualify_read_only(
            sql,
            schema="ontobricks_registry_dev",
            objects=frozenset({"triplestore_lakercm_v1"}),
        )
        self.assertIn("`ontobricks_registry_dev`.`triplestore_lakercm_v1`", out)


class RegistrationTests(unittest.TestCase):
    def test_query_lakehouse_is_registered_and_guarded(self):
        mem = types.ModuleType("agent.memory_tools")
        mem.get_memory_tools = lambda: []
        sys.modules["agent.memory_tools"] = mem
        registered = {getattr(t, "__name__", ""): t for t in tools.get_all_tools()}
        self.assertIn("query_lakehouse", registered)
        self.assertTrue(
            getattr(registered["query_lakehouse"], "injection_guarded", False)
        )


if __name__ == "__main__":
    unittest.main()
