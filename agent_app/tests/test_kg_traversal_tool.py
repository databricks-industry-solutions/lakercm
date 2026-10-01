"""Unit tests for the knowledge graph traversal tool in agent/tools.py.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_kg_traversal_tool

Covers path parsing, edge validation, type checking, injection resistance, and graceful
degradation. Heavy deps are stubbed, so this runs offline.
"""

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
_KG_SCHEMA = "ontobricks_registry_dev"


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
        schema_name="lakercm_dev",
        dbsql_mcp_warehouse_id=_PIN,
        kg_enabled=True,
        kg_schema=_KG_SCHEMA,
    )
    sys.modules["config"] = cfg

    svc = types.ModuleType("services")
    # get_all_tools() pulls in agent.memory_tools, which imports
    # services.store.get_store, and agent.tools lazily imports
    # services.embeddings. A non-package stub makes either a
    # ModuleNotFoundError ("'services' is not a package"), so declare
    # __path__ and register every submodule -- same shape as
    # test_review_remediation_tool.py.
    svc.__path__ = []
    svc_db = types.ModuleType("services.lakehouse_db")
    svc_db.get_db = lambda: _DB_HOLDER["db"]
    svc_store = types.ModuleType("services.store")
    svc_store.get_store = lambda: None
    svc_emb = types.ModuleType("services.embeddings")
    svc_emb.embed_one = lambda text: [0.0] * 1024
    svc_emb.to_pgvector_literal = lambda vec: "[]"
    svc.lakehouse_db = svc_db
    svc.store = svc_store
    svc.embeddings = svc_emb
    sys.modules["services"] = svc
    sys.modules["services.lakehouse_db"] = svc_db
    sys.modules["services.store"] = svc_store
    sys.modules["services.embeddings"] = svc_emb


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


class _FakeConfig:
    host = "https://ws.example.com/"

    def authenticate(self):
        return {"Authorization": "Bearer t"}


class KGTraversalTest(unittest.TestCase):
    def setUp(self):
        self.http = None
        self._patches = [
            mock.patch.object(
                tools,
                "_get_workspace_client",
                return_value=types.SimpleNamespace(config=_FakeConfig()),
            ),
            mock.patch.object(tools, "_REYDEN_POLL_SECONDS", 0.0),
            mock.patch.object(tools, "_span", lambda name: contextlib.nullcontext()),
            mock.patch.dict(os.environ, {"MLFLOW_TRACING_SQL_WAREHOUSE_ID": "std-wh"}),
        ]
        for p in self._patches:
            p.start()
        tools.settings.dbsql_mcp_warehouse_id = _PIN
        tools.settings.kg_enabled = True
        tools.settings.kg_schema = _KG_SCHEMA
        _DB_HOLDER["db"] = None

    def tearDown(self):
        for p in self._patches:
            p.stop()
        tools._dbsql_http = None

    def serve(self, *contents, sse=False):
        self.http = FakeHttp(*contents, sse=sse)
        tools._dbsql_http = self.http
        return self.http

    def test_kg_enabled_false_tool_not_present(self):
        """When kg_enabled is False, the tool should not be registered."""
        tools.settings.kg_enabled = False
        all_tools = tools.get_all_tools()
        tool_names = [tools._tool_name(t) for t in all_tools]
        self.assertNotIn("traverse_claims_graph", tool_names)

    def test_kg_enabled_true_tool_present(self):
        """When kg_enabled is True, the tool should be registered."""
        tools.settings.kg_enabled = True
        all_tools = tools.get_all_tools()
        tool_names = [tools._tool_name(t) for t in all_tools]
        self.assertIn("traverse_claims_graph", tool_names)

    def test_kg_status_reports_registered_when_enabled(self):
        """/health must be able to say the tool is actually live in this pod."""
        tools.settings.kg_enabled = True
        tools.settings.kg_schema = _KG_SCHEMA
        tools.get_all_tools()
        st = tools.kg_status()
        self.assertIs(st["enabled"], True)
        self.assertIs(st["tool_registered"], True)
        self.assertEqual(st["schema"], _KG_SCHEMA)
        self.assertEqual(st["reason"], "ok")

    def test_kg_status_distinguishes_flag_off_from_uninitialized(self):
        """An explicit off and a never-ran get_all_tools want different fixes."""
        tools.settings.kg_enabled = False
        tools.get_all_tools()
        st = tools.kg_status()
        self.assertIs(st["enabled"], False)
        self.assertIs(st["tool_registered"], False)
        self.assertIn("LAKERCM_KG_ENABLED", st["reason"])
        self.assertNotEqual(st["reason"], "not initialized")

    def test_kg_status_flags_the_registered_but_schemaless_half_state(self):
        """The gate is the flag alone, but the tool also needs a schema.

        So this pair is reachable and was invisible: the tool registers, the model
        happily calls it, and every call returns unavailable. #88 shipped exactly
        this class of dark failure (both env vars missing), which is why the
        reason string has to name the missing variable.
        """
        tools.settings.kg_enabled = True
        tools.settings.kg_schema = ""
        tools.get_all_tools()
        st = tools.kg_status()
        self.assertIs(st["enabled"], True)
        self.assertIs(st["tool_registered"], True)
        self.assertIn("LAKERCM_KG_SCHEMA", st["reason"])
        # Restore for the rest of the suite.
        tools.settings.kg_schema = _KG_SCHEMA

    def test_kg_status_returns_a_copy(self):
        """A caller must not be able to corrupt the module's boot record."""
        tools.get_all_tools()
        tools.kg_status()["enabled"] = "corrupted"
        self.assertIsNot(tools.kg_status()["enabled"], "corrupted")

    def test_an_anchored_traversal_matches_by_local_name(self):
        """`start` must work as a bare identifier, as the docstring promises.

        The store holds full URIs, so the original exact-equality anchor
        (`subject = 'POL-OK-PT-021'`) matched NOTHING and the traversal returned
        zero endpoints -- silently, and indistinguishably from a real dead end.
        """
        self.serve(_content(json.dumps({"data_array": [], "columns": []})))
        tools.traverse_claims_graph(path="hasDiagnosis", start="POL-OK-PT-021")
        sql = json.dumps(self.http.bodies[0])
        self.assertIn("ELEMENT_AT(SPLIT(subject, '/'), -1)", sql)
        self.assertNotIn("AND subject = 'POL-OK-PT-021'", sql)

    def test_a_full_uri_start_is_reduced_to_its_local_name(self):
        # Both documented forms have to behave identically.
        self.serve(_content(json.dumps({"data_array": [], "columns": []})))
        tools.traverse_claims_graph(
            path="hasDiagnosis",
            start="https://lakercm.example/ontology/Document/synthetic-1-0001-referral.pdf",
        )
        sql = json.dumps(self.http.bodies[0])
        self.assertIn("synthetic-1-0001-referral.pdf", sql)
        self.assertNotIn("https://lakercm.example", sql)

    def test_basic_single_hop_path(self):
        """Test a single-hop traversal hasDiagnosis."""
        self.serve(
            _content(
                columns=[
                    "end_type",
                    "end_id",
                    "path_count",
                    "example_path",
                    "rdfs_label",
                    "citation_label",
                ],
                rows=[
                    [
                        "DiagnosisCode",
                        "DC-123",
                        "5",
                        "doc1->hasDiagnosis->DC-123",
                        "Diabetes",
                        None,
                    ],
                ],
            )
        )
        result = tools.traverse_claims_graph(
            path="hasDiagnosis", start="doc1", limit=25
        )
        data = json.loads(result)
        self.assertEqual(data["endpoints_count"], 1)
        self.assertEqual(len(data["endpoints"]), 1)
        self.assertEqual(data["endpoints"][0]["end_id"], "DC-123")

    def test_three_hop_path(self):
        """Test a multi-hop traversal: ~deniedFor,hasDiagnosis,~governsDiagnosis."""
        self.serve(
            _content(
                columns=[
                    "end_type",
                    "end_id",
                    "path_count",
                    "example_path",
                    "rdfs_label",
                    "citation_label",
                ],
                rows=[
                    [
                        "PayerPolicy",
                        "PP-001",
                        "2",
                        "path",
                        "Some Policy",
                        "Veridane POL-001",
                    ],
                ],
            )
        )
        result = tools.traverse_claims_graph(
            path="~deniedFor,hasDiagnosis,~governsDiagnosis", start="denial1", limit=25
        )
        data = json.loads(result)
        self.assertEqual(data["endpoints_count"], 1)

    def test_invalid_edge_refused(self):
        """Unknown edges are refused with a helpful message."""
        result = tools.traverse_claims_graph(path="unknownEdge", start="doc1", limit=25)
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("unknownEdge", data["error"])
        self.assertIn("valid edges", data["error"].lower())

    def test_type_mismatch_refused(self):
        """Type mismatches in the chain are refused."""
        # DiagnosisCode endpoints cannot have hasProcedure edges (only Documents do)
        result = tools.traverse_claims_graph(
            path="hasDiagnosis,hasProcedure", start="doc1", limit=25
        )
        data = json.loads(result)
        self.assertIn("error", data)

    def test_empty_path_refused(self):
        """Empty path is refused."""
        result = tools.traverse_claims_graph(path="", start="doc1", limit=25)
        data = json.loads(result)
        self.assertIn("error", data)

    def test_four_hop_limit_exceeded(self):
        """More than 3 hops without start is refused."""
        result = tools.traverse_claims_graph(
            path="hasDiagnosis,hasProcedure,billedTo,issuedBy", start="", limit=25
        )
        data = json.loads(result)
        self.assertIn("error", data)

    def test_injection_in_path_refused(self):
        """SQL injection attempts in the path are refused."""
        http = self.serve()
        for malicious_path in [
            "hasDiagnosis; DROP TABLE x",
            "hasDiagnosis, (SELECT 1)",
            "gold_claim_codes",
            "' UNION SELECT * FROM gold_claim_codes --",
            "hasDiagnosis--",
        ]:
            with self.subTest(path=malicious_path):
                result = tools.traverse_claims_graph(
                    path=malicious_path, start="doc1", limit=25
                )
                data = json.loads(result)
                self.assertIn("error", data)
                # Should never reach HTTP
                self.assertEqual(len(http.bodies), 0)

    def test_injection_in_start_escaped(self):
        """SQL injection in start is escaped, not sent as text."""
        http = self.serve(
            _content(
                columns=[
                    "end_type",
                    "end_id",
                    "path_count",
                    "example_path",
                    "rdfs_label",
                    "citation_label",
                ],
                rows=[],
            )
        )
        result = tools.traverse_claims_graph(
            path="hasDiagnosis", start="o'brien.pdf", limit=25
        )
        # Should succeed with start escaped
        data = json.loads(result)
        self.assertEqual(data["endpoints_count"], 0)
        # Verify the start was passed as a literal, not text
        self.assertTrue(len(http.bodies) > 0)
        query = http.bodies[0]["json"]["params"]["arguments"]["query"]
        # The query should contain the escaped version
        self.assertIn("'o''brien.pdf'", query)

    def test_limit_clamped(self):
        """Limits are clamped to valid range."""
        http = self.serve(
            _content(
                columns=[
                    "end_type",
                    "end_id",
                    "path_count",
                    "example_path",
                    "rdfs_label",
                    "citation_label",
                ],
                rows=[],
            )
        )
        tools.traverse_claims_graph(path="hasDiagnosis", start="doc1", limit=500)
        # Should have clamped to max
        query = http.bodies[0]["json"]["params"]["arguments"]["query"]
        self.assertIn("LIMIT 200", query)

    def test_graceful_degradation_no_warehouse(self):
        """Unavailable Reyden warehouse returns degradation message."""
        tools.settings.dbsql_mcp_warehouse_id = ""
        result = tools.traverse_claims_graph(
            path="hasDiagnosis", start="doc1", limit=25
        )
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertNotIn("triplestore", data["error"].lower())
        self.assertNotIn("ontobricks", data["error"].lower())

    def test_graceful_degradation_kg_disabled(self):
        """KG disabled returns degradation message."""
        tools.settings.kg_enabled = False
        result = tools.traverse_claims_graph(
            path="hasDiagnosis", start="doc1", limit=25
        )
        data = json.loads(result)
        self.assertIn("error", data)

    def test_graceful_degradation_no_schema(self):
        """Missing KG schema returns degradation message."""
        tools.settings.kg_schema = ""
        result = tools.traverse_claims_graph(
            path="hasDiagnosis", start="doc1", limit=25
        )
        data = json.loads(result)
        self.assertIn("error", data)

    def test_query_failure_degradation(self):
        """Query failure returns degradation message without table names."""
        self.serve(
            _content(
                state="FAILED",
                error="TABLE_OR_VIEW_NOT_FOUND: ontobricks_registry_dev.triplestore_lakercm_v1",
            )
        )
        result = tools.traverse_claims_graph(
            path="hasDiagnosis", start="doc1", limit=25
        )
        data = json.loads(result)
        self.assertIn("error", data)
        # Should NOT contain table name
        self.assertNotIn("triplestore", data["error"].lower())

    def test_registered_tool_is_injection_guarded(self):
        """The traverse_claims_graph tool is injection-guarded by default."""
        all_tools = tools.get_all_tools()
        for t in all_tools:
            if tools._tool_name(t) == "traverse_claims_graph":
                self.assertTrue(
                    hasattr(t, "injection_guarded")
                    or hasattr(t.func, "injection_guarded")
                )
                break
        else:
            self.fail("traverse_claims_graph not found in tools")

    def test_query_has_right_structure(self):
        """Generated SQL has expected structure: fully qualified, one CTE per hop, etc."""
        http = self.serve(
            _content(
                columns=[
                    "end_type",
                    "end_id",
                    "path_count",
                    "example_path",
                    "rdfs_label",
                    "citation_label",
                ],
                rows=[],
            )
        )
        tools.traverse_claims_graph(
            path="hasDiagnosis,~governsDiagnosis", start="doc1", limit=10
        )
        query = http.bodies[0]["json"]["params"]["arguments"]["query"]
        # Should be fully qualified to the KG schema
        self.assertIn(f"`{tools.settings.catalog}`.`{_KG_SCHEMA}`", query)
        # Should have CTEs (one per hop)
        self.assertIn("WITH", query)
        # Should use the proper triplestore view
        self.assertIn("triplestore", query.lower())
        # Should have the warehouse pin
        # params["_meta"], not the top level -- see _mcp_call in agent/tools.py
        # and the equivalent assertion in test_dbsql_mcp_tool.py.
        self.assertEqual(
            http.bodies[0]["json"]["params"]["_meta"]["warehouse_id"], _PIN
        )
        # Should NOT have a https://lakercm.example prefix literal
        self.assertNotIn("https://lakercm.example", query)


if __name__ == "__main__":
    unittest.main()
