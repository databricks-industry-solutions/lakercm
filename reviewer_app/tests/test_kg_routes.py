"""Unit tests for the reviewer app's knowledge-graph routes (routes/kg.py).

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_kg_routes

The assertion that matters most is the JOIN KEY. medical_documents.document_name
is the ORIGINAL upload filename ("invoice.pdf"); gold derives its own
document_name from the volume path, which the upload prefixed
("{email}_{ts}_invoice.pdf"). Joining those returns zero rows, silently -- the
panel would render an empty graph for every document and look like a data
problem. The bridge is basename(file_path).

The second is that this feature cannot ship dark. #88 put the KG tool in the
agent with both env vars unset: it was never registered and nothing said so. So
/api/kg/status reports why it is unavailable, and an unavailable graph is a
stated condition rather than an empty panel.
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


_CFG = types.SimpleNamespace(
    kg_enabled=True,
    kg_schema="ontobricks_registry_dev",
    catalog="cat",
    get_warehouse_id=lambda: "wh1",
)


def _install_stubs() -> None:
    fastapi = types.ModuleType("fastapi")

    class _Router:
        def __init__(self, *a, **k):
            pass

        get = post = delete = put = api_route = staticmethod(_identity_decorator)

    fastapi.APIRouter = _Router
    fastapi.HTTPException = _HTTPException
    fastapi.Depends = lambda *a, **k: None
    fastapi.Request = object
    sys.modules["fastapi"] = fastapi

    deps = types.ModuleType("dependencies")
    deps.get_workspace_client = lambda: None
    deps.get_lakercm_db = lambda: None
    sys.modules["dependencies"] = deps

    cfg = types.ModuleType("config")
    cfg.settings = _CFG
    sys.modules["config"] = cfg

    svc = types.ModuleType("services")
    svc.__path__ = []
    wh = types.ModuleType("services.warehouse")
    wh.warehouse_rows = lambda *a, **k: []
    svc.warehouse = wh
    sys.modules["services"] = svc
    sys.modules["services.warehouse"] = wh


from tests._isolation import IsolatedModules  # noqa: E402

_ISOLATION = IsolatedModules()
kg = None


def setUpModule():
    global kg
    _ISOLATION.start(purge_first_party=True)
    _install_stubs()
    from routes import kg as stubbed_kg

    kg = stubbed_kg


def tearDownModule():
    _ISOLATION.stop()


class FakeDB:
    def __init__(self, doc=None, by_name=None, lookup_error=None):
        self._doc = doc
        self._by_name = by_name or {}
        self._lookup_error = lookup_error
        self.looked_up = None

    def get_document_by_id(self, document_id):
        return self._doc

    def get_documents_by_names(self, names):
        self.looked_up = list(names)
        if self._lookup_error:
            raise self._lookup_error
        return {n: self._by_name[n] for n in names if n in self._by_name}


def _run(coro):
    return asyncio.run(coro)


# A document whose upload name and lakehouse basename DIFFER, which is the whole
# point: joining on document_name would miss it.
DOC = {
    "id": "d1",
    "document_name": "invoice.pdf",
    "file_path": "dbfs:/Volumes/cat/lakercm_dev/documents_input/"
    "user_at_x_com_20260930_120000_invoice.pdf",
}
LAKEHOUSE_NAME = "user_at_x_com_20260930_120000_invoice.pdf"


class TestTheJoinKey(unittest.TestCase):
    def setUp(self):
        _CFG.kg_enabled = True
        _CFG.kg_schema = "ontobricks_registry_dev"
        self.calls = []

        def spy(_client, _wh, sql, params=None):
            self.calls.append((sql, params))
            return []

        kg.warehouse_rows = spy

    def test_it_binds_the_lakehouse_basename_not_the_upload_name(self):
        _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=FakeDB(DOC),
                workspace_client=object(),
            )
        )
        self.assertTrue(self.calls, "no query was issued")
        for _sql, params in self.calls:
            self.assertEqual(params["doc"], LAKEHOUSE_NAME)
            self.assertNotEqual(
                params["doc"],
                DOC["document_name"],
                "bound the upload filename -- this join returns zero rows",
            )

    def test_the_document_name_is_bound_never_interpolated(self):
        # A value that reaches the statement as text is an injection surface, and
        # a filename is user-supplied.
        _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=FakeDB(DOC),
                workspace_client=object(),
            )
        )
        for sql, _params in self.calls:
            self.assertNotIn(LAKEHOUSE_NAME, sql)
            self.assertIn(":doc", sql)

    def test_predicates_are_split_on_both_separators(self):
        # This store mixes them: predicates use '/', rdf:type objects use '#'.
        # Splitting on '/' alone matched nothing -- the bug #88 fixed agent-side.
        _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=FakeDB(DOC),
                workspace_client=object(),
            )
        )
        for sql, _params in self.calls:
            self.assertIn("'[/#]'", sql)

    def test_it_anchors_on_the_document_URI_not_a_bare_name(self):
        """The store holds FULL URIs; bare equality matches nothing, silently.

        This is the bug the first cut of this route shipped with. Asserting only
        that the right VALUE is bound (the test above) does not catch it -- the
        value was already correct. What was wrong is the comparison, so that is
        what this pins. A document with 15 triples returned `edges: []`, which is
        indistinguishable from one the graph has not ingested.
        """
        _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=FakeDB(DOC),
                workspace_client=object(),
            )
        )
        for sql, _params in self.calls:
            self.assertIn("/Document/", sql)
            self.assertIn("LIKE", sql.upper())
            # The exact-equality forms that silently match nothing.
            self.assertNotIn("subject = :doc", sql)
            self.assertNotIn("object = :doc", sql)

    def test_neighbours_are_stripped_to_their_local_name(self):
        # Objects are URIs too, so without this the UI renders
        # "https://lakercm.example/ontology/DiagnosisCode/M54.50" as a spoke label.
        _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=FakeDB(DOC),
                workspace_client=object(),
            )
        )
        # The NEIGHBOURHOOD query (the one with the UNION of both directions);
        # the linked query strips its own projections, which is correct there.
        neigh = [sql for sql, _p in self.calls if "UNION ALL" in sql]
        self.assertEqual(len(neigh), 1, "expected exactly one neighbourhood query")
        self.assertIn("SPLIT(object, '/')", neigh[0])
        self.assertIn("SPLIT(subject, '/')", neigh[0])

    def test_a_missing_document_404s(self):
        with self.assertRaises(_HTTPException) as ctx:
            _run(
                kg.document_neighbourhood(
                    request=object(),
                    document_id="d1",
                    db=FakeDB(None),
                    workspace_client=object(),
                )
            )
        self.assertEqual(ctx.exception.status_code, 404)


class TestUnavailableIsStatedNotInferred(unittest.TestCase):
    def tearDown(self):
        _CFG.kg_enabled = True
        _CFG.kg_schema = "ontobricks_registry_dev"

    def test_flag_off_reports_why(self):
        _CFG.kg_enabled = False
        status = _run(kg.kg_status())
        self.assertIs(status["available"], False)
        self.assertIn("LAKERCM_KG_ENABLED", status["reason"])

    def test_missing_schema_reports_why(self):
        # The half-state: the flag is on but the schema never reached the pod, so
        # the view cannot be qualified. Exactly #88's failure shape.
        _CFG.kg_schema = ""
        status = _run(kg.kg_status())
        self.assertIs(status["available"], False)
        self.assertIn("LAKERCM_KG_SCHEMA", status["reason"])

    def test_enabled_and_configured_is_ok(self):
        status = _run(kg.kg_status())
        self.assertIs(status["available"], True)
        self.assertEqual(status["reason"], "ok")

    def test_a_disabled_graph_queries_nothing(self):
        _CFG.kg_enabled = False
        called = []
        kg.warehouse_rows = lambda *a, **k: called.append(1) or []
        out = _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=FakeDB(DOC),
                workspace_client=object(),
            )
        )
        self.assertIs(out["available"], False)
        self.assertEqual(called, [], "queried the warehouse with the graph off")


class TestTheResultShape(unittest.TestCase):
    def setUp(self):
        _CFG.kg_enabled = True
        _CFG.kg_schema = "ontobricks_registry_dev"

    def test_unknown_edges_are_dropped(self):
        # EDGE_LABELS is an allowlist so an internal predicate cannot appear in
        # the UI just because it exists in the store.
        rows = [
            {"edge": "documentsPatient", "neighbour": "abc123", "direction": "out"},
            {"edge": "someInternalPredicate", "neighbour": "x", "direction": "out"},
        ]
        kg.warehouse_rows = lambda *a, **k: rows
        out = _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=FakeDB(DOC),
                workspace_client=object(),
            )
        )
        self.assertEqual([e["edge"] for e in out["edges"]], ["documentsPatient"])
        self.assertEqual(out["edges"][0]["label"], "patient")

    def test_a_warehouse_failure_degrades_instead_of_raising(self):
        # A cold warehouse or a mid-rebuild graph must not break the review page.
        def boom(*_a, **_k):
            raise RuntimeError("warehouse is starting")

        kg.warehouse_rows = boom
        out = _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=FakeDB(DOC),
                workspace_client=object(),
            )
        )
        self.assertIs(out["available"], True)
        self.assertEqual(out["edges"], [])
        self.assertIn("error", out)


class TestLinkedDocuments(unittest.TestCase):
    """Who else shares each spoke -- the answer a click on a node gives.

    Before this, the panel listed the patient's other documents by NAME and
    resolved each to an id by searching the browser's cached page of the
    document list. That page excludes auto-verified documents, so six of the
    seven links on a family document did nothing at all when clicked.
    """

    LINKED_ROWS = [
        {
            "edge": "documentsPatient",
            "neighbour": "837274cc5d1a635f",
            "document_name": "b.pdf",
            "total": "2",
        },
        {
            "edge": "documentsPatient",
            "neighbour": "837274cc5d1a635f",
            "document_name": "a.pdf",
            "total": "2",
        },
        {
            "edge": "hasDiagnosis",
            "neighbour": "M54.50",
            "document_name": "a.pdf",
            "total": "40",
        },
        # Not an allowlisted edge: must never surface, however it got here.
        {"edge": "type", "neighbour": "Document", "document_name": "z.pdf"},
    ]

    def setUp(self):
        _CFG.kg_enabled = True
        _CFG.kg_schema = "ontobricks_registry_dev"

        def by_query(_client, _wh, sql, params=None):
            return self.LINKED_ROWS if "ROW_NUMBER" in sql else []

        kg.warehouse_rows = by_query

    def _route(self, db):
        return _run(
            kg.document_neighbourhood(
                request=object(),
                document_id="d1",
                db=db,
                workspace_client=object(),
            )
        )

    def test_the_join_follows_the_edge_not_just_the_object(self):
        # Every document has rdf:type ...#Document. Joining on the object alone
        # would link each document to the whole store through that one triple.
        sql = kg._linked_sql("`c`.`s`.`t`")
        self.assertIn("= nb.edge", sql)
        for edge in kg.EDGE_LABELS:
            self.assertIn(f"'{edge}'", sql)

    def test_it_excludes_this_document_and_caps_each_spoke(self):
        # Capped per spoke, not with one LIMIT: a payer with thousands of claims
        # must not crowd the patient's handful out of the result.
        sql = kg._linked_sql("`c`.`s`.`t`")
        self.assertIn("NOT LIKE CONCAT('%/Document/', :doc)", sql)
        self.assertIn("PARTITION BY edge, uri", sql)
        self.assertIn(f"rn <= {kg.MAX_LINKED_PER_NODE}", sql)

    def test_linked_documents_carry_their_reviewer_id(self):
        db = FakeDB(
            DOC,
            by_name={
                "a.pdf": {"id": "id-a", "status": "auto_verified", "label": "x"},
            },
        )
        out = self._route(db)
        patient = out["linked"]["documentsPatient:837274cc5d1a635f"]
        self.assertEqual(patient["total"], 2)
        self.assertEqual(
            patient["documents"][0],
            {"name": "a.pdf", "id": "id-a", "status": "auto_verified", "label": "x"},
        )
        # Known to the graph, not yet to the review queue: listed, not openable.
        self.assertEqual(patient["documents"][1]["name"], "b.pdf")
        self.assertIsNone(patient["documents"][1]["id"])
        # One lookup for every name across every spoke, not one per spoke.
        self.assertEqual(db.looked_up, ["a.pdf", "b.pdf"])

    def test_total_counts_what_the_cap_did_not_return(self):
        out = self._route(FakeDB(DOC))
        dx = out["linked"]["hasDiagnosis:M54.50"]
        self.assertEqual(len(dx["documents"]), 1)
        self.assertEqual(dx["total"], 40)

    def test_unknown_edges_are_not_linked(self):
        out = self._route(FakeDB(DOC))
        self.assertFalse([k for k in out["linked"] if k.startswith("type:")])

    def test_siblings_are_still_the_patients_documents(self):
        out = self._route(FakeDB(DOC))
        self.assertEqual(out["siblings"], ["a.pdf", "b.pdf"])

    def test_a_failed_id_lookup_keeps_the_documents_listed(self):
        # Lakebase being unreachable costs the reviewer the click-through, not
        # the graph.
        out = self._route(FakeDB(DOC, lookup_error=RuntimeError("pool exhausted")))
        self.assertNotIn("error", out)
        docs = out["linked"]["documentsPatient:837274cc5d1a635f"]["documents"]
        self.assertEqual([d["name"] for d in docs], ["a.pdf", "b.pdf"])
        self.assertTrue(all(d["id"] is None for d in docs))

    def test_the_hub_is_named_by_its_document_type(self):
        out = self._route(FakeDB({**DOC, "extracted_label": "denial_management"}))
        self.assertEqual(out["node_type"], "denial_management")


if __name__ == "__main__":
    unittest.main()
