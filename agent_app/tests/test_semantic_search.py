"""Unit tests for the semantic_search_documents agent tool + RRF fusion.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_semantic_search

Heavy deps (langchain_core, databricks.sdk, config, services.lakehouse_db,
services.embeddings) are stubbed in sys.modules before import — same approach as
tests.test_reviewer_action_tools — so this runs offline with no app deps and no
network. The fake DB routes on SQL substring (vector arm vs keyword arm).
"""

from __future__ import annotations

import json
import os
import sys
import types
import unittest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

_DB_HOLDER: dict = {"db": None}


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
        auto_verdict_threshold=0.92, pipeline_id="", schema_name="lakercm"
    )
    sys.modules["config"] = cfg

    svc = types.ModuleType("services")
    svc_db = types.ModuleType("services.lakehouse_db")
    svc_db.get_db = lambda: _DB_HOLDER["db"]
    svc.lakehouse_db = svc_db
    # Stub the embeddings helper the tool lazily imports.
    svc_emb = types.ModuleType("services.embeddings")
    svc_emb.embed_one = lambda text: [0.1, 0.2, 0.3]
    svc_emb.to_pgvector_literal = lambda vec: "[" + ",".join(map(str, vec)) + "]"
    svc.embeddings = svc_emb
    sys.modules["services"] = svc
    sys.modules["services.lakehouse_db"] = svc_db
    sys.modules["services.embeddings"] = svc_emb


from tests._isolation import IsolatedModules  # noqa: E402

# The stubs are live only while this module's tests run (setUpModule ..
# tearDownModule); installed at import time they leaked into every other module
# of a single pytest run. `tools` is imported fresh against them.
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


def _row(path, **kw):
    """A public.document_chunks row as the search arms project it."""
    r = {
        "document_path": path,
        "chunk_id": kw.get("chunk_id", f"chunk-{path}"),
        "document_id": kw.get("document_id", f"id-{path}"),
        "document_name": kw.get("document_name", path),
        "user_email": kw.get("user_email", "reviewer@example.com"),
        "chunk_position": kw.get("chunk_position", 0),
        "chunk_to_retrieve": kw.get("chunk_to_retrieve", f"text for {path}"),
        "page_id": kw.get("page_id", 1),
        "image_uri": kw.get("image_uri", f"/Volumes/pages/{path}.png"),
        "score": kw.get("score", 0.5),
    }
    return r


def _gold(path, **kw):
    """A gold_extraction_labels_sync row as _curated_by_path projects it."""
    return {
        "document_path": path,
        "label": kw.get("label", "denial_management"),
        "confidence_score": kw.get("confidence_score", 0.9),
    }


class FakeDB:
    """Routes execute_query on SQL content: vector arm, keyword arm, gold lookup.

    The gold lookup is the small curated-field enrichment the document-level tool
    runs over its final result set; it carries neither "<=>" nor a tsquery, so it
    needs its own route or it would be served chunk rows.
    """

    def __init__(
        self,
        vector_rows=None,
        keyword_rows=None,
        gold_rows=None,
        raise_on=None,
        bm25=False,
    ):
        self.vector_rows = vector_rows or []
        self.keyword_rows = keyword_rows or []
        self.gold_rows = gold_rows or []
        self.raise_on = raise_on
        self.bm25 = bm25  # "vector" | "keyword" | "all" | None

    def execute_query(self, query, params=None):
        # The BM25 capability probe. Routed explicitly: it carries neither "<=>"
        # nor a tsquery, so without this it would be served keyword rows and the
        # detected backend would depend on unrelated fixtures.
        if "pg_class" in query:
            return [{"c": 1}] if self.bm25 else []
        if "gold_extraction_labels_sync" in query:
            return list(self.gold_rows)
        is_vector = "<=>" in query
        if (
            self.raise_on == "all"
            or (self.raise_on == "vector" and is_vector)
            or (self.raise_on == "keyword" and not is_vector)
        ):
            raise RuntimeError("relation does not exist")
        return list(self.vector_rows if is_vector else self.keyword_rows)


class SemanticSearchToolTests(unittest.TestCase):
    def setUp(self):
        tools._bm25_available = None  # detected once per process; reset per test

    def tearDown(self):
        _DB_HOLDER["db"] = None
        tools._bm25_available = None

    def test_empty_query_returns_zero(self):
        _DB_HOLDER["db"] = FakeDB()
        out = json.loads(tools.semantic_search_documents("   "))
        self.assertEqual(out["count"], 0)

    def test_semantic_mode_returns_vector_order(self):
        _DB_HOLDER["db"] = FakeDB(
            vector_rows=[_row("a.pdf"), _row("b.pdf")],
            keyword_rows=[_row("z.pdf")],
        )
        out = json.loads(tools.semantic_search_documents("denial", mode="semantic"))
        paths = [r["document_path"] for r in out["results"]]
        self.assertEqual(paths, ["a.pdf", "b.pdf"])  # keyword arm ignored
        # curated output only — matched_text is the embedding_text, no raw fields
        self.assertEqual(out["results"][0]["matched_text"], "text for a.pdf")
        self.assertNotIn("elements", out["results"][0])

    def test_keyword_mode_returns_keyword_only(self):
        _DB_HOLDER["db"] = FakeDB(
            vector_rows=[_row("a.pdf")],
            keyword_rows=[_row("k.pdf")],
        )
        out = json.loads(tools.semantic_search_documents("code X", mode="keyword"))
        self.assertEqual([r["document_path"] for r in out["results"]], ["k.pdf"])

    def test_hybrid_rrf_ranks_doc_in_both_first(self):
        # 'shared.pdf' appears in both arms → RRF should rank it above singletons.
        _DB_HOLDER["db"] = FakeDB(
            vector_rows=[_row("v1.pdf"), _row("shared.pdf")],
            keyword_rows=[_row("shared.pdf"), _row("k1.pdf")],
        )
        out = json.loads(tools.semantic_search_documents("appeal", mode="hybrid"))
        self.assertEqual(out["results"][0]["document_path"], "shared.pdf")
        # all distinct docs represented
        self.assertEqual(
            {r["document_path"] for r in out["results"]},
            {"shared.pdf", "v1.pdf", "k1.pdf"},
        )

    def test_bad_mode_defaults_to_hybrid(self):
        _DB_HOLDER["db"] = FakeDB(
            vector_rows=[_row("a.pdf")], keyword_rows=[_row("b.pdf")]
        )
        out = json.loads(tools.semantic_search_documents("x", mode="nonsense"))
        self.assertEqual(out["mode"], "hybrid")

    def test_db_failure_is_graceful(self):
        _DB_HOLDER["db"] = FakeDB(raise_on="all")
        out = json.loads(tools.semantic_search_documents("anything"))
        self.assertEqual(out["count"], 0)
        self.assertIn("note", out)

    def test_limit_clamped(self):
        rows = [_row(f"d{i}.pdf") for i in range(80)]
        _DB_HOLDER["db"] = FakeDB(vector_rows=rows, keyword_rows=[])
        out = json.loads(
            tools.semantic_search_documents("x", limit=999, mode="semantic")
        )
        self.assertLessEqual(len(out["results"]), 50)  # hard cap

    def test_rrf_merge_unit(self):
        # doc present in both lists outranks docs present in one
        fused = tools._rrf_merge(["x", "y"], ["y", "z"], limit=3)
        self.assertEqual(fused[0], "y")
        self.assertEqual(set(fused), {"x", "y", "z"})


class RegistrationTests(unittest.TestCase):
    def test_tool_registered(self):
        # get_all_tools imports memory_tools; stub it to isolate registration.
        mem = types.ModuleType("agent.memory_tools")
        mem.get_memory_tools = lambda: []
        sys.modules["agent.memory_tools"] = mem
        names = [getattr(t, "__name__", "") for t in tools.get_all_tools()]
        self.assertIn("semantic_search_documents", names)


if __name__ == "__main__":
    unittest.main()
