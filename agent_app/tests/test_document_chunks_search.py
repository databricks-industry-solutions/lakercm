"""Unit tests for the search_document_chunks agent tool + document collapse.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_document_chunks_search

Same offline approach as tests.test_semantic_search: heavy deps are stubbed in
sys.modules for the life of this module only, and the fake DB routes on SQL
content. Nothing here touches Lakebase or the FM endpoint.

What this locks down that test_semantic_search does not:
  * chunk-level results are NOT collapsed — two chunks of one document both come
    back, which is the whole point of the "chat with this document" path;
  * document-level results ARE collapsed — the same two chunks yield ONE
    document, ranked by its best chunk;
  * document_path scoping reaches the SQL as a predicate AND a bound parameter.
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
    svc_emb = types.ModuleType("services.embeddings")
    svc_emb.embed_one = lambda text: [0.1, 0.2, 0.3]
    svc_emb.to_pgvector_literal = lambda vec: "[" + ",".join(map(str, vec)) + "]"
    svc.embeddings = svc_emb
    sys.modules["services"] = svc
    sys.modules["services.lakehouse_db"] = svc_db
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


def _chunk(path, chunk_id, **kw):
    return {
        "document_path": path,
        "chunk_id": chunk_id,
        "document_id": kw.get("document_id", f"id-{path}"),
        "document_name": kw.get("document_name", path),
        "user_email": kw.get("user_email", "reviewer@example.com"),
        "chunk_position": kw.get("chunk_position", 0),
        "chunk_to_retrieve": kw.get("chunk_to_retrieve", f"{chunk_id} of {path}"),
        "page_id": kw.get("page_id", 1),
        "image_uri": kw.get("image_uri", f"/Volumes/pages/{path}-1.png"),
        "score": kw.get("score", 0.5),
    }


class RecordingDB:
    """Routes on SQL content and records every (query, params) pair."""

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
        self.bm25 = bm25
        self.calls: list[tuple[str, tuple]] = []

    def execute_query(self, query, params=None):
        self.calls.append((query, params or ()))

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

    def arm_queries(self) -> list[str]:
        """Just the two search arms — not the gold lookup or the BM25 probe."""
        return [
            q
            for q, _ in self.calls
            if "gold_extraction_labels_sync" not in q and "pg_class" not in q
        ]


class ChunkSearchTests(unittest.TestCase):
    def setUp(self):
        tools._bm25_available = None  # detected once per process; reset per test

    def tearDown(self):
        _DB_HOLDER["db"] = None
        tools._bm25_available = None

    def test_empty_query_returns_zero(self):
        _DB_HOLDER["db"] = RecordingDB()
        out = json.loads(tools.search_document_chunks("   "))
        self.assertEqual(out["count"], 0)

    def test_two_chunks_of_one_document_are_both_returned(self):
        # The counterpart to the document-level collapse: chunk search must keep
        # every passage, or "quote both places it says X" is unanswerable.
        _DB_HOLDER["db"] = RecordingDB(
            vector_rows=[_chunk("a.pdf", "c1"), _chunk("a.pdf", "c2")]
        )
        out = json.loads(tools.search_document_chunks("appeal", mode="semantic"))
        self.assertEqual(out["count"], 2)
        self.assertEqual(
            [r["text"] for r in out["results"]], ["c1 of a.pdf", "c2 of a.pdf"]
        )

    def test_keyword_mode_uses_keyword_arm_only(self):
        _DB_HOLDER["db"] = RecordingDB(
            vector_rows=[_chunk("v.pdf", "c1")],
            keyword_rows=[_chunk("k.pdf", "c9")],
        )
        out = json.loads(tools.search_document_chunks("CO-197", mode="keyword"))
        self.assertEqual([r["document_path"] for r in out["results"]], ["k.pdf"])

    def test_hybrid_rrf_ranks_chunk_in_both_arms_first(self):
        shared = _chunk("s.pdf", "shared")
        _DB_HOLDER["db"] = RecordingDB(
            vector_rows=[_chunk("v.pdf", "c1"), shared],
            keyword_rows=[shared, _chunk("k.pdf", "c2")],
        )
        out = json.loads(tools.search_document_chunks("deadline", mode="hybrid"))
        self.assertEqual(out["results"][0]["text"], "shared of s.pdf")
        self.assertEqual(len(out["results"]), 3)

    def test_document_scope_reaches_sql_and_params(self):
        db = RecordingDB(vector_rows=[_chunk("a.pdf", "c1")])
        _DB_HOLDER["db"] = db
        out = json.loads(
            tools.search_document_chunks("x", document_path="a.pdf", mode="hybrid")
        )
        self.assertEqual(out["document_path"], "a.pdf")
        arms = db.arm_queries()
        self.assertEqual(len(arms), 2)  # vector + keyword
        for q, params in [
            c for c in db.calls if "gold_ext" not in c[0] and "pg_class" not in c[0]
        ]:
            self.assertIn("document_path = %s", q)
            self.assertIn("a.pdf", params)

    def test_a_bare_file_name_also_matches_the_full_stored_path(self):
        # The regression this guards. public.document_chunks keys on the full
        # dbfs:/Volumes/.../<name>.pdf path, so a bare name under plain equality
        # matched NOTHING and the "chat with this document" path silently
        # returned zero for every question. Verified in the browser: the agent
        # received only document_name from get_active_review_context, passed it
        # as document_path, and got count=0 on a document that has chunks.
        db = RecordingDB(vector_rows=[_chunk("a.pdf", "c1")])
        _DB_HOLDER["db"] = db
        tools.search_document_chunks("x", document_path="denial.pdf", mode="semantic")
        for q, params in [
            c for c in db.calls if "gold_ext" not in c[0] and "pg_class" not in c[0]
        ]:
            self.assertIn("document_path LIKE %s", q)
            self.assertIn("%/denial.pdf", params)
            # The exact form is kept alongside it, so a stored path that somehow
            # has no directory part still resolves.
            self.assertIn("denial.pdf", params)

    def test_a_full_path_stays_an_exact_match(self):
        # The common case, and the index-friendly one: no LIKE, no wildcard.
        db = RecordingDB(vector_rows=[_chunk("a.pdf", "c1")])
        _DB_HOLDER["db"] = db
        full = "dbfs:/Volumes/cat/schema/documents_input/denial.pdf"
        tools.search_document_chunks("x", document_path=full, mode="semantic")
        for q, params in [
            c for c in db.calls if "gold_ext" not in c[0] and "pg_class" not in c[0]
        ]:
            self.assertIn("document_path = %s", q)
            self.assertNotIn("LIKE", q)
            self.assertIn(full, params)

    def test_a_whitespace_only_scope_is_no_scope(self):
        db = RecordingDB(vector_rows=[_chunk("a.pdf", "c1")])
        _DB_HOLDER["db"] = db
        tools.search_document_chunks("x", document_path="   ", mode="semantic")
        for q in db.arm_queries():
            # Not a bare "document_path" check: that name is in the SELECT list.
            self.assertNotIn("AND document_path", q)
            self.assertNotIn("LIKE", q)

    def test_unscoped_search_adds_no_document_predicate(self):
        db = RecordingDB(vector_rows=[_chunk("a.pdf", "c1")])
        _DB_HOLDER["db"] = db
        tools.search_document_chunks("x", mode="semantic")
        for q in db.arm_queries():
            self.assertNotIn("document_path = %s", q)

    def test_db_failure_is_graceful(self):
        _DB_HOLDER["db"] = RecordingDB(raise_on="all")
        out = json.loads(tools.search_document_chunks("anything"))
        self.assertEqual(out["count"], 0)
        self.assertIn("note", out)

    def test_limit_clamped(self):
        rows = [_chunk("d.pdf", f"c{i}") for i in range(80)]
        _DB_HOLDER["db"] = RecordingDB(vector_rows=rows)
        out = json.loads(tools.search_document_chunks("x", limit=999, mode="semantic"))
        self.assertLessEqual(len(out["results"]), 50)

    def test_bad_mode_defaults_to_hybrid(self):
        _DB_HOLDER["db"] = RecordingDB(vector_rows=[_chunk("a.pdf", "c1")])
        out = json.loads(tools.search_document_chunks("x", mode="nonsense"))
        self.assertEqual(out["mode"], "hybrid")


class DocumentCollapseTests(unittest.TestCase):
    """semantic_search_documents now reads the same chunk table."""

    def setUp(self):
        tools._bm25_available = None  # detected once per process; reset per test

    def tearDown(self):
        _DB_HOLDER["db"] = None
        tools._bm25_available = None

    def test_many_chunks_collapse_to_one_document_ranked_by_best(self):
        _DB_HOLDER["db"] = RecordingDB(
            vector_rows=[
                _chunk("a.pdf", "c1", chunk_to_retrieve="best passage"),
                _chunk("a.pdf", "c2", chunk_to_retrieve="worse passage"),
                _chunk("b.pdf", "c1"),
            ]
        )
        out = json.loads(tools.semantic_search_documents("x", mode="semantic"))
        self.assertEqual(
            [r["document_path"] for r in out["results"]], ["a.pdf", "b.pdf"]
        )
        # The arm returns best-first, so the FIRST chunk seen is the document's best.
        self.assertEqual(out["results"][0]["matched_text"], "best passage")

    def test_curated_fields_come_from_the_gold_replica(self):
        _DB_HOLDER["db"] = RecordingDB(
            vector_rows=[_chunk("a.pdf", "c1")],
            gold_rows=[
                {
                    "document_path": "a.pdf",
                    "label": "denial_management",
                    "confidence_score": 0.77,
                }
            ],
        )
        out = json.loads(tools.semantic_search_documents("x", mode="semantic"))
        self.assertEqual(out["results"][0]["label"], "denial_management")
        self.assertEqual(out["results"][0]["confidence_score"], 0.77)

    def test_missing_gold_replica_still_returns_results(self):
        # Fresh workspace: the synced table does not exist yet. Losing the
        # enrichment must not lose the search.
        class NoGoldDB(RecordingDB):
            def execute_query(self, query, params=None):
                if "gold_extraction_labels_sync" in query:
                    raise RuntimeError("relation does not exist")
                return super().execute_query(query, params)

        _DB_HOLDER["db"] = NoGoldDB(vector_rows=[_chunk("a.pdf", "c1")])
        out = json.loads(tools.semantic_search_documents("x", mode="semantic"))
        self.assertEqual(out["count"], 1)
        self.assertIsNone(out["results"][0]["label"])


class EmbeddingOutageTests(unittest.TestCase):
    """embed_one never raises — it returns a zero vector. That must never reach SQL:
    pgvector cosine against a zero vector is NaN, so ORDER BY ties every row and the
    arm hands back an arbitrary page of chunks as "the top matches"."""

    def setUp(self):
        self._real = sys.modules["services.embeddings"].embed_one
        sys.modules["services.embeddings"].embed_one = lambda text: [0.0] * 1024

    def tearDown(self):
        sys.modules["services.embeddings"].embed_one = self._real
        _DB_HOLDER["db"] = None

    def test_zero_vector_never_reaches_sql(self):
        db = RecordingDB(keyword_rows=[_chunk("a.pdf", "c1")])
        _DB_HOLDER["db"] = db
        tools.search_document_chunks("x", mode="hybrid")
        self.assertFalse(
            any("<=>" in q for q in db.arm_queries()),
            "a zero query vector was sent to pgvector",
        )

    def test_hybrid_degrades_to_keyword_with_a_note(self):
        _DB_HOLDER["db"] = RecordingDB(keyword_rows=[_chunk("a.pdf", "c1")])
        out = json.loads(tools.search_document_chunks("x", mode="hybrid"))
        self.assertEqual(out["count"], 1)  # keyword results still served
        self.assertIn("embedding endpoint is unavailable", out["note"])

    def test_semantic_reports_the_outage_not_an_empty_result(self):
        # An empty list would be read as "no document matches"; it must say outage.
        _DB_HOLDER["db"] = RecordingDB(vector_rows=[_chunk("a.pdf", "c1")])
        out = json.loads(tools.search_document_chunks("x", mode="semantic"))
        self.assertEqual(out["count"], 0)
        self.assertIn("NOT evidence", out["note"])

    def test_document_level_search_degrades_too(self):
        _DB_HOLDER["db"] = RecordingDB(keyword_rows=[_chunk("a.pdf", "c1")])
        out = json.loads(tools.semantic_search_documents("x", mode="hybrid"))
        self.assertEqual(out["count"], 1)
        self.assertIn("embedding endpoint is unavailable", out["note"])


class ExplainingPassageTests(unittest.TestCase):
    def setUp(self):
        tools._bm25_available = None  # detected once per process; reset per test

    def tearDown(self):
        _DB_HOLDER["db"] = None
        tools._bm25_available = None

    def test_passage_comes_from_the_better_ranking_arm(self):
        # d.pdf is rank 0 in keyword but rank 1 in vector, so the keyword passage is
        # what explains the match. Preferring the vector arm would show a passage
        # that need not contain the searched term at all.
        _DB_HOLDER["db"] = RecordingDB(
            vector_rows=[
                _chunk("other.pdf", "v0"),
                _chunk("d.pdf", "vc", chunk_to_retrieve="semantic passage"),
            ],
            keyword_rows=[
                _chunk("d.pdf", "kc", chunk_to_retrieve="exact CO-197 passage"),
                _chunk("other2.pdf", "k0"),
            ],
        )
        out = json.loads(tools.semantic_search_documents("CO-197", mode="hybrid"))
        hit = [r for r in out["results"] if r["document_path"] == "d.pdf"][0]
        self.assertEqual(hit["matched_text"], "exact CO-197 passage")


class ScopeMissTests(unittest.TestCase):
    def setUp(self):
        tools._bm25_available = None  # detected once per process; reset per test

    def tearDown(self):
        _DB_HOLDER["db"] = None
        tools._bm25_available = None

    def test_unindexed_path_is_distinguished_from_no_match(self):
        # Zero hits in a named document must not read as "the document says nothing
        # about X" when the path simply is not indexed.
        _DB_HOLDER["db"] = RecordingDB()
        out = json.loads(
            tools.search_document_chunks("x", document_path="typo.pdf", mode="keyword")
        )
        self.assertEqual(out["count"], 0)
        self.assertIn("No chunks are indexed", out["note"])
        self.assertIn("NOT evidence", out["note"])
        self.assertIn("typo.pdf", out["note"])


class KeywordBackendTests(unittest.TestCase):
    """Migration 000035 only creates the BM25 index when Lakebase Search is enabled
    on the project — a one-way UI toggle with no DAB field and no API — so BOTH
    backends are live states the agent must handle, not a migration window."""

    def setUp(self):
        tools._bm25_available = None

    def tearDown(self):
        _DB_HOLDER["db"] = None
        tools._bm25_available = None

    def _keyword_sql(self, *, bm25):
        db = RecordingDB(keyword_rows=[_chunk("a.pdf", "c1")], bm25=bm25)
        _DB_HOLDER["db"] = db
        tools.search_document_chunks("CO-197", mode="keyword")
        arms = db.arm_queries()
        self.assertEqual(len(arms), 1)
        return arms[0]

    def test_bm25_sorts_ASCENDING(self):
        # `<@>` is DISTANCE-like. The docs' own RRF example ranks it with a plain
        # ascending ORDER BY, so DESC would return the WORST matches — the single
        # most damaging way to get this wrong, and invisible without a check.
        sql = self._keyword_sql(bm25=True)
        self.assertIn("to_bm25query", sql)
        self.assertIn("<@>", sql)
        self.assertIn("ORDER BY score ASC", sql)
        self.assertNotIn("ORDER BY score DESC", sql)

    def test_bm25_query_names_the_index(self):
        # to_bm25query takes the index NAME as an argument, so the constant and the
        # migration's index name must agree or the arm breaks at runtime.
        self.assertIn(tools._BM25_INDEX, self._keyword_sql(bm25=True))

    def test_postgres_fts_fallback_sorts_DESCENDING(self):
        sql = self._keyword_sql(bm25=False)
        self.assertIn("ts_rank", sql)
        self.assertIn("ORDER BY score DESC", sql)
        self.assertNotIn("to_bm25query", sql)

    def test_bm25_arm_has_NO_match_predicate(self):
        # Deliberate, and it matters: plainto_tsquery ANDs every term, so on a
        # natural-language question it would demand all words in one chunk, return
        # nothing, and silently delete the keyword arm from the hybrid — defeating
        # the partial-match ranking BM25 exists to provide. The docs' keyword and
        # hybrid examples both rank unfiltered.
        sql = self._keyword_sql(bm25=True)
        self.assertNotIn("@@", sql)
        self.assertNotIn("plainto_tsquery", sql)

    def test_fts_fallback_KEEPS_the_match_predicate(self):
        # ts_rank scores 0 for a non-match, so without @@ the fallback would return
        # zero-score rows as results.
        sql = self._keyword_sql(bm25=False)
        self.assertIn("content_tsv @@ plainto_tsquery", sql)

    def test_backend_is_probed_once_not_per_query(self):
        db = RecordingDB(keyword_rows=[_chunk("a.pdf", "c1")], bm25=True)
        _DB_HOLDER["db"] = db
        for _ in range(3):
            tools.search_document_chunks("x", mode="keyword")
        probes = [q for q, _ in db.calls if "pg_class" in q]
        self.assertEqual(len(probes), 1, "capability probe should be cached")

    def test_probe_failure_falls_back_to_fts(self):
        class NoProbeDB(RecordingDB):
            def execute_query(self, query, params=None):
                if "pg_class" in query:
                    raise RuntimeError("permission denied for table pg_class")
                return super().execute_query(query, params)

        _DB_HOLDER["db"] = NoProbeDB(keyword_rows=[_chunk("a.pdf", "c1")])
        out = json.loads(tools.search_document_chunks("x", mode="keyword"))
        self.assertEqual(out["count"], 1)  # still serves results
        self.assertFalse(tools._bm25_available)


class RegistrationTests(unittest.TestCase):
    def test_both_search_tools_registered(self):
        mem = types.ModuleType("agent.memory_tools")
        mem.get_memory_tools = lambda: []
        sys.modules["agent.memory_tools"] = mem
        names = [getattr(t, "__name__", "") for t in tools.get_all_tools()]
        self.assertIn("search_document_chunks", names)
        self.assertIn("semantic_search_documents", names)


if __name__ == "__main__":
    unittest.main()
