"""Unit tests for the warehouse-backed analytics routes.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_analytics_routes

The Statement Execution API returns every value as a STRING — these tests
pin the coercion paths that a naive port would get wrong (bool("false") is
True), the named-parameter binding (user input must never be interpolated),
the offset-past-end total_count fallback, and the response shapes.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

_REVIEWER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REVIEWER_DIR not in sys.path:
    sys.path.insert(0, _REVIEWER_DIR)

from databricks.sdk.service.sql import StatementState  # noqa: E402

from tests._isolation import IsolatedModules  # noqa: E402


def _install_config_stub() -> None:
    """Stub config before routes.analytics imports `settings` (the real module
    reads deployment env at import time)."""
    cfg = types.ModuleType("config")

    class _Settings:
        lakercm_schema = "lakercm"
        catalog = "test_catalog"
        lakercm_schema = "test_schema"
        # Required by services.lakehouse_db at import time (routes.analytics →
        # dependencies → lakehouse_db); analytics itself never reads it.
        auto_verdict_threshold = 0.85
        automated_reviewer_email = "<automated>"

        def get_warehouse_id(self):
            return "wh-test"

    cfg.settings = _Settings()  # type: ignore[attr-defined]
    sys.modules["config"] = cfg


# The stub is live only while this module's tests run; installed at import time
# it leaked into every other module of a single pytest run, and the modules
# that stub config only "if not already there" then ran on this one's settings.
_ISOLATION = IsolatedModules()
analytics = None


def setUpModule():
    global analytics
    _ISOLATION.start(purge_first_party=True)
    _install_config_stub()
    import routes.analytics as stubbed_analytics

    analytics = stubbed_analytics


def tearDownModule():
    _ISOLATION.stop()


def _response(columns, rows, state=StatementState.SUCCEEDED):
    return SimpleNamespace(
        status=SimpleNamespace(state=state, error=None),
        result=(
            SimpleNamespace(data_array=rows, row_count=len(rows))
            if rows is not None
            else None
        ),
        manifest=SimpleNamespace(
            schema=SimpleNamespace(columns=[SimpleNamespace(name=c) for c in columns])
        ),
        statement_id="stmt-1",
    )


def _arrow_response(columns, row_count):
    """A Lakehouse//RT-shaped result: rows exist, but only as an Arrow
    attachment — `data_array` is None. Reading this as "no rows" is what
    rendered a populated gold layer as an all-zeros dashboard."""
    return SimpleNamespace(
        status=SimpleNamespace(state=StatementState.SUCCEEDED, error=None),
        result=SimpleNamespace(data_array=None, row_count=row_count),
        manifest=SimpleNamespace(
            schema=SimpleNamespace(columns=[SimpleNamespace(name=c) for c in columns])
        ),
        statement_id="stmt-arrow",
    )


class _FakeClient:
    """WorkspaceClient stand-in capturing executed statements."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # (statement, parameters)
        self.result_shapes = []  # (format, disposition) per call
        self.statement_execution = SimpleNamespace(
            execute_statement=self._execute,
            get_statement=MagicMock(),
        )

    def _execute(
        self,
        warehouse_id,
        statement,
        parameters=None,
        wait_timeout=None,
        format=None,
        disposition=None,
    ):
        self.calls.append((statement, parameters))
        self.result_shapes.append((format, disposition))
        return self._responses.pop(0)


class AnalyticsRoutesTest(unittest.TestCase):
    def setUp(self):
        analytics._reviewers_cache["ts"] = 0.0
        analytics._reviewers_cache["values"] = []
        self._orig_resolve = analytics.resolve_user_identity
        analytics.resolve_user_identity = lambda raw: {
            "email": {"12345": "amey@x.com"}.get(raw, raw),
            "display_name": None,
        }

    def tearDown(self):
        analytics.resolve_user_identity = self._orig_resolve

    # ── result disposition (Lakehouse//RT regression) ──────────────────────

    def test_every_query_pins_an_inline_json_result(self):
        """The analytics reads run on the Reyden (Lakehouse//RT) warehouse,
        which defaults to an Arrow attachment. The shape must be pinned, not
        inherited from the warehouse."""
        client = _FakeClient([_response(["verdict"], [["correct"]])])
        analytics._execute_warehouse_query(client, "SELECT verdict FROM fact_review")
        self.assertTrue(client.result_shapes, "no statement was executed")
        for fmt, disp in client.result_shapes:
            self.assertEqual(getattr(fmt, "value", fmt), "JSON_ARRAY")
            self.assertEqual(getattr(disp, "value", disp), "INLINE")

    def test_rows_without_an_inline_array_raise_instead_of_reading_as_zero(self):
        """A result that reports rows but carries no data_array means the
        pinned format did not apply. Returning [] there is what turned a
        populated gold layer into an all-zeros dashboard, so it must fail loud."""
        client = _FakeClient([_arrow_response(["n"], row_count=7)])
        with self.assertRaises(Exception) as ctx:
            analytics._execute_warehouse_query(client, "SELECT count(*) AS n")
        self.assertIn("no inline", str(getattr(ctx.exception, "detail", ctx.exception)))

    def test_a_genuinely_empty_result_still_returns_no_rows(self):
        client = _FakeClient([_arrow_response(["n"], row_count=0)])
        self.assertEqual(
            analytics._execute_warehouse_query(client, "SELECT count(*) AS n"), []
        )

    # ── /summary ──────────────────────────────────────────────────────────

    def test_summary_coerces_string_booleans_and_counts(self):
        rows = [
            ["correct", "false", "42", "42"],
            ["partially_correct", "false", "8", "8"],
            ["incorrect", "false", "2", "2"],
            ["correct", "true", "902", "902"],
        ]
        client = _FakeClient(
            [
                _response(
                    ["verdict", "is_automated", "total_reviews", "total_docs"], rows
                )
            ]
        )
        resp = asyncio.run(
            analytics.get_analytics_summary(reviewer=None, workspace_client=client)
        )
        self.assertEqual(resp.total_reviews, 954)
        self.assertEqual(resp.human_reviewed, 52)
        self.assertEqual(resp.auto_reviewed, 902)  # "true" rows — not bool("false")
        self.assertEqual(resp.human_correct_count, 42)
        self.assertEqual(resp.human_accuracy_pct, 80.8)
        self.assertEqual(resp.auto_accuracy_pct, 100.0)
        self.assertIn("MEASURE(review_count)", client.calls[0][0])

    def test_summary_empty_fact_returns_zeros(self):
        client = _FakeClient(
            [
                _response(
                    ["verdict", "is_automated", "total_reviews", "total_docs"], None
                )
            ]
        )
        resp = asyncio.run(
            analytics.get_analytics_summary(reviewer=None, workspace_client=client)
        )
        self.assertEqual(resp.total_reviews, 0)
        self.assertEqual(resp.accuracy_pct, 0.0)

    def test_summary_reviewer_filter_binds_in_list(self):
        reviewers = _response(["reviewer_email"], [["12345"], ["amey@x.com"]])
        summary = _response(
            ["verdict", "is_automated", "total_reviews", "total_docs"],
            [["correct", "false", "2", "2"]],
        )
        client = _FakeClient([reviewers, summary])
        resp = asyncio.run(
            analytics.get_analytics_summary(
                reviewer="amey@x.com", workspace_client=client
            )
        )
        self.assertEqual(resp.human_reviewed, 2)
        stmt, params = client.calls[1]
        # Legacy numeric id AND email both bound as named markers, not inlined.
        self.assertIn("reviewer_email IN (:r0, :r1)", stmt)
        bound = {p.name: p.value for p in params}
        self.assertEqual(set(bound.values()), {"12345", "amey@x.com"})
        self.assertNotIn("amey@x.com", stmt)

    # ── /recent-reviews ───────────────────────────────────────────────────

    def test_recent_reviews_coercion_and_window_count(self):
        cols = [
            "id",
            "document_id",
            "document_name",
            "reviewer_email",
            "verdict",
            "reasoning",
            "is_automated",
            "created_at",
            "total_count",
        ]
        rows = [
            [
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
                "doc.pdf",
                "amey@x.com",
                "correct",
                None,
                "false",
                "2026-06-08T18:16:34.830Z",
                "52",
            ]
        ]
        client = _FakeClient([_response(cols, rows)])
        resp = asyncio.run(
            analytics.get_recent_reviews(
                reviewer=None,
                verdict=None,
                date_from=None,
                date_to=None,
                search=None,
                include_automated=False,
                limit=20,
                offset=0,
                workspace_client=client,
            )
        )
        self.assertEqual(resp.total_count, 52)
        self.assertFalse(resp.reviews[0].is_automated)  # "false" must coerce False
        self.assertEqual(resp.reviews[0].created_at.tzinfo is not None, True)
        stmt, params = client.calls[0]
        self.assertIn("NOT is_automated", stmt)
        self.assertIn("ORDER BY created_at DESC, id DESC", stmt)
        self.assertIsNone(params)

    def test_recent_reviews_binds_user_input(self):
        client = _FakeClient([_response(["id"], None)])
        asyncio.run(
            analytics.get_recent_reviews(
                reviewer=None,
                verdict="correct",
                date_from="2026-01-01",
                date_to="2026-06-01",
                search="O'Brien; DROP TABLE--",
                include_automated=True,
                limit=5,
                offset=0,
                workspace_client=client,
            )
        )
        stmt, params = client.calls[0]
        bound = {p.name: p.value for p in params}
        self.assertEqual(bound["search"], "O'Brien; DROP TABLE--")
        self.assertEqual(bound["verdict"], "correct")
        self.assertIn("DATE_ADD(CAST(:date_to AS DATE), 1)", stmt)
        self.assertIn("ILIKE '%' || :search || '%'", stmt)
        self.assertNotIn("DROP TABLE", stmt)  # never interpolated
        self.assertNotIn("NOT is_automated", stmt)  # include_automated=True

    def test_recent_reviews_offset_past_end_falls_back_to_count(self):
        empty_page = _response(["id"], None)
        count = _response(["cnt"], [["52"]])
        client = _FakeClient([empty_page, count])
        resp = asyncio.run(
            analytics.get_recent_reviews(
                reviewer=None,
                verdict=None,
                date_from=None,
                date_to=None,
                search=None,
                include_automated=False,
                limit=20,
                offset=100,
                workspace_client=client,
            )
        )
        self.assertEqual(resp.total_count, 52)
        self.assertEqual(resp.reviews, [])
        self.assertEqual(len(client.calls), 2)
        self.assertIn("COUNT(*)", client.calls[1][0])

    # ── /processing-metrics ───────────────────────────────────────────────

    def test_processing_metrics_shapes_and_empty_review_block(self):
        cols = [
            "kind",
            "total",
            "avg_seconds",
            "min_seconds",
            "max_seconds",
            "median_seconds",
        ]
        rows = [
            ["pipeline", "473", "395.5", "20.4", "1408.5", "199.6"],
            ["review", "0", None, None, None, None],
        ]
        client = _FakeClient([_response(cols, rows)])
        resp = asyncio.run(analytics.get_processing_metrics(workspace_client=client))
        self.assertEqual(resp.pipeline.total, 473)
        self.assertEqual(resp.pipeline.median_seconds, 199.6)
        self.assertEqual(resp.review.total, 0)
        self.assertIsNone(resp.review.avg_seconds)  # empty ⇒ total=0 + all-None

    # ── /trend ────────────────────────────────────────────────────────────

    def test_trend_buckets_months_and_accuracy(self):
        cols = ["review_month", "verdict", "cnt"]
        rows = [
            ["2026-04", "correct", "34"],
            ["2026-04", "partially_correct", "2"],
            ["2026-04", "incorrect", "1"],
            ["2026-06", "correct", "7"],
        ]
        client = _FakeClient([_response(cols, rows)])
        resp = asyncio.run(
            analytics.get_analytics_trend(reviewer=None, workspace_client=client)
        )
        self.assertEqual(len(resp.trend), 2)
        self.assertEqual(resp.trend[0].review_month, "2026-04")
        self.assertEqual(resp.trend[0].total, 37)
        self.assertEqual(resp.trend[0].accuracy_pct, 91.9)
        self.assertIn("NOT is_automated", client.calls[0][0])

    # ── /reviewers ────────────────────────────────────────────────────────

    def test_reviewers_dedupes_raw_ids_by_resolved_email(self):
        client = _FakeClient(
            [_response(["reviewer_email"], [["12345"], ["amey@x.com"], ["mk@x.com"]])]
        )
        resp = asyncio.run(analytics.get_reviewer_list(workspace_client=client))
        values = [r.value for r in resp.reviewers]
        self.assertEqual(sorted(values), ["amey@x.com", "mk@x.com"])  # 12345 merged

    def test_reviewers_cache_avoids_requery(self):
        client = _FakeClient([_response(["reviewer_email"], [["amey@x.com"]])])
        asyncio.run(analytics.get_reviewer_list(workspace_client=client))
        # Second call must hit the TTL cache (no responses left to pop).
        asyncio.run(analytics.get_reviewer_list(workspace_client=client))
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
