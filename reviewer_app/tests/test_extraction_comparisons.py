"""get_extraction_comparisons must degrade while the synced table catches up.

Run from reviewer_app/:
    python -m pytest tests/test_extraction_comparisons.py

ai_classify v2.1's confidence and rationale are added to gold_extraction_labels
by the pipeline, but they reach the Lakebase copy the reviewer app reads only
after the pipeline's next update and the sync after it. In that window the
columns do not exist in Postgres. Selecting them unconditionally would not
degrade the classification display -- it would 500 the whole
/api/documents/{id}/comparisons endpoint, taking the entire extraction panel
down with it, on every document, for a column nobody has seen yet.

So the query asks first (gold_sync_has_column) and substitutes typed NULLs when
the answer is no. These tests pin both halves of that, plus the two traps around
it: the percent-sign rule in the method's own docstring, and the fact that the
response model drops keys it has not declared.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


class _FakeDb:
    """Just enough of the db class to capture the SQL the method builds."""

    GOLD_SYNC = "lakercm_dev.gold_extraction_labels_sync"

    def __init__(self, has_column: bool):
        self._has_column = has_column
        self.executed: list[str] = []

    def gold_sync_has_column(self, column: str) -> bool:
        return self._has_column

    def _execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        self.executed.append(query)
        return []

    @property
    def sql(self) -> str:
        # The comparisons query is the one carrying the gold table name; the
        # column probe (when it runs for real) reads information_schema.
        return next(q for q in self.executed if "gold_extraction_labels_sync" in q)


def _run(has_column: bool) -> _FakeDb:
    from services.lakehouse_db import LakeRCMDatabase

    db = _FakeDb(has_column)
    # Unbound call against the fake: exercises the real method body without
    # needing a connection pool.
    LakeRCMDatabase.get_extraction_comparisons(db, "/Volumes/x/doc.pdf")
    return db


class TestClassifyColumnsDegradeGracefully(unittest.TestCase):
    def test_the_query_selects_the_classify_columns_once_they_exist(self):
        db = _run(has_column=True)
        self.assertIn("classify_confidence, classify_rationale", db.sql)
        self.assertNotIn("NULL::DOUBLE PRECISION", db.sql)

    def test_the_query_stands_them_in_as_null_until_the_sync_catches_up(self):
        # The important half: this must not raise, and must not ask Postgres for
        # a column that is not there yet.
        db = _run(has_column=False)
        self.assertIn("NULL::DOUBLE PRECISION AS classify_confidence", db.sql)
        self.assertIn("NULL::TEXT AS classify_rationale", db.sql)

    def test_the_sql_carries_no_percent_sign_beyond_its_placeholder(self):
        # The method docstring is explicit: psycopg reads '%' as the start of a
        # parameter placeholder BEFORE it parses SQL comments, so one stray
        # percent sign anywhere -- including in a comment -- fails the call with
        # "incomplete placeholder" and answers 500. Both branches add new lines
        # into exactly that blast radius.
        for has_column in (True, False):
            sql = _run(has_column).sql
            self.assertNotIn(
                "%",
                sql.replace("%s", ""),
                f"percent sign leaked (has_column={has_column})",
            )


class TestANullPageImageDoesNotKillTheReviewForm(unittest.TestCase):
    """One unrendered page must not take down the whole document.

    ai_parse_document emits one page_images entry per page whether or not it
    produced an image, so a NULL element means "page N did not render" and the
    POSITION is load-bearing: routes/documents.py serves page_images[page] and
    falls back to the raw upload when that entry is falsy.

    Typed List[str], pydantic rejected the null and failed validation for the
    entire comparison payload -- so /comparisons 500'd and the review form
    disappeared completely, not just one image. Observed live on dev: 2 of 3,419
    documents, which read as a broken app rather than an unrendered page.
    """

    def test_a_null_entry_validates_and_keeps_its_position(self):
        from schemas import ExtractionComparisonItem

        item = ExtractionComparisonItem(
            document_path="/Volumes/x/doc.pdf",
            document_name="doc.pdf",
            page_images=["/Volumes/x/page_images/abc.jpg", None],
        )
        # Two pages, second unrendered -- and still index 1, not dropped.
        self.assertEqual(len(item.page_images), 2)
        self.assertIsNone(item.page_images[1])
        self.assertTrue(item.page_images[0].endswith("abc.jpg"))

    def test_a_whole_batch_survives_one_bad_row(self):
        # The real failure mode: routes/documents.py builds
        # [ExtractionComparisonItem(**row) for row in rows], so one invalid row
        # raised and took every other row with it.
        from schemas import ExtractionComparisonItem

        rows = [
            {
                "document_path": "/Volumes/x/a.pdf",
                "document_name": "a.pdf",
                "page_images": ["/p/1.jpg"],
            },
            {
                "document_path": "/Volumes/x/b.pdf",
                "document_name": "b.pdf",
                "page_images": ["/p/1.jpg", None],
            },
        ]
        items = [ExtractionComparisonItem(**r) for r in rows]
        self.assertEqual(len(items), 2)

    def test_an_all_null_array_is_still_accepted(self):
        # A document where nothing rendered is a legitimate state, not an error.
        from schemas import ExtractionComparisonItem

        item = ExtractionComparisonItem(
            document_path="/Volumes/x/doc.pdf",
            document_name="doc.pdf",
            page_images=[None, None],
        )
        self.assertEqual(item.page_images, [None, None])


class TestSchemaDeclaresTheNewFields(unittest.TestCase):
    def test_a_row_without_the_classify_keys_still_validates(self):
        from schemas import ExtractionComparisonItem

        item = ExtractionComparisonItem(
            document_path="/Volumes/x/doc.pdf", document_name="doc.pdf"
        )
        self.assertIsNone(item.classify_confidence)
        self.assertIsNone(item.classify_rationale)

    def test_the_fields_survive_model_construction(self):
        # The model IGNORES undeclared keys, so selecting a column in SQL is not
        # enough to get it to the UI -- it has to be declared here too. This is
        # the assertion that would have caught that class of bug the first time.
        from schemas import ExtractionComparisonItem

        item = ExtractionComparisonItem(
            document_path="/Volumes/x/doc.pdf",
            document_name="doc.pdf",
            classify_confidence=0.68,
            classify_rationale="States the claim was denied for want of prior auth.",
        )
        self.assertAlmostEqual(item.classify_confidence, 0.68)
        self.assertIn("prior auth", item.classify_rationale)


if __name__ == "__main__":
    unittest.main()
