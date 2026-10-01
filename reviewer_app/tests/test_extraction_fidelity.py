"""fetch_extraction_fidelity must degrade to "no badge", never to an error.

Run from reviewer_app/:
    python -m pytest tests/test_extraction_fidelity.py

The fidelity signal is advisory by construction. It is scored against the
synthetic generator's manifest -- the only ground truth that survives
ai_parse_document normalising a glyph -- so it is absent for every real document
and for every target whose analytics pipeline has not yet materialised
gold_extraction_fidelity. "We could not check" and "the extraction was faithful"
must therefore render identically: no badge. If either raised, a lagging
analytics pipeline would put an error state on every document in the app.

The 'true'/'false' string assertion is not hypothetical: the Statement Execution
API returns booleans as strings, so ``bool(row['rewritten_and_auto_verified'])``
is True for the string 'false' -- which would claim a document auto-verified with
an altered code when it had actually been held. load_terminology carries the same
note about is_billable for the same reason.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from services import review_proposals as rp  # noqa: E402


def _rows(*rows):
    """A warehouse_rows stand-in returning the given rows."""
    return mock.patch.object(rp, "warehouse_rows", return_value=list(rows))


class TestFetchExtractionFidelity(unittest.TestCase):
    def _call(self):
        return rp.fetch_extraction_fidelity(
            mock.Mock(), "wh-1", "dbfs:/Volumes/c/s/v/synthetic-1-0001-referral.pdf"
        )

    def test_missing_table_returns_none_without_raising(self) -> None:
        """The dataset is new; every target has a window where it does not exist."""
        with mock.patch.object(
            rp, "warehouse_rows", side_effect=Exception("TABLE_OR_VIEW_NOT_FOUND")
        ):
            self.assertIsNone(self._call())

    def test_no_row_returns_none(self) -> None:
        with _rows():
            self.assertIsNone(self._call())

    def test_faithful_extraction_returns_none(self) -> None:
        """Nothing to tell the reviewer -- the stored codes match the page."""
        with _rows({"fidelity_status": "faithful", "codes_rewritten": []}):
            self.assertIsNone(self._call())

    def test_real_document_returns_none(self) -> None:
        """A real document has no planted ground truth, so there is no finding."""
        with _rows({"fidelity_status": "no_ground_truth", "codes_rewritten": []}):
            self.assertIsNone(self._call())

    def test_rewritten_code_is_reported_with_both_sides(self) -> None:
        """The reviewer needs what was printed AND what got stored."""
        with _rows(
            {
                "fidelity_status": "rewritten",
                "codes_rewritten": [{"source_code": "I1O", "stored_as": "I10"}],
                "rewritten_and_auto_verified": "true",
            }
        ):
            found = self._call()
        self.assertIsNotNone(found)
        self.assertEqual(found["fidelity_status"], "rewritten")
        self.assertEqual(
            found["codes_rewritten"], [{"source_code": "I1O", "stored_as": "I10"}]
        )
        self.assertTrue(found["rewritten_and_auto_verified"])

    def test_false_string_is_not_truthy(self) -> None:
        """Statement Execution returns booleans as strings; bool('false') is True."""
        with _rows(
            {
                "fidelity_status": "rewritten",
                "codes_rewritten": [{"source_code": "I1O", "stored_as": "I10"}],
                "rewritten_and_auto_verified": "false",
            }
        ):
            found = self._call()
        self.assertIsNotNone(found)
        self.assertFalse(
            found["rewritten_and_auto_verified"],
            "the string 'false' was read as True -- the badge would claim an "
            "altered code auto-verified when the document was actually held",
        )

    def test_dropped_status_is_reported(self) -> None:
        """Codes present on the page that never reached the claim also count."""
        with _rows(
            {
                "fidelity_status": "dropped",
                "codes_rewritten": [],
                "rewritten_and_auto_verified": "false",
            }
        ):
            found = self._call()
        self.assertIsNotNone(found)
        self.assertEqual(found["fidelity_status"], "dropped")
        self.assertEqual(found["codes_rewritten"], [])

    def test_malformed_entries_are_skipped(self) -> None:
        """A NULL stored_as or a non-dict entry must not crash the badge."""
        with _rows(
            {
                "fidelity_status": "rewritten",
                "codes_rewritten": [
                    {"source_code": "I1O", "stored_as": "I10"},
                    {"source_code": None, "stored_as": "X"},
                    "not-a-dict",
                ],
                "rewritten_and_auto_verified": "true",
            }
        ):
            found = self._call()
        self.assertEqual(len(found["codes_rewritten"]), 1)


if __name__ == "__main__":
    unittest.main()
