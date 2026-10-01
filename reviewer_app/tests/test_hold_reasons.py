"""Unit tests for services.hold_reasons — the held-document verdict.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_hold_reasons

No I/O, so nothing is stubbed. The tests that matter most are the last two: a
regression pinning the exact document that prompted this work, and a property
over a generated matrix asserting that the sentence "below the ... threshold"
cannot be produced unless the comparison actually holds. Example-based tests
would let a future refactor reintroduce the contradiction on an input nobody
thought to write down.
"""

from __future__ import annotations

import os
import sys
import unittest

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import hold_reasons as hr  # noqa: E402
from services.remediation import (  # noqa: E402
    NEEDS_JUDGMENT,
    NOT_RESOLVABLE,
    REASON_INVALID_CODE,
    REASON_MISSING_MEMBER_ID,
    REASON_NON_BILLABLE,
    Candidate,
    Remediation,
)

THRESHOLD = 0.92


def _rem(reason, **kw):
    defaults = dict(
        review_reason=reason,
        resolution=NEEDS_JUDGMENT,
        guidance="guidance text from remediation",
    )
    defaults.update(kw)
    return Remediation(**defaults)


class CoerceBoolTests(unittest.TestCase):
    """The Statement Execution API hands back stringified booleans, and the old
    frontend `=== true` collapsed NULL and "true" into the same answer."""

    def test_real_booleans_pass_through(self):
        self.assertIs(hr.coerce_bool(True), True)
        self.assertIs(hr.coerce_bool(False), False)

    def test_stringified_booleans_are_read_correctly(self):
        self.assertIs(hr.coerce_bool("true"), True)
        self.assertIs(hr.coerce_bool("True"), True)
        self.assertIs(hr.coerce_bool(" TRUE "), True)
        # bool("false") is True in Python; this is the bug being prevented.
        self.assertIs(hr.coerce_bool("false"), False)
        self.assertIs(hr.coerce_bool("False"), False)

    def test_unknown_stays_unknown(self):
        for value in (None, "", "   ", "maybe", object()):
            self.assertIsNone(hr.coerce_bool(value), value)


class FormatPctTests(unittest.TestCase):
    """Floors, so a score can never be reported as having cleared a bar it
    missed. 0.9996 is the exact value from the document that prompted this."""

    def test_it_never_rounds_up_to_a_clean_hundred(self):
        self.assertEqual(hr.format_pct(0.9996), "99.96%")
        self.assertEqual(hr.format_pct(0.99999), "99.99%")

    def test_exact_values_stay_clean(self):
        self.assertEqual(hr.format_pct(1.0), "100%")
        self.assertEqual(hr.format_pct(0.92), "92%")
        self.assertEqual(hr.format_pct(0.84), "84%")

    def test_just_below_the_threshold_is_visible(self):
        self.assertEqual(hr.format_pct(0.9199), "91.99%")

    def test_missing_score(self):
        self.assertIsNone(hr.format_pct(None))
        self.assertIsNone(hr.format_pct("nonsense"))


class StateTests(unittest.TestCase):
    def test_auto_verified(self):
        a = hr.assess_document(
            is_automated=True, confidence_score=0.9996, threshold=THRESHOLD
        )
        self.assertEqual(a.state, hr.STATE_AUTO_VERIFIED)
        self.assertEqual(a.blocking, ())
        self.assertIsNone(a.primary)
        self.assertFalse(a.unexplained)

    def test_stringified_true_is_auto_verified(self):
        a = hr.assess_document(
            is_automated="true", confidence_score=0.9996, threshold=THRESHOLD
        )
        self.assertEqual(a.state, hr.STATE_AUTO_VERIFIED)

    def test_stringified_false_is_held(self):
        a = hr.assess_document(
            is_automated="false", confidence_score=0.5, threshold=THRESHOLD
        )
        self.assertEqual(a.state, hr.STATE_HELD)

    def test_unknown_routing_renders_no_verdict(self):
        """A NULL is_automated used to read as "held" and get a fabricated
        reason. It must now produce no verdict at all."""
        a = hr.assess_document(
            is_automated=None, confidence_score=0.9996, threshold=THRESHOLD
        )
        self.assertEqual(a.state, hr.STATE_UNKNOWN)
        self.assertEqual(a.blocking, ())
        self.assertIsNone(a.primary)


class RecordedReasonTests(unittest.TestCase):
    def test_guidance_comes_from_remediation_verbatim(self):
        """Not re-worded here: the reviewer's sentence and the agent's proposal
        must come from one string or they will drift."""
        rem = _rem(
            REASON_INVALID_CODE,
            observed_code="M5450",
            code_system="ICD-10-CM",
            field_name="diagnosis_codes",
            candidates=(
                Candidate(
                    code="M54.50",
                    description="Low back pain",
                    method="truncation_prefix",
                ),
            ),
        )
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.99,
            threshold=THRESHOLD,
            review_reasons=[REASON_INVALID_CODE],
            remediations=[rem],
        )
        (item,) = a.blocking
        self.assertEqual(item.code, REASON_INVALID_CODE)
        self.assertEqual(item.guidance, "guidance text from remediation")
        self.assertEqual(item.observed_code, "M5450")
        self.assertEqual(item.field_name, "diagnosis_codes")
        self.assertIn("M5450", item.detail)
        self.assertEqual(len(item.candidates), 1)

    def test_one_item_per_flagged_code_not_per_reason(self):
        rems = [
            _rem(REASON_INVALID_CODE, observed_code="M5450"),
            _rem(REASON_INVALID_CODE, observed_code="I1O"),
        ]
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.99,
            threshold=THRESHOLD,
            review_reasons=[REASON_INVALID_CODE],
            remediations=rems,
        )
        self.assertEqual(len(a.blocking), 2)
        self.assertEqual({i.observed_code for i in a.blocking}, {"M5450", "I1O"})

    def test_a_recorded_reason_survives_missing_code_detail(self):
        """validated_codes unavailable must not make the flag disappear."""
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.99,
            threshold=THRESHOLD,
            review_reasons=[REASON_INVALID_CODE],
            remediations=[],
        )
        (item,) = a.blocking
        self.assertEqual(item.code, REASON_INVALID_CODE)
        self.assertTrue(item.detail)

    def test_missing_member_id_is_not_resolvable(self):
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.99,
            threshold=THRESHOLD,
            review_reasons=[REASON_MISSING_MEMBER_ID],
        )
        (item,) = a.blocking
        self.assertEqual(item.resolution, NOT_RESOLVABLE)

    def test_recorded_low_confidence_states_both_numbers(self):
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.841,
            threshold=THRESHOLD,
            review_reasons=[hr.REASON_LOW_CONFIDENCE],
        )
        (item,) = a.blocking
        self.assertEqual(item.code, hr.REASON_LOW_CONFIDENCE)
        self.assertIn("84.1%", item.detail)
        self.assertIn("92%", item.detail)
        self.assertIn("below", item.detail)

    def test_an_unrecognised_reason_is_echoed_never_dropped(self):
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.99,
            threshold=THRESHOLD,
            review_reasons=["duplicate_claim_suspected"],
        )
        (item,) = a.blocking
        self.assertEqual(item.code, hr.REASON_UNRECOGNIZED)
        self.assertIn("duplicate_claim_suspected", item.detail)
        self.assertEqual(item.observed_code, "duplicate_claim_suspected")

    def test_precedence_picks_the_hardest_blocker(self):
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.5,
            threshold=THRESHOLD,
            review_reasons=[
                hr.REASON_LOW_CONFIDENCE,
                REASON_NON_BILLABLE,
                REASON_MISSING_MEMBER_ID,
            ],
        )
        self.assertEqual(len(a.blocking), 3)
        self.assertEqual(a.primary.code, REASON_MISSING_MEMBER_ID)


class DerivedAndRefusedTests(unittest.TestCase):
    def test_held_with_no_reason_and_low_score_derives_low_confidence(self):
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.84,
            threshold=THRESHOLD,
            review_reasons=[],
        )
        (item,) = a.blocking
        self.assertEqual(item.code, hr.REASON_LOW_CONFIDENCE)
        self.assertEqual(item.source, hr.SOURCE_DERIVED_CONFIDENCE)
        self.assertFalse(a.unexplained)

    def test_held_with_no_reason_and_no_score(self):
        a = hr.assess_document(
            is_automated=False,
            confidence_score=None,
            threshold=THRESHOLD,
            review_reasons=[],
        )
        (item,) = a.blocking
        self.assertEqual(item.code, hr.REASON_CONFIDENCE_UNAVAILABLE)
        self.assertNotIn("below", item.detail)

    def test_the_document_that_prompted_this_refuses_to_guess(self):
        """THE regression test.

        Held, no recorded reason, confidence 0.9996 against a 0.92 threshold.
        The old banner rendered "Extraction confidence was 100%, below the 92%
        auto-verify threshold". Both halves were wrong: 100% was a rounding
        artifact and nothing established that the score was low.
        """
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.9996,
            threshold=THRESHOLD,
            review_reasons=[],
        )
        self.assertTrue(a.unexplained)
        (item,) = a.blocking
        self.assertEqual(item.code, hr.REASON_NOT_RECORDED)
        self.assertEqual(item.source, hr.SOURCE_UNEXPLAINED)
        self.assertEqual(item.resolution, NOT_RESOLVABLE)
        self.assertNotIn("below", item.detail)
        self.assertIn("99.96%", item.detail)
        self.assertNotIn("100%", item.detail)

    def test_a_recorded_low_confidence_that_disagrees_reports_the_disagreement(
        self,
    ):
        """Inconsistent data must not be laundered into a confident sentence."""
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.98,
            threshold=THRESHOLD,
            review_reasons=[hr.REASON_LOW_CONFIDENCE],
        )
        (item,) = a.blocking
        self.assertNotIn("below", item.detail)
        self.assertIn("disagree", item.detail)


class AdvisoryTests(unittest.TestCase):
    UNCAPTURED = {"uncaptured_codes": ["E11.9", "I10"]}
    FIDELITY = {"codes_rewritten": [{"source_code": "I1O", "stored_as": "I10"}]}

    def test_advisories_never_gate_auto_verification(self):
        a = hr.assess_document(
            is_automated=True,
            confidence_score=0.9996,
            threshold=THRESHOLD,
            uncaptured=self.UNCAPTURED,
            fidelity=self.FIDELITY,
        )
        self.assertEqual(a.state, hr.STATE_AUTO_VERIFIED)
        self.assertEqual(a.blocking, ())
        self.assertEqual(len(a.advisory), 2)

    def test_advisories_never_appear_as_blocking_or_as_the_chip(self):
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.5,
            threshold=THRESHOLD,
            review_reasons=[hr.REASON_LOW_CONFIDENCE],
            uncaptured=self.UNCAPTURED,
            fidelity=self.FIDELITY,
        )
        codes = {i.code for i in a.blocking}
        self.assertFalse(codes & hr.ADVISORY_FINDINGS)
        self.assertNotIn(a.primary.code, hr.ADVISORY_FINDINGS)
        for item in a.advisory:
            self.assertEqual(item.severity, hr.SEVERITY_ADVISORY)

    def test_partial_rewrite_pairs_are_ignored(self):
        """Mirrors the FidelityNotice guard: a pair missing either side is not
        renderable, and half a comparison is worse than none."""
        a = hr.assess_document(
            is_automated=True,
            confidence_score=0.99,
            threshold=THRESHOLD,
            fidelity={"codes_rewritten": [{"source_code": "I1O"}, None]},
        )
        self.assertEqual(a.advisory, ())

    def test_advisory_degradation_is_reported_not_hidden(self):
        a = hr.assess_document(
            is_automated=True,
            confidence_score=0.99,
            threshold=THRESHOLD,
            degraded=["advisory_unavailable"],
        )
        self.assertEqual(a.degraded, ("advisory_unavailable",))


class InvariantTests(unittest.TestCase):
    def test_a_held_document_always_carries_at_least_one_reason(self):
        for confidence in (None, 0.0, 0.5, 0.9199, 0.92, 0.9996, 1.0):
            for reasons in ([], [REASON_INVALID_CODE], ["something_new"]):
                a = hr.assess_document(
                    is_automated=False,
                    confidence_score=confidence,
                    threshold=THRESHOLD,
                    review_reasons=reasons,
                )
                with self.subTest(confidence=confidence, reasons=reasons):
                    self.assertTrue(a.blocking)
                    self.assertIsNotNone(a.primary)

    def test_the_contradiction_is_unrepresentable(self):
        """The property that replaces the bug.

        Nothing the module can emit may claim a score is below the threshold
        unless it actually is. Generated over the full cross-product rather
        than asserted on examples, because the original bug lived on an input
        nobody had written a test for.
        """
        reason_sets = (
            [],
            [hr.REASON_LOW_CONFIDENCE],
            [hr.REASON_CONFIDENCE_UNAVAILABLE],
            [REASON_INVALID_CODE],
            [REASON_INVALID_CODE, hr.REASON_LOW_CONFIDENCE],
            ["future_reason"],
        )
        confidences = (None, 0.0, 0.5, 0.84, 0.9199, 0.92, 0.9996, 1.0)
        for automated in (False, "false"):
            for confidence in confidences:
                for reasons in reason_sets:
                    a = hr.assess_document(
                        is_automated=automated,
                        confidence_score=confidence,
                        threshold=THRESHOLD,
                        review_reasons=reasons,
                    )
                    for item in a.blocking + a.advisory:
                        if "below" not in item.detail:
                            continue
                        with self.subTest(confidence=confidence, reasons=reasons):
                            self.assertIsNotNone(
                                confidence,
                                "claimed 'below' with no score at all",
                            )
                            self.assertLess(
                                confidence,
                                THRESHOLD,
                                f"claimed 'below' for {confidence}",
                            )

    def test_no_emitted_text_overstates_a_score_as_one_hundred(self):
        a = hr.assess_document(
            is_automated=False,
            confidence_score=0.9996,
            threshold=THRESHOLD,
            review_reasons=[],
        )
        for item in a.blocking:
            self.assertNotIn("100%", item.detail)


class PrimaryForListTests(unittest.TestCase):
    """The queue chip. Same derivation as the detail page so a card and the
    document it opens cannot disagree."""

    def test_pending_with_a_recorded_reason(self):
        item = hr.primary_for_list(
            effective_status="pending",
            is_automated=False,
            confidence_score=0.99,
            review_reasons=[REASON_NON_BILLABLE],
            threshold=THRESHOLD,
        )
        self.assertEqual(item.code, REASON_NON_BILLABLE)
        self.assertEqual(item.title, "Non-billable code")

    def test_pending_held_on_score_alone(self):
        item = hr.primary_for_list(
            effective_status="pending",
            is_automated=False,
            confidence_score=0.5,
            review_reasons=[],
            threshold=THRESHOLD,
        )
        self.assertEqual(item.code, hr.REASON_LOW_CONFIDENCE)

    def test_pending_but_unexplained(self):
        item = hr.primary_for_list(
            effective_status="pending",
            is_automated=False,
            confidence_score=0.9996,
            review_reasons=[],
            threshold=THRESHOLD,
        )
        self.assertEqual(item.code, hr.REASON_NOT_RECORDED)

    def test_non_pending_statuses_get_no_chip(self):
        for status in ("auto_verified", "reviewed", "processing", "failed", None):
            self.assertIsNone(
                hr.primary_for_list(
                    effective_status=status,
                    is_automated=True,
                    confidence_score=0.99,
                    review_reasons=[],
                    threshold=THRESHOLD,
                ),
                status,
            )

    def test_it_agrees_with_the_detail_assessment(self):
        kwargs = dict(
            is_automated=False,
            confidence_score=0.5,
            review_reasons=[REASON_INVALID_CODE, hr.REASON_LOW_CONFIDENCE],
            threshold=THRESHOLD,
        )
        chip = hr.primary_for_list(effective_status="pending", **kwargs)
        detail = hr.assess_document(**kwargs)
        self.assertEqual(chip.code, detail.primary.code)


if __name__ == "__main__":
    unittest.main()
