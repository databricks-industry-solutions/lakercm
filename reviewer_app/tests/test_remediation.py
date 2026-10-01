"""Unit tests for services.remediation — the deterministic candidate resolver.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_remediation

The module has no I/O, so nothing is stubbed. The terminology fixture is the
REAL seeded corpus (scripts/seed_reference_data.py), not a hand-written one:
the resolver's whole job is to agree with the terminology the pipeline
validates against, and a fixture that drifted from it would pass while the
resolver was wrong. The two malformed codes and the non-billable parents used
below are the exact values synthetic_data/docgen.py plants.
"""

from __future__ import annotations

import os
import sys
import unittest

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
# scripts/ goes on the path directly (not as a package): seed_reference_data
# imports its _warehouse_sql sibling by bare name. Same as
# scripts/tests/test_seed_reference_data.py.
_SCRIPTS_DIR = os.path.join(os.path.dirname(_APP_DIR), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from services import remediation as rem  # noqa: E402

from seed_reference_data import CPT_HCPCS_ROWS, ICD10_ROWS  # noqa: E402


def _terminology() -> rem.Terminology:
    """The live seeded terminology, in the shapes the two ref tables have."""
    rows = [
        rem.TerminologyRow(
            code=code,
            description=desc,
            code_system=rem.ICD10,
            is_billable=billable,
        )
        for code, desc, _category, billable in ICD10_ROWS
    ]
    # ref_cpt_hcpcs has no is_billable column — every procedure row is billable.
    rows += [
        rem.TerminologyRow(
            code=row[0],
            description=row[1],
            code_system=rem.CPT_HCPCS,
        )
        for row in CPT_HCPCS_ROWS
    ]
    return rem.Terminology(rows)


TERMINOLOGY = _terminology()


class TestTerminologyIndex(unittest.TestCase):
    def test_indexes_both_systems(self):
        self.assertTrue(TERMINOLOGY.rows(rem.ICD10))
        self.assertTrue(TERMINOLOGY.rows(rem.CPT_HCPCS))

    def test_lookup_is_case_and_whitespace_insensitive(self):
        self.assertIsNotNone(TERMINOLOGY.get(" i10 ", rem.ICD10))

    def test_a_dotless_spelling_resolves_to_the_dotted_row(self):
        """OCR drops the dot; the terminology stores the dotted form."""
        dotted = TERMINOLOGY.get("M54.50", rem.ICD10)
        dotless = TERMINOLOGY.get("M5450", rem.ICD10)
        self.assertIsNotNone(dotted)
        self.assertIs(dotless, dotted)

    def test_rows_are_ordered_deterministically(self):
        codes = [r.code for r in TERMINOLOGY.rows(rem.ICD10)]
        self.assertEqual(codes, sorted(codes))


class TestOcrVariants(unittest.TestCase):
    def test_letter_o_read_for_a_zero_is_corrected(self):
        self.assertIn("I10", rem._ocr_variants("I1O"))

    def test_the_leading_letter_is_not_turned_into_a_digit(self):
        """I10 is a real code that STARTS with a letter I.

        A position-blind letter->digit pass rewrites it to 110 and then
        "corrects" a valid code into an invalid one. The corpus plants I1O for
        I10 specifically, so both directions have to stay straight.
        """
        for variant in rem._ocr_variants("I10"):
            self.assertFalse(variant.startswith("1"), variant)

    def test_a_digit_read_for_the_leading_letter_is_offered(self):
        self.assertIn("I10", rem._ocr_variants("110"))

    def test_an_unambiguous_code_yields_no_variants(self):
        self.assertEqual(rem._ocr_variants("99213"), [])

    def test_empty_input_is_safe(self):
        self.assertEqual(rem._ocr_variants(""), [])


class TestInvalidCode(unittest.TestCase):
    """The malformed_code quirk: MALFORMED_CODES in synthetic_data/docgen.py."""

    def test_the_planted_ocr_code_resolves_to_exactly_one_reading(self):
        r = rem.remediate_code("invalid_code", "I1O", rem.ICD10, TERMINOLOGY)
        self.assertEqual(r.resolution, rem.DETERMINISTIC)
        self.assertEqual([c.code for c in r.candidates], ["I10"])
        self.assertEqual(r.candidates[0].method, rem.METHOD_OCR_NORMALIZE)

    def test_the_planted_truncated_code_is_ambiguous_not_guessed(self):
        """9921 prefixes several E&M codes. The resolver must not pick one."""
        r = rem.remediate_code("invalid_code", "9921", rem.CPT_HCPCS, TERMINOLOGY)
        codes = [c.code for c in r.candidates]
        self.assertGreater(len(codes), 1, codes)
        self.assertEqual(r.resolution, rem.NEEDS_JUDGMENT)
        for code in codes:
            self.assertTrue(code.startswith("9921"), code)

    def test_an_ocr_reading_outranks_a_truncation(self):
        """Both methods can fire; the exact-hit-after-normalizing is likelier."""
        r = rem.remediate_code("invalid_code", "I1O", rem.ICD10, TERMINOLOGY)
        self.assertEqual(r.candidates[0].method, rem.METHOD_OCR_NORMALIZE)

    def test_an_unrecoverable_code_says_so(self):
        r = rem.remediate_code("invalid_code", "ZZZZZ", rem.CPT_HCPCS, TERMINOLOGY)
        self.assertEqual(r.resolution, rem.NOT_RESOLVABLE)
        self.assertEqual(r.candidates, ())

    def test_a_modifier_suffix_is_stripped_before_lookup(self):
        """99213-25 is 99213 plus a modifier; silver looks up the base code."""
        r = rem.remediate_code("invalid_code", "9921-25", rem.CPT_HCPCS, TERMINOLOGY)
        self.assertEqual(r.observed_code, "9921")

    def test_candidates_never_include_the_observed_code(self):
        r = rem.remediate_code("invalid_code", "99213", rem.CPT_HCPCS, TERMINOLOGY)
        self.assertNotIn("99213", [c.code for c in r.candidates])

    def test_a_long_shortlist_is_capped_and_flagged(self):
        """A one-character observation prefixes most of a system."""
        r = rem.remediate_code("invalid_code", "9", rem.CPT_HCPCS, TERMINOLOGY)
        self.assertLessEqual(len(r.candidates), rem.MAX_CANDIDATES)
        if len(r.candidates) == rem.MAX_CANDIDATES:
            self.assertTrue(r.candidates_truncated)


class TestNonBillableParent(unittest.TestCase):
    """The non_billable_code quirk: a parent billed for its billable child."""

    def _parents(self):
        return [c for c, _d, _cat, billable in ICD10_ROWS if not billable]

    def test_every_seeded_parent_has_at_least_one_billable_child(self):
        """Otherwise the quirk is unfixable and the demo case is dead.

        This is the invariant the terminology has to hold for the workflow to
        mean anything: if a parent has no billable child, the reviewer is shown
        a problem with no available remedy.
        """
        for parent in self._parents():
            r = rem.remediate_code("non_billable_code", parent, rem.ICD10, TERMINOLOGY)
            self.assertTrue(
                r.candidates,
                f"{parent} is non-billable but has no billable child seeded",
            )

    def test_children_extend_the_parent_and_are_all_billable(self):
        billable = {c for c, _d, _cat, b in ICD10_ROWS if b}
        for parent in self._parents():
            r = rem.remediate_code("non_billable_code", parent, rem.ICD10, TERMINOLOGY)
            for cand in r.candidates:
                self.assertTrue(cand.code.startswith(parent), cand.code)
                self.assertIn(cand.code, billable)
                self.assertEqual(cand.method, rem.METHOD_BILLABLE_CHILD)

    def test_a_parent_with_several_children_needs_judgment(self):
        """M25.56 (knee) splits by laterality — the resolver must not choose.

        Asserted, not conditional: if the terminology ever collapses back to a
        single knee child this whole path stops being exercised, and the case
        that justifies a human (or the agent) reading the document silently
        disappears from the demo.
        """
        r = rem.remediate_code("non_billable_code", "M25.56", rem.ICD10, TERMINOLOGY)
        self.assertGreater(len(r.candidates), 1, [c.code for c in r.candidates])
        self.assertEqual(r.resolution, rem.NEEDS_JUDGMENT)
        self.assertIn("laterality", r.guidance)
        self.assertEqual(sorted(c.code for c in r.candidates), ["M25.561", "M25.562"])

    def test_the_parent_itself_is_never_a_candidate(self):
        for parent in self._parents():
            r = rem.remediate_code("non_billable_code", parent, rem.ICD10, TERMINOLOGY)
            self.assertNotIn(parent, [c.code for c in r.candidates])

    def test_a_billable_code_has_no_children_to_offer(self):
        r = rem.remediate_code("non_billable_code", "M54.50", rem.ICD10, TERMINOLOGY)
        self.assertEqual(r.resolution, rem.NOT_RESOLVABLE)


class TestMissingMemberId(unittest.TestCase):
    def test_it_is_never_resolvable_and_offers_nothing(self):
        r = rem.remediate_missing_member_id()
        self.assertEqual(r.resolution, rem.NOT_RESOLVABLE)
        self.assertEqual(r.candidates, ())

    def test_the_guidance_forbids_proposing_a_value(self):
        """The agent reads this verbatim. It has to say 'do not', not hint."""
        guidance = rem.remediate_missing_member_id().guidance.lower()
        self.assertIn("do not propose", guidance)

    def test_the_guidance_points_at_where_to_actually_get_it(self):
        guidance = rem.remediate_missing_member_id().guidance.lower()
        self.assertTrue(
            "intake" in guidance or "payer" in guidance,
            "the decline has to be actionable, not just a refusal",
        )


class TestRemediateDocument(unittest.TestCase):
    def _codes(self, *entries):
        return list(entries)

    def test_a_clean_document_produces_nothing(self):
        self.assertEqual(rem.remediate_document([], [], TERMINOLOGY), [])

    def test_missing_member_id_needs_no_codes(self):
        out = rem.remediate_document(["missing_member_id"], [], TERMINOLOGY)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].review_reason, "missing_member_id")

    def test_one_remediation_per_code_not_per_field_mention(self):
        """ai_extract repeats a code across every field that carries it.

        silver_validate_codes dedups its COUNTs for exactly this reason; the
        reviewer needs one decision per code too.
        """
        entries = self._codes(
            {
                "field_name": "primary_diagnosis",
                "code": "M54.5",
                "code_system": rem.ICD10,
                "code_valid": True,
                "is_non_billable": True,
            },
            {
                "field_name": "pain_code",
                "code": "M54.5",
                "code_system": rem.ICD10,
                "code_valid": True,
                "is_non_billable": True,
            },
        )
        out = rem.remediate_document(["non_billable_code"], entries, TERMINOLOGY)
        self.assertEqual(len(out), 1)

    def test_an_invalid_code_is_not_also_treated_as_non_billable(self):
        """code_valid=False means it is not in the terminology at all, so
        billability does not apply to it."""
        entries = self._codes(
            {
                "field_name": "dx",
                "code": "I1O",
                "code_system": rem.ICD10,
                "code_valid": False,
                "is_non_billable": False,
            }
        )
        out = rem.remediate_document(
            ["invalid_code", "non_billable_code"], entries, TERMINOLOGY
        )
        self.assertEqual([r.review_reason for r in out], ["invalid_code"])

    def test_only_reasons_the_document_actually_has_are_remediated(self):
        """A non-billable code present on a document NOT flagged for it is not
        turned into a finding — the pipeline's reasons are authoritative."""
        entries = self._codes(
            {
                "field_name": "dx",
                "code": "M54.5",
                "code_system": rem.ICD10,
                "code_valid": True,
                "is_non_billable": True,
            }
        )
        out = rem.remediate_document(["missing_member_id"], entries, TERMINOLOGY)
        self.assertEqual([r.review_reason for r in out], ["missing_member_id"])

    def test_both_code_reasons_on_one_document(self):
        entries = self._codes(
            {
                "field_name": "dx",
                "code": "M54.5",
                "code_system": rem.ICD10,
                "code_valid": True,
                "is_non_billable": True,
            },
            {
                "field_name": "proc",
                "code": "9921",
                "code_system": rem.CPT_HCPCS,
                "code_valid": False,
                "is_non_billable": False,
            },
        )
        out = rem.remediate_document(
            ["invalid_code", "non_billable_code"], entries, TERMINOLOGY
        )
        self.assertEqual(
            sorted(r.review_reason for r in out),
            ["invalid_code", "non_billable_code"],
        )

    def test_string_boolean_flags_from_the_warehouse_are_honoured(self):
        """The warehouse returns struct booleans as \"true\"/\"false\" strings,
        and bool(\"false\") is True. An invalid code arriving with code_valid
        \"false\" must still be remediated, not read as valid."""
        entries = [
            {
                "field_name": "procedure_code",
                "code": "9921",
                "code_system": rem.CPT_HCPCS,
                "code_valid": "false",
                "is_non_billable": "false",
            }
        ]
        out = rem.remediate_document(["invalid_code"], entries, TERMINOLOGY)
        self.assertEqual([r.review_reason for r in out], ["invalid_code"])
        self.assertTrue(out[0].candidates)

    def test_a_string_true_code_valid_is_not_flagged_invalid(self):
        entries = [
            {
                "field_name": "dx",
                "code": "M25.561",
                "code_system": rem.ICD10,
                "code_valid": "true",
                "is_non_billable": "false",
            }
        ]
        self.assertEqual(
            rem.remediate_document(["invalid_code"], entries, TERMINOLOGY), []
        )

    def test_malformed_entries_are_skipped_not_fatal(self):
        entries = [None, "nonsense", {}, {"code": ""}]
        out = rem.remediate_document(["invalid_code"], entries, TERMINOLOGY)
        self.assertEqual(out, [])

    def test_raw_code_is_used_when_the_looked_up_code_is_absent(self):
        entries = self._codes(
            {
                "field_name": "dx",
                "raw_code": "I1O",
                "code_system": rem.ICD10,
                "code_valid": False,
                "is_non_billable": False,
            }
        )
        out = rem.remediate_document(["invalid_code"], entries, TERMINOLOGY)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].observed_code, "I1O")

    def test_the_field_name_is_carried_through_for_the_ui(self):
        entries = self._codes(
            {
                "field_name": "primary_diagnosis",
                "code": "M54.5",
                "code_system": rem.ICD10,
                "code_valid": True,
                "is_non_billable": True,
            }
        )
        out = rem.remediate_document(["non_billable_code"], entries, TERMINOLOGY)
        self.assertEqual(out[0].field_name, "primary_diagnosis")


class TestSerialization(unittest.TestCase):
    def test_as_dict_is_json_safe(self):
        import json

        r = rem.remediate_code("invalid_code", "I1O", rem.ICD10, TERMINOLOGY)
        json.dumps(r.as_dict())

    def test_an_unknown_reason_is_reported_not_invented(self):
        r = rem.remediate_code("some_new_reason", "X", rem.ICD10, TERMINOLOGY)
        self.assertEqual(r.resolution, rem.NOT_RESOLVABLE)
        self.assertIn("some_new_reason", r.guidance)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
