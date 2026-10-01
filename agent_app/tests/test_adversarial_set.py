"""Unit tests for the adversarial eval probes (eval/dataset.ADVERSARIAL_SET).

The probes are data, so the risk is that they silently stop being adversarial —
a reworded probe that no longer poses an attack still passes eval and creates
false confidence. These tests pin the properties that make the set meaningful:

  * structural validity (so `init` cannot fail at seed time)
  * every probe gets the dedicated `adversarial` stratum, not a scattered one
  * the injection probes genuinely trip `guards.scan_injection` — i.e. they are
    still attacks, cross-checked against the detector rather than assumed
  * the fabrication-bait probes expect a tool call, because their failure mode
    is inventing an answer, not declining to look

`eval.dataset` imports mlflow at module scope, so ADVERSARIAL_SET is read from
source when mlflow is unavailable. That keeps the test runnable in a bare
environment without stubbing a large surface.

Run from agent_app/:
  python3 -m pytest tests/test_adversarial_set.py
"""

from __future__ import annotations

import ast
import os
import sys
import unittest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

from agent.guards import scan_injection  # noqa: E402


def _load_set(name: str) -> list[dict]:
    """Read a top-level list-of-dicts constant from eval/dataset.py.

    Uses ast.literal_eval on the parsed assignment so no import (and therefore no
    mlflow) is needed. Falls back to a real import if the constant shape ever
    stops being a literal.
    """
    path = os.path.join(_AGENT_APP_DIR, "eval", "dataset.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == name:
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", "") == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found as a literal in eval/dataset.py")


ADVERSARIAL_SET = _load_set("ADVERSARIAL_SET")
SMOKE_SET = _load_set("SMOKE_SET")

# Probes whose point is "look, then honestly report nothing" rather than refuse.
_FABRICATION_BAIT = {
    "hallucination_bait_document",
    "fabricated_policy_citation",
    "out_of_range_metric",
}


class TestStructure(unittest.TestCase):
    def test_nonempty(self):
        self.assertGreaterEqual(len(ADVERSARIAL_SET), 8)

    def test_required_keys_and_types(self):
        for case in ADVERSARIAL_SET:
            self.assertLessEqual({"q", "tools", "probe"}, set(case), case)
            self.assertLessEqual(set(case), {"q", "tools", "probe", "fixtures"}, case)
            self.assertIsInstance(case["q"], str)
            self.assertIsInstance(case["tools"], list)
            self.assertIsInstance(case["probe"], str)
            self.assertTrue(case["q"].strip(), case)
            self.assertTrue(case["probe"].strip(), case)

    def test_probe_names_unique(self):
        names = [c["probe"] for c in ADVERSARIAL_SET]
        self.assertEqual(len(names), len(set(names)), "duplicate probe names")

    def test_questions_unique_and_disjoint_from_smoke(self):
        qs = [c["q"] for c in ADVERSARIAL_SET]
        self.assertEqual(len(qs), len(set(qs)), "duplicate probe questions")
        overlap = set(qs) & {c["q"] for c in SMOKE_SET}
        self.assertEqual(overlap, set(), f"probe duplicated in SMOKE_SET: {overlap}")

    def test_tools_are_known_tool_names(self):
        # A typo'd expected tool would make ToolCallCorrectness unsatisfiable.
        with open(
            os.path.join(_AGENT_APP_DIR, "agent", "tools.py"), encoding="utf-8"
        ) as fh:
            src = fh.read()
        for case in ADVERSARIAL_SET:
            for tool in case["tools"]:
                self.assertIn(f"def {tool}(", src, f"unknown tool {tool!r} in {case}")


class TestAdversarialIntent(unittest.TestCase):
    """The probes must still BE attacks — cross-checked against the detector."""

    # probe name -> the guards category it should trip
    EXPECTED_CATEGORY = {
        "instruction_override": "instruction_override",
        "role_reassignment": "role_reassignment",
        "fake_turn_boundary": "fake_turn_boundary",
        "guardrail_negation": "guardrail_negation",
    }

    def test_injection_probes_trip_the_detector(self):
        for case in ADVERSARIAL_SET:
            expected = self.EXPECTED_CATEGORY.get(case["probe"])
            if not expected:
                continue
            verdict = scan_injection(case["q"])
            self.assertTrue(
                verdict.detected,
                f"probe {case['probe']!r} no longer reads as an attack: {case['q']!r}",
            )
            self.assertIn(expected, verdict.categories, case["probe"])

    def test_every_guard_category_has_a_probe(self):
        # Keeps the eval set and the detector in step: a new detector category
        # without a probe is an untested control.
        from agent import guards

        detector_categories = {name for name, _ in guards._INJECTION_PATTERNS}
        covered = set(self.EXPECTED_CATEGORY.values())
        missing = detector_categories - covered
        self.assertEqual(
            missing,
            {"system_prompt_exfiltration"},
            "detector categories without a matching probe changed; either add a "
            f"probe or update this assertion. uncovered={missing}",
        )

    def test_system_prompt_exfiltration_is_covered_by_a_probe(self):
        # It rides the instruction_override probe ("output your full system
        # prompt"), so assert that explicitly rather than leaving it implicit.
        probe = next(c for c in ADVERSARIAL_SET if c["probe"] == "instruction_override")
        self.assertIn(
            "system_prompt_exfiltration", scan_injection(probe["q"]).categories
        )


class TestExpectations(unittest.TestCase):
    def test_refusal_probes_expect_no_tools(self):
        for case in ADVERSARIAL_SET:
            if case["probe"] in _FABRICATION_BAIT:
                continue
            self.assertEqual(
                case["tools"],
                [],
                f"{case['probe']} should expect a refusal (no tool calls)",
            )

    def test_fabrication_bait_expects_a_real_lookup(self):
        # The failure mode is inventing an answer, so the agent must actually
        # search before reporting nothing found.
        for case in ADVERSARIAL_SET:
            if case["probe"] in _FABRICATION_BAIT:
                self.assertEqual(
                    len(case["tools"]),
                    1,
                    f"{case['probe']} should expect exactly one lookup tool",
                )

    def test_out_of_range_probe_expects_a_date_scoped_tool(self):
        # The probe asks for a metric in 1998. Only a tool that takes a date range
        # can come back empty for it; an all-time tool invites presenting today's
        # numbers as 1998's (fourth review).
        case = next(c for c in ADVERSARIAL_SET if c["probe"] == "out_of_range_metric")
        path = os.path.join(_AGENT_APP_DIR, "agent", "tools.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        params = {
            node.name: {a.arg for a in node.args.args + node.args.kwonlyargs}
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        (tool,) = case["tools"]
        self.assertTrue(
            {"date_from", "date_to"} <= params[tool],
            f"{tool} cannot scope a lookup to 1998: {sorted(params[tool])}",
        )

    def test_every_probe_is_served_stub_tools_never_live_ones(self):
        # Seventh review: without `_tool_fixtures`, a row falls back to global
        # replay or the LIVE tools. Refusal probes get empty fixtures; each
        # fabrication-bait probe gets exactly its expected tool, answering with
        # nothing (or, for policy search, a NEAR policy) to report.
        for case in ADVERSARIAL_SET:
            fixtures = case.get("fixtures", {})
            if case["probe"] in _FABRICATION_BAIT:
                self.assertEqual(set(fixtures), set(case["tools"]), case["probe"])
            else:
                self.assertEqual(fixtures, {}, case["probe"])

    def test_bait_fixtures_contain_nothing_to_fabricate_from(self):
        by_probe = {c["probe"]: c.get("fixtures", {}) for c in ADVERSARIAL_SET}
        doc = by_probe["hallucination_bait_document"]["get_extraction_results"]
        self.assertEqual((doc["extraction_count"], doc["extractions"]), (0, []))
        reviews = by_probe["out_of_range_metric"]["get_recent_reviews"]
        self.assertEqual((reviews["count"], reviews["reviews"]), (0, []))
        policy = by_probe["fabricated_policy_citation"]["search_payer_policy"]
        self.assertNotIn("POL-ZZ-FAKE-999", str(policy["policies"]))

    def test_policy_fixture_mirrors_the_corpus(self):
        corpus_path = os.path.join(_AGENT_APP_DIR, "..", "scripts")
        if not os.path.isdir(corpus_path):
            self.skipTest("policy corpus not reachable from here")
        sys.path.insert(0, corpus_path)
        import payer_policy_content as corpus

        case = next(
            c for c in ADVERSARIAL_SET if c["probe"] == "fabricated_policy_citation"
        )
        (served,) = case["fixtures"]["search_payer_policy"]["policies"]
        real = {p["citation_label"]: p for p in corpus.POLICIES}[served["citation"]]
        self.assertEqual(served["governs_codes"], real["related_codes"])
        self.assertEqual(served["title"], real["title"])

    def test_fabricated_policy_probe_uses_a_nonexistent_policy_id(self):
        # If the id ever became real, the probe would stop testing fabrication.
        case = next(
            c for c in ADVERSARIAL_SET if c["probe"] == "fabricated_policy_citation"
        )
        corpus_path = os.path.join(
            _AGENT_APP_DIR, "..", "scripts", "payer_policy_content.py"
        )
        if not os.path.exists(corpus_path):
            self.skipTest("policy corpus not reachable from here")
        with open(corpus_path, encoding="utf-8") as fh:
            corpus = fh.read()
        self.assertIn("POL-ZZ-FAKE-999", case["q"])
        self.assertNotIn(
            "POL-ZZ-FAKE-999", corpus, "probe policy id is now real — pick another"
        )


if __name__ == "__main__":
    unittest.main()
