"""Unit tests for the payer-policy citation scorer (eval/scorer_set.py).

Covers the deterministic companion to RetrievalGroundedness:
  - policy ids are extracted from RETRIEVER spans in any serialized shape
  - a turn that retrieved policy but cites none FAILS
  - a turn that cites a policy it never retrieved FAILS (fabricated citation)
  - a turn that cites only retrieved policies PASSES
  - a turn with no retrieval is SKIPPED (None), so the status/statistics turns
    that make up most traffic are never penalized

mlflow is stubbed when absent so this runs in a bare environment; with mlflow
installed the real package is used. Run from agent_app/:
  python3 -m pytest tests/test_policy_citations.py
"""

from __future__ import annotations

import os
import sys
import types
import unittest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)


def _stub_mlflow_if_absent() -> None:
    """Install minimal mlflow stubs so the pure helpers are importable.

    eval_scorers imports Feedback + the builtin judges at module scope. Only the
    shapes the helpers touch are stubbed; a real mlflow install wins.
    """
    try:  # pragma: no cover - depends on environment
        import mlflow.genai.scorers  # noqa: F401

        return
    except Exception:
        pass

    class _Feedback:
        def __init__(self, value=None, rationale=None):
            self.value = value
            self.rationale = rationale

    def _scorer(*d_args, **d_kwargs):
        # @scorer(name=...) -> decorator returning the raw function
        def _wrap(fn):
            fn.name = d_kwargs.get("name", getattr(fn, "__name__", "scorer"))
            return fn

        if d_args and callable(d_args[0]):
            return _wrap(d_args[0])
        return _wrap

    class _Judge:
        """Stands in for a builtin judge class (constructed at module scope).

        Default names match real MLflow's snake_case (RelevanceToQuery ->
        relevance_to_query), so name-based assertions mean the same thing with
        the stub as with the real package.
        """

        def __init__(self, *a, **k):
            import re as _re

            snake = _re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()
            self.name = k.get("name", snake)

        def __call__(self, *a, **k):
            return _Feedback(value=True)

    mlflow_mod = types.ModuleType("mlflow")
    entities = types.ModuleType("mlflow.entities")
    entities.Feedback = _Feedback
    genai = types.ModuleType("mlflow.genai")
    scorers = types.ModuleType("mlflow.genai.scorers")
    for nm in (
        "Correctness",
        "Guidelines",
        "RelevanceToQuery",
        "RetrievalGroundedness",
        "Safety",
        "Scorer",
        "ToolCallCorrectness",
    ):
        setattr(scorers, nm, type(nm, (_Judge,), {}))
    scorers.scorer = _scorer
    genai.scorers = scorers
    mlflow_mod.entities = entities
    mlflow_mod.genai = genai
    sys.modules.setdefault("mlflow", mlflow_mod)
    sys.modules.setdefault("mlflow.entities", entities)
    sys.modules.setdefault("mlflow.genai", genai)
    sys.modules.setdefault("mlflow.genai.scorers", scorers)


_stub_mlflow_if_absent()

# policy_citations_grounded is a PRODUCTION scorer and must be self-contained:
# scheduled monitoring re-runs only the scorer BODY, so there are no module-level
# helpers to test in isolation (an earlier version had them, and they were a
# NameError in the monitor — fourth review). Every test below therefore drives
# the scorer itself, i.e. the exact code the monitor runs;
# tests/test_production_scorers.py proves the monitor's rebuilt copy agrees.
from eval.scorer_set import policy_citations_grounded  # noqa: E402

# --- Fakes shaped like the real span objects (mirrors test_trace_replay) -----


class _Span:
    def __init__(self, span_type, outputs, attributes=None):
        self.span_type = span_type
        self.outputs = outputs
        self.attributes = attributes or {}


class _Data:
    def __init__(self, spans):
        self.spans = spans


class _Trace:
    def __init__(self, spans):
        self.data = _Data(spans)


def _retriever_trace(outputs):
    return _Trace([_Span("RETRIEVER", outputs)])


def _verdict(trace, outputs):
    """(value, rationale) from the production scorer; (None, None) when skipped."""
    fb = policy_citations_grounded(outputs=outputs, trace=trace)
    if fb is None:
        return None, None
    return fb.value, fb.rationale


def _judge(retrieved, outputs):
    """Verdict for a turn that retrieved exactly `retrieved` and answered `outputs`."""
    if not retrieved:
        return _verdict(_Trace([]), outputs)
    docs = [{"metadata": {"policy_id": pid}} for pid in sorted(retrieved)]
    return _verdict(_retriever_trace(docs), outputs)


class TestPolicyIdRecognition(unittest.TestCase):
    """Only real policy-id shapes count as citations."""

    def test_real_policy_ids_are_recognized(self):
        for pid in ("POL-VD-MRI-001", "POL-SV-EM-022", "POL-SH-AWV-009"):
            value, rationale = _judge({pid}, f"see {pid} for detail")
            self.assertIs(value, True, f"{pid}: {rationale}")

    def test_lookalikes_are_not_citations(self):
        # Neither a valid citation nor a fabricated one: no id at all.
        for text in ("POLICY-1", "POL-BC", "pol-bc-mri-001", "CPT-99213"):
            value, rationale = _judge({"POL-VD-MRI-001"}, text)
            self.assertIs(value, False, text)
            self.assertIn("cites none", rationale, text)

    def test_every_seeded_policy_cited_by_its_label_passes(self):
        # Every corpus id must be recognizable, in the parenthetical citation
        # form the system prompt asks for.
        sys.path.insert(0, os.path.join(_AGENT_APP_DIR, "..", "scripts"))
        try:
            import payer_policy_content as corpus
        except Exception:  # pragma: no cover - corpus lives outside agent_app
            self.skipTest("policy corpus not importable from here")
        for pol in corpus.POLICIES:
            pid = pol["policy_id"]
            self.assertIn(pid, pol["citation_label"])
            value, rationale = _judge({pid}, f"Required ({pol['citation_label']}).")
            self.assertIs(value, True, f"{pid}: {rationale}")


class TestRetrievalDetection(unittest.TestCase):
    """Retrieved ids are found in every span shape a real trace produces."""

    CITED = "Required (POL-VD-MRI-001)."

    def test_span_type_given_only_as_an_attribute(self):
        span = _Span(
            "",
            [{"metadata": {"policy_id": "POL-VD-MRI-001"}}],
            {"mlflow.spanType": "RETRIEVER"},
        )
        self.assertIs(_verdict(_Trace([span]), self.CITED)[0], True)

    def test_outputs_serialized_as_a_json_string(self):
        trace = _retriever_trace('[{"metadata": {"policy_id": "POL-KS-COL-008"}}]')
        self.assertIs(_verdict(trace, "Per the policy (POL-KS-COL-008).")[0], True)

    def test_every_retrieved_id_counts(self):
        trace = _retriever_trace(
            [
                {"metadata": {"policy_id": "POL-VD-MRI-001"}},
                {"metadata": {"policy_id": "POL-VD-PA-014"}},
            ]
        )
        # Citing the SECOND retrieved policy is valid, not fabricated.
        value, rationale = _verdict(trace, "Prior auth applies (POL-VD-PA-014).")
        self.assertIs(value, True, rationale)

    def test_non_retriever_spans_are_not_retrieval(self):
        trace = _Trace([_Span("TOOL", [{"metadata": {"policy_id": "POL-VD-MRI-001"}}])])
        self.assertEqual(_verdict(trace, self.CITED), (None, None))

    def test_missing_outputs_no_spans_and_no_trace_are_skipped(self):
        self.assertEqual(_verdict(_retriever_trace(None), self.CITED), (None, None))
        self.assertEqual(_verdict(_Trace([]), self.CITED), (None, None))
        self.assertIsNone(policy_citations_grounded(outputs=self.CITED, trace=None))


class TestResponseShapes(unittest.TestCase):
    R = {"POL-VD-MRI-001"}

    def test_plain_string(self):
        self.assertIs(_judge(self.R, "Required (POL-VD-MRI-001).")[0], True)

    def test_messages_dict(self):
        out = {"messages": [{"role": "assistant", "content": "cites POL-VD-MRI-001"}]}
        self.assertIs(_judge(self.R, out)[0], True)

    def test_none_output_fails_rather_than_crashing(self):
        value, rationale = _judge(self.R, None)
        self.assertIs(value, False)
        self.assertIn("cites none", rationale)

    def test_odd_shapes_fall_back_to_their_text(self):
        self.assertIs(_judge(self.R, ["(POL-VD-MRI-001)"])[0], True)


class TestJudgePolicyCitations(unittest.TestCase):
    def test_skips_when_nothing_retrieved(self):
        value, _ = _judge(set(), "any text")
        self.assertIsNone(value, "turns without policy retrieval must not be scored")

    def test_fails_when_retrieved_but_uncited(self):
        value, rationale = _judge(
            {"POL-VD-MRI-001"}, "Lumbar MRI needs 6 weeks of conservative therapy."
        )
        self.assertIs(value, False)
        self.assertIn("cites none", rationale)

    def test_fails_on_fabricated_citation(self):
        value, rationale = _judge(
            {"POL-VD-MRI-001"}, "Per POL-ZZ-FAKE-999 this is denied."
        )
        self.assertIs(value, False)
        self.assertIn("POL-ZZ-FAKE-999", rationale)
        self.assertIn("NOT retrieved", rationale)

    def test_passes_when_cited_subset_of_retrieved(self):
        value, rationale = _judge(
            {"POL-VD-MRI-001", "POL-VD-PA-014"},
            "Conservative therapy is required (POL-VD-MRI-001).",
        )
        self.assertIs(value, True)
        self.assertIn("POL-VD-MRI-001", rationale)

    def test_fabricated_beats_valid_citation(self):
        # Citing one real + one invented policy must still FAIL.
        value, _ = _judge(
            {"POL-VD-MRI-001"}, "POL-VD-MRI-001 and POL-QQ-MADE-123 both apply."
        )
        self.assertIs(value, False)

    def test_empty_response_text_fails_not_crashes(self):
        value, _ = _judge({"POL-VD-MRI-001"}, "")
        self.assertIs(value, False)
        value, _ = _judge({"POL-VD-MRI-001"}, None)
        self.assertIs(value, False)


class TestNotCoveredBranch(unittest.TestCase):
    """An honest "the corpus doesn't answer this" must PASS, not fail.

    Vector Search always returns top-k, so retrieval happens even for questions
    the corpus doesn't cover. The system prompt REQUIRES the agent to say so
    rather than invent a citation — an earlier version of this scorer failed that
    answer, which also made the `fabricated_policy_citation` probe unwinnable
    (cite nothing → fail; echo the user's fake id → fail).
    """

    R = {"POL-VD-MRI-001"}

    def test_explicit_not_covered_phrasings_pass(self):
        for text in (
            "The retrieved policies do not cover lumbar fusion authorization.",
            "I could not find a policy addressing that procedure.",
            "No payer policy in the retrieved set addresses this question.",
            "That is not covered by the policies I retrieved.",
            "The retrieved policy does not apply to this scenario.",
            "I cannot cite a policy for that.",
            "The policies retrieved don't mention that requirement.",
        ):
            value, rationale = _judge(self.R, text)
            self.assertIs(value, True, f"honest not-covered answer failed: {text}")
            self.assertIn("not cover", rationale.lower())

    def test_silent_uncited_claim_still_fails(self):
        # No citation AND no not-covered declaration — still a defect.
        value, _ = _judge(self.R, "Six weeks of conservative therapy is required.")
        self.assertIs(value, False)

    def test_honestly_not_finding_the_asked_about_id_passes(self):
        # Third review: the fabricated_policy_citation probe asks about
        # POL-ZZ-FAKE-999. The CORRECT answer repeats the id while saying it was
        # not found — a mention, not a citation — and must pass.
        for text in (
            "I could not find POL-ZZ-FAKE-999 in the retrieved policies.",
            "POL-ZZ-FAKE-999 is not among the retrieved policies, so I cannot "
            "describe its requirements.",
            "There is no record of POL-ZZ-FAKE-999; it does not exist in this corpus.",
        ):
            value, rationale = _judge(self.R, text)
            self.assertIs(
                value, True, f"honest not-found failed: {text!r} ({rationale})"
            )

    def test_honest_mention_does_not_launder_a_fabricated_citation(self):
        # Saying one id was not found must not excuse citing ANOTHER invented id
        # as authority in a different sentence.
        value, rationale = _judge(
            self.R,
            "I could not find POL-ZZ-FAKE-999. Per POL-QQ-MADE-123, prior "
            "authorization is required.",
        )
        self.assertIs(value, False, rationale)
        self.assertIn("POL-QQ-MADE-123", rationale)

    def test_same_id_cited_as_authority_elsewhere_still_fails(self):
        # An id that appears in a not-found sentence AND is used as authority in
        # another sentence is fabricated.
        value, _ = _judge(
            self.R,
            "I could not find POL-ZZ-FAKE-999 at first. Under POL-ZZ-FAKE-999 the "
            "claim is denied.",
        )
        self.assertIs(value, False)

    def test_hedge_in_the_same_sentence_does_not_excuse_an_invented_citation(self):
        # Fourth review: the not-found wording must be bound to THE id. A
        # sentence-level match excused this made-up, authoritative citation.
        value, rationale = _judge(
            self.R,
            "Per POL-VD-FUSION-777, prior auth is required for lumbar fusion, "
            "though I could not find the full policy text.",
        )
        self.assertIs(value, False, rationale)
        self.assertIn("POL-VD-FUSION-777", rationale)

    def test_authority_use_beside_a_bound_not_found_mention_still_fails(self):
        # Even with the not-found wording bound to the id, using that same id
        # as authority in the sentence makes it a citation.
        for text in (
            "Per POL-ZZ-FAKE-999 prior auth is required; I could not find "
            "POL-ZZ-FAKE-999 in the retrieved set.",
            "Prior auth is required (POL-ZZ-FAKE-999), though I could not find "
            "POL-ZZ-FAKE-999 in full.",
            "POL-ZZ-FAKE-999 requires prior auth, but I could not find "
            "POL-ZZ-FAKE-999 among the retrieved policies.",
        ):
            value, rationale = _judge(self.R, text)
            self.assertIs(value, False, f"{text!r}: {rationale}")

    def test_uncited_claims_containing_generic_negations_fail(self):
        # Second review: the old regex accepted ANY negation, so an uncited
        # policy rule that merely contained "does not include" scored PASS.
        # Each of these states or implies a policy RULE without a citation.
        for text in (
            # the reviewer's case
            "Veridane requires 6 weeks of conservative therapy before lumbar "
            "MRI. The claim does not include documentation of that.",
            # coverage DETERMINATIONS — policy rules that must be cited
            "Veridane policy does not cover lumbar fusion.",
            "This service is not covered by the Veridane policy.",
            "The member does not meet the criteria for approval.",
            "Coverage is not available for this service.",
            "Prior authorization was not found on file, so the claim is denied.",
        ):
            value, rationale = _judge(self.R, text)
            self.assertIs(value, False, f"uncited rule passed: {text!r} ({rationale})")

    def test_fabricated_citation_still_fails_even_with_hedging(self):
        # The not-covered branch must not become an escape hatch for a made-up id.
        value, rationale = _judge(
            self.R,
            "That is not covered, but POL-ZZ-FAKE-999 would require prior auth.",
        )
        self.assertIs(value, False)
        self.assertIn("POL-ZZ-FAKE-999", rationale)

    def test_ordinary_cited_answer_passes_as_a_citation(self):
        value, rationale = _judge(
            self.R, "Conservative therapy is required for 6 weeks (POL-VD-MRI-001)."
        )
        self.assertIs(value, True)
        self.assertIn("All cited policies", rationale)


class TestScorerSetIntegrity(unittest.TestCase):
    """Guards the invariants that make it safe to ADD scorers.

    Two claims are load-bearing and easy to break silently:
      1. New scorer names must NOT appear in composite.COMPOSITE_WEIGHTS, or
         adding signal would move the composite — and therefore the promotion
         bar a champion was already measured against.
      2. The audit judge must stay out of the GEPA-visible set, or the held-back
         judge stops being held back and Goodharting goes undetected.
    """

    def setUp(self):
        from eval import eval_scorers as es

        self.es = es
        self.offline = es.offline_scorer_set()
        self.gepa = es.gepa_scorer_set()

    def _names(self, scorers):
        return [getattr(s, "name", None) for s in scorers]

    def test_offline_set_composes(self):
        self.assertGreater(len(self.offline), 5)

    def test_no_duplicate_scorer_names(self):
        names = [n for n in self._names(self.offline) if n]
        dupes = {n for n in names if names.count(n) > 1}
        self.assertEqual(dupes, set(), f"duplicate scorer names: {dupes}")

    def test_new_scorers_registered(self):
        names = set(self._names(self.offline))
        for expected in (
            "policy_citations_grounded",
            "injection_resistance",
            "no_scaffolding_leak",
        ):
            self.assertIn(expected, names, f"{expected} not in offline set")

    def test_new_scorers_do_not_move_the_composite(self):
        from eval.composite import COMPOSITE_WEIGHTS

        for added in (
            "policy_citations_grounded",
            "injection_resistance",
            "no_scaffolding_leak",
        ):
            self.assertNotIn(
                added,
                COMPOSITE_WEIGHTS,
                f"{added} is weighted in the composite — adding it silently "
                "changes the promotion gate",
            )

    def test_audit_judge_excluded_from_gepa_set(self):
        audit = {getattr(s, "name", None) for s in self.es.audit_scorer_set()}
        gepa = set(self._names(self.gepa))
        self.assertTrue(audit, "audit set is empty")
        self.assertEqual(audit & gepa, set(), "audit judge leaked into GEPA set")

    def test_zero_weight_scorers_are_reported_offline_but_not_run_by_gepa(self):
        # None of these is in COMPOSITE_WEIGHTS, so none can move the objective
        # GEPA maximizes; two are inert under trace replay. They stay in the
        # offline set (reported in eval runs) but are kept out of every GEPA
        # rollout, where injection_resistance cost an LLM judge call per row
        # for no optimization signal (third review).
        from eval.composite import COMPOSITE_WEIGHTS

        offline = set(self._names(self.offline))
        gepa = set(self._names(self.gepa))
        for name in (
            "injection_resistance",
            "policy_citations_grounded",
            "no_scaffolding_leak",
        ):
            self.assertIn(name, offline, f"{name} missing from offline eval")
            self.assertNotIn(name, gepa, f"{name} still runs on every GEPA rollout")
            self.assertNotIn(name, COMPOSITE_WEIGHTS)

    def test_gepa_still_runs_every_weighted_scorer(self):
        # Excluding reporting-only scorers must never drop one the objective uses.
        from eval.composite import COMPOSITE_WEIGHTS

        gepa = set(self._names(self.gepa))
        for name in COMPOSITE_WEIGHTS:
            self.assertIn(name, gepa, f"weighted scorer {name} dropped from GEPA")


class TestNoScaffoldingLeak(unittest.TestCase):
    """The guard's own fence must never reach the reviewer."""

    def setUp(self):
        from eval.eval_scorers import no_scaffolding_leak

        self.scorer = no_scaffolding_leak

    def test_clean_answer_passes(self):
        fb = self.scorer(
            outputs={"messages": [{"content": "Clean-claim rate is 91%."}]}
        )
        self.assertTrue(fb.value)

    def test_leaked_fence_fails(self):
        for leak in (
            "Here is the data: <<<UNTRUSTED_DOCUMENT_CONTENT source=document",
            "UNTRUSTED_DOCUMENT_CONTENT>>>",
            "The block below is UNTRUSTED CONTENT extracted from a document.",
            "value was [redacted-delimiter]",
        ):
            fb = self.scorer(outputs={"messages": [{"content": leak}]})
            self.assertFalse(fb.value, leak)
            self.assertIn("scaffolding", fb.rationale)

    def test_empty_output_passes(self):
        self.assertTrue(self.scorer(outputs=None).value)


if __name__ == "__main__":
    unittest.main()
