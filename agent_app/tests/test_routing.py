"""Unit tests for complexity-tiered model routing (agent/routing.py).

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_routing

Heavy deps (config, agent.llm, services.store) are stubbed in sys.modules before
import — same approach as tests.test_semantic_search / tests.test_delete_thread —
so this runs offline with no app deps and no network. The classifier LLM and the
Store are fakes whose behavior each test controls via module-level holders.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import types
import unittest

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

# --- controllable holders ----------------------------------------------------
_CLS: dict = {"reply": "high", "err": False, "calls": 0, "last_kwargs": None}
_STORE_HOLDER: dict = {"store": None}


class _FakeLLM:
    def __init__(self, **kwargs):
        _CLS["last_kwargs"] = kwargs

    async def ainvoke(self, messages):
        _CLS["calls"] += 1
        if _CLS["err"]:
            raise RuntimeError("classifier endpoint unreachable")
        return types.SimpleNamespace(content=_CLS["reply"])


class _Item:
    def __init__(self, value):
        self.value = value


class _FakeStore:
    def __init__(self, raise_on=False):
        self.data: dict = {}
        self.raise_on = raise_on

    def get(self, namespace, key):
        if self.raise_on:
            raise RuntimeError("store down")
        return self.data.get((tuple(namespace), key))

    def put(self, namespace, key, value, index=None):
        if self.raise_on:
            raise RuntimeError("store down")
        # Mirror the real contract we rely on: index=False (no embedding) and no
        # 'content' key in the persisted value (PHI/embedding safety).
        assert index is False, "pin must be written with index=False"
        assert "content" not in value, "pin value must not carry a 'content' key"
        self.data[(tuple(namespace), key)] = _Item(dict(value))


def _install_stubs() -> None:
    cfg = types.ModuleType("config")
    cfg.settings = types.SimpleNamespace(
        routing_enabled=True,
        routing_default_tier="med",
        classifier_endpoint="databricks-gpt-5-6-luna",
        classifier_reasoning_effort="medium",
        classifier_base_path="/serving-endpoints",
        classifier_timeout_seconds=3.0,
        routing_pin_ttl_seconds=1800.0,
        llm_endpoint="blend-endpoint",
        llm_endpoint_low="",
        llm_endpoint_med="",
        llm_endpoint_high="",
        effort_low="low",
        effort_med="medium",
        effort_high="high",
        agent_llm_effort="high",
    )
    sys.modules["config"] = cfg

    # agent.llm.build_chat_openai — imported lazily inside classify_complexity.
    agent_llm = types.ModuleType("agent.llm")
    agent_llm.build_chat_openai = lambda **kwargs: _FakeLLM(**kwargs)
    sys.modules["agent.llm"] = agent_llm

    # services.store.get_store — imported lazily inside the pin helpers.
    svc = types.ModuleType("services")
    svc_store = types.ModuleType("services.store")

    def _get_store():
        store = _STORE_HOLDER["store"]
        if store is None:
            raise RuntimeError("no store configured for this test")
        return store

    svc_store.get_store = _get_store
    svc.store = svc_store
    sys.modules["services"] = svc
    sys.modules["services.store"] = svc_store


from tests._isolation import IsolatedModules  # noqa: E402

# The stubs are live only while this module's tests run (setUpModule ..
# tearDownModule); installed at import time they replaced config and services
# for every later module of a single pytest run. `routing` is imported fresh.
_ISOLATION = IsolatedModules()
routing = None


def setUpModule():
    global routing
    _ISOLATION.start(fresh=("agent.routing",))
    _install_stubs()
    from agent import routing as stubbed_routing

    routing = stubbed_routing


def tearDownModule():
    _ISOLATION.stop()


def _run(coro):
    return asyncio.run(coro)


class HighRiskTests(unittest.TestCase):
    def test_cues_force_high(self):
        self.assertTrue(
            routing.is_high_risk("please reconcile codes across these 3 docs")
        )
        self.assertTrue(routing.is_high_risk("is there a discrepancy here?"))
        self.assertTrue(routing.is_high_risk("check for FRAUD in this file"))

    def test_domain_nouns_do_not_trigger(self):
        # denial-management is the product's bread and butter — must NOT be high-risk
        self.assertFalse(routing.is_high_risk("summarize this denial letter"))
        self.assertFalse(routing.is_high_risk("what is the appeal deadline?"))
        self.assertFalse(routing.is_high_risk("what is the member id"))
        self.assertFalse(routing.is_high_risk(""))

    def test_axes_are_reported_separately(self):
        # complexity vs stakes are different signals and must be attributable
        self.assertEqual(routing.high_risk_kind("reconcile these two"), "complexity")
        self.assertEqual(routing.high_risk_kind("is this fraud?"), "stakes")
        self.assertIsNone(routing.high_risk_kind("what is the member id"))
        # both present → attributed to complexity (the reason a bigger model helps)
        self.assertEqual(
            routing.high_risk_kind("reconcile these for fraud"), "complexity"
        )

    def test_word_boundary_not_substring(self):
        # Dropped cues — everyday review vocabulary must NOT force HIGH.
        self.assertFalse(routing.is_high_risk("show me the audit trail"))
        self.assertFalse(routing.is_high_risk("status across the queue"))
        # Substring bleed the old `in` check got wrong.
        self.assertFalse(routing.is_high_risk("compare allowances for this code"))
        # ...while the genuine cross-document cues still fire.
        self.assertTrue(routing.is_high_risk("compare these two remits"))
        self.assertTrue(routing.is_high_risk("reconciliation across all claims"))
        self.assertTrue(routing.is_high_risk("these totals are inconsistent"))
        self.assertTrue(routing.is_high_risk("cross-referencing the EOBs"))


class TierMapTests(unittest.TestCase):
    def tearDown(self):
        routing.settings.llm_endpoint_low = ""
        routing.settings.llm_endpoint_med = ""
        routing.settings.llm_endpoint_high = ""
        routing.settings.effort_low = "low"
        routing.settings.effort_med = "medium"

    def test_phase1_effort_only(self):
        # endpoints empty → None (existing blend), effort by tier
        self.assertEqual(routing.tier_to_endpoint_effort("low"), (None, "medium"))
        self.assertEqual(routing.tier_to_endpoint_effort("med"), (None, "medium"))
        self.assertEqual(routing.tier_to_endpoint_effort("high"), (None, "high"))

    def test_phase2_endpoints(self):
        routing.settings.llm_endpoint_low = "svc-low"
        routing.settings.llm_endpoint_med = "svc-med"
        routing.settings.llm_endpoint_high = "svc-high"
        self.assertEqual(routing.tier_to_endpoint_effort("low"), ("svc-low", "low"))
        self.assertEqual(routing.tier_to_endpoint_effort("med"), ("svc-med", "medium"))
        self.assertEqual(routing.tier_to_endpoint_effort("high"), ("svc-high", "high"))

    def test_low_without_its_own_service_resolves_like_med(self):
        # No low-tier service → LOW must behave exactly like MED (today's
        # behavior), even when a med service exists: low effort never reaches it
        # by accident, and never reaches the blend.
        self.assertEqual(
            routing.tier_to_endpoint_effort("low"),
            routing.tier_to_endpoint_effort("med"),
        )
        routing.settings.llm_endpoint_med = "svc-med"
        self.assertEqual(routing.tier_to_endpoint_effort("low"), ("svc-med", "medium"))

    def test_unsafe_effort_never_reaches_the_blend(self):
        # A misconfigured effort var must not put none/low on the blend...
        routing.settings.effort_med = "low"
        self.assertEqual(routing.tier_to_endpoint_effort("med"), (None, "medium"))
        # ...but a dedicated tier service may run at low effort.
        routing.settings.llm_endpoint_med = "svc-med"
        self.assertEqual(routing.tier_to_endpoint_effort("med"), ("svc-med", "low"))

    def test_bad_tier_defaults_to_med(self):
        self.assertEqual(routing.tier_to_endpoint_effort("nonsense"), (None, "medium"))


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        _CLS.update(reply="high", err=False, calls=0, last_kwargs=None)

    def test_parses_variants(self):
        for reply, expected in [
            ("high", "high"),
            ("low", "low"),
            ("medium", "med"),
            ("  HIGH\n", "high"),
            ("tier: low", "low"),
            ("med", "med"),
            # UNPARSEABLE → None. The caller converts this into the observable
            # `unparseable` source; it must NOT look like a successful 'med'.
            ("banana", None),
            ("", None),
            ("I am sorry, I cannot help with that.", None),
        ]:
            _CLS["reply"] = reply
            self.assertEqual(_run(routing.classify_complexity("x")), expected)

    def test_uses_serving_endpoints_base_path(self):
        _run(routing.classify_complexity("x"))
        self.assertEqual(_CLS["last_kwargs"].get("base_path"), "/serving-endpoints")
        self.assertEqual(_CLS["last_kwargs"].get("model"), "databricks-gpt-5-6-luna")
        # never the gateway /responses path for the classifier
        self.assertFalse(_CLS["last_kwargs"].get("use_responses_api"))

    def test_classifier_runs_at_medium_effort_with_real_budget(self):
        _run(routing.classify_complexity("x"))
        kw = _CLS["last_kwargs"]
        self.assertEqual(kw.get("reasoning_effort"), "medium")
        # reasoning tokens count against max_tokens: 16 would starve a reasoning
        # model into an empty reply (-> unparseable -> HIGH on every hard turn)
        self.assertGreaterEqual(kw.get("max_tokens"), 1024)
        self.assertNotIn("temperature", kw)  # Luna rejects non-default values

    def test_unsupported_classifier_effort_is_coerced(self):
        routing.settings.classifier_reasoning_effort = "minimal"  # Luna 400s
        try:
            _run(routing.classify_complexity("x"))
            self.assertEqual(_CLS["last_kwargs"].get("reasoning_effort"), "medium")
        finally:
            routing.settings.classifier_reasoning_effort = "medium"

    def test_empty_classifier_effort_omits_the_parameter(self):
        # e.g. a Claude classifier, which 400s on reasoning_effort
        routing.settings.classifier_reasoning_effort = ""
        try:
            _run(routing.classify_complexity("x"))
            self.assertNotIn("reasoning_effort", _CLS["last_kwargs"])
        finally:
            routing.settings.classifier_reasoning_effort = "medium"

    def test_reasoning_blocks_are_not_parsed_as_the_answer(self):
        # A reasoning summary that mentions "high" must not decide the tier.
        _CLS["reply"] = [
            {"type": "reasoning", "summary": "not high-stakes, a simple lookup"},
            {"type": "text", "text": "low"},
        ]
        self.assertEqual(_run(routing.classify_complexity("x")), "low")

    def test_raises_on_transport_error(self):
        _CLS["err"] = True
        with self.assertRaises(Exception):
            _run(routing.classify_complexity("x"))


class ResolveTierTests(unittest.TestCase):
    def setUp(self):
        _CLS.update(reply="high", err=False, calls=0, last_kwargs=None)
        _STORE_HOLDER["store"] = _FakeStore()

    # --- the UI tier selector ------------------------------------------------
    def test_an_explicit_selection_is_honoured_without_classifying(self):
        """The whole point of the control: it decides, and it costs no classifier
        call. A selector that still pays for classification is a lie about cost."""
        tier, source = _run(
            routing.resolve_tier("t1", "summarize this document", requested="low")
        )
        self.assertEqual((tier, source), ("low", "user_selected"))
        self.assertEqual(_CLS["calls"], 0)

    def test_an_explicit_selection_carries_forward(self):
        _run(routing.resolve_tier("t1", "summarize this", requested="high"))
        self.assertEqual(routing.get_pinned_tier("t1"), "high")

    def test_a_safety_cue_beats_an_explicit_cheaper_choice(self):
        """Deliberate product decision: safety outranks the selector. The SOURCE
        has to say it was an override, or the UI cannot explain why the control it
        is showing did not take effect and the rate is not measurable."""
        tier, source = _run(
            routing.resolve_tier("t1", "reconcile these claims", requested="low")
        )
        self.assertEqual(tier, "high")
        self.assertEqual(source, "user_override_escalated")
        self.assertEqual(_CLS["calls"], 0)

    def test_asking_for_high_on_a_cue_turn_is_not_an_override(self):
        """Nothing was overridden — the reviewer and the cue agree, so reporting an
        escalation would put a warning on a screen with nothing wrong."""
        tier, source = _run(
            routing.resolve_tier("t1", "reconcile these claims", requested="high")
        )
        self.assertEqual((tier, source), ("high", "complexity_cue"))

    def test_auto_reproduces_the_pre_selector_behaviour(self):
        for requested in (None, routing.normalize_requested_tier("auto")):
            with self.subTest(requested=requested):
                _STORE_HOLDER["store"] = _FakeStore()
                _CLS.update(calls=0)
                tier, source = _run(
                    routing.resolve_tier(
                        "t1", "walk me through this", requested=requested
                    )
                )
                self.assertEqual(source, "classified")
                self.assertEqual(_CLS["calls"], 1)

    def test_first_turn_classifies_and_pins(self):
        tier, source = _run(
            routing.resolve_tier("t1", "walk me through this whole appeal")
        )
        self.assertEqual((tier, source), ("high", "classified"))
        self.assertEqual(_CLS["calls"], 1)
        # persisted
        self.assertEqual(routing.get_pinned_tier("t1"), "high")

    def test_second_turn_reuses_pin_no_reclassify(self):
        _CLS["reply"] = "med"
        _run(routing.resolve_tier("t2", "hello"))
        self.assertEqual(_CLS["calls"], 1)
        # second turn: pinned, classifier NOT called again
        tier, source = _run(routing.resolve_tier("t2", "another simple question"))
        self.assertEqual((tier, source), ("med", "pinned"))
        self.assertEqual(_CLS["calls"], 1)

    def test_high_risk_override_beats_pin_and_reclassify(self):
        # pin low first
        _CLS["reply"] = "low"
        _run(routing.resolve_tier("t3", "what is the id"))
        self.assertEqual(routing.get_pinned_tier("t3"), "low")
        # a high-risk turn overrides upward WITHOUT calling the classifier
        calls_before = _CLS["calls"]
        tier, source = _run(
            routing.resolve_tier("t3", "reconcile these across all pages")
        )
        # source now names the AXIS that fired (was: high_risk_override)
        self.assertEqual((tier, source), ("high", "complexity_cue"))
        self.assertEqual(_CLS["calls"], calls_before)  # no LLM call
        self.assertEqual(routing.get_pinned_tier("t3"), "high")  # pinned upward

    def test_cue_sources_name_the_axis(self):
        _CLS["reply"] = "low"
        tier, source = _run(routing.resolve_tier("tc1", "reconcile these remits"))
        self.assertEqual((tier, source), ("high", "complexity_cue"))
        tier, source = _run(routing.resolve_tier("tc2", "check for fraud"))
        self.assertEqual((tier, source), ("high", "stakes_cue"))

    def test_expired_pin_is_reclassified_and_can_come_down(self):
        # pin high, then age the pin past its TTL
        _CLS["reply"] = "high"
        _run(routing.resolve_tier("tt1", "walk me through everything"))
        self.assertEqual(routing.get_pinned_tier("tt1"), "high")
        item = _STORE_HOLDER["store"].data[(routing.ROUTING_NAMESPACE, "tt1")]
        item.value["pinned_at"] = time.time() - 10_000  # older than the 1800s TTL
        # expired → not honored, so the next turn re-decides...
        self.assertIsNone(routing.get_pinned_tier("tt1"))
        _CLS["reply"] = "low"
        tier, source = _run(routing.resolve_tier("tt1", "thanks, next document"))
        # ...and an expired pin must NOT drag the fresh one back up
        self.assertEqual((tier, source), ("low", "classified"))
        self.assertEqual(routing.get_pinned_tier("tt1"), "low")

    def test_ttl_zero_disables_expiry(self):
        routing.settings.routing_pin_ttl_seconds = 0
        try:
            _CLS["reply"] = "high"
            _run(routing.resolve_tier("tt2", "reconcile"))
            item = _STORE_HOLDER["store"].data[(routing.ROUTING_NAMESPACE, "tt2")]
            item.value["pinned_at"] = time.time() - 10_000
            self.assertEqual(routing.get_pinned_tier("tt2"), "high")
        finally:
            routing.settings.routing_pin_ttl_seconds = 1800.0

    def test_legacy_pin_without_timestamp_is_honored(self):
        # pins written before pinned_at existed must not be invalidated
        store = _STORE_HOLDER["store"]
        store.data[(routing.ROUTING_NAMESPACE, "tt3")] = _Item({"tier": "high"})
        self.assertEqual(routing.get_pinned_tier("tt3"), "high")

    def test_unparseable_classifier_resolves_up_observably(self):
        _CLS["reply"] = "banana"
        tier, source = _run(routing.resolve_tier("t9", "what is the member id"))
        # rubric says "when uncertain, choose the HIGHER tier"; and the source
        # must distinguish this from a real classification
        self.assertEqual((tier, source), ("high", "unparseable"))
        # pinned → a broken classifier costs ONE call per thread, not per turn
        self.assertEqual(routing.get_pinned_tier("t9"), "high")

    def test_classifier_error_defaults_observably(self):
        _CLS["err"] = True
        tier, source = _run(routing.resolve_tier("t4", "some question"))
        self.assertEqual((tier, source), ("med", "default_on_error"))

    def test_store_unavailable_is_graceful(self):
        _STORE_HOLDER["store"] = _FakeStore(raise_on=True)
        _CLS["reply"] = "high"
        # classifies and returns despite the store raising on get + put
        tier, source = _run(routing.resolve_tier("t5", "walk me through this"))
        self.assertEqual((tier, source), ("high", "classified"))

    def test_pin_is_upward_only(self):
        routing.pin_tier("t6", "high")
        routing.pin_tier("t6", "low")  # must not downgrade
        self.assertEqual(routing.get_pinned_tier("t6"), "high")


class ModuleTests(unittest.TestCase):
    def test_namespace_and_tiers(self):
        self.assertEqual(routing.ROUTING_NAMESPACE, ("agent_routing",))
        self.assertEqual(set(routing._TIER_RANK), {"low", "med", "high"})


class NormalizeRequestedTierTests(unittest.TestCase):
    """The UI says "Medium"; the tiers are named "med". Anything unrecognised has
    to mean "no request" rather than raising: a stray value from a stale client
    must not fail a turn, and classifying is what the reviewer would have got."""

    def test_auto_and_blank_mean_no_request(self):
        for value in ("auto", "AUTO", " auto ", "", "   ", None, 7, object()):
            self.assertIsNone(routing.normalize_requested_tier(value), value)

    def test_medium_maps_to_med(self):
        self.assertEqual(routing.normalize_requested_tier("medium"), "med")
        self.assertEqual(routing.normalize_requested_tier("Medium"), "med")

    def test_the_wire_tiers_pass_through(self):
        for value in ("low", "med", "high", "HIGH", " low "):
            self.assertEqual(
                routing.normalize_requested_tier(value), value.strip().lower()
            )

    def test_garbage_is_not_a_tier(self):
        for value in ("urgent", "medium-high", "lowest"):
            self.assertIsNone(routing.normalize_requested_tier(value), value)


if __name__ == "__main__":
    unittest.main()
