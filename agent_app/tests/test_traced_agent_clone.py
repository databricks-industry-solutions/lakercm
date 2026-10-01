"""The per-request clone has to carry the tier agents.

`add_langgraph_fastapi_endpoint` runs `request_agent = agent.clone()` on EVERY
request, and the base `LangGraphAgent.clone()` rebuilds through
`type(self)(name=, graph=, description=, config=)`. It cannot know about an extra
`__init__` parameter, so a subclass that adds one gets the default back. Its own
docstring says so: "Subclasses that add required __init__ parameters must
override clone() to pass those parameters through."

`_TracedAGUIAgent` adds `tier_agents` and did not override `clone()`, so
`self._tier_agents` was None on every turn the app ever served. Complexity-tiered
routing was dead in production while looking entirely healthy:

  * boot logged "AG-UI complexity routing enabled: 3 distinct tier graph(s)";
  * /health reported {"enabled": true, "distinct_graphs": 3} (module state, not
    the per-request object);
  * the clone is still a _TracedAGUIAgent, so run() executed, the
    `agent_turn_agui` span was created, and trace_marker dispatched normally;
  * only `if self._tier_agents:` failed -- and a skipped branch logs nothing.

So the single visible symptom was the ABSENCE of a routing_marker event: the UI
tier selector rendered, the client sent forwardedProps.agent_tier, and the server
ignored it 100% of the time.

These tests drive the REAL base clone() rather than asserting on source text, so
they fail if a future ag_ui_langgraph changes clone()'s contract.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_traced_agent_clone
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

_HAVE_AGUI = importlib.util.find_spec("ag_ui_langgraph") is not None


class _FakeGraph:
    """Minimal object satisfying LangGraphAgent.__init__'s graph introspection."""

    def __init__(self):
        self.nodes = {"agent": SimpleNamespace()}
        self.name = "fake"


@unittest.skipUnless(_HAVE_AGUI, "ag_ui_langgraph not installed")
class TestTierAgentsSurviveClone(unittest.TestCase):
    def _classes(self):
        from ag_ui_langgraph import LangGraphAgent

        class WithoutOverride(LangGraphAgent):
            """The shape the bug had: extra kwarg, inherited clone()."""

            def __init__(self, *args, tier_agents=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._tier_agents = tier_agents or None

        class WithOverride(WithoutOverride):
            def clone(self):
                cloned = super().clone()
                cloned._tier_agents = self._tier_agents
                return cloned

        return WithoutOverride, WithOverride

    def _make(self, cls, tiers):
        return cls(name="t", graph=_FakeGraph(), description="d", tier_agents=tiers)

    def test_the_base_clone_drops_an_extra_init_parameter(self):
        # Pins the upstream behaviour this fix exists for. If a future release
        # starts preserving subclass attributes, this fails and the override can
        # be reconsidered rather than cargo-culted.
        without, _ = self._classes()
        agent = self._make(without, {"low": 1, "med": 2, "high": 3})
        self.assertTrue(agent._tier_agents)
        self.assertFalse(getattr(agent.clone(), "_tier_agents", None))

    def test_the_override_carries_the_tier_agents(self):
        _, with_override = self._classes()
        tiers = {"low": 1, "med": 2, "high": 3}
        agent = self._make(with_override, tiers)
        self.assertEqual(agent.clone()._tier_agents, tiers)

    def test_the_clone_keeps_the_subclass_so_run_is_still_traced(self):
        # Why nothing else looked broken: the clone IS the subclass, so run(),
        # the agent_turn_agui span and trace_marker all kept working.
        _, with_override = self._classes()
        agent = self._make(with_override, {"low": 1})
        self.assertIsInstance(agent.clone(), type(agent))

    def test_routing_off_stays_off_through_a_clone(self):
        # None must not become a truthy empty container: that would make run()
        # enter the routing block with nothing to dispatch to.
        _, with_override = self._classes()
        agent = self._make(with_override, None)
        self.assertIsNone(agent._tier_agents)
        self.assertIsNone(agent.clone()._tier_agents)


@unittest.skipUnless(_HAVE_AGUI, "ag_ui_langgraph not installed")
class TestTracedAgentDeclaresClone(unittest.TestCase):
    def test_main_overrides_clone(self):
        # Source-level, because importing agent_app.main mounts endpoints and
        # compiles graphs at import time, which a unit test must not do.
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "main.py").read_text()
        cls = src.split("class _TracedAGUIAgent", 1)[1]
        body = cls.split("\nclass ", 1)[0]
        self.assertIn("def clone(self)", body)
        self.assertIn("_tier_agents", body.split("def clone(self)", 1)[1])


if __name__ == "__main__":
    unittest.main()
