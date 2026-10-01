"""/health must say whether complexity routing is actually live.

Routing failing open is silent by construction: the boot block swallows any
tier-graph error into a warning, the UI keeps rendering its Low/Medium/High
selector, the client keeps sending forwardedProps.agent_tier — and the server
just ignores it. The only external symptom is the ABSENCE of a routing_marker
custom event, which nobody sees without diffing an SSE stream.

That is exactly how it was found: the selector looked fine, a Low-tier turn about
fraud produced an obviously High-tier answer, and only the raw stream showed
trace_marker present and routing_marker missing, with no "tier" anywhere.

Source-level assertions rather than an import: agent_app/main.py mounts endpoints
and compiles graphs at import time, which a unit test should not do. What matters
is structural anyway — that EVERY path out of the boot block records its outcome,
so a future fourth path cannot leave /health confidently wrong.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_routing_health_observability
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"
SRC = MAIN.read_text()
TREE = ast.parse(SRC)


def _health_fn() -> ast.AsyncFunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.name == "health":
                return node
    raise AssertionError("agent_app/main.py has no health() handler")


class TestHealthReportsRoutingState(unittest.TestCase):
    def test_health_returns_a_routing_key(self):
        body = ast.unparse(_health_fn())
        self.assertIn("'routing'", body.replace('"', "'"))
        self.assertIn("_ROUTING_BOOT", body)

    def test_routing_boot_has_a_never_initialized_default(self):
        # "Not initialized" must be distinguishable from an explicit off and from
        # a crash: the three want different fixes, and collapsing them into a
        # bare False is what made this invisible in the first place.
        # AnnAssign, not Assign: it carries a `: dict` annotation.
        assigns = [
            n
            for n in TREE.body
            if isinstance(n, ast.AnnAssign)
            and getattr(n.target, "id", None) == "_ROUTING_BOOT"
        ]
        self.assertEqual(len(assigns), 1, "_ROUTING_BOOT must be declared once")
        literal = ast.literal_eval(assigns[0].value)
        self.assertIs(literal["enabled"], False)
        self.assertIn("reason", literal)

    def test_it_is_declared_before_the_health_handler(self):
        # Read at request time, so definition order is only a readability
        # concern — but a reader should not have to prove that.
        self.assertLess(SRC.index("_ROUTING_BOOT"), SRC.index("async def health"))

    def test_every_boot_outcome_is_recorded(self):
        # Success, explicitly-off, and failure. Three, not two: an explicit off
        # and a swallowed exception look identical on /health otherwise.
        updates = SRC.count("_ROUTING_BOOT.update(")
        self.assertGreaterEqual(
            updates,
            3,
            "each path out of the tier-graph boot block must record its outcome",
        )

    def test_the_success_path_overwrites_the_default_reason(self):
        # {"enabled": true, "reason": "not initialized"} was the live response and
        # reads as a contradiction; the success branch must state its own reason.
        block = SRC.split("_ROUTING_BOOT.update(", 1)[1][:400]
        self.assertIn("enabled=True", block)
        self.assertIn("reason=", block)

    def test_the_request_path_is_measured_not_just_boot(self):
        # Boot state alone could not localise the real break: routing reported
        # enabled with 3 tier graphs while no turn emitted a routing_marker.
        self.assertIn("_ROUTING_TURNS", SRC)
        body = ast.unparse(_health_fn())
        self.assertIn("routing_turns", body.replace('"', "'"))

    def test_every_per_turn_outcome_is_distinguishable(self):
        # skipped vs failed vs resolved want different fixes, so each needs its
        # own counter rather than one "routing didn't happen" bucket.
        for key in ("total", "skipped_no_tier_agents", "resolved", "failed"):
            self.assertIn(key, SRC, f"no per-turn counter for {key!r}")

    def test_the_request_path_failure_records_the_exception(self):
        after = SRC.split("AG-UI tier routing failed", 1)[1][:400]
        self.assertIn("last_error", after)

    def test_the_failure_path_records_the_reason(self):
        # The except that swallows the outage has to leave evidence behind.
        after = SRC.split("tier-graph setup failed", 1)[1]
        self.assertIn("_ROUTING_BOOT.update(", after)
        self.assertIn("reason=", after)


if __name__ == "__main__":
    unittest.main()
