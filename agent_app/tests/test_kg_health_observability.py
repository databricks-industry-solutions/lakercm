"""/health must say whether the KG traversal tool is actually live in this pod.

PR #88 shipped traverse_claims_graph and it was DARK: config.py reads
LAKERCM_KG_ENABLED / LAKERCM_KG_SCHEMA from the environment, and the apps bundle
never set them, so kg_enabled stayed False in the running pod and the tool was
never registered. Nothing said so. The only way to tell was to coax the model into
calling a tool it did not have, which is the same invisibility that cost a
debugging cycle on complexity routing (see test_routing_health_observability.py).

There is a second, nastier half-state: get_all_tools() gates on the FLAG alone,
but traverse_claims_graph also requires a non-empty schema and returns
unavailable without one. So the tool can be registered, advertised to the model,
and refuse every single call. /health has to name that case, not just report a
boolean.

Source-level assertions rather than an import: agent_app/main.py mounts endpoints
and compiles graphs at import time, which a unit test should not do. The
behavioural half lives in test_kg_traversal_tool.py, which already has the stub
harness to call get_all_tools() for real.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_kg_health_observability
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_SRC = (ROOT / "main.py").read_text()
MAIN_TREE = ast.parse(MAIN_SRC)
TOOLS_SRC = (ROOT / "agent" / "tools.py").read_text()
TOOLS_TREE = ast.parse(TOOLS_SRC)


def _health_fn() -> ast.AsyncFunctionDef:
    for node in ast.walk(MAIN_TREE):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.name == "health":
                return node
    raise AssertionError("agent_app/main.py has no health() handler")


class TestHealthReportsKgState(unittest.TestCase):
    def test_health_returns_a_kg_key(self):
        body = ast.unparse(_health_fn()).replace('"', "'")
        self.assertIn("'kg'", body)
        self.assertIn("kg_status()", body)

    def test_health_imports_kg_status(self):
        # Via agent.tools, which owns the registration decision -- main.py must
        # not re-derive it from settings, or the two could disagree.
        self.assertIn("kg_status", MAIN_SRC.split("async def health")[0])


class TestKgBootRecord(unittest.TestCase):
    def test_kg_boot_has_a_never_initialized_default(self):
        assigns = [
            n
            for n in TOOLS_TREE.body
            if isinstance(n, ast.AnnAssign)
            and getattr(n.target, "id", None) == "_KG_BOOT"
        ]
        self.assertEqual(len(assigns), 1, "_KG_BOOT must be declared once")
        literal = ast.literal_eval(assigns[0].value)
        self.assertIs(literal["enabled"], False)
        self.assertIs(literal["tool_registered"], False)
        # "Never ran" must be distinguishable from an explicit off: a pod whose
        # get_all_tools() never executed is a different bug from a flag set false.
        self.assertEqual(literal["reason"], "not initialized")

    def test_kg_status_accessor_exists(self):
        names = [
            n.name
            for n in ast.walk(TOOLS_TREE)
            if isinstance(n, ast.FunctionDef) and n.name == "kg_status"
        ]
        self.assertEqual(len(names), 1, "agent/tools.py must expose kg_status()")

    def test_the_registration_decision_records_its_outcome(self):
        self.assertIn("_KG_BOOT.update(", TOOLS_SRC)

    def test_every_outcome_is_distinguishable(self):
        # Three reasons, not a bare boolean: flag off, schema missing, and ok.
        # Collapsing them is what made #88 invisible.
        block = TOOLS_SRC.split("_KG_BOOT.update(", 1)[1][:900]
        self.assertIn("LAKERCM_KG_ENABLED", block)
        self.assertIn("LAKERCM_KG_SCHEMA", block)
        self.assertIn('"ok"', block)

    def test_the_schema_is_reported_not_just_its_presence(self):
        # Pointing at the WRONG schema looks identical to pointing at the right
        # one unless /health echoes the value back.
        block = TOOLS_SRC.split("_KG_BOOT.update(", 1)[1][:900]
        self.assertIn("schema=", block)

    def test_tool_registration_is_reported_separately_from_the_flag(self):
        # enabled is the operator's intent; tool_registered is what the pod did.
        # They can diverge, so /health must not conflate them.
        block = TOOLS_SRC.split("_KG_BOOT.update(", 1)[1][:900]
        self.assertIn("tool_registered=", block)
        self.assertIn("enabled=", block)


if __name__ == "__main__":
    unittest.main()
