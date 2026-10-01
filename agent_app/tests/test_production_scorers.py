"""Every production @scorer must run where scheduled monitoring runs it.

Scheduled monitoring never imports eval/scorer_set.py. Registration serializes
only each @scorer function's BODY, and the monitor re-executes that body
(mlflow.genai.scorers.scorer_utils.recreate_function) in a namespace holding
just `mlflow` and a few entity classes. A body that calls a module-level helper,
reads a module-level regex, or relies on a module-level import is therefore a
NameError on every sampled trace, while every in-process test stays green,
because in-process the module globals ARE there. That is how the first versions
of policy_citations_grounded and no_scaffolding_leak were written (fourth
review): their assessments, and the alerts that read them, could never appear.

Two checks:
  * static (always runs, needs nothing installed): every name a scorer body
    reads from global scope is a builtin or in the monitor's namespace
  * dynamic (needs real mlflow): rebuild every production scorer from its own
    serialized form with mlflow's recreate_function, and check that the rebuilt
    copy returns what the in-process scorer returns

Run from agent_app/:
  python3 -m pytest tests/test_production_scorers.py
"""

from __future__ import annotations

import ast
import builtins
import copy
import os
import symtable
import sys
import textwrap
import unittest
from pathlib import Path

_AGENT_APP_DIR = Path(__file__).resolve().parents[1]
if str(_AGENT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_APP_DIR))

SCORER_SET = _AGENT_APP_DIR / "eval" / "scorer_set.py"

# The globals recreate_function gives the body it exec()s
# (mlflow/genai/scorers/scorer_utils.py). Anything else must be imported or
# defined INSIDE the body.
MONITOR_NAMESPACE = frozenset(
    {
        "mlflow",
        "Feedback",
        "Assessment",
        "AssessmentSource",
        "AssessmentError",
        "AssessmentSourceType",
        "Trace",
        "CategoricalRating",
    }
)
ALLOWED_GLOBALS = MONITOR_NAMESPACE | frozenset(dir(builtins))

# What production_schedule() registers as custom (@scorer) monitors.
PRODUCTION_CUSTOM_SCORERS = frozenset(
    {
        "no_sql_warehouse_regression",
        "latency_under_slo",
        "tool_call_budget",
        "policy_citations_grounded",
        "no_scaffolding_leak",
    }
)


def _is_scorer_decorator(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return isinstance(target, ast.Name) and target.id == "scorer"


def _scorer_functions(source: str) -> list[ast.FunctionDef]:
    return [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and any(_is_scorer_decorator(d) for d in node.decorator_list)
    ]


def _global_reads(fn: ast.FunctionDef) -> set[str]:
    """Names the function body reads from global scope, as the monitor runs it.

    Rebuilt the way recreate_function rebuilds it: no decorator, no docstring,
    no annotations (it prepends `from __future__ import annotations`, so they
    are never evaluated). Names bound anywhere inside the body, including
    in-body imports and nested helpers, are local or free, not global.
    """
    rebuilt = copy.deepcopy(fn)
    rebuilt.decorator_list = []
    rebuilt.returns = None
    args = rebuilt.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        arg.annotation = None
    body = rebuilt.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        rebuilt.body = body[1:] or [ast.Pass()]
    module = ast.Module(body=[rebuilt], type_ignores=[])
    top = symtable.symtable(ast.unparse(module), "<scorer>", "exec")
    (fn_table,) = [t for t in top.get_children() if t.get_name() == fn.name]

    def walk(table) -> set[str]:
        names = {
            s.get_name()
            for s in table.get_symbols()
            if s.is_global() and s.is_referenced()
        }
        for child in table.get_children():
            names |= walk(child)
        return names

    return walk(fn_table)


class TestProductionScorersAreSelfContained(unittest.TestCase):
    """Static check: nothing a scorer body needs is missing from the monitor."""

    @classmethod
    def setUpClass(cls):
        cls.functions = _scorer_functions(SCORER_SET.read_text(encoding="utf-8"))

    def test_finds_every_production_custom_scorer(self):
        names = {fn.name for fn in self.functions}
        self.assertTrue(
            PRODUCTION_CUSTOM_SCORERS <= names,
            f"missing: {sorted(PRODUCTION_CUSTOM_SCORERS - names)}",
        )

    def test_no_scorer_body_reads_a_name_the_monitor_lacks(self):
        problems = {
            fn.name: sorted(_global_reads(fn) - ALLOWED_GLOBALS)
            for fn in self.functions
        }
        problems = {name: leaked for name, leaked in problems.items() if leaked}
        self.assertEqual(
            problems,
            {},
            "these @scorer bodies read module-level names that scheduled "
            "monitoring does not provide; move them INSIDE the body",
        )

    def test_the_check_catches_the_original_bug(self):
        # The shape the first version had: a thin body calling module helpers.
        bad = _scorer_functions(textwrap.dedent('''
                @scorer(name="x")
                def x(outputs=None, trace=None):
                    """Docstring."""
                    value, why = judge(ids(trace), text(outputs))
                    return Feedback(value=value, rationale=_RE.pattern + why)
                '''))[0]
        self.assertEqual(
            _global_reads(bad) - ALLOWED_GLOBALS, {"judge", "ids", "text", "_RE"}
        )

    def test_in_body_imports_and_nested_helpers_are_fine(self):
        good = _scorer_functions(textwrap.dedent("""
                @scorer
                def y(outputs=None):
                    import re

                    def helper(s):
                        return re.sub("a", "b", s)

                    return Feedback(value=bool([c for c in helper(str(outputs))]))
                """))[0]
        self.assertEqual(_global_reads(good) - ALLOWED_GLOBALS, set())


def _real_mlflow_or_none():
    """The real mlflow scorer utilities, or None if absent or stubbed."""
    try:
        import mlflow
        from mlflow.genai.scorers.scorer_utils import recreate_function
    except Exception:
        return None
    if not getattr(mlflow, "__file__", None):  # a sibling test's bare stub
        return None
    return recreate_function


class _Span:
    def __init__(self, span_type, outputs=None, name="span", attributes=None):
        self.span_type = span_type
        self.outputs = outputs
        self.name = name
        self.attributes = attributes or {}


class _Data:
    def __init__(self, spans):
        self.spans = spans


class _Info:
    def __init__(self, execution_time_ms):
        self.execution_time_ms = execution_time_ms


class _Trace:
    def __init__(self, spans, execution_time_ms=1200):
        self.data = _Data(spans)
        self.info = _Info(execution_time_ms)


class TestRebuiltAsTheMonitorDoes(unittest.TestCase):
    """Dynamic check: the monitor's rebuilt copy returns what the scorer returns."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        recreate = _real_mlflow_or_none()
        if recreate is None:
            raise unittest.SkipTest("real mlflow not importable (absent or stubbed)")
        cls.recreate = staticmethod(recreate)
        import importlib

        cls.ss = importlib.import_module("eval.scorer_set")
        if not getattr(cls.ss, "__file__", None):
            raise unittest.SkipTest("eval.scorer_set is stubbed in sys.modules")

    def _rebuild(self, scorer_obj):
        d = scorer_obj.model_dump()
        return self.recreate(
            d["call_source"], d["call_signature"], d["original_func_name"]
        )

    def _assert_same(self, scorer_obj, **kwargs):
        rebuilt = self._rebuild(scorer_obj)(**kwargs)
        live = scorer_obj(**kwargs)
        if live is None:
            self.assertIsNone(rebuilt)
            return None
        self.assertEqual(
            (rebuilt.value, rebuilt.rationale), (live.value, live.rationale)
        )
        return rebuilt.value

    def test_every_production_custom_scorer_rebuilds(self):
        names = set()
        for scorer_obj, _rate in self.ss.production_schedule():
            dumped = scorer_obj.model_dump()
            if not dumped.get("call_source"):
                continue  # a builtin judge: serialized by class, not by source
            names.add(dumped["name"])
            self.assertTrue(callable(self._rebuild(scorer_obj)), dumped["name"])
        self.assertEqual(names, set(PRODUCTION_CUSTOM_SCORERS))

    def test_policy_citations_grounded(self):
        retrieved = _Trace(
            [_Span("RETRIEVER", [{"metadata": {"policy_id": "POL-VD-MRI-001"}}])]
        )
        cases = [
            ("Conservative therapy is required (POL-VD-MRI-001).", True),
            ("Six weeks is required.", False),
            ("Per POL-ZZ-FAKE-999 this is denied.", False),
            ("I could not find POL-ZZ-FAKE-999 in the retrieved policies.", True),
            ("The retrieved policies do not cover lumbar fusion.", True),
            (
                "Per POL-VD-FUSION-777, prior auth is required, though I could not "
                "find the full policy text.",
                False,
            ),
        ]
        for text, want in cases:
            with self.subTest(text=text):
                got = self._assert_same(
                    self.ss.policy_citations_grounded, outputs=text, trace=retrieved
                )
                self.assertIs(got, want)
        self.assertIsNone(
            self._assert_same(
                self.ss.policy_citations_grounded,
                outputs="anything",
                trace=_Trace([_Span("TOOL")]),
            )
        )

    def test_no_scaffolding_leak(self):
        for outputs, want in (
            ("UNTRUSTED_DOCUMENT_CONTENT>>>", False),
            ({"messages": [{"content": "value was [redacted-delimiter]"}]}, False),
            ("Clean-claim rate is 91%.", True),
            (None, True),
        ):
            with self.subTest(outputs=outputs):
                got = self._assert_same(self.ss.no_scaffolding_leak, outputs=outputs)
                self.assertIs(got, want)

    def test_code_scorers(self):
        warehouse = _Trace([_Span("TOOL", name="StatementExecution.execute")])
        clean = _Trace([_Span("TOOL", name="get_document_details")])
        self.assertIs(
            self._assert_same(self.ss.no_sql_warehouse_regression, trace=warehouse),
            False,
        )
        self.assertIs(
            self._assert_same(self.ss.no_sql_warehouse_regression, trace=clean), True
        )
        # The analytics tier reaches the Reyden warehouse through the pinned
        # system.ai.dbsql MCP service (agent/tools.py > _reyden_sql). Its span is
        # an intended path, not the retired direct-warehouse one.
        reyden = _Trace(
            [
                _Span("TOOL", name="query_lakehouse"),
                _Span(
                    "UNKNOWN",
                    name="dbsql_mcp_call",
                    attributes={
                        "dbsql_mcp.warehouse_id": "00000000abcd1234",
                        "dbsql_mcp.statement_id": "s-1",
                    },
                ),
            ]
        )
        direct = _Trace(
            [
                _Span(
                    "UNKNOWN",
                    name="POST",
                    attributes={"http.url": "https://h/api/2.0/sql/statements"},
                )
            ]
        )
        self.assertIs(
            self._assert_same(self.ss.no_sql_warehouse_regression, trace=reyden), True
        )
        self.assertIs(
            self._assert_same(self.ss.no_sql_warehouse_regression, trace=direct), False
        )
        self.assertIs(
            self._assert_same(
                self.ss.latency_under_slo, trace=_Trace([], execution_time_ms=12_000)
            ),
            False,
        )
        self.assertIs(
            self._assert_same(
                self.ss.tool_call_budget,
                trace=_Trace([_Span("TOOL") for _ in range(9)]),
            ),
            False,
        )


if __name__ == "__main__":
    unittest.main()
