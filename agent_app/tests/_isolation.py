"""Keep one test module's sys.modules stubs from leaking into the others.

Several test modules replace heavy dependencies (langchain_core, config,
services, mlflow, fastapi, langgraph) with bare stubs, then import the module
under test against them. They used to do it at import time and never undo it.
pytest imports every test module during collection, before any test runs, so
in a single `pytest agent_app/tests` or `pytest reviewer_app/tests` run (how CI
runs them) the stubs were live for the whole session: tests that pass on their
own failed, and the tests that need the real libraries skipped.

agent_app/tests/_isolation.py and reviewer_app/tests/_isolation.py are
identical: the two apps are separate packages with separate `tests` packages,
tested in separate processes.

Usage, from a module's setUpModule / tearDownModule (both unittest and pytest
honor them), so the stubs are live only while that module's own tests run:

    _ISOLATION = IsolatedModules()

    def setUpModule():
        global tools
        _ISOLATION.start(fresh=("agent.tools",))
        _install_stubs()
        from agent import tools

    def tearDownModule():
        _ISOLATION.stop()

`stop()` returns sys.modules to what `start()` saw: every replaced entry is
put back, and every stub or first-party module added in between (which may
have bound a stub) is dropped, along with the attribute its import set on the
parent package. Third-party modules first imported in between are kept:
re-importing a compiled extension (numpy, pydantic_core) is unsafe, and a real
library module does not bind the test's stubs.
"""

from __future__ import annotations

import os
import sys

# The app this copy serves: agent_app/ or reviewer_app/.
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ABSENT = object()


def _is_first_party(module) -> bool:
    """A stub, or one of the app's own modules (never the tests package)."""
    if getattr(module, "__spec__", None) is None:
        # A stub built with types.ModuleType: the import system gives every
        # module it loads a spec, builtins (sys, marshal) included. Keying on a
        # missing __file__ instead also matched sys itself, and purging it broke
        # every import that followed.
        return getattr(module, "__name__", "") != "__main__"
    path = getattr(module, "__file__", None)
    if not path:
        return False  # builtin, frozen, or a namespace package
    path = os.path.abspath(path)
    return (
        path.startswith(_APP_DIR + os.sep)
        and not path.startswith(_TESTS_DIR + os.sep)
        and "site-packages" not in path
    )


class IsolatedModules:
    """Snapshot sys.modules, and restore it exactly (see the module docstring)."""

    def __init__(self) -> None:
        self._before: dict[str, object] | None = None

    def start(
        self, fresh: tuple[str, ...] = (), purge_first_party: bool = False
    ) -> None:
        """Snapshot sys.modules; drop `fresh` so it is re-imported against stubs.

        `purge_first_party` drops every app module as well, for a module under
        test whose import chain (routes -> dependencies -> services) binds the
        stubbed config at several levels.
        """
        self._before = dict(sys.modules)
        names = list(fresh)
        if purge_first_party:
            names += [n for n, m in self._before.items() if _is_first_party(m)]
        for name in names:
            sys.modules.pop(name, None)
            self._set_parent_attr(name, _ABSENT)

    def stop(self) -> None:
        before = self._before
        if before is None:
            return
        self._before = None
        for name in list(sys.modules):
            if name in before:
                continue
            if _is_first_party(sys.modules[name]):
                del sys.modules[name]
                self._set_parent_attr(name, _ABSENT)
        for name, module in before.items():
            if sys.modules.get(name, _ABSENT) is not module:
                sys.modules[name] = module
                self._set_parent_attr(name, module)

    @staticmethod
    def _set_parent_attr(name: str, value) -> None:
        """Point (or un-point) the parent package's attribute for `name`.

        `from agent import tools` reads the ATTRIBUTE on the `agent` package
        before it looks at sys.modules, so a stale attribute would hand the
        stub-bound module to every later importer.
        """
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name) if parent_name else None
        if parent is None:
            return
        if value is _ABSENT:
            if child in getattr(parent, "__dict__", {}):
                try:
                    delattr(parent, child)
                except (AttributeError, TypeError):
                    pass
        else:
            try:
                setattr(parent, child, value)
            except (AttributeError, TypeError):
                pass
