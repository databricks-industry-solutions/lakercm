"""Unit tests for services.checkpointer.delete_thread.

Run from agent_app/ as the working directory:
  python3 -m unittest tests.test_delete_thread

Covers:
  - empty thread_id returns a skip (idempotent by design)
  - saver.delete_thread is used when available (langgraph-checkpoint-postgres
    2.0.10+)
  - raw-SQL fallback is used when the method is missing
  - None checkpointer (boot-time table probe failed) returns a skip
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Make agent_app/ importable from tests/ when running `python3 -m unittest`
# from agent_app/ as the CWD.
_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)


class _FakeSaver:
    """Stand-in for PostgresSaver with a controllable delete_thread method."""

    def __init__(self, has_delete_thread: bool):
        self.delete_thread_calls: list[str] = []
        if has_delete_thread:
            self.delete_thread = self._delete_thread  # type: ignore[assignment]

    def _delete_thread(self, thread_id: str) -> None:
        self.delete_thread_calls.append(thread_id)


def _install_langgraph_stub() -> None:
    if "langgraph.checkpoint.postgres" in sys.modules:
        return
    langgraph = types.ModuleType("langgraph")
    checkpoint = types.ModuleType("langgraph.checkpoint")
    postgres = types.ModuleType("langgraph.checkpoint.postgres")
    postgres.PostgresSaver = _FakeSaver  # type: ignore[attr-defined]
    # services/checkpointer.py also imports AsyncPostgresSaver from the `.aio`
    # SUBMODULE. A plain ModuleType has no __path__, so `import
    # langgraph.checkpoint.postgres.aio` fails with "not a package" unless we
    # register the submodule explicitly (and mark the parent as a package).
    postgres.__path__ = []  # type: ignore[attr-defined]
    aio = types.ModuleType("langgraph.checkpoint.postgres.aio")
    aio.AsyncPostgresSaver = _FakeSaver  # type: ignore[attr-defined]
    postgres.aio = aio  # type: ignore[attr-defined]
    sys.modules["langgraph"] = langgraph
    sys.modules["langgraph.checkpoint"] = checkpoint
    sys.modules["langgraph.checkpoint.postgres"] = postgres
    sys.modules["langgraph.checkpoint.postgres.aio"] = aio


def _install_lakehouse_db_stub() -> None:
    stub = types.ModuleType("services.lakehouse_db")
    stub.get_db = MagicMock()  # type: ignore[attr-defined]
    sys.modules["services.lakehouse_db"] = stub


from tests._isolation import IsolatedModules  # noqa: E402

# The stubs are live only while this module's tests run (setUpModule ..
# tearDownModule); installed at import time they leaked into every other module
# of a single pytest run.
_ISOLATION = IsolatedModules()


def setUpModule():
    _ISOLATION.start(fresh=("services.checkpointer",))
    _install_langgraph_stub()
    _install_lakehouse_db_stub()


def tearDownModule():
    _ISOLATION.stop()


def _fresh_checkpointer_module():
    """Return services.checkpointer with per-test singleton state reset."""
    if "services.checkpointer" in sys.modules:
        del sys.modules["services.checkpointer"]
    import importlib

    return importlib.import_module("services.checkpointer")


class DeleteThreadTests(unittest.TestCase):
    def test_empty_thread_id_is_skipped(self) -> None:
        mod = _fresh_checkpointer_module()
        result = mod.delete_thread("")
        self.assertEqual(result["status"], "skipped")

    def test_uses_saver_delete_thread_when_available(self) -> None:
        mod = _fresh_checkpointer_module()
        fake_saver = _FakeSaver(has_delete_thread=True)
        with patch.object(mod, "get_checkpointer", return_value=fake_saver):
            result = mod.delete_thread("conv-123")
        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["method"], "PostgresSaver.delete_thread")
        self.assertEqual(fake_saver.delete_thread_calls, ["conv-123"])

    def test_falls_back_to_raw_sql_when_method_missing(self) -> None:
        mod = _fresh_checkpointer_module()
        fake_saver = _FakeSaver(has_delete_thread=False)

        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        pool = MagicMock()
        pool.connection.return_value = conn
        fake_db = MagicMock()
        fake_db.get_pool.return_value = pool

        with (
            patch.object(mod, "get_checkpointer", return_value=fake_saver),
            patch.object(mod, "get_db", return_value=fake_db),
        ):
            result = mod.delete_thread("conv-xyz")

        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["method"], "raw-sql-fallback")
        tables = [
            call.args[0].split()[2]  # "DELETE FROM public.<table> WHERE ..."
            for call in cursor.execute.call_args_list
        ]
        self.assertEqual(
            tables,
            [
                "public.checkpoint_writes",
                "public.checkpoint_blobs",
                "public.checkpoints",
            ],
        )

    def test_none_checkpointer_is_skipped(self) -> None:
        mod = _fresh_checkpointer_module()
        with patch.object(mod, "get_checkpointer", return_value=None):
            result = mod.delete_thread("conv-1")
        self.assertEqual(result["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
