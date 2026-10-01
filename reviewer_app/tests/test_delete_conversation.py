"""Unit tests for the cascade-delete route handler in routes.chat.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_delete_conversation

Exercises the order-of-operations, fail-closed behavior, idempotent-missing
path, and audit writes on both success and failure paths. All external
collaborators (DB, agent HTTP client, Databricks SDK) are mocked.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_REVIEWER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REVIEWER_DIR not in sys.path:
    sys.path.insert(0, _REVIEWER_DIR)


def _install_stubs() -> None:
    """Stub heavy deps (fastapi, httpx, databricks sdk, settings, db layer) so
    importing routes.chat works without a full environment."""
    if "httpx" not in sys.modules:
        httpx_mod = types.ModuleType("httpx")
        httpx_mod.AsyncClient = MagicMock()  # type: ignore[attr-defined]
        httpx_mod.Timeout = MagicMock()  # type: ignore[attr-defined]
        sys.modules["httpx"] = httpx_mod

    if "config" not in sys.modules:
        cfg = types.ModuleType("config")

        class _Settings:
            lakercm_schema = "lakercm"
            agent_app_url = "https://agent.example/"
            # Read by services.lakehouse_db at import time (routes.chat →
            # dependencies → lakehouse_db). Missing, every test here failed with
            # AttributeError; nothing ran them in isolation to notice.
            auto_verdict_threshold = 0.92
            automated_reviewer_email = "<automated>"

        cfg.settings = _Settings()  # type: ignore[attr-defined]
        sys.modules["config"] = cfg

    if "databricks" not in sys.modules:
        dbx = types.ModuleType("databricks")
        sdk = types.ModuleType("databricks.sdk")

        class _WorkspaceClient:
            class config:
                @staticmethod
                def _header_factory():
                    return {}

        sdk.WorkspaceClient = _WorkspaceClient  # type: ignore[attr-defined]
        sys.modules["databricks"] = dbx
        sys.modules["databricks.sdk"] = sdk


from tests._isolation import IsolatedModules  # noqa: E402

# The stubs are live only while this module's tests run (setUpModule ..
# tearDownModule); installed at import time they leaked into every other module
# of a single pytest run.
_ISOLATION = IsolatedModules()


def setUpModule():
    _ISOLATION.start(purge_first_party=True)
    _install_stubs()


def tearDownModule():
    _ISOLATION.stop()


def _fresh_chat_module():
    for name in ("routes.chat", "services.agent_client"):
        sys.modules.pop(name, None)
    import importlib

    return importlib.import_module("routes.chat")


class FakeRequest:
    def __init__(self, email: str = "alice@example.com"):
        self.headers = {"x-forwarded-email": email}


class DeleteConversationTests(unittest.TestCase):
    CONV = "conv-abc-123"
    EMAIL = "alice@example.com"

    def _run(self, coro):
        return asyncio.run(coro)

    def test_idempotent_missing_returns_zero_rows(self) -> None:
        chat = _fresh_chat_module()
        db = MagicMock()
        db.get_conversation_history.return_value = []  # nothing owned
        agent_mock = AsyncMock()

        with (
            patch.object(chat, "agent_delete_thread", agent_mock),
            patch.object(chat, "get_current_user_email", return_value=self.EMAIL),
        ):
            result = self._run(chat.delete_conversation(self.CONV, FakeRequest(), db))

        self.assertEqual(result, {"status": "deleted", "rows": 0, "agent": "skipped"})
        agent_mock.assert_not_awaited()
        db.delete_conversation.assert_not_called()
        db.log_conversation_deletion.assert_not_called()

    def test_success_calls_agent_before_reviewer_and_audits(self) -> None:
        chat = _fresh_chat_module()
        db = MagicMock()
        db.get_conversation_history.return_value = [{"message_content": "hi"}]
        db.delete_conversation.return_value = 2

        call_order: list[str] = []

        async def fake_agent(thread_id, email):
            call_order.append("agent")
            self.assertEqual(thread_id, self.CONV)
            self.assertEqual(email, self.EMAIL)
            return {"status": "ok", "checkpointer": {"status": "deleted"}}

        def fake_reviewer_delete(conv, email):
            call_order.append("reviewer")
            return 2

        db.delete_conversation.side_effect = fake_reviewer_delete

        with (
            patch.object(chat, "agent_delete_thread", side_effect=fake_agent),
            patch.object(chat, "get_current_user_email", return_value=self.EMAIL),
        ):
            result = self._run(chat.delete_conversation(self.CONV, FakeRequest(), db))

        self.assertEqual(call_order, ["agent", "reviewer"])
        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["rows"], 2)
        db.log_conversation_deletion.assert_called_once()
        audit_kwargs = db.log_conversation_deletion.call_args.kwargs
        self.assertEqual(audit_kwargs["agent_cleanup_status"], "deleted")
        self.assertEqual(audit_kwargs["reviewer_rowcount"], 2)
        self.assertEqual(audit_kwargs["conversation_id"], self.CONV)

    def test_agent_failure_fails_closed_and_audits_failure(self) -> None:
        chat = _fresh_chat_module()
        db = MagicMock()
        db.get_conversation_history.return_value = [{"message_content": "hi"}]

        async def failing_agent(thread_id, email):
            raise RuntimeError("agent boom")

        with (
            patch.object(chat, "agent_delete_thread", side_effect=failing_agent),
            patch.object(chat, "get_current_user_email", return_value=self.EMAIL),
        ):
            with self.assertRaises(chat.HTTPException) as ctx:
                self._run(chat.delete_conversation(self.CONV, FakeRequest(), db))

        self.assertEqual(ctx.exception.status_code, 502)
        # Reviewer DELETE must NOT run when the agent fails.
        db.delete_conversation.assert_not_called()
        # Audit row recorded with agent_cleanup_status=failed and error_detail set.
        db.log_conversation_deletion.assert_called_once()
        audit_kwargs = db.log_conversation_deletion.call_args.kwargs
        self.assertEqual(audit_kwargs["agent_cleanup_status"], "failed")
        self.assertIn("agent boom", audit_kwargs["error_detail"])
        self.assertEqual(audit_kwargs["reviewer_rowcount"], 0)

    def test_audit_failure_does_not_break_success_response(self) -> None:
        """If the success-path audit INSERT fails, the delete still succeeds —
        the row is already gone, so a 5xx would mislead the user."""
        chat = _fresh_chat_module()
        db = MagicMock()
        db.get_conversation_history.return_value = [{"message_content": "hi"}]
        db.delete_conversation.return_value = 1
        db.log_conversation_deletion.side_effect = RuntimeError("audit down")

        async def fake_agent(thread_id, email):
            return {"status": "ok"}

        with (
            patch.object(chat, "agent_delete_thread", side_effect=fake_agent),
            patch.object(chat, "get_current_user_email", return_value=self.EMAIL),
        ):
            result = self._run(chat.delete_conversation(self.CONV, FakeRequest(), db))

        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["rows"], 1)


if __name__ == "__main__":
    unittest.main()
