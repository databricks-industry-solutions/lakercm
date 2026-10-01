"""Unit tests for the speech-to-text transcription route + service parser.

Run from reviewer_app/ as the working directory:
  python3 -m unittest tests.test_transcribe

Covers the route guards (200 / 400 / 413 / 403 / 429) and the shape-tolerant
`_extract_text` parser — the FM returns `message.content` as a LIST of blocks
for reasoning models (validated live against databricks-gemini-3-5-flash), which
the naive `content.strip()` would have crashed on.

All external collaborators (Databricks SDK, settings, the FM call) are mocked;
no network.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

_REVIEWER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REVIEWER_DIR not in sys.path:
    sys.path.insert(0, _REVIEWER_DIR)


def _install_stubs() -> None:
    """Stub heavy deps so importing the transcription modules works standalone."""
    # Sibling modules' stubs no longer leak in (tests/_isolation.py), but keep
    # this additive: set the attrs this module needs onto whatever settings
    # object is there rather than assuming a clean slate.
    cfg = sys.modules.get("config") or types.ModuleType("config")
    if not hasattr(cfg, "settings"):

        class _Settings:
            lakercm_schema = "lakercm"
            pass

        cfg.settings = _Settings()  # type: ignore[attr-defined]
    _s = cfg.settings  # type: ignore[attr-defined]
    _s.transcription_engine = getattr(_s, "transcription_engine", "fm")
    _s.transcription_endpoint = getattr(
        _s, "transcription_endpoint", "databricks-gemini-3-5-flash"
    )
    # small, so the 429 test is cheap
    _s.transcription_rate_per_min = 3
    sys.modules["config"] = cfg

    if "databricks" not in sys.modules:
        dbx = types.ModuleType("databricks")
        sdk = types.ModuleType("databricks.sdk")

        class _WorkspaceClient:  # never actually used — the FM call is mocked
            class config:
                host = "https://example.databricks.net"

                @staticmethod
                def _header_factory():
                    return {"Authorization": "Bearer x"}

        sdk.WorkspaceClient = _WorkspaceClient  # type: ignore[attr-defined]
        sys.modules["databricks"] = dbx
        sys.modules["databricks.sdk"] = sdk

    # dependencies.py imports LakeRCMDatabase by name only — stub the heavy DB
    # module so we don't drag in psycopg / real settings.
    if "services.lakehouse_db" not in sys.modules:
        lh = types.ModuleType("services.lakehouse_db")

        class _LakeRCMDatabase:
            pass

        lh.LakeRCMDatabase = _LakeRCMDatabase  # type: ignore[attr-defined]
        sys.modules["services.lakehouse_db"] = lh


from tests._isolation import IsolatedModules  # noqa: E402

# The stubs are live only while this module's tests run (setUpModule ..
# tearDownModule); installed at import time they leaked into every other module
# of a single pytest run. `svc` and `route` are imported fresh against them.
_ISOLATION = IsolatedModules()
svc = route = None


def setUpModule():
    global svc, route
    _ISOLATION.start(purge_first_party=True)
    _install_stubs()
    from routes import transcription as stubbed_route
    from services import transcription as stubbed_svc

    svc, route = stubbed_svc, stubbed_route


def tearDownModule():
    _ISOLATION.stop()


class FakeRequest:
    def __init__(self, email: str | None = "alice@example.com"):
        self.headers = {"x-forwarded-email": email} if email else {}


class FakeUpload:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data


def _wav(n_data_bytes: int = 64) -> bytes:
    return b"RIFF" + b"\x00" * n_data_bytes


class ExtractTextTests(unittest.TestCase):
    def test_list_of_blocks(self) -> None:
        content = [
            {
                "type": "text",
                "text": "The patient was prescribed ",
                "thoughtSignature": "abc",
            },
            {"type": "text", "text": "amoxicillin."},
        ]
        self.assertEqual(
            svc._extract_text(content), "The patient was prescribed amoxicillin."
        )

    def test_plain_string(self) -> None:
        self.assertEqual(svc._extract_text("hello world"), "hello world")

    def test_non_text_blocks_ignored(self) -> None:
        self.assertEqual(svc._extract_text([{"type": "image_url"}, {"foo": "bar"}]), "")

    def test_empty_and_none(self) -> None:
        self.assertEqual(svc._extract_text([]), "")
        self.assertEqual(svc._extract_text(None), "")

    def test_strip_preamble(self) -> None:
        self.assertEqual(svc._strip_preamble("Transcript: hello"), "hello")
        self.assertEqual(svc._strip_preamble("hello"), "hello")


class TranscribeRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        route._hits.clear()  # reset the per-user rate-limit window

    def _run(self, coro):
        return asyncio.run(coro)

    def test_200_returns_text_and_seq_id(self) -> None:
        with patch.object(route, "transcribe_audio", AsyncMock(return_value="hello")):
            result = self._run(
                route.transcribe(FakeRequest(), audio=FakeUpload(_wav()), seq_id=7)
            )
        self.assertEqual(result, {"text": "hello", "seq_id": 7})

    def test_400_non_wav(self) -> None:
        with patch.object(route, "transcribe_audio", AsyncMock(return_value="x")):
            with self.assertRaises(route.HTTPException) as ctx:
                self._run(
                    route.transcribe(
                        FakeRequest(), audio=FakeUpload(b"XXXXnotwav"), seq_id=0
                    )
                )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_413_oversize(self) -> None:
        big = _wav(route.MAX_BYTES + 1)
        with patch.object(route, "transcribe_audio", AsyncMock(return_value="x")):
            with self.assertRaises(route.HTTPException) as ctx:
                self._run(
                    route.transcribe(FakeRequest(), audio=FakeUpload(big), seq_id=0)
                )
        self.assertEqual(ctx.exception.status_code, 413)

    def test_403_demo_identity(self) -> None:
        with patch.object(route, "transcribe_audio", AsyncMock(return_value="x")):
            with self.assertRaises(route.HTTPException) as ctx:
                self._run(
                    route.transcribe(
                        FakeRequest(email=None), audio=FakeUpload(_wav()), seq_id=0
                    )
                )
        self.assertEqual(ctx.exception.status_code, 403)

    def test_429_rate_limit(self) -> None:
        # transcription_rate_per_min is stubbed to 3 → the 4th call in the
        # window is rejected.
        with patch.object(route, "transcribe_audio", AsyncMock(return_value="ok")):
            for _ in range(3):
                self._run(
                    route.transcribe(FakeRequest(), audio=FakeUpload(_wav()), seq_id=0)
                )
            with self.assertRaises(route.HTTPException) as ctx:
                self._run(
                    route.transcribe(FakeRequest(), audio=FakeUpload(_wav()), seq_id=0)
                )
        self.assertEqual(ctx.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
