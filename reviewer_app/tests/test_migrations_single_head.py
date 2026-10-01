"""The migration history has exactly one head.

Two heads made `alembic upgrade head` fail on every reviewer boot ("Multiple
head revisions are present"), and the app caught the error and started anyway:
the new dev branch got no tables, and nothing reported it except the
Lakebase change data feed finding nothing to feed.

Run from the repo root:
    python3 -m pytest reviewer_app/tests/test_migrations_single_head.py
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

try:
    from alembic.script import ScriptDirectory
except ImportError:  # pragma: no cover - alembic is a reviewer app dependency
    ScriptDirectory = None


@unittest.skipIf(ScriptDirectory is None, "alembic not installed")
class TestMigrationHistory(unittest.TestCase):
    def setUp(self):
        # 000026 reads AGENT_SP_CLIENT_ID at import time; loading the history
        # imports every revision file.
        env = {"AGENT_SP_CLIENT_ID": "test-agent-sp"}
        with mock.patch.dict(os.environ, env):
            self.script = ScriptDirectory(str(MIGRATIONS_DIR))
            self.heads = self.script.get_heads()

    def test_there_is_exactly_one_head(self):
        self.assertEqual(len(self.heads), 1, f"heads: {sorted(self.heads)}")

    def test_the_head_reaches_every_revision(self):
        with mock.patch.dict(os.environ, {"AGENT_SP_CLIENT_ID": "test-agent-sp"}):
            reachable = {
                rev.revision for rev in self.script.walk_revisions("base", "heads")
            }
            every = {rev.revision for rev in self.script.walk_revisions()}
        self.assertEqual(reachable, every)


if __name__ == "__main__":
    unittest.main()
