"""grant CREATE ON SCHEMA public to the agent SP so it can bootstrap its
LangGraph checkpointer/store tables on a fresh Lakebase instance.

Revision ID: 20260421_000020
Revises: 20260421_000019
Create Date: 2026-04-21 00:00:20.000000

Context: in this workspace the original agent SP (a7e5b316-...) was torn
down and replaced by 7b1edad0-.... Its LangGraph tables (checkpoints,
checkpoint_blobs, checkpoint_writes, store, store_vectors, plus the
pgvector tracker vector_migrations and the langgraph bookkeeping tables
checkpoint_migrations + store_migrations) remained owned by the deleted
SP, so the new agent got InsufficientPrivilege on first use. Cleanup
was done out-of-band (DROP + let the new agent re-create on first call)
because Alembic runs as the reviewer SP and cannot REASSIGN OWNED from
a principal that no longer exists.

What this migration does: grant CREATE ON SCHEMA public to the current
agent SP so that *if* LangGraph's setup() runs (i.e. the tables aren't
there yet), the agent can create them and own them. Safe when the
tables already exist — CREATE is a no-op against a table that already
belongs to another owner; it only matters at bootstrap.

If the tables are already present and owned by a defunct SP, that is an
out-of-band cleanup problem (see /tmp/fix_orphaned_tables.py used once
on this instance); no Alembic migration can reassign ownership away
from a deleted principal.
"""

import os

from alembic import op


revision = "20260421_000020"
down_revision = "20260421_000019"
branch_labels = None
depends_on = None

AGENT_SP = os.environ.get("AGENT_SP_CLIENT_ID", "").strip()


def _role_exists(role: str) -> bool:
    if not role:
        return False
    bind = op.get_bind()
    res = bind.exec_driver_sql(
        "SELECT 1 FROM pg_roles WHERE rolname = %(r)s", {"r": role}
    )
    return res.scalar() is not None


def upgrade():
    if not AGENT_SP or not _role_exists(AGENT_SP):
        return
    op.execute(f'GRANT USAGE ON SCHEMA public TO "{AGENT_SP}"')
    op.execute(f'GRANT CREATE ON SCHEMA public TO "{AGENT_SP}"')


def downgrade():
    if not AGENT_SP or not _role_exists(AGENT_SP):
        return
    op.execute(f'REVOKE CREATE ON SCHEMA public FROM "{AGENT_SP}"')
