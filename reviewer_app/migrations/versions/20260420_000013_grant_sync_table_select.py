"""grant select on UC→Lakebase sync tables to PUBLIC

Revision ID: 20260420_000013
Revises: 20260419_000012
Create Date: 2026-04-20 00:00:13.000000

Migration 20260224_000006 grants SELECT on all tables in the lakercm
schema to PUBLIC at migration time, plus ALTER DEFAULT PRIVILEGES for future
tables. Default privileges only apply to objects created by the migrating
role, though — the UC→Lakebase sync service owns the synced tables it
creates, so those tables do NOT inherit the default grant.

After any Lakebase recreate (or sync re-initialization), both apps fail with
`permission denied for table gold_extraction_labels_sync`. This migration
re-asserts USAGE + SELECT on the schema and all tables it currently contains,
so a fresh deploy restores app access without manual intervention.

If more synced tables are added later, add them explicitly here — the default
privilege mechanism can't help when the creator is the sync role.
"""

from alembic import op


revision = "20260420_000013"
down_revision = "20260419_000012"
branch_labels = None
depends_on = None


def upgrade():
    # CREATE IF NOT EXISTS to tolerate fresh Lakebase deploys where the
    # reverse-CDC sync hasn't materialised the schema yet (see migration
    # 20260224_000006).
    op.execute("CREATE SCHEMA IF NOT EXISTS lakercm")
    op.execute("GRANT USAGE ON SCHEMA lakercm TO PUBLIC")
    op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA lakercm TO PUBLIC")


def downgrade():
    op.execute("REVOKE SELECT ON ALL TABLES IN SCHEMA lakercm FROM PUBLIC")
    op.execute("REVOKE USAGE ON SCHEMA lakercm FROM PUBLIC")
