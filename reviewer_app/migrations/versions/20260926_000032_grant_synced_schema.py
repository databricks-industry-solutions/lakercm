"""grant USAGE + SELECT on this deploy's synced-gold schema

Revision ID: 20260926_000032
Revises: 20260924_000031
Create Date: 2026-09-26 00:00:32.000000

A synced table takes its Postgres schema from its Unity Catalog schema, so the
dev deploy's copy of gold_extraction_labels is
`lakercm_dev.gold_extraction_labels_sync` while prod's is `lakercm.*`.
Migrations 000006 and 000013 grant on the literal `lakercm` only, which is
why they applied cleanly on dev and left the dev schema unreadable.

This grants the schema named by LAKERCM_SCHEMA, which the bundle sets per
target (bundles/apps/resources/app.yml). On prod that schema IS `lakercm`,
so this repeats what 000013 already did — harmless, and it makes a recreated
synced table readable again without a new revision each time, since the sync
service owns the table it creates and ALTER DEFAULT PRIVILEGES never reaches it.

CREATE SCHEMA IF NOT EXISTS comes first for the same reason 000013 has it: on a
fresh project the app can boot before the sync has materialized anything, and
GRANT on a missing schema is an error.
"""

import os

from alembic import op

revision = "20260926_000032"
down_revision = "20260924_000031"
branch_labels = None
depends_on = None

SCHEMA = os.environ.get("LAKERCM_SCHEMA", "lakercm").strip() or "lakercm"


def upgrade():
    # Quoted: a schema name comes from configuration, not from a literal here.
    schema = f'"{SCHEMA}"'
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    op.execute(f"GRANT USAGE ON SCHEMA {schema} TO PUBLIC")
    op.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO PUBLIC")
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT SELECT ON TABLES TO PUBLIC"
    )


def downgrade():
    schema = f'"{SCHEMA}"'
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} REVOKE SELECT ON TABLES FROM PUBLIC"
    )
    op.execute(f"REVOKE SELECT ON ALL TABLES IN SCHEMA {schema} FROM PUBLIC")
    op.execute(f"REVOKE USAGE ON SCHEMA {schema} FROM PUBLIC")
