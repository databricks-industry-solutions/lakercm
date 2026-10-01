"""grant usage on lakercm schema to public

Revision ID: 20260224_000006
Revises: 20260220_000005
Create Date: 2026-02-24 00:00:06.000000

The synced table gold_extraction_labels_synced lives in the lakercm
schema in Lakebase. Grant USAGE + SELECT to PUBLIC so the app user can
query it regardless of which OAuth identity connects.
"""

from alembic import op

revision = "20260224_000006"
down_revision = "20260220_000005"
branch_labels = None
depends_on = None


def upgrade():
    # On a fresh Lakebase database, the `lakercm` schema doesn't exist
    # until reverse-CDC sync materialises it. Create-if-absent so the
    # migration runs on first deploys (the synced tables, when they later
    # land, will populate this schema; until then the GRANTs are no-ops
    # against an empty schema).
    op.execute("CREATE SCHEMA IF NOT EXISTS lakercm")
    op.execute("GRANT USAGE ON SCHEMA lakercm TO PUBLIC")
    op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA lakercm TO PUBLIC")
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA lakercm "
        "GRANT SELECT ON TABLES TO PUBLIC"
    )


def downgrade():
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA lakercm "
        "REVOKE SELECT ON TABLES FROM PUBLIC"
    )
    op.execute("REVOKE SELECT ON ALL TABLES IN SCHEMA lakercm FROM PUBLIC")
    op.execute("REVOKE USAGE ON SCHEMA lakercm FROM PUBLIC")
