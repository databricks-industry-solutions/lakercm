"""re-grant schema + table privileges to lakercm-checkpoint-runtime group role

Revision ID: 20260530_000028
Revises: 20260528_000027
Create Date: 2026-05-30 00:00:28.000000

Why this exists: migration 20260528_000027 granted these same privileges,
but a deploy's `scripts/add_lakebase_roles.py` reconcile step
DELETE+RECREATED the group role `lakercm-checkpoint-runtime` (its
binding had gone stale, `postgres_role=None`). Recreating a Postgres role
drops every GRANT it held — so the recreated role lost the SELECT grants
027 had applied. Alembic had already marked 027 applied, so
`upgrade head` would not re-run it, leaving the role stranded without
`public.*` SELECT and the agent hitting:

    psycopg.errors.InsufficientPrivilege: permission denied for table
    document_extraction_reviews   (in get_review_statistics)

This is a fresh revision (028) that re-applies the exact grants from 027.
GRANT statements are idempotent (no-op when the privilege already exists),
so this is safe even on environments where the role was never dropped.

Durability note: now that the role is bound correctly,
`add_lakebase_roles.py` skips recreation on future deploys
(it only delete+recreates on a stale binding), so this re-grant holds.
If the role is ever recreated again, add another re-grant revision or
move the grants into role provisioning (run as a PG superuser). See the
plan's "Durability follow-up".

The role name comes from env CHECKPOINT_ROLE_NAME (set in the bundle app
config, bundles/apps/resources/app.yml, for the boot path; the offline runner
defaults it to the stable constant). Empty/unset -> skip with a log.
"""

import os

from alembic import op

revision = "20260530_000028"
down_revision = "20260528_000027"
branch_labels = None
depends_on = None

ROLE_NAME = os.environ.get("CHECKPOINT_ROLE_NAME", "").strip()

# Source of truth: same list as
# reviewer_app/services/lakehouse_db.py:_AGENT_READ_TABLES
# and migration 20260528_000027.
AGENT_READ_TABLES = (
    "medical_documents",
    "document_extraction_reviews",
    "conversations",
    "lakebase_events",
)


def upgrade():
    if not ROLE_NAME:
        print(
            "[migration 20260530_000028] CHECKPOINT_ROLE_NAME unset — "
            "skipping group-role re-grant. The agent will lack public.* "
            "SELECT until this runs with the role name set."
        )
        return

    quoted = f'"{ROLE_NAME}"'  # hyphens in group display name → must double-quote
    op.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}")
    op.execute(f"GRANT CREATE ON SCHEMA public TO {quoted}")
    for table in AGENT_READ_TABLES:
        op.execute(f"GRANT SELECT ON public.{table} TO {quoted}")


def downgrade():
    # No-op: this revision only re-applies grants that 20260528_000027
    # is responsible for. Revoking here would undo 027's intent.
    pass
