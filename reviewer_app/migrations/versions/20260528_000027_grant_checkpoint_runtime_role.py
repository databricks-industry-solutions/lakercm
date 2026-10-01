"""grant schema + table privileges to the lakercm-checkpoint-runtime group role

Revision ID: 20260528_000027
Revises: 20260508_000026
Create Date: 2026-05-28 00:00:27.000000

Why this exists: replaces `_grant_agent_role_membership()` that used to run
on every reviewer-app pod boot inside `services/lakehouse_db.py.__init__`.
Apps mutating their own data layer on every startup is an anti-pattern
(MLflow trace pollution, scale-up races, audit-log noise); the canonical
home for these grants is alembic, which runs once at deploy time via the
reviewer-app's `_run_alembic_upgrade()` startup hook.

The group role `lakercm-checkpoint-runtime` is a workspace-group-backed
Lakebase Postgres role (registered via `scripts/add_lakebase_roles.py` with
`RoleIdentityType.GROUP`). The agent app's connection sessions-as this
group role (PGUSER set in bundles/apps/resources/agent_app.yml) so SP rotations don't
require re-granting — group membership is the stable identity.

Ordering constraint: this migration assumes the group role already
exists. The lakebase bundle's post-deploy step (scripts/deploy_all.py ->
_deploy_lib provision_lakebase_roles) invokes `scripts/add_lakebase_roles.py`
BEFORE the apps bundle is deployed, so by the time alembic runs at
reviewer-app boot the role exists. If anyone re-orders deploy_all to run apps
before role provisioning, this migration
will fail with `role "lakercm-checkpoint-runtime" does not exist`.

Idempotent: GRANT statements are no-ops when the privilege already exists.

The role name comes from env (the bundle app config sets
`CHECKPOINT_ROLE_NAME` in bundles/apps/resources/app.yml). Empty/unset →
skip with a log, same fallback semantics as the runtime function had.
"""

import os

from alembic import op

revision = "20260528_000027"
down_revision = "20260508_000026"
branch_labels = None
depends_on = None

ROLE_NAME = os.environ.get("CHECKPOINT_ROLE_NAME", "").strip()

# Source of truth: same list as
# reviewer_app/services/lakehouse_db.py:_AGENT_READ_TABLES.
AGENT_READ_TABLES = (
    "medical_documents",
    "document_extraction_reviews",
    "conversations",
    "lakebase_events",
)


def upgrade():
    if not ROLE_NAME:
        print(
            "[migration 20260528_000027] CHECKPOINT_ROLE_NAME unset — "
            "skipping group-role grants. Agent will fall back to the "
            "per-SP grants from migration 20260508_000026."
        )
        return

    quoted = f'"{ROLE_NAME}"'  # hyphens in group display name → must double-quote
    op.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}")
    op.execute(f"GRANT CREATE ON SCHEMA public TO {quoted}")
    for table in AGENT_READ_TABLES:
        op.execute(f"GRANT SELECT ON public.{table} TO {quoted}")


def downgrade():
    if not ROLE_NAME:
        return
    quoted = f'"{ROLE_NAME}"'
    for table in AGENT_READ_TABLES:
        op.execute(f"REVOKE SELECT ON public.{table} FROM {quoted}")
    op.execute(f"REVOKE CREATE ON SCHEMA public FROM {quoted}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {quoted}")
