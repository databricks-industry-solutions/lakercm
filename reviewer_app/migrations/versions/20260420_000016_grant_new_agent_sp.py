"""grant SELECT on medical_documents and document_extraction_reviews to the
current agent SP (resolved at runtime from AGENT_SP_CLIENT_ID).

Revision ID: 20260420_000016
Revises: 20260421_000015
Create Date: 2026-04-20 00:00:16.000000

The prior agent app was torn down and replaced, which gave it a new service
principal. Migration 000011 granted access to the now-deleted SP, so
after the rebuild the new agent SP had no read access to reviewer tables.
This migration grants the same scope to whichever SP is the current agent.

The SP UUID is read from the AGENT_SP_CLIENT_ID env var (set by the bundle's
config.env block per-target). This keeps the migration portable across
workspaces that have different agent SPs.

Note: like 000011, this migration is a no-op if run by a principal without
grant-option on these tables. In that case the grants must be applied
out-of-band by the table owner.
"""

import os

from alembic import op


revision = "20260420_000016"
down_revision = "20260421_000015"
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
    op.execute(f'GRANT SELECT ON public.medical_documents TO "{AGENT_SP}"')
    op.execute(f'GRANT SELECT ON public.document_extraction_reviews TO "{AGENT_SP}"')


def downgrade():
    if not AGENT_SP or not _role_exists(AGENT_SP):
        return
    op.execute(f'REVOKE SELECT ON public.document_extraction_reviews FROM "{AGENT_SP}"')
    op.execute(f'REVOKE SELECT ON public.medical_documents FROM "{AGENT_SP}"')
    op.execute(f'REVOKE USAGE ON SCHEMA public FROM "{AGENT_SP}"')
