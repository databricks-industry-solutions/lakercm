"""grant SELECT on medical_documents and document_extraction_reviews to the agent SP

Revision ID: 20260419_000011
Revises: 20260417_000010
Create Date: 2026-04-19 00:00:11.000000

The standalone chat-agent app runs under its own service principal and
needs read access to these two reviewer-owned tables. Scoped to the
agent SP specifically rather than PUBLIC to avoid widening data access
on shared Lakebase.

The agent SP UUID is read from AGENT_SP_CLIENT_ID at migration time
(injected via the bundle's app config.env). This makes the migration
portable across workspaces that have different SP UUIDs.

If the SP role doesn't exist yet (fresh Lakebase, role registration
pending), grants are skipped — this is a no-op until the SP is added
as a Postgres role via scripts/add_lakebase_roles.py + the next app
restart.

This migration is a no-op if run by the reviewer SP without
grant-option on the target tables. In that case grants must be
applied out-of-band by the table owner.
"""

import os

from alembic import op


revision = "20260419_000011"
down_revision = "20260417_000010"
branch_labels = None
depends_on = None

# Read the live agent SP at migration time. Fall back to empty so the
# migration becomes a tolerated no-op on workspaces where the env var
# isn't set (legacy local dev paths).
AGENT_SP = os.environ.get("AGENT_SP_CLIENT_ID", "").strip()


def _role_exists(role: str) -> bool:
    """True if a Postgres role exists in the current database."""
    if not role:
        return False
    bind = op.get_bind()
    res = bind.exec_driver_sql(
        "SELECT 1 FROM pg_roles WHERE rolname = %(r)s", {"r": role}
    )
    return res.scalar() is not None


def upgrade():
    if not AGENT_SP:
        # No SP configured — nothing to grant. Migration becomes a no-op.
        return
    if not _role_exists(AGENT_SP):
        # SP not yet a Postgres role. Skip; next deploy + app restart
        # picks it up after add_lakebase_roles.py runs.
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
