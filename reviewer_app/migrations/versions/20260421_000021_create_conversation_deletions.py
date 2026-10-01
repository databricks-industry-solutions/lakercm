"""create conversation_deletions audit table

Revision ID: 20260421_000021
Revises: 20260421_000020
Create Date: 2026-04-21 00:00:21.000000

Audit trail for cascade-delete of conversations. One row per attempt,
success or failure, so the table doubles as a HIPAA §164.312(b) log and
a retry surface — a 'failed' row means the agent-side cleanup errored
and the reviewer's conversations row was intentionally preserved.

Mirrors the RLS pattern from 20260316_000007 and 20260331_000008 so the
app service principal can INSERT via the PUBLIC role.
"""

from alembic import op


revision = "20260421_000021"
down_revision = "20260421_000020"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("DROP TABLE IF EXISTS public.conversation_deletions")
    op.execute(
        """
        CREATE TABLE public.conversation_deletions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id VARCHAR(64) NOT NULL,
            user_email VARCHAR(320) NOT NULL,
            deleted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            agent_cleanup_status VARCHAR(32) NOT NULL,
            reviewer_rowcount INT NOT NULL DEFAULT 0,
            error_detail TEXT
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_del_user "
        "ON public.conversation_deletions(user_email)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_del_time "
        "ON public.conversation_deletions(deleted_at DESC)"
    )
    op.execute(
        "CREATE POLICY lakercm_app_full_access "
        "ON public.conversation_deletions "
        "FOR ALL TO PUBLIC "
        "USING (true) WITH CHECK (true)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_conv_del_time")
    op.execute("DROP INDEX IF EXISTS idx_conv_del_user")
    op.execute("DROP TABLE IF EXISTS public.conversation_deletions")
