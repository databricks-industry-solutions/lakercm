"""create lakebase_events audit table

Revision ID: 20260506_000025
Revises: 20260429_000024
Create Date: 2026-05-06 00:00:25.000000

Stores Lakebase autoscaling endpoint state transitions captured by the
in-app `lakebase_monitor` background task. Powers the admin diagnostics
page (cold-start history, spin-up/spin-down deltas).

Mirrors the RLS pattern of conversation_deletions so the app SP can
INSERT via PUBLIC.
"""

from alembic import op


revision = "20260506_000025"
down_revision = "20260429_000024"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("DROP TABLE IF EXISTS public.lakebase_events")
    op.execute(
        """
        CREATE TABLE public.lakebase_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            endpoint_name TEXT NOT NULL,
            prev_state TEXT,
            new_state TEXT NOT NULL,
            cold_start_seconds DOUBLE PRECISION,
            metadata JSONB
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_lakebase_events_ts "
        "ON public.lakebase_events(ts DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_lakebase_events_endpoint "
        "ON public.lakebase_events(endpoint_name, ts DESC)"
    )
    op.execute(
        "CREATE POLICY lakercm_app_full_access "
        "ON public.lakebase_events "
        "FOR ALL TO PUBLIC "
        "USING (true) WITH CHECK (true)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_lakebase_events_endpoint")
    op.execute("DROP INDEX IF EXISTS idx_lakebase_events_ts")
    op.execute("DROP TABLE IF EXISTS public.lakebase_events")
