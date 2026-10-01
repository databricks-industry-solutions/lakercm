"""add trace_id to conversations for persistent thumbs feedback

Revision ID: 20260429_000023
Revises: 20260421_000022
Create Date: 2026-04-29 00:00:23.000000

The MLflow trace_id is captured per assistant turn from the SSE stream and
must be stored alongside the message so reloading a conversation preserves
the thumbs-up/down affordance on past turns. Without this column, only
just-streamed messages can be rated.
"""

from alembic import op


# revision identifiers, used by Alembic
revision = "20260429_000023"
down_revision = "20260421_000022"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE public.conversations "
        "ADD COLUMN IF NOT EXISTS trace_id VARCHAR(128)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_trace_id "
        "ON public.conversations(trace_id)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_conversations_trace_id")
    op.execute("ALTER TABLE public.conversations DROP COLUMN IF EXISTS trace_id")
