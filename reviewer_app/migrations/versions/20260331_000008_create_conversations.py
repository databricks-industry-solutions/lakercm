"""create conversations table for chat agent memory

Revision ID: 20260331_000008
Revises: 20260316_000007
Create Date: 2026-03-31 00:00:08.000000

Stores chat conversation messages in Lakebase for the LakeRCM Assistant.
Each row is a single message (user or assistant) within a conversation.
"""

from alembic import op

# revision identifiers, used by Alembic
revision = "20260331_000008"
down_revision = "20260316_000007"
branch_labels = None
depends_on = None


def upgrade():
    # Drop table if it was pre-created outside Alembic (e.g. by _ensure_tables)
    op.execute("DROP TABLE IF EXISTS public.conversations")
    op.execute(
        """
        CREATE TABLE public.conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id VARCHAR(64) NOT NULL,
            user_email VARCHAR(320) NOT NULL,
            document_id UUID,
            title VARCHAR(200),
            message_role VARCHAR(20) NOT NULL,
            message_content TEXT NOT NULL,
            tool_calls TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_user "
        "ON public.conversations(user_email)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_conv "
        "ON public.conversations(conversation_id)"
    )
    # RLS policy so all authenticated roles (including the app SP) can access
    op.execute(
        "CREATE POLICY lakercm_app_full_access "
        "ON public.conversations "
        "FOR ALL TO PUBLIC "
        "USING (true) WITH CHECK (true)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_conversations_conv")
    op.execute("DROP INDEX IF EXISTS idx_conversations_user")
    op.execute("DROP TABLE IF EXISTS public.conversations")
