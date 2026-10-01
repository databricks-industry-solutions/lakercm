"""re-grant reviewer-table SELECT + schema CREATE to the current agent SP

Revision ID: 20260508_000026
Revises: 20260506_000025
Create Date: 2026-05-08 00:00:26.000000

Why this exists: the agent app's service principal was replaced with a new
one. Migrations 000016
and 000020 granted USAGE / SELECT / CREATE to whatever AGENT_SP_CLIENT_ID
was set when they ran — they ran with the OLD SP UUID and are now marked
applied in alembic_version, so they will not re-run after the SP rotation.
The new agent SP therefore had no read access to public.medical_documents
or public.document_extraction_reviews, which broke the agent's tool calls
with "permission denied for table document_extraction_reviews".

This migration restates the same grants for the *current* AGENT_SP_CLIENT_ID
(read at runtime from env, set by the bundle's app.yaml render). Same
shape as 000016 + 000020 fused so future SP rotations have a single
follow-up migration to copy.

Idempotent: running GRANT USAGE / GRANT SELECT a second time on a principal
that already has the privilege is a no-op in PostgreSQL.
"""

import os

from alembic import op

revision = "20260508_000026"
down_revision = "20260506_000025"
branch_labels = None
depends_on = None

AGENT_SP = os.environ["AGENT_SP_CLIENT_ID"]


def upgrade():
    # Schema access (also lets the LangGraph checkpointer bootstrap its
    # tables on a fresh instance when needed).
    op.execute(f'GRANT USAGE ON SCHEMA public TO "{AGENT_SP}"')
    op.execute(f'GRANT CREATE ON SCHEMA public TO "{AGENT_SP}"')

    # Reviewer tables the agent's tools read from. SELECT only — the agent
    # never mutates reviewer data; review writes go through the reviewer
    # app's own SP (REVIEWER_SP_CLIENT_ID).
    op.execute(f'GRANT SELECT ON public.medical_documents TO "{AGENT_SP}"')
    op.execute(f'GRANT SELECT ON public.document_extraction_reviews TO "{AGENT_SP}"')
    # Chat memory + lifecycle tables the agent reads when surfacing
    # conversation history or recent state transitions to users.
    op.execute(f'GRANT SELECT ON public.conversations TO "{AGENT_SP}"')
    op.execute(f'GRANT SELECT ON public.lakebase_events TO "{AGENT_SP}"')


def downgrade():
    op.execute(f'REVOKE SELECT ON public.lakebase_events FROM "{AGENT_SP}"')
    op.execute(f'REVOKE SELECT ON public.conversations FROM "{AGENT_SP}"')
    op.execute(f'REVOKE SELECT ON public.document_extraction_reviews FROM "{AGENT_SP}"')
    op.execute(f'REVOKE SELECT ON public.medical_documents FROM "{AGENT_SP}"')
    op.execute(f'REVOKE CREATE ON SCHEMA public FROM "{AGENT_SP}"')
    # Leave USAGE on schema public — other roles may still need it.
