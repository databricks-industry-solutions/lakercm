"""create document_notes table for the reviewer notepad

Revision ID: 20260913_000029
Revises: 20260530_000028
Create Date: 2026-09-13 00:00:29.000000

Per-(document, reviewer) private notepad backing the in-document assistant's
notepad feature. Distinct from public.document_extraction_reviews, which is a
single shared verdict row per document: notes are personal scratch context and
never feed the accuracy stats or the gold sync.

The reviewer app (table owner) reads and writes it. The agent app only READS it
(via the get_active_review_context tool, to show the reviewer's notes to the
in-document assistant), so this grants SELECT — not write — to the agent's
Postgres identity. The agent sessions-as the checkpoint group role
(PGUSER=${var.checkpoint_role_name}); we also grant the agent SP directly as a
fallback for the pre-group-role config, mirroring migrations 000011 / 000027.
Both grants are guarded so they no-op on a fresh workspace where the roles
don't exist yet.
"""

import os

from alembic import op

# revision identifiers, used by Alembic
revision = "20260913_000029"
down_revision = "20260530_000028"
branch_labels = None
depends_on = None

# Agent Postgres identities that need read access to the notepad. The group
# role is the primary (PGUSER); the SP is the fallback. Same env vars the
# earlier agent-grant migrations read.
CHECKPOINT_ROLE = os.environ.get("CHECKPOINT_ROLE_NAME", "").strip()
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


def _grant_select(role: str) -> None:
    if not _role_exists(role):
        # Role not registered yet (fresh Lakebase). Skipped; a later re-grant
        # migration or the next deploy after role provisioning picks it up.
        print(
            f"[migration 20260913_000029] role {role!r} absent — "
            "skipping SELECT grant on public.document_notes."
        )
        return
    quoted = f'"{role}"'
    op.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}")
    op.execute(f"GRANT SELECT ON public.document_notes TO {quoted}")


def upgrade():
    # Drop if pre-created outside Alembic (mirrors create_conversations).
    op.execute("DROP TABLE IF EXISTS public.document_notes")
    op.execute("""
        CREATE TABLE public.document_notes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL,
            user_email VARCHAR(320) NOT NULL,
            note_text TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_document_notes_doc_user UNIQUE (document_id, user_email)
        )
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_notes_doc "
        "ON public.document_notes(document_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_notes_user "
        "ON public.document_notes(user_email)"
    )
    # Permissive policy so all authenticated roles keep access if RLS is ever
    # enabled on this table (matches public.conversations).
    op.execute(
        "CREATE POLICY lakercm_app_full_access "
        "ON public.document_notes "
        "FOR ALL TO PUBLIC "
        "USING (true) WITH CHECK (true)"
    )
    # Read-only access for the agent app's Postgres identities.
    _grant_select(CHECKPOINT_ROLE)
    _grant_select(AGENT_SP)


def downgrade():
    for role in (CHECKPOINT_ROLE, AGENT_SP):
        if _role_exists(role):
            op.execute(f'REVOKE SELECT ON public.document_notes FROM "{role}"')
    op.execute("DROP INDEX IF EXISTS idx_document_notes_user")
    op.execute("DROP INDEX IF EXISTS idx_document_notes_doc")
    op.execute("DROP TABLE IF EXISTS public.document_notes")
