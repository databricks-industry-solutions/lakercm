"""create document_review_proposals table for agent-assisted review

Revision ID: 20260928_000033
Revises: 20260926_000032
Create Date: 2026-09-28 00:00:33.000000

Append-only record of every fix the agent proposed for a held document, and
what the human did with it.

Why a separate table rather than columns on document_extraction_reviews: that
table is deliberately ONE row per document with ON CONFLICT DO UPDATE, so its
`id` survives and CDC emits an UPDATE on the same PK for the gold sync. It
holds the current verdict, not a history. Proposals need the opposite — every
attempt kept, including the ones that were rejected — so they live here and
join on document_id.

Without this table the analytics are impossible: once a reviewer approves a
staged card, the resulting correction is byte-identical to one they typed
themselves, so agent involvement leaves no trace and "was the proposal any
good?" cannot be asked.

Two integrity rules are enforced here rather than in prose:

- A ``not_resolvable`` row cannot carry a ``proposed_value``. This is what
  makes the missing_member_id refusal structural: a member ID cannot be
  derived, and the schema now makes it impossible to record a guess at one,
  whatever a caller (or a future prompt) tries to write.
- ``disposition`` and ``resolution`` are constrained to their known values, so
  a typo becomes an error at write time instead of a silently uncounted row in
  the acceptance-rate query.

The reviewer app owns writes (it owns the review path and the audit identity).
The agent app only READS, to see what has already been proposed on the document
in front of it — so this grants SELECT, matching migration 000029.
"""

import os

from alembic import op

# revision identifiers, used by Alembic
revision = "20260928_000033"
down_revision = "20260926_000032"
branch_labels = None
depends_on = None

# Agent Postgres identities that need read access. The group role is the
# primary (PGUSER); the SP is the fallback for the pre-group-role config. Same
# env vars the earlier agent-grant migrations read.
CHECKPOINT_ROLE = os.environ.get("CHECKPOINT_ROLE_NAME", "").strip()
AGENT_SP = os.environ.get("AGENT_SP_CLIENT_ID", "").strip()

# Kept in sync with services/remediation.py and services/review_proposals.py.
RESOLUTIONS = ("deterministic", "needs_judgment", "not_resolvable")
DISPOSITIONS = (
    "pending",  # staged, awaiting the reviewer
    "accepted",  # applied as proposed
    "modified",  # applied, but the reviewer changed the value
    "rejected",  # reviewer declined it
    "declined",  # the AGENT declined to propose (not_resolvable)
    "superseded",  # a newer proposal replaced it
)


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
        print(
            f"[migration 20260928_000033] role {role!r} absent — "
            "skipping SELECT grant on public.document_review_proposals."
        )
        return
    quoted = f'"{role}"'
    op.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}")
    op.execute(f"GRANT SELECT ON public.document_review_proposals TO {quoted}")


def _in_list(column: str, values) -> str:
    joined = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({joined})"


def upgrade():
    # Drop if pre-created outside Alembic (mirrors create_document_notes).
    op.execute("DROP TABLE IF EXISTS public.document_review_proposals")
    op.execute(f"""
        CREATE TABLE public.document_review_proposals (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL,

            -- Why the document was held (gold_extraction_labels.review_reasons).
            review_reason VARCHAR(64) NOT NULL,
            -- How much judgment the terminology left (services.remediation).
            resolution VARCHAR(32) NOT NULL,

            -- What was wrong. correction_key is the reviewer UI's id:<n> key,
            -- which is POSITIONAL into the identifiers array: field_name and
            -- observed_value are stored alongside it so a proposal can be
            -- re-validated before staging if extraction re-ran and reindexed.
            field_name TEXT,
            correction_key VARCHAR(32),
            observed_value TEXT,

            -- What was proposed, and the shortlist it was chosen from.
            proposed_value TEXT,
            rationale TEXT,
            candidates JSONB,

            -- Provenance.
            source VARCHAR(32) NOT NULL,
            model VARCHAR(128),
            proposed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            -- Withheld from the reviewer on purpose: the control slice that
            -- makes "the agent made review faster" measurable instead of
            -- asserted. Computed and scored like any other proposal, never
            -- shown, so its document's turnaround is un-assisted.
            withheld BOOLEAN NOT NULL DEFAULT FALSE,

            -- What the human did.
            disposition VARCHAR(32) NOT NULL DEFAULT 'pending',
            disposition_at TIMESTAMPTZ,
            disposition_by VARCHAR(320),
            human_value TEXT,

            CONSTRAINT ck_review_proposals_resolution
                CHECK ({_in_list("resolution", RESOLUTIONS)}),
            CONSTRAINT ck_review_proposals_disposition
                CHECK ({_in_list("disposition", DISPOSITIONS)}),
            -- A refusal cannot carry a value. This is the missing_member_id
            -- guarantee expressed as a constraint.
            CONSTRAINT ck_review_proposals_no_value_when_unresolvable
                CHECK (resolution <> 'not_resolvable' OR proposed_value IS NULL)
        )
        """)
    # Required for the Lakebase -> Unity Catalog change data feed. The CDF
    # config covers the whole `public` schema, but it SKIPS any table without
    # REPLICA IDENTITY FULL and does not re-check later — so without this the
    # table never reaches lb_document_review_proposals_history, and the
    # acceptance analytics would have no source, silently. Same reason
    # migrations 000014 / 000018 set it on the other two fed tables.
    op.execute("ALTER TABLE public.document_review_proposals REPLICA IDENTITY FULL")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_review_proposals_doc "
        "ON public.document_review_proposals(document_id)"
    )
    # The triage job and the queue badge both ask "what is still pending?".
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_review_proposals_pending "
        "ON public.document_review_proposals(disposition, proposed_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_review_proposals_reason "
        "ON public.document_review_proposals(review_reason)"
    )
    # Permissive policy so all authenticated roles keep access if RLS is ever
    # enabled on this table (matches public.document_notes).
    op.execute(
        "CREATE POLICY lakercm_app_full_access "
        "ON public.document_review_proposals "
        "FOR ALL TO PUBLIC "
        "USING (true) WITH CHECK (true)"
    )
    _grant_select(CHECKPOINT_ROLE)
    _grant_select(AGENT_SP)


def downgrade():
    for role in (CHECKPOINT_ROLE, AGENT_SP):
        if _role_exists(role):
            op.execute(
                "REVOKE SELECT ON public.document_review_proposals "
                f'FROM "{role}"'
            )
    op.execute("DROP INDEX IF EXISTS idx_review_proposals_reason")
    op.execute("DROP INDEX IF EXISTS idx_review_proposals_pending")
    op.execute("DROP INDEX IF EXISTS idx_review_proposals_doc")
    op.execute("DROP TABLE IF EXISTS public.document_review_proposals")
