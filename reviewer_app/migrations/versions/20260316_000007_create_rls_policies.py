"""create RLS policies for app service principal

Revision ID: 20260316_000007
Revises: 20260224_000006
Create Date: 2026-03-16 00:00:07.000000

When RLS is enabled on tables, non-owner roles are denied all row access
unless explicit policies exist. This migration creates permissive policies
that grant all authenticated roles (including the app service principal)
full access to the app's tables.
"""

from alembic import op

revision = "20260316_000007"
down_revision = "20260224_000006"
branch_labels = None
depends_on = None


def upgrade():
    # Drop first to be idempotent (policies may already exist from manual setup)
    for table in (
        "medical_documents",
        "document_extraction_reviews",
        "alembic_version",
    ):
        op.execute(
            f"DROP POLICY IF EXISTS lakercm_app_full_access ON public.{table}"
        )
        op.execute(
            f"CREATE POLICY lakercm_app_full_access "
            f"ON public.{table} "
            f"FOR ALL TO PUBLIC "
            f"USING (true) WITH CHECK (true)"
        )


def downgrade():
    op.execute(
        "DROP POLICY IF EXISTS lakercm_app_full_access "
        "ON public.medical_documents"
    )
    op.execute(
        "DROP POLICY IF EXISTS lakercm_app_full_access "
        "ON public.document_extraction_reviews"
    )
    op.execute(
        "DROP POLICY IF EXISTS lakercm_app_full_access " "ON public.alembic_version"
    )
