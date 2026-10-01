"""re-assert REPLICA IDENTITY FULL on CDC-sourced tables

Revision ID: 20260421_000018
Revises: 20260420_000017
Create Date: 2026-04-21 00:00:18.000000

Migration 14 set REPLICA IDENTITY FULL on medical_documents and
document_extraction_reviews, but at least one Lakebase instance had its
CDC-to-UC sink created before that migration ran. The sink silently
skips tables without REPLICA IDENTITY FULL and does not re-check when
the setting is changed post-hoc, so the medical_documents Change
History stalled while reviews flowed through.

Fix: run the ALTER again so instances currently past migration 14 with
missing/downgraded identity settings converge back to FULL. ALTER TABLE
... REPLICA IDENTITY FULL is idempotent at the Postgres level, but the
downstream CDC sink may still need a manual resync/recreate on the
Databricks side to pick up previously-skipped rows.
"""

from alembic import op


revision = "20260421_000018"
down_revision = "20260420_000017"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE public.medical_documents REPLICA IDENTITY FULL")
    op.execute("ALTER TABLE public.document_extraction_reviews REPLICA IDENTITY FULL")


def downgrade():
    pass
