"""enable pgvector extension for agent long-term memory store

Revision ID: 20260419_000012
Revises: 20260419_000011
Create Date: 2026-04-19 00:00:12.000000

LangGraph's PostgresStore uses pgvector for semantic search over long-term
memories. The extension must be enabled on the Lakebase database before
PostgresStore.setup() can create its index. CREATE EXTENSION is idempotent.

Note: requires a role that can create extensions — in Databricks Lakebase
this typically means the database owner or a superuser. If run by a role
without privilege, this migration will fail and must be applied out-of-band.
"""

from alembic import op


revision = "20260419_000012"
down_revision = "20260419_000011"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade():
    # Intentionally not dropping the extension on downgrade — it may be in
    # use by the agent's store tables. Drop manually if truly needed.
    pass
