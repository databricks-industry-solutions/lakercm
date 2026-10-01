"""merge the document_notes and document_embeddings heads

Revision ID: 20260924_000031
Revises: 20260913_000029, 20260914_000030
Create Date: 2026-09-24 00:00:31.000000

Migrations 000029 (document_notes) and 000030 (document_embeddings) were both
written against 000028, which left two heads. `alembic upgrade head` then
failed with "Multiple head revisions are present", and the reviewer app's boot
ran no migration at all: on the new dev branch no table was created, so the
Lakebase change data feed had nothing to feed. The two touch different tables,
so this revision only joins the branches. A database that has applied either
one, both, or neither upgrades cleanly; relinking 000030 after 000029 instead
would have stranded one that applied 000030 alone.
"""

revision = "20260924_000031"
down_revision = ("20260913_000029", "20260914_000030")
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
