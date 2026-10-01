"""create medical_documents table with content_hash

Revision ID: 20260220_000001
Revises: None
Create Date: 2026-02-20 00:00:01.000000

Initial LakeRCM schema: medical_documents table for document tracking.
"""

from alembic import op

# revision identifiers, used by Alembic
revision = "20260220_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create medical_documents table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS public.medical_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_email VARCHAR(320) NOT NULL,
            document_name VARCHAR(255) NOT NULL,
            file_path VARCHAR(1000) NOT NULL UNIQUE,
            file_size BIGINT NOT NULL,
            document_type VARCHAR(50),
            notes TEXT,
            content_hash VARCHAR(64),
            processing_status VARCHAR(20) DEFAULT 'pending'
                CHECK (processing_status IN ('pending', 'processing', 'ready', 'failed')),
            processing_error TEXT,
            num_pages INTEGER,
            element_count INTEGER,
            has_medical_entities BOOLEAN DEFAULT false,
            upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processing_timestamp TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            deleted_at TIMESTAMPTZ
        )
    """
    )

    # Indexes
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_medical_documents_user_email
        ON public.medical_documents(user_email, upload_timestamp DESC)
        WHERE deleted_at IS NULL
    """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_medical_documents_status
        ON public.medical_documents(processing_status, upload_timestamp DESC)
        WHERE deleted_at IS NULL
    """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_medical_documents_file_path
        ON public.medical_documents(file_path)
        WHERE deleted_at IS NULL
    """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_medical_documents_user_content_hash
        ON public.medical_documents(user_email, content_hash)
        WHERE deleted_at IS NULL AND content_hash IS NOT NULL
    """
    )

    # Trigger for updated_at
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_medical_documents_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """
    )

    op.execute(
        """
        CREATE TRIGGER trigger_update_medical_documents_updated_at
        BEFORE UPDATE ON public.medical_documents
        FOR EACH ROW
        EXECUTE FUNCTION update_medical_documents_updated_at()
    """
    )


def downgrade():
    op.execute(
        "DROP TRIGGER IF EXISTS trigger_update_medical_documents_updated_at ON public.medical_documents"
    )
    op.execute("DROP FUNCTION IF EXISTS update_medical_documents_updated_at")
    op.execute("DROP INDEX IF EXISTS public.idx_medical_documents_user_content_hash")
    op.execute("DROP INDEX IF EXISTS public.idx_medical_documents_file_path")
    op.execute("DROP INDEX IF EXISTS public.idx_medical_documents_status")
    op.execute("DROP INDEX IF EXISTS public.idx_medical_documents_user_email")
    op.execute("DROP TABLE IF EXISTS public.medical_documents")
