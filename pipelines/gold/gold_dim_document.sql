-- =============================================================================
-- Gold Layer: Document Dimension
-- =============================================================================
-- Kimball dimension, grain: one row per document. Natural UUID-string key
-- (surrogate keys would be unstable across MV recomputes and buy nothing at
-- this scale). Soft-deleted documents are KEPT with is_deleted = true so no
-- fact reference can dangle; facts exclude them at build time.
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW dim_document
COMMENT 'Document dimension: one row per document, current attributes, soft deletes flagged via is_deleted.'
TBLPROPERTIES (
  'quality' = 'gold',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

SELECT
  id                               AS document_id,
  document_name,
  document_type,
  user_email                       AS uploader_email,
  file_path,
  file_size                        AS file_size_bytes,
  notes,
  content_hash,
  processing_status,
  processing_error,
  num_pages,
  element_count,
  has_medical_entities,
  upload_timestamp,
  processing_timestamp,
  created_at,
  updated_at,
  deleted_at,
  (deleted_at IS NOT NULL)         AS is_deleted
FROM silver_documents
