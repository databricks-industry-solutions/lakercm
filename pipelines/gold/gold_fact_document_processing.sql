-- =============================================================================
-- Gold Layer: Document Processing Fact (accumulating snapshot)
-- =============================================================================
-- Grain: one row per non-soft-deleted document, all lifecycle statuses. Holds
-- the milestone timestamps and lag measures:
--   pipeline_seconds          upload → extraction complete; NULL unless both
--                             timestamps exist AND processing >= upload
--                             (exactly the old get_extraction_pipeline_metrics
--                             guards, so COUNT(pipeline_seconds) ≡ old total)
--   review_turnaround_seconds extraction complete → FIRST human review
--                             (created_at is immutable across overrides, so
--                             MIN over silver_reviews = first review)
-- unix_millis/1000.0 keeps the fractional seconds Postgres EXTRACT(EPOCH)
-- produced. Also feeds the document_ops_metrics metric view (status mix,
-- auto-verification rate, backlog age).
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW fact_document_processing
COMMENT 'Accumulating snapshot per document: lifecycle milestones, pipeline latency, and human-review turnaround. Excludes soft-deleted documents.'
TBLPROPERTIES (
  'quality' = 'gold',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH first_human_review AS (
  SELECT document_id, MIN(created_at) AS first_human_reviewed_at
  FROM silver_reviews
  WHERE NOT is_automated
  GROUP BY document_id
)

SELECT
  d.id                             AS document_id,
  d.document_type,
  d.user_email                     AS uploader_email,
  d.processing_status              AS current_status,
  d.upload_timestamp,
  d.processing_timestamp,
  fhr.first_human_reviewed_at,
  CASE WHEN d.upload_timestamp IS NOT NULL
        AND d.processing_timestamp IS NOT NULL
        AND d.processing_timestamp >= d.upload_timestamp
       THEN (unix_millis(d.processing_timestamp) - unix_millis(d.upload_timestamp)) / 1000.0
  END                              AS pipeline_seconds,
  CASE WHEN fhr.first_human_reviewed_at IS NOT NULL
        AND d.processing_timestamp IS NOT NULL
       THEN (unix_millis(fhr.first_human_reviewed_at) - unix_millis(d.processing_timestamp)) / 1000.0
  END                              AS review_turnaround_seconds
FROM silver_documents d
LEFT JOIN first_human_review fhr
  ON fhr.document_id = d.id
WHERE d.deleted_at IS NULL
