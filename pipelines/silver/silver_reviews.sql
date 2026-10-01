-- =============================================================================
-- Silver Layer: Extraction Reviews (current state from the Lakebase CDC mirror)
-- =============================================================================
-- One row per review id ≡ one row per reviewed document (the source upserts
-- on document_id preserving id + created_at, so verdict overrides arrive as
-- CDC UPDATEs on the same id). Same full-recompute QUALIFY dedupe as
-- silver_documents — idempotent against the double-written initial snapshot.
--
-- Deliberately NO document join here: /api/analytics/reviewers must keep
-- listing reviewers whose only reviews are on soft-deleted documents
-- (parity with the old Lakebase get_distinct_reviewers).
--
-- created_at = first-review time (immutable in Postgres across overrides);
-- updated_at = last verdict/corrections change. updated_at > created_at is
-- NOT evidence of an override (a 2026-04-21 CDC-backfill migration touched
-- every row) — override detection lives in gold_fact_review via verdict
-- change-detection over the raw event log.
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW silver_reviews (

  CONSTRAINT valid_verdict
    EXPECT (verdict IN ('correct', 'partially_correct', 'incorrect')),

  CONSTRAINT uuid_shaped_id
    EXPECT (id RLIKE '^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')
    ON VIOLATION DROP ROW,

  CONSTRAINT has_document_id
    EXPECT (document_id IS NOT NULL)
)
COMMENT 'Current state of Lakebase document_extraction_reviews: latest CDC event per review id; one row per reviewed document.'
TBLPROPERTIES (
  'quality' = 'silver',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH latest AS (
  SELECT *
  FROM ${cdc_reviews_table}
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY id ORDER BY _pg_lsn DESC, _sort_by DESC
  ) = 1
)

SELECT
  lower(regexp_replace(hex(id), '(.{8})(.{4})(.{4})(.{4})(.{12})', '$1-$2-$3-$4-$5'))          AS id,
  lower(regexp_replace(hex(document_id), '(.{8})(.{4})(.{4})(.{4})(.{12})', '$1-$2-$3-$4-$5')) AS document_id,
  reviewer_email,
  verdict,
  reasoning,
  corrections,
  COALESCE(is_automated, FALSE)    AS is_automated,
  created_at,
  updated_at,
  _pg_change_type                  AS _last_change_type,
  _pg_lsn                          AS _last_lsn,
  CAST(_timestamp AS TIMESTAMP)    AS _synced_at
FROM latest
WHERE _pg_change_type <> 'delete'
