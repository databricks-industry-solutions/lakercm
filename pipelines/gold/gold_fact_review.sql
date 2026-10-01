-- =============================================================================
-- Gold Layer: Review Fact
-- =============================================================================
-- Grain: one CURRENT verdict per (document × review channel).
--   Human branch: latest review per document from silver_reviews.
--   Auto branch:  synthesized from processing_status = 'auto_verified' —
--                 auto verdicts are never written to the reviews table, and
--                 the synthesized shape (id 'auto-<doc>', reviewer
--                 '<automated>', verdict 'correct') is the app's existing
--                 contract for RecentReviewItem. The '<automated>' literal is
--                 frozen here; the app's AUTOMATED_REVIEWER_EMAIL env override
--                 is dead after this migration.
-- A document that was auto-verified AND later human-reviewed yields two rows
-- (one per channel) — intentional parity with the old union semantics.
--
-- Soft-deleted documents are excluded in BOTH branches at build time so no
-- downstream consumer (metric view, /recent-reviews, /processing-metrics)
-- can leak them.
--
-- Override metrics come from verdict CHANGE-DETECTION over the raw CDC event
-- log (verdict IS DISTINCT FROM its predecessor per review id), NOT from
-- updated_at — a 2026-04-21 backfill migration touched updated_at fleet-wide.
-- The mirror only logs changes since it exists, so these counts start at 0
-- and accrue real overrides going forward.
--
-- turnaround_seconds keeps fractional precision via unix_millis (parity with
-- Postgres EXTRACT(EPOCH ...)) and — matching the old SQL — has deliberately
-- NO >= guard (a review predating processing_timestamp goes negative).
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW fact_review (

  CONSTRAINT valid_verdict
    EXPECT (verdict IN ('correct', 'partially_correct', 'incorrect'))
    ON VIOLATION DROP ROW,

  CONSTRAINT has_reviewed_at
    EXPECT (reviewed_at IS NOT NULL)
)
COMMENT 'Review fact: current human verdicts ∪ synthesized auto-verified verdicts, one row per document per channel. Excludes soft-deleted documents.'
TBLPROPERTIES (
  'quality' = 'gold',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH verdict_changes AS (
  -- Change-detection over the full event log (pre-dedupe): how many times did
  -- the verdict actually change per review id? Duplicate snapshot events have
  -- identical verdicts, so they contribute nothing.
  SELECT
    lower(regexp_replace(hex(id), '(.{8})(.{4})(.{4})(.{4})(.{12})', '$1-$2-$3-$4-$5')) AS review_id,
    SUM(CASE WHEN prev_verdict IS NOT NULL AND NOT (verdict <=> prev_verdict)
             THEN 1 ELSE 0 END) AS verdict_changes_count
  FROM (
    SELECT id, verdict,
           LAG(verdict) OVER (PARTITION BY id ORDER BY _pg_lsn, _sort_by) AS prev_verdict
    FROM ${cdc_reviews_table}
    WHERE _pg_change_type <> 'delete'
  )
  GROUP BY 1
)

SELECT
  r.id                             AS review_id,
  r.document_id,
  d.document_name,
  d.document_type,
  r.reviewer_email,
  r.verdict,
  r.reasoning,
  FALSE                            AS is_automated,
  r.created_at                     AS reviewed_at,
  CASE WHEN d.processing_timestamp IS NOT NULL
       THEN (unix_millis(r.created_at) - unix_millis(d.processing_timestamp)) / 1000.0
  END                              AS turnaround_seconds,
  CASE WHEN r.corrections IS NOT NULL
        AND startswith(r.corrections, '{')
        AND try_parse_json(r.corrections) IS NOT NULL
       THEN array_size(json_object_keys(r.corrections))
  END                              AS fields_corrected_count,
  COALESCE(vc.verdict_changes_count, 0)       AS verdict_changes_count,
  (COALESCE(vc.verdict_changes_count, 0) > 0) AS was_overridden
FROM silver_reviews r
JOIN silver_documents d
  ON d.id = r.document_id
LEFT JOIN verdict_changes vc
  ON vc.review_id = r.id
WHERE NOT r.is_automated
  AND d.deleted_at IS NULL

UNION ALL

SELECT
  CONCAT('auto-', d.id)            AS review_id,
  d.id                             AS document_id,
  d.document_name,
  d.document_type,
  '<automated>'                    AS reviewer_email,
  'correct'                        AS verdict,
  CAST(NULL AS STRING)             AS reasoning,
  TRUE                             AS is_automated,
  COALESCE(d.processing_timestamp, d.updated_at) AS reviewed_at,
  CAST(NULL AS DOUBLE)             AS turnaround_seconds,
  CAST(NULL AS INT)                AS fields_corrected_count,
  0                                AS verdict_changes_count,
  FALSE                            AS was_overridden
FROM silver_documents d
WHERE d.processing_status = 'auto_verified'
  AND d.deleted_at IS NULL
