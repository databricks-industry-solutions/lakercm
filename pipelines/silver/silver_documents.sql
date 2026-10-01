-- =============================================================================
-- Silver Layer: Documents (current state from the Lakebase CDC mirror)
-- =============================================================================
-- One row per document: the latest CDC event per id wins. The mirror
-- (${cdc_docs_table}) is an append-only event log written by the Lakebase
-- reverse-CDC sync — full row image per event (REPLICA IDENTITY FULL) plus
-- _pg_* metadata. Full recompute + QUALIFY dedupe is deliberately used
-- instead of streaming AUTO CDC: the mirror is service-owned and may be
-- re-snapshotted (the initial snapshot was double-written), which breaks a
-- streaming read but is a no-op for recompute at this scale (~2k events).
--
-- Hard deletes (winning event is a Postgres DELETE) drop the row here.
-- Soft deletes (deleted_at set) are RETAINED — silver stays a truthful
-- mirror; gold facts exclude them at build time.
--
-- UUIDs arrive as BINARY(16); decoded to canonical lowercase dashed strings
-- so keys match what the Postgres-facing app emits.
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW silver_documents (

  CONSTRAINT valid_status
    EXPECT (processing_status IN ('processing', 'pending', 'auto_verified', 'failed')),

  CONSTRAINT uuid_shaped_id
    EXPECT (id RLIKE '^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')
    ON VIOLATION DROP ROW
)
COMMENT 'Current state of Lakebase medical_documents: latest CDC event per id; hard deletes removed, soft deletes retained with deleted_at set.'
TBLPROPERTIES (
  'quality' = 'silver',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH latest AS (
  SELECT *
  FROM ${cdc_docs_table}
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY id ORDER BY _pg_lsn DESC, _sort_by DESC
  ) = 1
)

SELECT
  lower(regexp_replace(hex(id), '(.{8})(.{4})(.{4})(.{4})(.{12})', '$1-$2-$3-$4-$5')) AS id,
  user_email,
  document_name,
  file_path,
  file_size,
  document_type,
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
  _pg_change_type                  AS _last_change_type,
  _pg_lsn                          AS _last_lsn,
  CAST(_timestamp AS TIMESTAMP)    AS _synced_at
FROM latest
WHERE _pg_change_type <> 'delete'
