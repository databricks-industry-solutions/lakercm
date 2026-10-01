-- =============================================================================
-- Silver Layer: Agent review proposals (current state from the Lakebase mirror)
-- =============================================================================
-- One row per proposal id. Unlike silver_reviews, the source table is
-- APPEND-ONLY per proposal — a document accumulates one row per fix the agent
-- offered — so this is many rows per document, and that is the point: the
-- rejected proposals have to survive or the acceptance rate only ever counts
-- its own successes.
--
-- A proposal is still UPDATED once, when the reviewer dispositions it
-- (pending -> accepted/modified/rejected), so the same QUALIFY dedupe as
-- silver_reviews applies: latest CDC event per id.
--
-- `withheld` rows are KEPT here. They are the control arm — computed, stored,
-- never shown — and dropping them at silver would remove the comparison the
-- holdout exists to make. Consumers that model what a reviewer SAW must filter
-- them out explicitly; gold_fact_agent_proposal carries the flag through.
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW silver_review_proposals (

  CONSTRAINT valid_resolution
    EXPECT (resolution IN ('deterministic', 'needs_judgment', 'not_resolvable')),

  CONSTRAINT valid_disposition
    EXPECT (disposition IN (
      'pending', 'accepted', 'modified', 'rejected', 'declined', 'superseded'
    )),

  CONSTRAINT uuid_shaped_id
    EXPECT (id RLIKE '^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$')
    ON VIOLATION DROP ROW,

  CONSTRAINT has_document_id
    EXPECT (document_id IS NOT NULL),

  -- The missing_member_id guarantee, re-checked in the lakehouse. Postgres
  -- constrains it at write time; if a row ever arrives here with a value on an
  -- unresolvable proposal, something wrote around the constraint and the
  -- expectation surfaces it instead of it reaching a dashboard.
  CONSTRAINT no_value_when_unresolvable
    EXPECT (resolution <> 'not_resolvable' OR proposed_value IS NULL)
)
COMMENT 'Current state of Lakebase document_review_proposals: latest CDC event per proposal id. Append-only per document (many proposals per document), including rejected and withheld ones.'
TBLPROPERTIES (
  'quality' = 'silver',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH latest AS (
  SELECT *
  FROM ${cdc_proposals_table}
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY id ORDER BY _pg_lsn DESC, _sort_by DESC
  ) = 1
)

SELECT
  lower(regexp_replace(hex(id), '(.{8})(.{4})(.{4})(.{4})(.{12})', '$1-$2-$3-$4-$5'))          AS id,
  lower(regexp_replace(hex(document_id), '(.{8})(.{4})(.{4})(.{4})(.{12})', '$1-$2-$3-$4-$5')) AS document_id,
  review_reason,
  resolution,
  source,
  field_name,
  correction_key,
  observed_value,
  proposed_value,
  rationale,
  candidates,
  model,
  COALESCE(withheld, FALSE)        AS withheld,
  disposition,
  disposition_at,
  disposition_by,
  human_value,
  proposed_at,
  _pg_change_type                  AS _last_change_type,
  _pg_lsn                          AS _last_lsn,
  CAST(_timestamp AS TIMESTAMP)    AS _synced_at
FROM latest
WHERE _pg_change_type <> 'delete'
