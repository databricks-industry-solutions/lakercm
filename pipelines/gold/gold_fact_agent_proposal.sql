-- =============================================================================
-- Gold Layer: Agent proposal fact (one row per proposal, atomic grain)
-- =============================================================================
-- Whether agent-assisted review is WORKING, which is not the same question as
-- whether reviewers click Approve.
--
-- Acceptance rate alone is the weak measure, and it is worse than weak — it is
-- self-confirming. A system that puts a plausible answer next to an Approve
-- button manufactures agreement, so a high acceptance rate is consistent both
-- with a good assistant and with reviewers who have stopped reading. Those two
-- are told apart only by scoring the PROPOSAL against the truth and crossing it
-- with what the human did:
--
--                    | proposal correct        | proposal wrong
--   -----------------+-------------------------+------------------------------
--   human accepted   | working as intended     | AUTOMATION BIAS  <- the number
--   human rejected   | reviewer over-corrected | human caught it     that
--   or modified      |                         |                     matters
--
-- Ground truth comes from synthetic_document_manifest.expected_fix, recorded by
-- the generator at the moment the quirk was planted — the only time the right
-- answer is known. It is NULL where the answer is genuinely undetermined
-- (missing_member_id has none; the two non-textbook non_billable paths displace
-- an unrelated diagnosis), so is_scoreable guards every correctness column.
-- Those rows still count for acceptance. Not every proposal can be graded, and
-- pretending otherwise would mark correct proposals wrong.
--
-- Withheld rows are the CONTROL arm — computed, stored, never shown — so they
-- are kept and flagged, never filtered. was_shown is the treatment indicator;
-- comparing turnaround across it is the only defensible form of "the agent made
-- review faster", because reviewers get faster at a corpus on their own.
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW fact_agent_proposal (

  CONSTRAINT has_proposal_id
    EXPECT (proposal_id IS NOT NULL)
    ON VIOLATION DROP ROW,

  CONSTRAINT has_document
    EXPECT (document_id IS NOT NULL),

  -- A withheld proposal must never have been dispositioned by a human: if one
  -- has, it was shown, and the control arm is contaminated.
  CONSTRAINT withheld_is_never_human_dispositioned
    EXPECT (
      NOT withheld
      OR disposition NOT IN ('accepted', 'modified', 'rejected')
    )
)
COMMENT 'Gold fact: one row per agent review proposal, with what the reviewer did with it and — where ground truth exists — whether the proposal was actually right. Includes withheld (control-arm) and rejected proposals; filter on was_shown to model what a reviewer saw.'
TBLPROPERTIES (
  'quality' = 'gold',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH truth AS (
  SELECT
    document_path,
    -- The manifest is append-only per generation run; a regenerated corpus can
    -- carry more than one row per path. Latest wins, matching the document the
    -- pipeline actually parsed.
    max_by(expected_fix, doc_id) AS expected_fix,
    max_by(quirks, doc_id)       AS planted_quirks
  FROM ${manifest_table}
  GROUP BY document_path
),

joined AS (
  SELECT
    p.id                         AS proposal_id,
    p.document_id,
    d.document_name,
    d.document_type,
    p.review_reason,
    p.resolution,
    p.source,
    p.model,
    p.field_name,
    p.correction_key,
    p.observed_value,
    p.proposed_value,
    p.human_value,
    p.rationale,
    p.candidates,
    COALESCE(p.withheld, FALSE)  AS withheld,
    p.disposition,
    p.proposed_at,
    p.disposition_at,
    p.disposition_by,
    t.expected_fix,
    t.planted_quirks
  FROM silver_review_proposals p
  LEFT JOIN dim_document d
    ON d.document_id = p.document_id
  -- dim_document exposes the document's path as file_path (not document_path);
  -- it carries the dbfs: scheme and equals the manifest's document_path exactly.
  LEFT JOIN truth t
    ON t.document_path = d.file_path
),

scored AS (
  SELECT
    *,
    -- The treatment indicator. A withheld proposal was computed but never put
    -- in front of anyone.
    NOT withheld                                     AS was_shown,

    -- Whether this proposal can be graded at all.
    expected_fix IS NOT NULL                         AS is_scoreable,

    -- What the review ended up using: the reviewer's own value when they
    -- changed it, else what was proposed.
    COALESCE(human_value, proposed_value)            AS final_value,

    -- A refusal is a legitimate outcome, counted separately from a proposal
    -- that no one acted on. missing_member_id lands here by construction.
    resolution = 'not_resolvable'                    AS was_declined,

    -- Reviewer engagement, for the rows a human could act on.
    disposition IN ('accepted', 'modified', 'rejected') AS was_dispositioned,

    CASE
      WHEN disposition_at IS NOT NULL AND proposed_at IS NOT NULL
        THEN CAST(disposition_at AS DOUBLE) - CAST(proposed_at AS DOUBLE)
    END                                              AS seconds_to_disposition
  FROM joined
)

SELECT
  *,
  -- Correctness of the PROPOSAL, independent of whether anyone accepted it.
  CASE WHEN is_scoreable THEN proposed_value = expected_fix END
    AS proposal_correct,

  -- Correctness of the OUTCOME — what the document ended up coded as. Differs
  -- from proposal_correct precisely when the reviewer changed the value.
  CASE WHEN is_scoreable THEN final_value = expected_fix END
    AS outcome_correct,

  -- The four quadrants. Each is NULL rather than FALSE when the row cannot be
  -- graded, so an ungradeable proposal never silently counts as a success.
  CASE
    WHEN is_scoreable AND was_dispositioned
      THEN disposition = 'accepted' AND NOT (proposed_value = expected_fix)
  END                                                AS automation_bias,
  CASE
    WHEN is_scoreable AND was_dispositioned
      THEN disposition = 'accepted' AND proposed_value = expected_fix
  END                                                AS accepted_and_correct,
  CASE
    WHEN is_scoreable AND was_dispositioned
      THEN disposition <> 'accepted' AND NOT (proposed_value = expected_fix)
  END                                                AS human_caught_it,
  CASE
    WHEN is_scoreable AND was_dispositioned
      THEN disposition <> 'accepted' AND proposed_value = expected_fix
  END                                                AS reviewer_overcorrected,

  current_timestamp()                                AS _computed_at
FROM scored
