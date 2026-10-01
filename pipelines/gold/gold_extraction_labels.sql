-- =============================================================================
-- Gold Layer: Document Extraction Results
-- =============================================================================
-- One streaming table that joins the parallel silver branches, blends a
-- confidence score, and decides where each document goes: auto-verified, or
-- the human review queue. Streaming (not MV) because the Lakebase synced table
-- `gold_extraction_labels_sync` rejects MV sources for CONTINUOUS/TRIGGERED
-- policies; CDF on this table feeds the sync.
--
-- Sources:
--   silver_extract_identifiers   (streaming side — ai_extract output + the
--                                 three confidence signals)
--   silver_classify_label        (static lookup — ai_classify output)
--   silver_validate_codes        (static lookup — the code check)
--
-- Stream-static LEFT JOINs: extract is streaming; classify and the code check
-- are static lookups, which a triggered update refreshes before this flow reads
-- them. If classify hasn't landed yet for a doc, COALESCE drops in 'other' so
-- the has_label EXPECT never trips.
--
-- Confidence blend. A weighted MEAN over the signals a document actually has,
-- renormalized so a missing signal leaves the denominator as well as the
-- numerator:
--
--     extract 0.50 | classify 0.20 | parse 0.15 | completeness 0.15
--
-- Renormalizing, rather than defaulting an absent signal to zero, is what stops
-- a document that produced three of the four signals from being punished for the
-- fourth. It replaces two ad-hoc fallbacks that used to do this job unevenly:
-- parse no longer borrows extract's value, and the all-zero-extract case no
-- longer switches to a separate 0.5/0.5 formula. Both are now the same rule.
--
-- NULLIF(extract_confidence_mean, 0) is what makes the all-zero case work.
-- ai_extract sometimes returns 0.0 for EVERY field of a cleanly parsed document
-- — 15 of ~1000 prod documents and 1 of 12 in dev — which is "no scores", not
-- "no confidence": parse confidence and completeness were both ~1.0 on those.
-- Read literally, 0.50 * 0 capped the blend and sent a clean document to review.
-- Only extract gets this treatment; a completeness of 0.0 is a real measurement.
--
-- WHY THE CLASSIFY SIGNAL IS RESCALED BEFORE IT IS WEIGHTED. ai_classify v2.1's
-- confidence is not on the same scale as the other three. It is a calibrated
-- posterior over 11 mutually-exclusive labels, where ai_extract's is a per-span
-- score that saturates at 1.0 on a verbatim copy. Measured on 60 real documents
-- it ran min 0.55 / median 0.68 / max 0.78, across only 12 distinct values.
-- Blended raw it did two wrong things:
--   * a flawless document could not exceed 0.956, and the reviewer UI renders
--     this column as a percentage, so "99%" became unreachable; and
--   * because renormalization drops an absent signal's weight, a document the
--     classifier FAILED to score outranked one it scored 0.78 with confidence --
--     unscored beat scored.
-- Dividing by ${classify_confidence_ceiling} and clamping fixes both: measured
-- max 0.9888, median 0.9422, and an auto-verify rate of 65% at the unchanged
-- 0.92 threshold against the old blend's 55%. At or above the ceiling the
-- classify signal costs a document nothing (it ties an unscored one); below it
-- the penalty is at most 0.20 * (1 - confidence/ceiling).
--
-- The clamp is load-bearing, not cosmetic: a confidence of 0.90 against a 0.80
-- ceiling contributes 0.225 against a 0.20 weight and pushes a perfect document
-- to 1.025, violating the valid_confidence EXPECT. It is an explicit CASE and
-- NOT LEAST(1.0, ...) because Spark's least() SKIPS NULLs -- LEAST(1.0, NULL) is
-- 1.0, not NULL -- which would hand every unscored document a full-weight
-- numerator against a reduced denominator. DuckDB returns NULL there, so the
-- execution tests would have passed while production was wrong.
--
-- Result stays in [0, 1] so the auto_verdict_threshold and the
-- valid_confidence EXPECT both remain valid.
--
-- Routing. A document is auto-verified when review_reasons is EMPTY, and held
-- when it is not. Every reason is named, including the confidence one:
--   invalid_code            a code missing from the terminology, or malformed
--   non_billable_code       a code that exists but is not billable at that level
--   missing_member_id       no member or subscriber ID on the document
--   low_confidence          confidence below auto_verdict_threshold
--   confidence_unavailable  no confidence score could be computed
--
-- The last two are new, and they replace a silence. Previously a document held
-- on confidence carried an EMPTY review_reasons, and every consumer had to guess
-- what the emptiness meant. The reviewer app guessed "confidence was low" and
-- rendered "Extraction confidence was 100%, below the 92% auto-verify threshold"
-- on a document measuring 99.96% -- a sentence that is both self-contradictory
-- and unfalsifiable from the data it was drawn from. An empty list means "no
-- rule fired", which is a different claim from "the score was low".
--
-- So the reason is recorded where the decision is made. Consumers read it; none
-- of them re-derives it. The reviewer app and the agent read is_automated from
-- the Lakebase copy of this table, so the decision is made once, here.
--
-- Adding a VALUE to review_reasons is not a schema change, so the Lakebase
-- synced table needs no work. Existing rows keep their old arrays until a FULL
-- REFRESH -- a triggered update only computes new rows.
-- =============================================================================

CREATE OR REFRESH STREAMING TABLE gold_extraction_labels (

  CONSTRAINT valid_document_path
    EXPECT (document_path IS NOT NULL),

  CONSTRAINT valid_document_name
    EXPECT (document_name IS NOT NULL),

  CONSTRAINT has_label
    EXPECT (label IS NOT NULL AND label != ''),

  CONSTRAINT valid_confidence
    EXPECT (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1))
)
COMMENT 'Gold extraction results: one row per document with label, identifiers, blended confidence, the routing decision (is_automated) and the review_reasons behind it.'
TBLPROPERTIES (
  'quality' = 'gold',
  'pipelines.autoOptimize.managed' = 'true',
  'delta.enableChangeDataFeed' = 'true'
)
AS

WITH joined AS (
  SELECT
    e.document_path,
    e.document_name,
    e.user_email,
    COALESCE(c.label, 'other') AS label,
    -- Raw, exactly as ai_classify reported it. The reviewer UI shows THIS number;
    -- only the blend below sees the rescaled form. Carrying the rescaled value
    -- here instead would display a 0.68 classification as 85%.
    c.classify_confidence,
    c.classify_rationale,
    e.page_images,
    e.identifiers,
    e.elements,
    -- Blend inputs, hoisted so the weighted sum and the weight sum below each
    -- read one definition and cannot drift apart.
    NULLIF(e.extract_confidence_mean, 0)                          AS extract_confidence,
    e.parse_confidence_mean,
    e.completeness_score,
    CASE
      WHEN c.classify_confidence IS NULL                          THEN NULL
      WHEN c.classify_confidence >= ${classify_confidence_ceiling} THEN 1.0
      ELSE c.classify_confidence / ${classify_confidence_ceiling}
    END                                                           AS classify_confidence_scaled,
    e.extracted_at,
    -- Carried forward rather than folded into review_reasons here: the reason
    -- array needs confidence_score, which is a select-list alias of THIS select
    -- and so cannot be referenced from it. Building every reason one CTE down
    -- keeps all five in a single expression.
    COALESCE(v.codes_invalid, 0)     AS codes_invalid,
    COALESCE(v.codes_non_billable, 0) AS codes_non_billable
  FROM STREAM(silver_extract_identifiers) e
  LEFT JOIN silver_classify_label c
    ON c.document_path = e.document_path
  LEFT JOIN silver_validate_codes v
    ON v.document_path = e.document_path
),

-- Its own CTE because a select list cannot reference its own aliases, and
-- because the blend is easier to read and to test in one place. Deliberately
-- plain scalar SQL (CAST / COALESCE / NULLIF / CASE / arithmetic): this dataset
-- is one of the few that test_pipeline_sql_execution.py actually transpiles and
-- RUNS in DuckDB, and higher-order functions are the least reliable part of that
-- path -- so no zip_with/aggregate formulation here.
scored AS (
  SELECT
    *,
    CAST(
      (
          0.50 * COALESCE(extract_confidence, 0)
        + 0.20 * COALESCE(classify_confidence_scaled, 0)
        + 0.15 * COALESCE(parse_confidence_mean, 0)
        + 0.15 * COALESCE(completeness_score, 0)
      )
      / NULLIF(
          (
              CASE WHEN extract_confidence         IS NULL THEN 0 ELSE 0.50 END
            + CASE WHEN classify_confidence_scaled IS NULL THEN 0 ELSE 0.20 END
            + CASE WHEN parse_confidence_mean      IS NULL THEN 0 ELSE 0.15 END
            + CASE WHEN completeness_score         IS NULL THEN 0 ELSE 0.15 END
          ),
          0
        )
      AS DOUBLE
    ) AS confidence_score
  FROM joined
),

-- Downstream of `scored`, NOT of `joined`, and that ordering is load-bearing:
-- low_confidence and confidence_unavailable are predicates on confidence_score,
-- so the reasons have to be built against the blended value this pipeline
-- actually routes on. Against `joined` they would not even resolve -- the blend
-- moved out of that CTE when it became four signals.
reasoned AS (
  SELECT
    *,
    filter(
      array(
        CASE WHEN codes_invalid > 0 THEN 'invalid_code' END,
        CASE WHEN codes_non_billable > 0 THEN 'non_billable_code' END,
        -- Any member / subscriber / insured ID field with a value. A heading
        -- such as 'Member ID' is layout, not an ID: gold_claim_codes skips the
        -- same fields.
        CASE
          WHEN identifiers IS NOT NULL
           AND size(filter(
                 identifiers,
                 x -> lower(trim(x.name)) RLIKE '(^|_)(member|subscriber|insured)_?(id|number|no|num)($|_)'
                      AND NOT lower(trim(x.name)) RLIKE '^(section_header|section_heading|section_title|page_header|page_footer|page_number|header|footer|heading|title|document_title)(_|$)'
                      AND trim(x.value) != ''
               )) = 0
            THEN 'missing_member_id'
        END,
        -- The confidence hold, NAMED. It used to be recorded only as the
        -- ABSENCE of a reason, and the reviewer app inferred it from that
        -- emptiness -- telling reviewers a document measuring 99.96% was
        -- "below the 92% threshold". An empty list meant "no rule fired",
        -- which is not the same claim. Emitting these two makes
        -- `held <=> size(review_reasons) > 0` true by construction.
        CASE
          WHEN confidence_score IS NOT NULL
           AND confidence_score < ${auto_verdict_threshold}
            THEN 'low_confidence'
        END,
        CASE WHEN confidence_score IS NULL THEN 'confidence_unavailable' END
      ),
      r -> r IS NOT NULL
    ) AS review_reasons
  FROM scored
),

routed AS (
  SELECT
    *,
    -- Exactly equivalent to the previous
    -- `confidence_score IS NOT NULL AND >= threshold AND size(...) = 0`:
    -- both dropped conjuncts are now entailed by low_confidence and
    -- confidence_unavailable, which are complete and disjoint over the
    -- complement. Routing does not change; the invariant becomes structural
    -- instead of coincidental, and there is one definition of "held".
    size(review_reasons) = 0                           AS auto_verified
  FROM reasoned
)

SELECT
  document_path,
  document_name,
  user_email,
  label,
  page_images,
  identifiers,
  elements,
  confidence_score,
  extracted_at,
  auto_verified                                        AS is_automated,
  CASE WHEN auto_verified THEN 'correct'     END       AS verdict,
  CASE WHEN auto_verified THEN '<automated>' END       AS reviewer_email,
  CASE WHEN auto_verified THEN '<automated>' END       AS reasoning,
  -- Last: the Lakebase synced table takes a new trailing column as an additive
  -- schema change, reaching it on the sync after the pipeline's next update. A
  -- column inserted mid-list instead is a destroy + create of the table the
  -- reviewer app, the agent's document tools and the embedding job all read, and
  -- that needs lifecycle.prevent_destroy lowered on purpose. APPEND ONLY.
  review_reasons,
  classify_confidence,
  classify_rationale,
  -- Pseudonymous patient key: what ties one patient's documents together in the
  -- knowledge graph. Appended last, per the APPEND ONLY note above.
  --
  -- Derived from the EXTRACTED member id, not from the synthetic manifest,
  -- because this has to work for a real uploaded document too -- there is no
  -- manifest row for those. The member id is already payer-scoped (its prefix
  -- encodes the payer), so it needs no further qualification to be unique.
  --
  -- BE PRECISE ABOUT WHAT THIS IS. A SHA-256 digest of an identifier is
  -- PSEUDONYMISATION, not HIPAA Safe Harbor de-identification: Safe Harbor's
  -- re-identification code may not be derived from the individual's
  -- information, and this is. It is used anyway, deliberately, because:
  --   * a truly random surrogate would have to be minted once and remembered,
  --     and this is a recomputable materialized view -- uuid() here would
  --     reassign every patient on each refresh and churn the whole graph;
  --   * the corpus is synthetic, so no real PHI exists to protect;
  --   * the digest, not the member id, is what reaches the triplestore, so the
  --     graph still carries no patient name, date of birth or member id.
  -- The member id -> key mapping is not stored separately: it is recomputable
  -- from `identifiers`, which stays in gold and never leaves it.
  -- Do NOT describe this column as de-identified.
  -- Pure array functions, deliberately: the same filter() idiom the
  -- missing_member_id reason above uses. A scalar subquery with explode()
  -- correlated on `identifiers` would parse but not run -- Spark does not
  -- support that lateral correlation inside a scalar subquery.
  CASE
    WHEN identifiers IS NULL THEN NULL
    WHEN size(filter(
           identifiers,
           x -> lower(trim(x.name)) RLIKE '(^|_)(member|subscriber|insured)_?(id|number|no|num)($|_)'
                AND NOT lower(trim(x.name)) RLIKE '^(section_header|section_heading|section_title|page_header|page_footer|page_number|header|footer|heading|title|document_title)(_|$)'
                AND trim(x.value) != ''
         )) = 0 THEN NULL
    ELSE element_at(
      transform(
        filter(
          identifiers,
          x -> lower(trim(x.name)) RLIKE '(^|_)(member|subscriber|insured)_?(id|number|no|num)($|_)'
               AND NOT lower(trim(x.name)) RLIKE '^(section_header|section_heading|section_title|page_header|page_footer|page_number|header|footer|heading|title|document_title)(_|$)'
               AND trim(x.value) != ''
        ),
        x -> substr(
          sha2(upper(regexp_replace(trim(x.value), '[^A-Za-z0-9]', '')), 256), 1, 16
        )
      ),
      1
    )
  END                                                  AS patient_key
FROM routed
