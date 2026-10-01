-- =============================================================================
-- Gold Layer (observability): Extraction fidelity against planted ground truth
-- =============================================================================
-- Answers one question the rest of the medallion structurally cannot: does the
-- code we are about to bill still say what the source document said?
--
-- WHY THIS EXISTS -- and why it is not a hold reason in the medallion path.
--
-- silver_extract_identifiers instructs the model, in as many words, to "COPY
-- CODES CHARACTER FOR CHARACTER ... 'I1O' with a letter O stays 'I1O'". On a
-- measured 100-document batch that instruction was violated for 9 of the 12
-- documents carrying a planted 'I1O' (letter O) for ICD-10 'I10' (digit zero):
-- the stored code came back as the VALID 'I10', validated clean, and the
-- document auto-verified at confidence 0.9996.
--
-- The rewrite is not happening where the prompt can reach it. For 7 of those 9,
-- ai_parse_document had ALREADY normalised the glyph -- the parsed element text
-- contains 'I10' and no 'I1O' at all, so ai_extract copied faithfully from text
-- that was already wrong. Two consequences follow, both measured, both the
-- reason this dataset is shaped the way it is:
--
--   1. A verbatim check is useless. Comparing each extracted code against the
--      concatenated parsed text flags 0 of 96 code mentions on that batch,
--      because wherever the extractor emits 'I10' the parsed text contains
--      'I10' somewhere too.
--   2. Confidence cannot see it. Across the 12, parse_confidence_mean spans
--      0.99918-0.99946 and the per-identifier confidence on the rewritten
--      diagnosis field is 1.0 -- for the escaped and the caught documents
--      alike. The parser reports total certainty about the character it
--      changed.
--
-- So downstream of ai_parse_document the defect is genuinely unobservable, and
-- the ONLY surviving ground truth is what the generator planted. That makes
-- this a SYNTHETIC-CORPUS instrument, not a production control: a real document
-- has no manifest row and lands here as 'no_ground_truth'. It is deliberately
-- kept out of silver_validate_codes and gold_extraction_labels' review_reasons
-- so no routing decision is ever taken on a signal real documents cannot carry.
-- What it gives the demo is the honest version of the story: the rewrite rate is
-- measured and attributable instead of invisible.
--
-- Lives in the analytics pipeline (not documents) because that is where the
-- other manifest-scored dataset lives -- see gold_fact_agent_proposal, which
-- reads ${manifest_table} the same way.
--
-- DETECTION. Not edit distance: glyph confusion is the whole mechanism, so the
-- test is a confusable-collapse. Fold the letters a digit is misread as
-- (O->0, I->1, L->1, S->5, B->8, Z->2) in BOTH the planted and the extracted
-- code; if the folded forms match while the raw forms differ, the same glyph
-- slot was resolved two different ways. 'I1O' and 'I10' both fold to '110'.
-- A truncation ('9921' for '99213') folds to different lengths and is reported
-- as dropped, not rewritten -- which is right, and validation already catches
-- that one on its own (10 of 10 on the measured batch).
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW gold_extraction_fidelity (

  CONSTRAINT valid_document_path
    EXPECT (document_path IS NOT NULL),

  CONSTRAINT known_fidelity_status
    EXPECT (fidelity_status IN ('faithful', 'rewritten', 'dropped', 'no_ground_truth'))
)
COMMENT 'Extraction fidelity for synthetic documents: planted source codes vs the codes actually stored, with silently rewritten glyphs (I1O -> I10) named per document. Observability only - never a routing input, and no_ground_truth for any real document.'
TBLPROPERTIES (
  'quality' = 'gold',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH truth AS (
  SELECT
    document_path,
    -- Append-only per generation run; a regenerated corpus can carry more than
    -- one row per path. Latest wins, matching gold_fact_agent_proposal.
    max_by(diagnosis_codes, doc_id) AS src_diagnosis_codes,
    max_by(procedure_codes, doc_id) AS src_procedure_codes,
    max_by(quirks, doc_id)          AS planted_quirks
  FROM ${manifest_table}
  GROUP BY document_path
),

-- The planted codes as a set. The manifest stores them comma-joined per field.
planted AS (
  SELECT
    document_path,
    planted_quirks,
    array_distinct(
      filter(
        transform(
          split(concat_ws(',', src_diagnosis_codes, src_procedure_codes), ','),
          x -> upper(trim(x))
        ),
        x -> x != ''
      )
    ) AS source_codes
  FROM truth
),

-- What the pipeline actually stored. raw_code is the model's own output, before
-- silver_validate_codes' dotted/modifier normalisation -- which is the value
-- that has to be compared against the page.
stored AS (
  SELECT
    document_path,
    array_distinct(
      transform(validated_codes, c -> upper(c.raw_code))
    ) AS extracted_codes
  FROM ${validate_codes_table}
),

paired AS (
  SELECT
    p.document_path,
    p.planted_quirks,
    p.source_codes,
    COALESCE(s.extracted_codes, array())                     AS extracted_codes,
    -- Planted codes that survived character-for-character.
    filter(p.source_codes, c -> array_contains(
      COALESCE(s.extracted_codes, array()), c))               AS codes_matched,
    -- Planted codes that did not survive verbatim but reappear with a confusable
    -- glyph resolved differently. Carries both sides so the UI can show
    -- 'I1O -> I10' without recomputing the fold.
    filter(
      transform(
        filter(p.source_codes,
               c -> NOT array_contains(COALESCE(s.extracted_codes, array()), c)),
        c -> named_struct(
          'source_code', c,
          -- get(), not element_at(): element_at throws INVALID_ARRAY_INDEX on an
          -- empty array, and "no confusable counterpart" is the common case.
          -- get() is 0-based and returns NULL out of bounds, which is exactly
          -- the sentinel the two filters below branch on.
          'stored_as', get(
            filter(
              COALESCE(s.extracted_codes, array()),
              e -> translate(e, 'OILSBZ', '011582')
                   = translate(c, 'OILSBZ', '011582')
            ),
            0
          )
        )
      ),
      r -> r.stored_as IS NOT NULL
    )                                                        AS codes_rewritten,
    -- Planted, absent, and with no confusable counterpart: a truncation or a
    -- straight miss. Reported for completeness; validation already holds these.
    filter(
      transform(
        filter(p.source_codes,
               c -> NOT array_contains(COALESCE(s.extracted_codes, array()), c)),
        c -> named_struct(
          'source_code', c,
          -- get(), not element_at(): element_at throws INVALID_ARRAY_INDEX on an
          -- empty array, and "no confusable counterpart" is the common case.
          -- get() is 0-based and returns NULL out of bounds, which is exactly
          -- the sentinel the two filters below branch on.
          'stored_as', get(
            filter(
              COALESCE(s.extracted_codes, array()),
              e -> translate(e, 'OILSBZ', '011582')
                   = translate(c, 'OILSBZ', '011582')
            ),
            0
          )
        )
      ),
      r -> r.stored_as IS NULL
    )                                                        AS codes_dropped
  FROM planted p
  LEFT JOIN stored s ON s.document_path = p.document_path
),

-- is_automated comes from gold, not recomputed here: the point of the headline
-- metric is the combination (a code was rewritten AND the document still
-- auto-verified), and that routing decision is made once, in
-- gold_extraction_labels.
routed AS (
  SELECT
    pr.*,
    g.is_automated,
    g.confidence_score,
    g.review_reasons
  FROM paired pr
  LEFT JOIN ${extraction_labels_table} g
    ON g.document_path = pr.document_path
)

SELECT
  document_path,
  planted_quirks,
  source_codes,
  extracted_codes,
  codes_matched,
  codes_rewritten,
  codes_dropped,
  size(codes_rewritten)                       AS rewritten_count,
  size(codes_dropped)                         AS dropped_count,
  is_automated,
  confidence_score,
  review_reasons,
  -- The one that matters: a clinical code was silently altered and nothing
  -- stopped the document. Kept as its own column so a dashboard tile and the
  -- reviewer badge read the same predicate.
  (size(codes_rewritten) > 0 AND COALESCE(is_automated, FALSE))
                                              AS rewritten_and_auto_verified,
  CASE
    WHEN source_codes IS NULL OR size(source_codes) = 0 THEN 'no_ground_truth'
    WHEN size(codes_rewritten) > 0                      THEN 'rewritten'
    WHEN size(codes_dropped) > 0                        THEN 'dropped'
    ELSE 'faithful'
  END                                         AS fidelity_status,
  current_timestamp()                         AS evaluated_at
FROM routed
