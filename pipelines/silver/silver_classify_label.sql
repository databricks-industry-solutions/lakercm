-- =============================================================================
-- Silver Layer (parallel branch): Document-Type Classification
-- =============================================================================
-- One ai_classify per document, runs in parallel with silver_extract_identifiers.
-- Both branches fan out from STREAM(bronze_doc_parsed); gold_extraction_labels
-- joins them back together (stream-static LEFT JOIN, classify as static side).
--
-- ai_classify constrains output to a fixed enum, unlike the free-form "label"
-- field the previous combined ai_extract used. Add a new label here when a
-- new document type is introduced; gold's COALESCE to 'other' prevents
-- downstream has_label EXPECT violations on novel docs.
--
-- VERSION 2.1 (was 2.0). Four things change together, none separable:
--
--   1. VARIANT input. v2.1 accepts parsed_content directly, so the doc_text
--      flatten CTE this file used to carry is gone. That CTE concatenated every
--      element's content with spaces -- throwing away the layout the classifier
--      can otherwise use -- and it DROPPED any document whose elements produced
--      an empty string. Those rows then reached gold with a NULL label and
--      tripped its has_label EXPECT. Passing the VARIANT keeps the structure and
--      keeps the row (it lands as 'other' via the COALESCE below).
--   2. confidence_score, via enableConfidenceScores (2.1-only). gold folds it
--      into its confidence blend.
--   3. rationale, via enableRationales (2.1-only). The reviewer app shows it, so
--      a reviewer can see WHY a document was typed the way it was.
--   4. The response shape moves from {"response":[label]} to
--      {"response":[{value, confidence_score, rationale}]}. The read is
--      :response[0]:value::STRING, NOT :response[0]::STRING -- the old read
--      against a 2.1 response would store the whole per-label JSON object as the
--      label and silently corrupt routing.
--
-- Labels are the object form (label -> description) rather than a bare array.
-- Descriptions are documented to improve accuracy materially on ambiguous
-- categories, and this taxonomy has a genuinely ambiguous pair: a referral that
-- carries an authorization reads as both referral_workqueue and
-- prior_authorization. Measured against the real corpus the classifier resolves
-- that pair to prior_authorization at only 0.55-0.66 confidence.
--
-- NOTE ON THE CONFIDENCE SCALE: this is a calibrated posterior over 11
-- mutually-exclusive labels, so it does NOT behave like ai_extract's per-span
-- confidence (which saturates at 1.0 on a verbatim copy). Measured over 60 real
-- documents it ran min 0.55 / median 0.68 / max 0.78 across only 12 distinct
-- values. gold therefore normalizes it against classify_confidence_ceiling
-- before blending; read the comment there before changing either side.
-- =============================================================================

CREATE OR REFRESH STREAMING TABLE silver_classify_label (

  CONSTRAINT valid_document_path
    EXPECT (document_path IS NOT NULL),

  CONSTRAINT has_label
    EXPECT (label IS NOT NULL AND label != ''),

  -- Bounded 0-1 by contract. EXPECT rather than DROP ROW because a NULL
  -- confidence is legitimate: gold renormalizes its blend over whichever
  -- signals are present, so a row with no confidence is still useful.
  CONSTRAINT valid_classify_confidence
    EXPECT (classify_confidence IS NULL
            OR (classify_confidence >= 0 AND classify_confidence <= 1))
)
COMMENT 'Document-type classification (ai_classify v2.1): label, per-label confidence and a grounded rationale. Joined into gold_extraction_labels as the static side.'
TBLPROPERTIES (
  'quality' = 'silver',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

SELECT
  document_path,
  document_name,
  user_email,
  COALESCE(cls:response[0]:value::STRING, 'other') AS label,
  cls:response[0]:confidence_score::DOUBLE         AS classify_confidence,
  cls:response[0]:rationale::STRING                AS classify_rationale,
  current_timestamp()                              AS classified_at
FROM (
  SELECT
    document_path,
    document_name,
    user_email,
    ai_classify(
      parsed_content,
      -- v2.x takes labels as a JSON STRING; the v1 ARRAY(...) form does not
      -- compile ("labels requires STRING, but got ARRAY<STRING>" -- the flow
      -- failed analysis on its first deploy with a v2 version, dev CD).
      '{
        "denial_management": "Claim denial letters, appeals, and denial work items",
        "referral_workqueue": "Referral routing or work-queue items",
        "invoice": "Bills or invoices carrying charges and amounts",
        "prior_authorization": "Prior-authorization requests or approvals for a service or drug",
        "explanation_of_benefits": "Payer EOB summarizing how a claim was adjudicated",
        "clinical_notes": "Provider clinical or progress notes",
        "lab_results": "Laboratory test results",
        "patient_record": "Demographic or registration patient records",
        "eligibility_verification": "Coverage or eligibility checks",
        "claims_summary": "Summary of one or more submitted claims",
        "other": "Anything that does not fit the categories above"
      }',
      map(
        'version', '2.1',
        'multilabel', 'false',                -- single best-fit type; gold stores one label
        'enableConfidenceScores', 'true',     -- 2.1-only; feeds gold's confidence blend
        'enableRationales', 'true',           -- 2.1-only; shown to the reviewer
        'instructions', 'US healthcare claims/administration documents (payer/provider). Classify by the document''s primary administrative purpose.'
      )
    ) AS cls
  -- Bound to one micro-batch per trigger: maxFilesPerTrigger = 1 ensures each
  -- batch ≈ one bronze commit (~50 docs) ≈ ~10 sec of ai_classify work. This
  -- granularity keeps streaming latencies bounded and avoids unbounded batches.
  FROM STREAM(bronze_doc_parsed) WITH (maxFilesPerTrigger = 1)
  WHERE parsed_content IS NOT NULL
)
