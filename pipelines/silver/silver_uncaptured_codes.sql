-- =============================================================================
-- Silver Layer: Billable diagnoses printed on the page that never reached the claim
-- =============================================================================
-- The complement of silver_validate_codes. That dataset asks "is every code we
-- captured a real one?"; this asks "did we capture every real code that was
-- there?" -- and on the current corpus the answer is no for 4.0% of documents.
--
-- Measured over 2,169 documents: 87 lose at least one dotted ICD-10 code that is
-- printed in the parsed text AND present in ref_icd10_cm, 111 diagnosis mentions
-- in total, and 52 of those losses are on documents that AUTO-VERIFIED -- shipped
-- without a person ever seeing them. Examples are not marginal: J45.909 asthma,
-- I25.10 coronary artery disease, I48.91 atrial fibrillation, D64.9 anemia.
-- For revenue-cycle work that is under-coding, and it is invisible today because
-- every existing check reasons about the codes that WERE captured.
--
-- PRODUCTION-REAL, unlike gold_extraction_fidelity. The signal is "a valid code
-- is in the page text but not in the stored codes", which is computable from
-- pipeline data alone -- no generator manifest, so it holds for real customer
-- documents. That is also why it sits in the documents pipeline rather than
-- analytics.
--
-- PRECISION, measured against the generator manifest: 97 of the 111 flagged
-- mentions were codes actually planted as that document's diagnoses, so 87.4%.
-- The remaining 12.6% are codes printed in REFERENCE context -- a payer policy
-- table or a coverage list naming codes that were never assigned to this patient.
-- That is the known false-positive mode, and the reason this dataset is
-- observability and NOT a review_reasons entry: at 87.4% it would hold some
-- documents wrongly, and routing is a deliberate decision to take with that
-- number in hand rather than a side effect of adding a dataset.
--
-- DOTTED ICD-10 ONLY, deliberately. silver_validate_codes documents at length why
-- scanning free text for bare 5-digit codes is unsafe: a ZIP code and a CPT code
-- share ^[0-9]{5}$, and a 5-digit payer id (60054) was read as a procedure code,
-- found invalid, and marked a whole claim unclean. In free page text there is no
-- field name to gate on at all, so the only shape safe to scan is the
-- self-identifying one: [A-Z][0-9][0-9A-Z].[0-9A-Z]{1,4}. The cost is real -- this
-- misses dropped CPTs, which gold_extraction_fidelity does catch on synthetic
-- corpora (99214 and 99203 on two documents) -- so the two checks are
-- complementary, not redundant.
--
-- ONE ROW PER AFFECTED DOCUMENT, not per document. Consumers LEFT JOIN, so an
-- empty-array row for the ~96% with nothing to report would be pure volume.
--
-- MATERIALIZED VIEW: it aggregates (collect_list ... GROUP BY), and a global
-- aggregation on a stream resolves to complete mode, which fails at analysis.
-- Same reasoning as silver_validate_codes -- see its header.
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW silver_uncaptured_codes (

  CONSTRAINT valid_document_path
    EXPECT (document_path IS NOT NULL),

  CONSTRAINT has_a_finding
    EXPECT (uncaptured_count > 0)
)
COMMENT 'Billable ICD-10 diagnoses present in the parsed page text that never reached the captured codes: the under-coding counterpart to silver_validate_codes. One row per AFFECTED document. Observability, not a routing input - 87.4% precision, with reference-context codes as the known false positive.'
TBLPROPERTIES (
  'quality' = 'silver',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

-- Every document's page text, from the parser's own elements rather than the
-- extracted identifiers: the question is what the PAGE said, so the identifiers
-- (which are what extraction chose to keep) cannot be the source.
WITH page AS (
  SELECT
    document_path,
    document_name,
    user_email,
    -- A tab or line break separates tokens the way a space does, matching the
    -- normalisation in silver_validate_codes' parts CTE.
    upper(
      translate(
        concat_ws(' ', transform(elements, x -> x.text_content)),
        concat(chr(9), chr(10), chr(13)),
        '   '
      )
    ) AS page_text
  FROM silver_extract_identifiers
),

-- Code-SHAPED tokens. [.] rather than an escaped dot: Spark and DuckDB (the test
-- harness) read backslash escapes differently, and silver_validate_codes avoids
-- them throughout for the same reason.
candidates AS (
  SELECT DISTINCT
    p.document_path,
    p.document_name,
    p.user_email,
    tok AS candidate_code
  FROM page p
  LATERAL VIEW explode(
    array_distinct(
      regexp_extract_all(p.page_text, '[A-Z][0-9][0-9A-Z][.][0-9A-Z]{1,4}', 0)
    )
  ) e AS tok
),

-- What the pipeline actually kept. LEFT JOIN, not INNER: 28 of 2,197 documents
-- have NO row in silver_validate_codes at all because they captured zero
-- classified codes -- and those are the worst cases, one of them auto-verified
-- having stored nothing while three codes were printed on it. An inner join here
-- would silently drop exactly the documents this dataset exists to surface.
captured AS (
  SELECT
    document_path,
    array_distinct(transform(validated_codes, c -> upper(c.code))) AS kept_codes
  FROM silver_validate_codes
),

-- Shape alone is not enough: the token has to be a code the terminology knows,
-- which is what separates a real dropped diagnosis from an invoice line that
-- happens to look like one.
uncaptured AS (
  SELECT
    c.document_path,
    c.document_name,
    c.user_email,
    c.candidate_code,
    icd.description AS code_description,
    icd.is_billable
  FROM candidates c
  JOIN ${ref_catalog}.${ref_schema}.ref_icd10_cm icd
    ON upper(icd.code) = c.candidate_code
  LEFT JOIN captured k
    ON k.document_path = c.document_path
  WHERE NOT array_contains(COALESCE(k.kept_codes, array()), c.candidate_code)
)

SELECT
  document_path,
  any_value(document_name) AS document_name,
  any_value(user_email)    AS user_email,
  collect_list(
    named_struct(
      'code',             candidate_code,
      'code_description', code_description,
      'is_billable',      is_billable
    )
  ) AS uncaptured_codes,
  COUNT(DISTINCT candidate_code) AS uncaptured_count,
  -- Split out because a non-billable parent code left on the page is a coding
  -- nicety, whereas a billable one is revenue the claim never asked for.
  COUNT(DISTINCT CASE WHEN is_billable THEN candidate_code END)
                                 AS uncaptured_billable_count,
  current_timestamp()            AS evaluated_at
FROM uncaptured
GROUP BY document_path
