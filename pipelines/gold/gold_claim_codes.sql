-- =============================================================================
-- Gold Layer: Claim Coding Facts (code grounding + claims attributes)
-- =============================================================================
-- Grain: one row per extracted document. Joins the code-grounding silver step
-- (silver_validate_codes) back onto gold_extraction_labels and plucks the
-- claims-domain attributes out of the extracted identifier array, so the
-- semantic layer / Genie / dashboards can report real claims KPIs:
--   clean-claim rate, invalid-code rate, coding-specificity (non-billable)
--   rate, denial mix by payer, prior-auth approval rate.
--
-- It also CODES the free-text denial reason to a CARC (X12 835 Claim Adjustment
-- Reason Code) + a coarse denial_category. denial_reason itself may embed
-- patient/provider names, so it is masked in gold_claim_codes_secure and excluded
-- from Genie; the coded columns are the structured, non-PHI substitute.
--
-- EXTERNAL DEPENDENCY: reads ${ref_catalog}.${ref_schema}.ref_carc, seeded by the
-- reference_data bundle (scripts/seed_reference_data.py) — which deploys before
-- the pipelines bundle for exactly this reason.
--
-- WHY A SEPARATE TABLE (not extra columns on gold_extraction_labels):
--   gold_extraction_labels is a STREAMING table whose CDF feeds the Lakebase
--   synced table. Adding columns there forces a full refresh and changes the
--   sync's schema. This is purely additive — an MV that nothing else depends
--   on — so it can be dropped/recreated freely.
--
-- MV (not streaming): it re-reads two upstream streaming tables and does a
-- non-streaming aggregate/pluck. Triggered mode makes full recompute cheap,
-- and nothing downstream needs CDF off this table.
--
-- ATTRIBUTE PLUCKING IS HEURISTIC. ai_extract emits free-form snake_case
-- names, so each attribute is chosen by RANKING candidate fields on name fit
-- (see the attribute-plucking block); a document-level field breaks ties over a
-- row_N_* field. Behavior is executed against fixtures in
-- scripts/tests/test_pipeline_sql_execution.py — extend a pattern there first
-- when the generator introduces a new field spelling.
-- =============================================================================

CREATE OR REFRESH MATERIALIZED VIEW gold_claim_codes
COMMENT 'Per-document claim coding facts: grounded ICD-10/CPT code validation rollup, plucked claims attributes (payer, claim id, denial reason, auth status, billed amount), and CARC-coded denial reasons (denial_carc_code, denial_category). Source for the claims_coding_metrics semantic layer.'
TBLPROPERTIES (
  'quality' = 'gold',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH flat AS (
  SELECT
    g.document_path,
    lower(trim(id.name))                      AS field_name,
    trim(id.value)                            AS field_value,
    lower(trim(id.name)) RLIKE '^row_[0-9]+_' AS is_row_field
  FROM gold_extraction_labels g
  LATERAL VIEW explode(g.identifiers) t AS id
  WHERE id.value IS NOT NULL AND trim(id.value) != ''
    -- Headings, page headers and footers, titles and page numbers are layout,
    -- not values. On the first synthetic denial the 'Denial reason' HEADING
    -- (section_header_denial_reason) outranked the header table's reason row
    -- ('CARC 204: Service not covered ...'), so the denial lost its CARC and
    -- was coded 'other'. gold_extraction_labels skips the same fields.
    AND NOT lower(trim(id.name)) RLIKE '^(section_header|section_heading|section_title|page_header|page_footer|page_number|header|footer|heading|title|document_title)(_|$)'
),

-- ---------------------------------------------------------------------------
-- Attribute plucking, RANKED BY FIELD-NAME FIT.
--
-- ai_extract emits free-form snake_case names, so each attribute is a set of
-- candidate fields. An earlier version took MAX(field_value) over every field
-- whose name merely CONTAINED the keyword, which returns the lexicographically
-- largest VALUE, not the right field: payer_name='Kestrel' + payer_address=
-- 'PO Box 981106' yielded payer='PO Box 981106' (reproduced by executing this
-- SQL in DuckDB; see scripts/tests/test_pipeline_sql_execution.py), corrupting
-- every payer KPI. Now:
--   name_score 30 = exact canonical name   (payer, payer_name, insurer, ...)
--   name_score 20 = keyword match, EXCLUDING contact/identifier fields
--                   (address, phone, fax, zip, *_id, number, email, ...)
--   billed_amount additionally ranks true charges above balances: total_charge
--   is what was billed; balance_due is what remains owed after payment.
-- Ranking = name_score * 10, +5 for a document-level (non row_N_) field, so name
-- fit dominates and document-level only breaks ties. Deterministic tie-break on
-- field_name, then field_value.
-- ---------------------------------------------------------------------------
attr_candidates AS (
  SELECT document_path, 'payer' AS attribute, field_name, field_value, is_row_field,
    CASE
      WHEN field_name RLIKE '^(payer|payer_name|insurer|insurer_name|insurance_company|insurance_carrier|carrier|carrier_name|health_plan|insurance_plan|plan_name)$' THEN 30
      WHEN field_name RLIKE '(payer|insurer|insurance|carrier|health_plan)'
           AND NOT field_name RLIKE '(address|addr|street|city|state|zip|postal|phone|fax|email|contact|website|url|(^|_)id($|_)|number|(^|_)no($|_)|num($|_)|type|status|policy)'
        THEN 20
    END AS name_score
  FROM flat
  UNION ALL
  SELECT document_path, 'claim_id', field_name, field_value, is_row_field,
    CASE
      WHEN field_name RLIKE '^(claim_id|claim_number|claim_no|claim_num)$' THEN 30
      -- Token-bounded: RLIKE is a substring search, and 'claim_no' matched
      -- inside claim_notes, plucking note text as the claim id (fifth review).
      WHEN field_name RLIKE '(^|_)claim_(id|number|no|num)($|_)' THEN 20
    END
  FROM flat
  UNION ALL
  -- denial_reason is the FREE TEXT; denial_code is the code. They used to share
  -- a score, and the field_name tie-break picked 'denial_code', so a document
  -- carrying both lost its reason text (third review). The code is plucked as its
  -- own attribute and fed to CARC coding alongside the text (see denial_src).
  SELECT document_path, 'denial_reason', field_name, field_value, is_row_field,
    CASE
      WHEN field_name RLIKE '^(denial_reason|reason_for_denial|rejection_reason|denial_description|denial_reason_text)$' THEN 30
      WHEN field_name RLIKE '(denial_reason|reason_for_denial|rejection_reason|denial_desc)'
           AND NOT field_name RLIKE '(_code($|_)|carc|rarc)' THEN 20
    END
  FROM flat
  UNION ALL
  SELECT document_path, 'denial_code', field_name, field_value, is_row_field,
    CASE
      WHEN field_name RLIKE '^(denial_code|denial_reason_code|carc|carc_code|adjustment_reason_code|reason_code)$' THEN 30
      -- '(^|_)carc($|_)', not 'carc': carcinoma_stage is not a denial code.
      WHEN field_name RLIKE '(denial_code|(^|_)carc($|_)|adjustment_reason)' THEN 20
    END
  FROM flat
  UNION ALL
  SELECT document_path, 'auth_status', field_name, field_value, is_row_field,
    CASE
      WHEN field_name RLIKE '^(auth_status|authorization_status|prior_auth_status|precert_status)$' THEN 30
      WHEN field_name RLIKE '(auth_status|authorization_status|precert_status)' THEN 20
    END
  FROM flat
  UNION ALL
  SELECT document_path, 'patient_mrn', field_name, field_value, is_row_field,
    CASE
      WHEN field_name RLIKE '^(patient_mrn|mrn|medical_record_number|member_id|patient_id|subscriber_id)$' THEN 30
      -- Token-bounded, so member_identity_verified = 'yes' is not an MRN.
      WHEN field_name RLIKE '(^|_)(patient_mrn|mrn|member_id|patient_id|subscriber_id)($|_)' THEN 20
    END
  FROM flat
  UNION ALL
  SELECT document_path, 'billed_amount', field_name, field_value, is_row_field,
    CASE
      WHEN field_name RLIKE '^(total_charge|total_charges|billed_amount|total_billed|amount_billed|charge_amount)$' THEN 30
      WHEN field_name RLIKE '(total_charge|billed_amount|total_billed|amount_billed|charge_amount)' THEN 25
      WHEN field_name RLIKE '(total_amount|amount_due|balance_due)' THEN 10
    END
  FROM flat
),

attr_ranked AS (
  SELECT
    document_path,
    attribute,
    field_value,
    ROW_NUMBER() OVER (
      PARTITION BY document_path, attribute
      ORDER BY name_score * 10 + CASE WHEN is_row_field THEN 0 ELSE 5 END DESC,
               field_name,
               field_value
    ) AS rn
  FROM attr_candidates
  WHERE name_score IS NOT NULL
),

attrs AS (
  SELECT
    document_path,
    MAX(CASE WHEN attribute = 'payer'         THEN field_value END) AS payer,
    MAX(CASE WHEN attribute = 'claim_id'      THEN field_value END) AS claim_id,
    MAX(CASE WHEN attribute = 'denial_reason' THEN field_value END) AS denial_reason,
    MAX(CASE WHEN attribute = 'denial_code'   THEN field_value END) AS denial_code,
    MAX(CASE WHEN attribute = 'auth_status'   THEN field_value END) AS auth_status,
    MAX(CASE WHEN attribute = 'patient_mrn'   THEN field_value END) AS patient_mrn,
    MAX(CASE WHEN attribute = 'billed_amount' THEN field_value END) AS billed_amount_raw
  FROM attr_ranked
  WHERE rn = 1
  GROUP BY document_path
),

-- ---------------------------------------------------------------------------
-- Denial-reason CARC coding. Rules live in ref_carc (seeded from
-- scripts/denial_code_content.py) and this block mirrors, exactly, the Python
-- emulation unit-tested in scripts/tests/test_denial_coding.py:
--   1. an EXPLICIT CARC in the text ("CO-50", "CARC 197") wins, but only if it
--      is in the curated ref_carc list (precision over recall);
--   2. otherwise the lowest match_priority whose pattern matches wins, ties
--      broken by code;
--   3. a present-but-unmatched reason becomes denial_category = 'other'.
-- The explicit code is normalized through INT so 'CO-050' matches code '50'.
-- The explicit-code regex is tuned for precision (5010 group codes only, a
-- separator before the number, no decimals) so creatinine 'Cr 1.8' or the lab
-- value 'CO2 24' cannot outrank the keyword rules; see denial_code_content.py.
-- ---------------------------------------------------------------------------
-- Code AND text together: an explicit CARC usually lives in the denial_code field
-- ("CO-50") while the words live in denial_reason, and either may be missing.
-- Joined with '. ' so each stays its own clause: a rule anchored at a clause
-- start ("Duplicate claim") still matches after a code prefix.
denial_text AS (
  SELECT
    document_path,
    lower(concat_ws('. ', denial_code, denial_reason)) AS reason_lc,
    -- A BARE CARC number in the dedicated code field (reason_code = '197')
    -- is explicit too: without a group prefix the regex below never saw it,
    -- and the denial fell through to 'other' (fifth review). Only a pure
    -- 1-3 digit value counts; RARC codes (N130, MA04) are alphanumeric.
    CASE WHEN trim(denial_code) RLIKE '^[0-9]{1,3}$' THEN trim(denial_code) END
      AS bare_code
  FROM attrs
  WHERE denial_reason IS NOT NULL OR denial_code IS NOT NULL
),

denial_src AS (
  SELECT
    document_path,
    reason_lc,
    CAST(
      COALESCE(
        try_cast(
          regexp_extract(
            reason_lc,
            '(^|[^a-z])(carc|co|pr|oa|pi)[- #:]+([0-9]{1,3})([^0-9.]|[.][^0-9]|[.]$|$)',
            3
          ) AS INT
        ),
        try_cast(bare_code AS INT)
      ) AS STRING
    ) AS explicit_code
  FROM denial_text
),

denial_ranked AS (
  SELECT
    s.document_path,
    r.code                                      AS denial_carc_code,
    r.description                               AS denial_carc_description,
    r.denial_category,
    CASE WHEN r.code = s.explicit_code
         THEN 'explicit_code' ELSE 'keyword_rule' END AS denial_coding_method,
    ROW_NUMBER() OVER (
      PARTITION BY s.document_path
      ORDER BY CASE WHEN r.code = s.explicit_code THEN 0 ELSE r.match_priority END,
               r.code
    ) AS rn
  FROM denial_src s
  JOIN ${ref_catalog}.${ref_schema}.ref_carc r
    ON r.code = s.explicit_code
    OR (r.match_pattern IS NOT NULL AND s.reason_lc RLIKE r.match_pattern)
),

denial_coded AS (
  SELECT document_path, denial_carc_code, denial_carc_description,
         denial_category, denial_coding_method
  FROM denial_ranked
  WHERE rn = 1
)

SELECT
  g.document_path,
  g.document_name,
  g.user_email,
  g.label                                         AS document_type,
  g.confidence_score,
  g.is_automated,
  g.extracted_at,

  -- Code grounding rollup (NULL-safe: documents with no code-shaped
  -- identifiers have no silver_validate_codes row at all).
  COALESCE(v.codes_total, 0)                      AS codes_total,
  COALESCE(v.codes_valid, 0)                      AS codes_valid,
  COALESCE(v.codes_invalid, 0)                    AS codes_invalid,
  COALESCE(v.codes_non_billable, 0)               AS codes_non_billable,
  v.validated_codes,

  -- Clean claim = every grounded code resolved AND none flagged non-billable.
  -- NULL (not FALSE) when the document carries no codes, so rate measures
  -- naturally exclude non-coded documents from the denominator.
  CASE
    WHEN COALESCE(v.codes_total, 0) = 0 THEN NULL
    ELSE (v.codes_invalid = 0 AND v.codes_non_billable = 0)
  END                                             AS is_clean_claim,
  CASE
    WHEN COALESCE(v.codes_total, 0) = 0 THEN NULL
    ELSE (v.codes_invalid > 0)
  END                                             AS has_invalid_code,

  a.payer,
  a.claim_id,
  a.denial_reason,
  -- Structured, NON-PHI denial coding (see the denial CTEs above). NULL when the
  -- document carries no denial reason; 'other' when it does but nothing matched.
  d.denial_carc_code,
  d.denial_carc_description,
  CASE
    WHEN a.denial_reason IS NULL AND a.denial_code IS NULL THEN NULL
    ELSE COALESCE(d.denial_category, 'other')
  END                                             AS denial_category,
  d.denial_coding_method,
  lower(a.auth_status)                            AS auth_status,
  a.patient_mrn,
  -- Strip currency symbols/commas before casting; try_cast keeps junk as NULL.
  try_cast(regexp_replace(a.billed_amount_raw, '[^0-9.-]', '') AS DOUBLE)
                                                  AS billed_amount
FROM gold_extraction_labels g
LEFT JOIN silver_validate_codes v
  ON v.document_path = g.document_path
LEFT JOIN attrs a
  ON a.document_path = g.document_path
LEFT JOIN denial_coded d
  ON d.document_path = g.document_path
