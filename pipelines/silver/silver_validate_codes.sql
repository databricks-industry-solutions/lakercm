-- =============================================================================
-- Silver Layer: Healthcare code-set grounding & validation
-- =============================================================================
-- Consumes silver_extract_identifiers, isolates the code-bearing identifiers
-- (ICD-10-CM diagnosis, CPT / HCPCS procedure), classifies each by regex,
-- and validates it against the seeded reference tables (ref_icd10_cm,
-- ref_cpt_hcpcs — created by the reference_data bundle's seed post-step).
--
-- Output: one row per document with a validated_codes[] struct array (one entry
-- per field occurrence) plus summary counts over DISTINCT codes — a code named
-- by three fields is one code. This is the deterministic "OCR -> claims intelligence"
-- layer: an extracted code is grounded (matched + described), invalid
-- (code-shaped but not in the terminology), or non-billable (parent code).
--
-- Consumed downstream as a STATIC side (like silver_classify_label): gold /
-- KPIs / the reviewer UI join on document_path. Reference tables are read
-- fully-qualified via the ${ref_catalog}.${ref_schema} pipeline config keys
-- (documents_pipeline.yml) — they are external UC tables, not pipeline
-- datasets, so they must be seeded before this step runs.
--
-- Classification (value pattern is authoritative; a field-name hint only
-- gates the ambiguous bare-5-digit CPT case so ZIPs / dollar amounts in a
-- 5-digit-shaped field are not misread as procedure codes):
--   ICD-10-CM : ^[A-Z][0-9][0-9A-Z]([.][0-9A-Z]{1,4})?$   (every letter is used:
--               U is the special-purpose range, and U07.1 COVID-19 and U09.9
--               post-COVID condition are billable ICD-10-CM codes)
--   HCPCS     : ^[A-Z][0-9]{4}$
--   CPT       : ^[0-9]{5}$   AND field name mentions cpt/hcpcs/procedure/service
--   CPT, malformed : ^[0-9]{4}$ in an explicit procedure field (counted invalid)
-- A value written 'code — description' contributes the code before each spaced
-- dash.
-- =============================================================================

-- MATERIALIZED VIEW, not a streaming table. This dataset AGGREGATES
-- (collect_list ... GROUP BY document_path, plus a DISTINCT to de-duplicate
-- repeated codes). Streaming tables are append-only, so a global aggregation
-- without a watermark resolves to complete mode and DISTINCT is unsupported on a
-- stream — declaring this STREAMING fails the pipeline at analysis rather than at
-- runtime, taking the whole documents pipeline (and the semantic/governance
-- post-steps that depend on it) down with it. Every other aggregating dataset in
-- this pipeline is an MV for the same reason (gold_fact_review,
-- gold_fact_document_processing, gold_claim_codes). Full recompute per trigger is
-- cheap at demo volume and nothing downstream needs CDF off this table.
CREATE OR REFRESH MATERIALIZED VIEW silver_validate_codes (

  CONSTRAINT valid_document_path
    EXPECT (document_path IS NOT NULL)
)
COMMENT 'Code-set grounding: extracted ICD-10-CM / CPT-HCPCS codes validated against the reference terminology, one row per document with a validated_codes[] array and summary counts.'
TBLPROPERTIES (
  'quality' = 'silver',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH parts AS (
  SELECT
    document_path,
    document_name,
    user_email,
    lower(trim(id.name))    AS field_name,
    -- A tab or line break separates entries the way a space does. Normalizing
    -- them here keeps every pattern below free of backslash escapes, which
    -- Spark and DuckDB (the test harness) read differently.
    upper(trim(translate(code_part, concat(chr(9), chr(10), chr(13)), '   ')))
                            AS part,
    id.confidence           AS extract_confidence
  FROM silver_extract_identifiers
  LATERAL VIEW explode(identifiers) t AS id
  -- One value can hold several codes ('99213, 36415', 'M54.50; I10'). Each
  -- part is classified and validated on its own; before, the whole value
  -- matched no shape and every code in it was silently dropped (ninth review).
  -- Only list separators split: a space or a dash can belong to one code
  -- ('20610 RT', '99213-25'), and splitting on spaces would read the B12 in
  -- 'Vitamin B12 deficiency' as a diagnosis code.
  LATERAL VIEW explode(split(id.value, '[,;]')) p AS code_part
  WHERE id.value IS NOT NULL AND trim(code_part) != ''
),

-- ai_extract often returns a code together with its description ('R07.9 —
-- Chest pain'), sometimes several in one value ('Z00.00 — Encounter ...
-- E78.5 — Hyperlipidemia'). Such a part matches no code shape as a whole, so
-- every code written this way was dropped: nine of the eighteen on the first
-- synthetic batch, among them the non-billable M54.5 it was built to catch.
-- Each token that stands before a spaced dash (em dash, en dash or hyphen) is
-- taken instead; a part with no spaced dash is classified whole, as before.
-- The shape checks below still decide what is a code, so a word that happens
-- to precede a dash ('PAIN - ...') drops out there.
exploded AS (
  SELECT
    document_path,
    document_name,
    user_email,
    field_name,
    trim(token)             AS raw_value,
    extract_confidence
  FROM parts
  LATERAL VIEW explode(
    CASE
      WHEN part RLIKE '(^| )[A-Z0-9][A-Z0-9.]{2,7} +[—–-] '
        THEN regexp_extract_all(part, '(^| )([A-Z0-9][A-Z0-9.]{2,7}) +[—–-] ', 2)
      ELSE array(part)
    END
  ) c AS token
  WHERE trim(token) != ''
),

-- Field ROLE flags, computed once. Classification is decided by (value shape x
-- field role), because shape alone is ambiguous: an undotted ICD-10 code and a
-- HCPCS code share `^[A-Z][0-9]{4}$`, and a ZIP code and a CPT code share
-- `^[0-9]{5}$`. Pinned by executing this SQL on fixtures in
-- scripts/tests/test_pipeline_sql_execution.py (TestPipelineExecution,
-- TestFieldPrecisionExecution).
field_roles AS (
  SELECT
    document_path,
    document_name,
    user_email,
    field_name,
    raw_value,
    extract_confidence,
    -- Never a clinical code, whatever the value looks like (zip_code = '60601'
    -- was once classified as CPT because its name contains 'code').
    field_name RLIKE '(zip|postal|area_code|phone|fax|country)' AS is_non_code_field,
    -- Whole diagnosis words only. A bare 'diag' substring also matched
    -- DIAGNOSTIC procedure fields: diagnostic_procedure_code = 93000 lost its
    -- CPT code, and diagnostic_service_code = G0439 was misread as ICD-10
    -- G04.39 and flagged invalid (third review).
    field_name RLIKE '(diagnos(is|es|e)|(^|_)diag(_|$|[0-9])|icd|(^|_)dx($|_|[0-9]))'
                                                                  AS is_dx_field
  FROM exploded
),

classified AS (
  SELECT
    *,
    CASE
      WHEN is_non_code_field THEN NULL
      -- DOTTED ICD-10 (M54.50, J45.909) is self-identifying. Any first letter:
      -- excluding U as "reserved" silently dropped U07.1 (COVID-19) and U09.9,
      -- both billable in ICD-10-CM (eighth review).
      WHEN raw_value RLIKE '^[A-Z][0-9][0-9A-Z][.][0-9A-Z]{1,4}$'
        THEN 'ICD-10-CM'
      -- UNDOTTED ICD-10 in a diagnosis field: the STANDARD form on real claims
      -- (837 / CMS-1500 carry M5450, not M54.50), plus bare 3-character codes
      -- (I10, Z23). Checked BEFORE the HCPCS branch, which would otherwise claim
      -- 'M5450' (same letter+4-digit shape) and flag a valid diagnosis invalid.
      -- Bare values stay gated on a diagnosis field: 'B12' is vitamin B-12, and
      -- room numbers and plan tiers share the shape.
      WHEN is_dx_field AND raw_value RLIKE '^[A-Z][0-9][0-9A-Z]([0-9A-Z]{1,4})?$'
        THEN 'ICD-10-CM'
      -- CPT/HCPCS only in a procedure-role field — never a diagnosis field.
      WHEN NOT is_dx_field
        AND field_name RLIKE '(cpt|hcpcs|procedure|(^|_)proc(_|$)|service|drug|code)'
        AND NOT field_name RLIKE '(reason|denial|carc|rarc|status|npi|tax)'
        -- claim / group / member / policy / plan / auth exclude only an
        -- IDENTIFIER of that thing (claim_number, group_code, plan_code,
        -- prior_auth_number). As bare substrings they also dropped real
        -- procedure fields: planned_procedure_code and
        -- prior_auth_procedure_code (fifth review), claim_line_procedure_code
        -- and claim_cpt_code (seventh review). Payer, provider, facility and
        -- similar codes are identifiers too: 5-digit payer ids (such as 60054)
        -- were read as CPT codes, found invalid, and marked the
        -- claim unclean (ninth review).
        AND NOT field_name RLIKE '(^|_)(claim|group|member|policy|plan|auth|authorization|precert|payer|payor|provider|facility|account|location|site|clinic|office|vendor|employer|subscriber|patient|pharmacy|network|contract)_(code|number|no|num|id)($|_)'
        -- Billed codes routinely carry 2-character modifiers ('99213-25',
        -- '20610 RT'); the base code is what gets validated. Rejecting them
        -- dropped the claim out of every clean-claim denominator.
        AND raw_value RLIKE '^([A-Z][0-9]{4}|[0-9]{5})([- ][0-9A-Z]{2})*$'
        THEN 'CPT/HCPCS'
      -- A 4-digit value in an explicit procedure or CPT field is a CPT code with
      -- a digit missing ('9921' for 99213). It was skipped, so the claim passed
      -- as clean; now it is looked up, fails, and counts as an invalid code.
      -- Explicit procedure fields only: revenue codes, bill types, years and
      -- counts are legitimately four digits.
      WHEN NOT is_dx_field
        AND field_name RLIKE '(cpt|hcpcs|procedure|(^|_)proc(_|$))'
        AND NOT field_name RLIKE '(reason|denial|carc|rarc|status|npi|tax|revenue|modifier|date|year|time|count|unit|qty|quantity|amount|charge|fee|total)'
        AND NOT field_name RLIKE '(^|_)(claim|group|member|policy|plan|auth|authorization|precert|payer|payor|provider|facility|account|location|site|clinic|office|vendor|employer|subscriber|patient|pharmacy|network|contract)_(code|number|no|num|id)($|_)'
        AND raw_value RLIKE '^[0-9]{4}$'
        THEN 'CPT/HCPCS'
      ELSE NULL
    END AS code_system
  FROM field_roles
),

-- Normalize to the reference tables' form: ref_icd10_cm stores DOTTED codes, so
-- an undotted 'M5450' is looked up as 'M54.50' (dot after the 3rd character),
-- and a CPT/HCPCS code is looked up without its modifiers ('99213-25' -> 99213).
-- A malformed 4-digit code keeps its own value, so the invalid entry names it.
normalized AS (
  SELECT
    *,
    CASE
      WHEN code_system = 'ICD-10-CM' AND instr(raw_value, '.') = 0
           AND length(raw_value) > 3
        THEN concat(substr(raw_value, 1, 3), '.', substr(raw_value, 4))
      WHEN code_system = 'CPT/HCPCS'
        THEN COALESCE(
          NULLIF(regexp_extract(raw_value, '^([A-Z][0-9]{4}|[0-9]{5})', 1), ''),
          raw_value
        )
      ELSE raw_value
    END AS lookup_code
  FROM classified
  WHERE code_system IS NOT NULL
),

validated AS (
  SELECT
    n.document_path,
    n.document_name,
    n.user_email,
    n.field_name,
    n.lookup_code AS code,
    n.raw_value   AS raw_code,
    n.code_system,
    n.extract_confidence,
    COALESCE(icd.description, cpt.description)      AS code_description,
    COALESCE(icd.category, cpt.category)            AS code_category,
    -- Non-billable parent codes (e.g. M54.5) are matched but flagged.
    (icd.code IS NOT NULL AND icd.is_billable = FALSE) AS is_non_billable,
    (icd.code IS NOT NULL OR cpt.code IS NOT NULL)  AS code_valid
  FROM (SELECT DISTINCT * FROM normalized) n
  -- Join on n.lookup_code — a REAL column. An earlier version joined on c.code,
  -- a select-list alias the ON clause cannot see (unresolved column at analysis).
  LEFT JOIN ${ref_catalog}.${ref_schema}.ref_icd10_cm icd
    ON n.code_system = 'ICD-10-CM' AND icd.code = n.lookup_code
  LEFT JOIN ${ref_catalog}.${ref_schema}.ref_cpt_hcpcs cpt
    ON n.code_system = 'CPT/HCPCS' AND cpt.code = n.lookup_code
)

SELECT
  document_path,
  any_value(document_name) AS document_name,
  any_value(user_email)    AS user_email,
  collect_list(
    named_struct(
      'field_name',        field_name,
      'code',              code,
      'raw_code',          raw_code,
      'code_system',       code_system,
      'code_valid',        code_valid,
      'is_non_billable',   is_non_billable,
      'code_description',  code_description,
      'code_category',     code_category,
      'extract_confidence', extract_confidence
    )
  ) AS validated_codes,
  -- Counts are per DISTINCT code, not per field mention. ai_extract repeats a
  -- code across every field that carries it (one synthetic prior auth had M54.5
  -- in pain_code, primary_diagnosis and the header table), which counted it
  -- three times and inflated every rate built on these columns. The dedup key is
  -- the code as looked up, so 'M5450' and 'M54.50' — and '99213-25' and '99213'
  -- — are one code, which is what they are on the claim. validated_codes above
  -- keeps every field occurrence: the invalid-code TVF names the field.
  COUNT(DISTINCT concat(code_system, ':', code))          AS codes_total,
  COUNT(DISTINCT CASE WHEN code_valid
                      THEN concat(code_system, ':', code) END)      AS codes_valid,
  COUNT(DISTINCT CASE WHEN NOT code_valid
                      THEN concat(code_system, ':', code) END)      AS codes_invalid,
  COUNT(DISTINCT CASE WHEN is_non_billable
                      THEN concat(code_system, ':', code) END)      AS codes_non_billable,
  bool_or(NOT code_valid)               AS has_invalid_code,
  current_timestamp()                   AS validated_at
FROM validated
GROUP BY document_path
