-- =============================================================================
-- Silver Layer (parallel branch): Field Extraction (ai_extract, identifiers only)
-- =============================================================================
-- One ai_extract per document, runs in parallel with silver_classify_label.
-- The label lives in silver_classify_label; this branch produces identifiers
-- and the supporting confidence signals. gold_extraction_labels is the
-- consumer (stream-static LEFT JOIN, extract as streaming side).
--
-- ai_extract is invoked with version 2.1 + mode=precision + enableCitations +
-- enableConfidenceScores.
-- Each identifier's value comes back wrapped as {value, confidence_score,
-- citation_ids}; metadata.citations resolves citation_ids to bounding boxes
-- (input is parsed_content from ai_parse_document, so citations are bbox).
-- The output identifier struct is denormalized to {name, value, confidence,
-- citations[]} so downstream consumers (gold, reviewer UI) need no join.
--
-- Three intermediate signals are emitted so gold can blend them without
-- re-deriving:
--   completeness_score      = fraction of identifiers with non-empty value.
--   parse_confidence_mean   = mean of per-element OCR/layout confidences from
--                             ai_parse_document v2.0. Skips NULLs so a single
--                             missing element doesn't poison the mean.
--   extract_confidence_mean = mean of per-identifier confidence_score values
--                             from ai_extract v2.1. NULL when the model
--                             returned no per-field scores.
-- =============================================================================

CREATE OR REFRESH STREAMING TABLE silver_extract_identifiers (

  CONSTRAINT valid_document_path
    EXPECT (document_path IS NOT NULL),

  CONSTRAINT valid_completeness
    EXPECT (completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 1))
)
COMMENT 'Field extraction (ai_extract, identifiers only). Streaming side of the gold_extraction_labels stream-static join.'
TBLPROPERTIES (
  'quality' = 'silver',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH extracted AS (
  SELECT
    document_path,
    document_name,
    user_email,
    -- Per-page rendered PNG URIs (populated when ai_parse_document imageOutputPath
    -- is set), ordered by page id. Carried to gold so the reviewer can overlay
    -- boxes on the parser's render canvas instead of the raw upload.
    transform(
      from_json(
        to_json(parsed_content:document:pages),
        'ARRAY<STRUCT<id: INT, image_uri: STRING>>'
      ),
      p -> p.image_uri
    ) AS page_images,
    transform(
      from_json(
        to_json(parsed_content:document:elements),
        'ARRAY<STRUCT<id: BIGINT, type: STRING, content: STRING, description: STRING, confidence: DOUBLE, bbox: ARRAY<STRUCT<coord: ARRAY<INT>, page_id: INT>>>>'
      ),
      el -> named_struct(
        'page_number', COALESCE(el.bbox[0].page_id + 1, 1),
        'element_type', el.type,
        'text_content', el.content,
        'bounding_box', named_struct(
          'x', CAST(el.bbox[0].coord[0] AS DOUBLE),
          'y', CAST(el.bbox[0].coord[1] AS DOUBLE),
          'width', CAST(el.bbox[0].coord[2] - el.bbox[0].coord[0] AS DOUBLE),
          'height', CAST(el.bbox[0].coord[3] - el.bbox[0].coord[1] AS DOUBLE)
        ),
        'confidence_score', el.confidence
      )
    ) AS elements,
    ai_extract(
      parsed_content,
      '{
        "identifiers": {
          "type": "array",
          "description": "ALL data points extracted from the document as name/value pairs. Include every field, table row, date, ID, status, and count.",
          "items": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string",
                "description": "Field name in snake_case (e.g. patient_name, row_1_coverage, auth_status, diagnosis_code)"
              },
              "value": {
                "type": "string",
                "description": "Raw verbatim value exactly as it appears in the document"
              }
            }
          }
        }
      }',
      map(
        'version', '2.1',
        -- Precision mode: an agentic extraction path rather than chunk-and-merge,
        -- built for long documents, high-volume outputs and schemas that need
        -- reasoning. That is exactly this call: an unbounded identifiers array
        -- over dense, multi-page, multi-table claim documents, including every
        -- row of every table.
        --
        -- Verified on a real parsed_content VARIANT that it does NOT change the
        -- citation contract: metadata.chunk_type stays "bbox" with coords
        -- byte-identical to the baseline call, so the citation_index from_json
        -- below and the reviewer's bbox overlay are unaffected. What it CAN move
        -- is the per-identifier confidence_score distribution, which
        -- extract_confidence_mean feeds and gold weights at 0.50 -- so this
        -- ships as its own commit, to keep any shift in the auto-verify rate
        -- attributable to it rather than to the classify signal.
        'mode', 'precision',
        'enableCitations', 'true',
        'enableConfidenceScores', 'true',
        'instructions', concat(
          'You are a healthcare document data extractor. Extract EVERY data point from this document.\n\n',
          'Populate "identifiers" with ALL of the following as snake_case name/value pairs:\n',
          '   - Every labeled field and its value (e.g. name: "patient_name", value: "<patient_last>, <patient_first>")\n',
          '   - Every table row - use a row prefix for each column (e.g. "row_1_patient_name", "row_1_coverage")\n',
          '   - Every numeric amount, date, ID, status, and count you see\n',
          '   - Every section header that has associated data\n',
          '   - Communication log entries, triage notes, scheduling details\n',
          '   - Insurance plan names, authorization types, priority levels, diagnosis codes\n',
          '   DO NOT summarize. Extract the raw values verbatim. Include ALL rows from tables.\n\n',
          -- KEPT, but known to be insufficient on its own — do not read this
          -- instruction as a control. Measured on a 100-document batch: of the 12
          -- documents carrying a planted 'I1O' (letter O) for ICD-10 'I10', 9 came
          -- back as the valid 'I10', validated clean, and auto-verified at 0.9996.
          -- For 7 of those 9 the instruction never had a chance: ai_parse_document
          -- had already normalised the glyph, so the parsed text handed to
          -- ai_extract contained 'I10' and no 'I1O' at all, and copying verbatim
          -- reproduced a value that was wrong before extraction began. The
          -- instruction still earns its place for the cases the model CAN affect
          -- (the 2 where the text kept 'I1O', and truncations like '9921', caught
          -- 10 of 10) — it just cannot see a parse-layer rewrite.
          -- gold_extraction_fidelity measures what escapes; nothing downstream of
          -- the parser can, which is why it scores against the generator manifest
          -- rather than any pipeline column.
          'COPY CODES CHARACTER FOR CHARACTER. Never repair a code that looks wrong:\n',
          '"I1O" with a letter O stays "I1O", a 4-digit "9921" stays "9921", and a\n',
          'code with a trailing modifier keeps it. Downstream validation compares each\n',
          'code against the terminology and needs to see the defect to flag it; a\n',
          'silently corrected code turns a rejectable claim into a clean one.\n\n',
          'DO NOT emit figure descriptions, logo content, or any other interpretation\n',
          'of decorative or visual imagery (e.g. "a blue circle with a white J"). Only\n',
          'emit fielded data the document is conveying — skip anything that describes\n',
          'what an image looks like rather than what data it carries.'
        )
      )
    ) AS e
  -- Bound to one micro-batch per trigger: maxFilesPerTrigger = 1 ensures each
  -- batch ≈ one bronze commit (~50 docs) ≈ ~4.5 min of ai_extract work. This
  -- granularity keeps streaming latencies bounded and avoids the previous
  -- 4h+ commits that had no progress events.
  FROM STREAM(bronze_doc_parsed) WITH (maxFilesPerTrigger = 1)
  WHERE parsed_content IS NOT NULL
),

parsed AS (
  SELECT
    document_path,
    document_name,
    user_email,
    page_images,
    elements,
    -- Drop visual / decorative extractions before downstream aggregates see
    -- them. Patterns match the snake_case names ai_extract emits for figure
    -- and logo descriptions, which the prompt also instructs the model to
    -- skip; the SQL filter is the deterministic safety net. Filtering here
    -- (in the parsed CTE) means completeness_score and extract_confidence_mean
    -- both compute against the cleaned population.
    filter(
      from_json(
        to_json(e:response:identifiers),
        'ARRAY<STRUCT<
          name:  STRUCT<value: STRING, confidence_score: DOUBLE, citation_ids: ARRAY<INT>>,
          value: STRUCT<value: STRING, confidence_score: DOUBLE, citation_ids: ARRAY<INT>>
        >>'
      ),
      x -> NOT (
        lower(trim(x.name.value))
          RLIKE '^(figure_description|figure_content|logo)(_|$)'
      )
    ) AS raw_identifiers,
    from_json(
      to_json(e:metadata:citations),
      'ARRAY<STRUCT<
        id: INT,
        bbox: ARRAY<STRUCT<coord: ARRAY<INT>, page_id: INT>>
      >>'
    ) AS citation_index
  FROM extracted
  WHERE e IS NOT NULL
),

scored AS (
  SELECT
    document_path,
    document_name,
    user_email,
    page_images,
    array_distinct(
      transform(
        raw_identifiers,
        x -> named_struct(
          'name',       lower(trim(x.name.value)),
          'value',      trim(x.value.value),
          'confidence', x.value.confidence_score,
          'citations',  CASE
                          WHEN citation_index IS NULL OR x.value.citation_ids IS NULL THEN NULL
                          ELSE flatten(
                            transform(
                              filter(citation_index, c -> array_contains(x.value.citation_ids, c.id)),
                              c -> c.bbox
                            )
                          )
                        END
        )
      )
    ) AS identifiers,
    elements,
    CASE
      WHEN raw_identifiers IS NULL OR size(raw_identifiers) = 0 THEN 0.0
      ELSE CAST(
        size(filter(
          raw_identifiers,
          x -> x.value.value IS NOT NULL AND trim(x.value.value) != ''
        )) AS DOUBLE
      ) / CAST(size(raw_identifiers) AS DOUBLE)
    END AS completeness_score,
    CASE
      WHEN elements IS NULL
        OR size(filter(elements, x -> x.confidence_score IS NOT NULL)) = 0
      THEN NULL
      ELSE aggregate(
        filter(elements, x -> x.confidence_score IS NOT NULL),
        CAST(0.0 AS DOUBLE),
        (acc, x) -> acc + x.confidence_score,
        acc -> acc / CAST(
          size(filter(elements, x -> x.confidence_score IS NOT NULL))
          AS DOUBLE
        )
      )
    END AS parse_confidence_mean,
    CASE
      WHEN raw_identifiers IS NULL
        OR size(filter(raw_identifiers, x -> x.value.confidence_score IS NOT NULL)) = 0
      THEN NULL
      ELSE aggregate(
        filter(raw_identifiers, x -> x.value.confidence_score IS NOT NULL),
        CAST(0.0 AS DOUBLE),
        (acc, x) -> acc + x.value.confidence_score,
        acc -> acc / CAST(
          size(filter(raw_identifiers, x -> x.value.confidence_score IS NOT NULL))
          AS DOUBLE
        )
      )
    END AS extract_confidence_mean
  FROM parsed
)

SELECT
  document_path,
  document_name,
  user_email,
  page_images,
  identifiers,
  elements,
  completeness_score,
  parse_confidence_mean,
  extract_confidence_mean,
  current_timestamp() AS extracted_at
FROM scored
