-- =============================================================================
-- Bronze Layer: Document Parsing (ai_parse_document)
-- =============================================================================
-- Auto Loader reads document binaries from the documents_input UC Volume via
-- STREAM read_files() in binaryFile format. The pipeline runs in triggered
-- mode: the lakercm_documents_refresh job kicks it on a file-arrival
-- trigger (external/bulk drops), and the app fires pipelines.start_update()
-- on every manual upload for low-latency UX. Each run processes whatever files
-- Auto Loader has discovered since the last managed checkpoint.
--
-- Why Auto Loader (vs the prior STREAM(bronze_uploads) approach):
--   - Single landing zone: app uploads AND external/bulk drops both land in
--     the volume and get picked up automatically. The old design required the
--     app to also INSERT into bronze_uploads via SQL warehouse, so any file
--     placed in the volume by another process was never ingested.
--   - No SQL warehouse hop on the upload hot path.
--   - File-event-based discovery (managed checkpoint) — no full re-list.
--
-- ai_parse_document() runs OCR + layout extraction on the binary content.
-- No model serving endpoint required — Databricks built-in Document AI.
--
-- user_email is parsed from the filename prefix per the app's
-- {safe_email}_{YYYYMMDD_HHMMSS}_{filename} convention. Files written by
-- external processes that don't follow the convention will populate
-- user_email with whatever sits before the first underscore — Lakebase's
-- _backfill_streamed_documents() COALESCEs missing values to 'streamed'.
--
-- Feeds: silver_classify_label, silver_extract_identifiers
-- =============================================================================

CREATE OR REFRESH STREAMING TABLE bronze_doc_parsed (

  CONSTRAINT valid_document_path
    EXPECT (document_path IS NOT NULL AND document_path != ''),

  CONSTRAINT has_content
    EXPECT (file_size_bytes > 0)
    ON VIOLATION DROP ROW,

  CONSTRAINT has_parsed_content
    EXPECT (parsed_content IS NOT NULL)
    ON VIOLATION DROP ROW
)
COMMENT 'Parsed document output from the documents_input UC volume via ai_parse_document'
TBLPROPERTIES (
  'quality' = 'bronze',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

SELECT
  path AS document_path,
  element_at(split(path, '/'), -1) AS document_name,
  -- The uploader, but ONLY when the filename actually encodes one. The reviewer
  -- app names an upload '<email-local-part>_<timestamp>_<original>', so the
  -- leading segment is the uploader. A document written straight into the volume
  -- has no uploader in its name, and split_part with no delimiter present
  -- returns the WHOLE string — so every generated document reported a
  -- user_email of 'synthetic-1790440247-0011-denial.pdf'. That only went
  -- unnoticed while the corpus happened to be reviewer-app uploads. NULL is the
  -- honest answer: nobody uploaded it.
  -- contains(), not LIKE '%_%': in SQL LIKE, underscore is a single-character
  -- WILDCARD, so '%_%' matches any non-empty string and the guard silently
  -- passed every filename it was written to exclude.
  CASE
    WHEN contains(element_at(split(path, '/'), -1), '_')
      THEN split_part(element_at(split(path, '/'), -1), '_', 1)
    ELSE NULL
  END AS user_email,
  CAST(length(content) AS BIGINT) AS file_size_bytes,
  ai_parse_document(content, map(
    'version', '2.0',
    'descriptionElementTypes', '',                      -- no AI figure/logo descriptions (unused downstream)
    'imageOutputPath', '${documents_page_images_path}'  -- render one PNG per page (reviewer overlay)
  )) AS parsed_content,
  modificationTime AS uploaded_at,
  current_timestamp() AS extracted_at
FROM STREAM read_files(
  '${documents_input_volume_path}',
  format => 'binaryFile',
  pathGlobFilter => '*.{pdf,png,jpg,jpeg,PDF,PNG,JPG,JPEG}',
  includeExistingFiles => true,
  maxFilesPerTrigger => '50'                            -- bound the slow parse per micro-batch
)
