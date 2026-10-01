-- =============================================================================
-- Silver Layer (parallel branch): Retrieval Chunks (ai_prep_search)
-- =============================================================================
-- One ai_prep_search per document, runs in parallel with
-- silver_extract_identifiers and silver_classify_label. All three branches fan
-- out from STREAM(bronze_doc_parsed) with no dependency on each other.
--
-- Unlike the other two branches, this one does NOT converge into
-- gold_extraction_labels. Its consumer is the load_chunks task (jobs bundle →
-- jobs/load_document_chunks.py), which embeds chunk_to_embed and upserts into
-- Lakebase public.document_chunks for hybrid vector + keyword search.
--
-- Curated fields (label, confidence_score, verdict) are deliberately NOT joined
-- in here: that would couple this branch to silver_classify_label and destroy
-- the parallelism. The agent recovers them with a Lakebase join against
-- gold_extraction_labels_sync at query time instead.
--
-- ai_prep_search takes the parsed_content VARIANT directly (no flatten CTE, as
-- ai_classify needs) and returns chunks that are "embedding-ready" — it does
-- NOT produce embeddings. The embedding call lives in the loader.
--
-- The chunk text is RAW parsed document content, by design: this is the corpus
-- that answers "what does THIS document say", which the curated gold summary in
-- gold_extraction_labels cannot. Access is governed by the Lakebase grants on
-- public.document_chunks, like every other document table — LakeRCM is a shared
-- review queue, so no search tool filters by user_email.
--
-- ⚠ ai_prep_search is Beta and gated on the workspace Previews page
-- (DBR 18.2+ / serverless environmentVersion 3+; this pipeline pins 4).
--
-- 'version' is PINNED to 2.0, which a live probe confirmed (2026-09-30): the
-- supported set is exactly {1.0, 2.0} -- an unsupported value is rejected at
-- compile time with the valid set enumerated, so the value is discoverable
-- rather than guessed. Pinning is not cosmetic here. The two versions have
-- DIFFERENT response contracts: 2.0 returns {document: {contents, pages},
-- error_status} -- the shape the from_json below parses -- while 1.0 returns
-- {error_message, response}, which has no `document` key at all. A bare call
-- follows the workspace default, so were that default ever 1.0,
-- `prep:document:contents` would be NULL, from_json would yield NULL, explode
-- would drop every row, and this table would go SILENTLY EMPTY: no error, no
-- failed task, and document search returning nothing forever. The unpinned call
-- resolved to the 2.0 shape when probed, so this pin preserves today's
-- behaviour and removes the dependency on a default nobody here controls.
-- =============================================================================

CREATE OR REFRESH STREAMING TABLE silver_doc_chunks (

  CONSTRAINT valid_document_path
    EXPECT (document_path IS NOT NULL),

  CONSTRAINT has_chunk_text
    EXPECT (chunk_to_embed IS NOT NULL AND chunk_to_embed != ''),

  -- chunk_to_retrieve is what the agent quotes. Reported rather than dropped so a
  -- parser change that stops populating it is visible in the pipeline's data
  -- quality metrics; the Lakebase column is nullable so it cannot wedge the load.
  CONSTRAINT has_retrievable_text
    EXPECT (chunk_to_retrieve IS NOT NULL AND chunk_to_retrieve != '')
)
COMMENT 'Retrieval chunks (ai_prep_search). Loaded into Lakebase public.document_chunks for hybrid vector + keyword document search.'
TBLPROPERTIES (
  'quality' = 'silver',
  'pipelines.autoOptimize.managed' = 'true'
)
AS

WITH prepped AS (
  SELECT
    document_path,
    document_name,
    user_email,
    ai_prep_search(parsed_content, map('version', '2.0')) AS prep
  -- Bound to one micro-batch per trigger: maxFilesPerTrigger = 1 ensures each
  -- batch ≈ one bronze commit (~50 docs) ≈ ~15 sec of ai_prep_search work. This
  -- granularity keeps streaming latencies bounded and avoids the previous 3h28m+
  -- commits that had no progress events.
  FROM STREAM(bronze_doc_parsed) WITH (maxFilesPerTrigger = 1)
  WHERE parsed_content IS NOT NULL
),

-- Read the chunk array with the same from_json(to_json(VARIANT)) idiom the
-- sibling branches use for parsed_content:document:elements / :pages, so the
-- fields come back typed and the rest of the query needs no variant_get.
-- `metadata` is omitted: ai_prep_search only populates it when the 'schema'
-- option requests document-level extraction, and its shape is unspecified
-- otherwise. The Lakebase table keeps a nullable column for it so adopting
-- 'schema' later needs no migration.
typed AS (
  SELECT
    document_path,
    document_name,
    user_email,
    from_json(
      to_json(prep:document:contents),
      'ARRAY<STRUCT<chunk_id: STRING, chunk_position: INT, chunk_to_retrieve: STRING, chunk_to_embed: STRING, pages: ARRAY<STRUCT<page_id: INT, image_uri: STRING>>>>'
    ) AS contents
  FROM prepped
)

SELECT
  chunk.chunk_id,
  chunk.chunk_position,
  chunk.chunk_to_embed,
  chunk.chunk_to_retrieve,
  -- First page the chunk came from, so a retrieved chunk cites a rendered page
  -- raster (bronze writes these via ai_parse_document imageOutputPath). NULL
  -- when ai_prep_search attributes no page to the chunk.
  chunk.pages[0].page_id AS page_id,
  chunk.pages[0].image_uri AS image_uri,
  document_path,
  document_name,
  user_email,
  current_timestamp() AS prepped_at
FROM typed
LATERAL VIEW explode(contents) c AS chunk
-- A failed prep is filtered structurally, NOT by testing prep:error_status:
-- explode drops any document whose contents array is NULL or empty, which is the
-- shape a failure produces. The docs describe error_status as an OBJECT, and if
-- it is present-but-empty on success rather than NULL, an `error_status IS NULL`
-- predicate would silently discard every chunk in the pipeline. The structural
-- filter is correct either way; the has_chunk_text EXPECT above reports anything
-- that slips through.
WHERE chunk.chunk_to_embed IS NOT NULL
  AND length(trim(chunk.chunk_to_embed)) > 0
