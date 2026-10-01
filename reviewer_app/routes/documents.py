"""
LakeRCM Document Management Routes

Handles document upload to Unity Catalog volumes, retrieval,
extraction comparisons, and human review submission.
"""

import hashlib
import io
import logging
from typing import Optional
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Request,
    HTTPException,
    Depends,
    BackgroundTasks,
)
from fastapi.responses import Response
from databricks.sdk import WorkspaceClient
from datetime import datetime

from schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentStatusCountsResponse,
    DocumentDetailResponse,
    DocumentExtraction,
    ExtractionComparisonItem,
    ExtractionComparisonResponse,
    HumanReviewSubmission,
    HumanReviewResponse,
    DocumentNoteSubmission,
    DocumentNoteResponse,
    ReviewDraftSubmission,
    ReviewDraftResponse,
    ProcessingStatus,
)
from dependencies import (
    get_workspace_client,
    get_lakercm_db,
    get_current_user_email,
    resolve_user_identity,
)
from config import settings
from services.pipeline_trigger import trigger_analytics_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents")


def _trigger_documents_pipeline(workspace_client: WorkspaceClient) -> None:
    """Fire-and-forget pipeline kick for low-latency manual uploads.

    The pipeline runs in triggered mode; the cron job refreshes it every
    5 minutes. To give app uploads sub-minute visibility, every successful
    upload also asks the pipeline to start an update immediately. If an
    update is already in flight, Auto Loader will discover the new file
    during that run — we treat the API's "update already running" response
    as success, not an error.
    """
    pipeline_id = settings.pipeline_id
    if not pipeline_id:
        logger.warning("PIPELINE_ID not configured; skipping pipeline trigger")
        return
    try:
        workspace_client.pipelines.start_update(
            pipeline_id=pipeline_id, full_refresh=False
        )
        logger.info("Triggered documents pipeline update: %s", pipeline_id)
    except Exception as e:
        # Most common case here is the SDK raising on a "update already
        # active" 409. That's the expected idempotent path — Auto Loader
        # picks up the new file in the in-flight run.
        msg = str(e).lower()
        if "already" in msg or "conflict" in msg or "409" in msg:
            logger.debug(
                "Pipeline %s already updating; new file will be in current run",
                pipeline_id,
            )
        else:
            logger.warning(
                "pipelines.start_update(%s) failed: %s (the volume "
                "file-arrival trigger on lakercm-documents-refresh "
                "will pick up the file)",
                pipeline_id,
                e,
            )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    workspace_client: WorkspaceClient = Depends(get_workspace_client),
    db=Depends(get_lakercm_db),
):
    user_email = get_current_user_email(request)

    allowed_types = ["application/pdf", "image/png", "image/jpeg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, detail="Only PDF, PNG, and JPEG files are allowed"
        )

    file_content = await file.read()
    file_size = len(file_content)
    max_size = 10 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size must be less than {max_size / (1024*1024):.0f}MB",
        )

    try:
        content_hash = hashlib.sha256(file_content).hexdigest()

        existing = db.find_document_by_content_hash(content_hash)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Duplicate file detected. A document with identical content "
                    f"was already uploaded as '{existing['document_name']}' "
                    f"on {existing['upload_timestamp']}."
                ),
            )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_email = user_email.split("@")[0]
        unique_filename = f"{safe_email}_{timestamp}_{file.filename}"

        volume_path = f"/Volumes/{settings.catalog}/{settings.lakercm_schema}/documents_input/{unique_filename}"
        # Stored form must match Auto Loader's path column (a dbfs: URI) —
        # status sync and the streamed-doc backfill join medical_documents.
        # file_path against gold_extraction_labels_sync.document_path by exact
        # string equality, so a bare /Volumes/... row is never matched: it
        # stays 'processing' forever and the backfill inserts a duplicate.
        file_path = f"dbfs:{volume_path}"

        workspace_client.files.upload(
            file_path=volume_path, contents=io.BytesIO(file_content), overwrite=False
        )

        logger.info("Uploaded document to volume: %s", volume_path)

        document_data = {
            "user_email": user_email,
            "document_name": file.filename,
            "file_path": file_path,
            "file_size": file_size,
            "document_type": document_type,
            "notes": notes,
            "processing_status": ProcessingStatus.PROCESSING,
            "content_hash": content_hash,
        }

        document_id = db.create_document_record(document_data)

        # Ask the pipeline to start an update — Auto Loader will pick the
        # file out of the volume. Idempotent: if an update is already
        # running, that's fine. BackgroundTasks runs after the response
        # is sent so the user doesn't wait on the SDK call.
        background_tasks.add_task(_trigger_documents_pipeline, workspace_client)

        return DocumentUploadResponse(
            id=document_id,
            filename=file.filename,
            volume_path=volume_path,
            file_size=file_size,
            status=ProcessingStatus.PROCESSING,
            message="Document uploaded successfully. Processing will begin shortly.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload document: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to upload document: {str(e)}"
        )
    finally:
        await file.close()


_VALID_STATUS_FILTERS = {"processing", "pending", "reviewed", "auto_verified"}


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    limit: int = 9,
    offset: int = 0,
    include_auto_verified: bool = False,
    status: Optional[str] = None,
    search: Optional[str] = None,
    db=Depends(get_lakercm_db),
):
    """List documents with live status derived from the gold sync table.

    Query params:
      limit/offset : pagination (UI defaults to 9 per page)
      include_auto_verified : include auto-verified docs
      status : 'processing' | 'pending' | 'reviewed' | 'auto_verified' | None
      search : case-insensitive substring on document_name / file_path / notes
    """
    try:
        status_filter = status if status in _VALID_STATUS_FILTERS else None
        search_term = search.strip() if search else None
        if not search_term:
            search_term = None

        documents = db.list_all_documents(
            include_auto_verified=include_auto_verified,
            limit=limit,
            offset=offset,
            status_filter=status_filter,
            search=search_term,
        )

        total = db.count_all_documents(
            include_auto_verified=include_auto_verified,
            status_filter=status_filter,
            search=search_term,
        )

        doc_ids = [str(d["id"]) for d in documents]
        latest_reviews = db.get_latest_review_per_document(doc_ids)

        for doc in documents:
            review = latest_reviews.get(str(doc["id"]))
            if review:
                doc["review_verdict"] = review["verdict"]
                doc["review_reviewer_email"] = review["reviewer_email"]
                identity = resolve_user_identity(review["reviewer_email"])
                doc["review_reviewer_name"] = (
                    identity.get("display_name") or review["reviewer_email"]
                )

        return DocumentListResponse(
            documents=documents, total_count=total, limit=limit, offset=offset
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/status-counts", response_model=DocumentStatusCountsResponse)
async def get_document_status_counts(
    include_auto_verified: bool = False,
    db=Depends(get_lakercm_db),
):
    """Counts for the dashboard Documents Processed tile."""
    try:
        counts = db.get_status_counts(include_auto_verified=include_auto_verified)
        return DocumentStatusCountsResponse(**counts)
    except Exception as e:
        logger.error("Failed to get status counts: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get status counts: {str(e)}"
        )


@router.get("/drafts", response_model=dict)
async def list_review_drafts(
    request: Request,
    db=Depends(get_lakercm_db),
):
    """Document ids the current reviewer has an unsubmitted draft for.

    Declared BEFORE GET /{document_id}: FastAPI matches in declaration order, so
    the other way round this path would be swallowed as a document id named
    "drafts" and 404 on a UUID parse. Same reason /status-counts sits above it.

    Never fails the list page: an error returns an empty set, because a missing
    badge is a cosmetic loss and a 500 here would break the queue.
    """
    user_email = get_current_user_email(request)
    try:
        return {"document_ids": db.list_review_draft_ids(user_email)}
    except Exception as e:
        logger.warning("draft id lookup failed: %s", e)
        return {"document_ids": []}


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    request: Request,
    document_id: str,
    db=Depends(get_lakercm_db),
):
    try:
        document = db.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.get("processing_status") not in ("pending", "auto_verified"):
            return DocumentDetailResponse(
                document=document, elements=[], extractions=DocumentExtraction()
            )

        elements = db.get_document_elements(document.get("file_path"))
        extractions = db.get_document_extractions(document.get("file_path"))

        return DocumentDetailResponse(
            document=document, elements=elements, extractions=extractions
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.get("/{document_id}/image")
async def get_document_image(
    request: Request,
    document_id: str,
    page: int = 0,
    workspace_client: WorkspaceClient = Depends(get_workspace_client),
    db=Depends(get_lakercm_db),
):
    try:
        document = db.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = document.get("file_path")
        document_name = document.get("document_name", "file")

        ext = document_name.rsplit(".", 1)[-1].lower() if "." in document_name else ""
        content_type_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "pdf": "application/pdf",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")

        # Serve the parser's rendered page PNG when available so bounding boxes
        # overlay correctly (box coords live in the parser's render canvas, not
        # the raw upload's pixel space). Fall back to the original upload for
        # legacy docs that predate page rendering. get_document_page_images is
        # fail-soft (returns [] if the column/row is absent).
        serve_path = file_path
        served_content_type = content_type
        page_images = db.get_document_page_images(file_path)
        if page_images:
            idx = page if 0 <= page < len(page_images) else 0
            if page_images[idx]:
                serve_path = page_images[idx]
                served_content_type = "image/png"

        download = workspace_client.files.download(serve_path)

        if download.contents is None:
            raise HTTPException(status_code=404, detail="File content not available")

        content = download.contents.read()
        return Response(
            content=content,
            media_type=served_content_type,
            headers={
                "Content-Disposition": f'inline; filename="{document_name}"',
                "Cache-Control": "private, max-age=3600",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to serve document image: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to load document image: {str(e)}"
        )


@router.get("/{document_id}/comparisons", response_model=ExtractionComparisonResponse)
async def get_extraction_comparisons(
    request: Request,
    document_id: str,
    db=Depends(get_lakercm_db),
):
    try:
        document = db.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = document.get("file_path")
        document_name = document.get("document_name", "")

        rows = db.get_extraction_comparisons(file_path)
        items = [ExtractionComparisonItem(**row) for row in rows]

        return ExtractionComparisonResponse(
            document_id=document_id,
            document_name=document_name,
            total_items=len(items),
            items=items,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get extraction comparisons: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get extraction comparisons: {str(e)}"
        )


@router.post("/{document_id}/review", response_model=HumanReviewResponse)
async def submit_extraction_review(
    request: Request,
    document_id: str,
    body: HumanReviewSubmission,
    background_tasks: BackgroundTasks,
    db=Depends(get_lakercm_db),
    workspace_client=Depends(get_workspace_client),
):
    user_email = get_current_user_email(request)

    try:
        document = db.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if body.verdict != "correct" and not body.reasoning:
            raise HTTPException(
                status_code=422,
                detail="Reasoning is required when verdict is not 'correct'",
            )

        result = db.upsert_extraction_review(
            document_id=document_id,
            reviewer_email=user_email,
            verdict=body.verdict.value,
            reasoning=body.reasoning,
            corrections=body.corrections,
        )

        # An agent proposal the reviewer approved and then edited is 'modified',
        # not 'accepted'. Only the submitted corrections reveal that, so it is
        # reconciled here. Best-effort: the measurement must never fail a review
        # that has already been written.
        try:
            db.reconcile_accepted_proposals(document_id, body.corrections)
        except Exception as e:
            logger.warning("proposal reconciliation skipped for %s: %s", document_id, e)

        # The draft has served its purpose: the verdict is now on the shared
        # review row. Leaving it behind would be actively wrong, because the
        # client prefers a draft over a submitted review on load (a draft is
        # normally the NEWER edit) and would resurrect the pre-submit state.
        # Best-effort for the same reason as the reconciliation above: never fail
        # a review that has already been written.
        try:
            db.delete_review_draft(document_id, user_email)
        except Exception as e:
            logger.warning("draft cleanup skipped for %s: %s", document_id, e)

        # Refresh review analytics event-driven — the file-arrival path only
        # covers document ingestion, not a human verdict landing in Lakebase.
        background_tasks.add_task(trigger_analytics_pipeline, workspace_client)

        return HumanReviewResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to submit extraction review: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to submit extraction review: {str(e)}"
        )


@router.get("/{document_id}/review", response_model=HumanReviewResponse)
async def get_extraction_review(
    request: Request,
    document_id: str,
    db=Depends(get_lakercm_db),
):
    try:
        document = db.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        review = db.get_extraction_review(document_id)

        if not review:
            raise HTTPException(
                status_code=404, detail="No review found for this document"
            )

        return HumanReviewResponse(**review)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get extraction review: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get extraction review: {str(e)}"
        )


@router.get("/{document_id}/notes", response_model=DocumentNoteResponse)
async def get_document_notes(
    request: Request,
    document_id: str,
    db=Depends(get_lakercm_db),
):
    """Return the current reviewer's private notepad for a document.

    Always 200s: an empty notepad (no row yet) returns note_text="" rather
    than 404, so the frontend can render the editor without special-casing.
    """
    user_email = get_current_user_email(request)
    try:
        document = db.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        note = db.get_document_notes(document_id, user_email)
        if not note:
            return DocumentNoteResponse(
                document_id=document_id, user_email=user_email, note_text=""
            )
        return DocumentNoteResponse(**note)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document notes: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get document notes: {str(e)}"
        )


@router.post("/{document_id}/notes", response_model=DocumentNoteResponse)
async def save_document_notes(
    request: Request,
    document_id: str,
    body: DocumentNoteSubmission,
    db=Depends(get_lakercm_db),
):
    """Create or replace the current reviewer's notepad for a document."""
    user_email = get_current_user_email(request)
    try:
        document = db.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        result = db.upsert_document_notes(
            document_id=document_id,
            user_email=user_email,
            note_text=body.note_text or "",
        )
        return DocumentNoteResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save document notes: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save document notes: {str(e)}"
        )


@router.get("/{document_id}/draft", response_model=ReviewDraftResponse)
async def get_review_draft(
    request: Request,
    document_id: str,
    db=Depends(get_lakercm_db),
):
    """Return the current reviewer's UNSUBMITTED draft for a document.

    Always 200s, like the notepad: no row yet returns an empty draft with
    exists=False rather than a 404, so the client has one shape to handle and
    never treats a missing draft as an error.
    """
    user_email = get_current_user_email(request)
    try:
        document = db.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        draft = db.get_review_draft(document_id, user_email)
        if not draft:
            return ReviewDraftResponse(
                document_id=document_id, user_email=user_email, exists=False
            )
        return ReviewDraftResponse(**draft, exists=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get review draft: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get review draft: {str(e)}"
        )


@router.post("/{document_id}/draft", response_model=ReviewDraftResponse)
async def save_review_draft(
    request: Request,
    document_id: str,
    body: ReviewDraftSubmission,
    db=Depends(get_lakercm_db),
):
    """Create or replace the current reviewer's draft for a document.

    POST rather than PUT deliberately: the client flushes a pending draft on
    visibilitychange via navigator.sendBeacon, which can only issue a POST.
    Losing the last edit when a reviewer switches tabs is the exact failure this
    endpoint exists to prevent, so the verb follows the transport.

    Writing a draft NEVER submits a review: no verdict validation, no analytics
    trigger, no proposal reconciliation. That asymmetry with the /review
    endpoint is the point.
    """
    user_email = get_current_user_email(request)
    try:
        document = db.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        result = db.upsert_review_draft(
            document_id=document_id,
            user_email=user_email,
            verdict=body.verdict.value if body.verdict else None,
            reasoning=body.reasoning,
            corrections=body.corrections,
        )
        return ReviewDraftResponse(**result, exists=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save review draft: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save review draft: {str(e)}"
        )


@router.delete("/{document_id}/draft", response_model=dict)
async def discard_review_draft(
    request: Request,
    document_id: str,
    db=Depends(get_lakercm_db),
):
    """Discard the current reviewer's draft without submitting anything."""
    user_email = get_current_user_email(request)
    try:
        deleted = db.delete_review_draft(document_id, user_email)
        return {"deleted": deleted}
    except Exception as e:
        logger.error("Failed to discard review draft: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to discard review draft: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(
    request: Request,
    document_id: str,
    db=Depends(get_lakercm_db),
):
    try:
        document = db.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        db.soft_delete_document(document_id)

        return {"message": "Document deleted successfully", "document_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )


@router.post("/sync-status")
async def sync_document_status(
    db=Depends(get_lakercm_db),
):
    """Sync processing status from the Lakebase-synced gold table.

    For each processing / stale-timestamp / unreviewed-pending doc:
      - the pipeline auto-verified it (is_automated) → 'auto_verified'
      - else                                        → 'pending' (awaiting review)

    Reads exclusively from Lakebase. The pipeline makes the call
    (gold_extraction_labels.sql): confidence at or above the threshold AND no
    invalid or non-billable code or missing member ID.
    """
    try:
        processing = db.get_pending_documents()
        needs_resync = db.get_documents_needing_timestamp_resync()
        pending_recheck = db.get_unreviewed_ready_documents()

        docs_by_path: dict = {}
        for doc in processing:
            docs_by_path[doc["file_path"]] = {"id": doc["id"], "kind": "processing"}
        for doc in needs_resync:
            if doc["file_path"] not in docs_by_path:
                docs_by_path[doc["file_path"]] = {"id": doc["id"], "kind": "resync"}
        for doc in pending_recheck:
            if doc["file_path"] not in docs_by_path:
                docs_by_path[doc["file_path"]] = {
                    "id": doc["id"],
                    "kind": "pending_recheck",
                }

        if not docs_by_path:
            return {
                "pending": 0,
                "auto_verified": 0,
                "message": "No documents to sync",
            }

        paths = list(docs_by_path.keys())
        label_rows = db.get_gold_labels_by_paths(paths)

        pending_count = 0
        auto_count = 0
        pending_paths: list = []
        auto_verified_paths: list = []

        for row in label_rows:
            path = row["document_path"]
            doc = docs_by_path.get(path)
            if not doc:
                continue
            extracted_at = row.get("extracted_at")
            if not extracted_at:
                continue
            target_status = "auto_verified" if row.get("is_automated") else "pending"
            # pending_recheck rows are already 'pending' — only write when
            # promoting to auto_verified. Avoids redundant writes and
            # never downgrades.
            if doc["kind"] == "pending_recheck" and target_status != "auto_verified":
                continue
            if db.update_document_status_with_timestamp(
                path, target_status, extracted_at
            ):
                if target_status == "auto_verified":
                    auto_count += 1
                    auto_verified_paths.append(path)
                else:
                    pending_count += 1
                    pending_paths.append(path)
                logger.info(
                    "Synced status=%s extracted_at=%s: %s",
                    target_status,
                    extracted_at,
                    path,
                )

        return {
            "pending": pending_count,
            "auto_verified": auto_count,
            "total_checked": len(paths),
            "pending_paths": pending_paths,
            "auto_verified_paths": auto_verified_paths,
        }

    except Exception as e:
        logger.error("Failed to sync document status: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to sync document status: {str(e)}"
        )
