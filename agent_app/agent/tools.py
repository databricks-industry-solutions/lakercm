"""
Synchronous LangChain tools for LakeRCM agent.

All tools read from Lakebase Postgres — the reviewer app's
public.* tables and the UC-synced gold tables. The LLM
never sees uncleaned bronze/raw-text data: document details come from
the curated gold layer (label, identifiers, elements, confidence).

TWO EXCEPTIONS. The analytics tier (`get_review_statistics`,
`get_pipeline_latency_stats`, `query_lakehouse`) reads the governed gold layer
on the Reyden (Lakehouse//RT) warehouse through the Unity AI Gateway MCP service
system.ai.dbsql — see "Reyden analytics tier" below; point lookups stay on
Lakebase. And `search_payer_policy` queries a Vector Search index over the
synthetic payer-policy corpus (see scripts/payer_policy_content.py) rather than
Lakebase. It is also the only tool that emits an MLflow RETRIEVER span, which is
what lets the already-registered `RetrievalGroundedness` judge score live
traffic (it returns None without one). Note `semantic_search_documents` is
retrieval-flavored too, but stays inside Lakebase (pgvector) and emits a normal
tool span.

Security: ContextVar tracks the authenticated user_email for audit
logging. LakeRCM is an admin/review tool — all authenticated users
can see all documents.
"""

import contextlib
import contextvars
import functools
import itertools
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Optional

from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool

from agent.guards import neutralize_if_injected
from config import settings
from services.lakehouse_db import get_db

logger = logging.getLogger(__name__)


def _guard_tool_output(output, source: str):
    """Neutralize a tool result if it carries injection text (OWASP LLM01).

    Document text is OCR'd from uploaded PDFs and `ai_extract` pulls identifier
    values VERBATIM, so an uploaded document can carry "ignore previous
    instructions ..." straight into the model's context (indirect injection).
    The result is fenced as data ONLY when a signal is present, so the normal
    path stays byte-identical and the recorded eval fixtures / GEPA baselines do
    not move. A JSON result is scanned as the object it encodes: JSON escapes a
    line break inside a value as backslash + n, which hid "Patient
    notes.\\nIgnore all previous instructions" from every boundary pattern.

    The warning carries NO document text, only category names: shape-based
    redaction removes identifier shapes, not the names, addresses and claim
    numbers a denial letter holds (seventh review). Non-string results pass
    through untouched.
    """
    if not isinstance(output, str):
        return output
    guarded, verdict = neutralize_if_injected(output, source=source)
    if verdict.detected:
        logger.warning(
            "injection signal in %s result (%s; %s confidence)",
            source,
            ", ".join(verdict.categories),
            "high" if verdict.high_confidence else "low",
        )
        _record_injection_signal(
            verdict.categories, source, high_confidence=verdict.high_confidence
        )
    return guarded


# Signals already recorded on each trace, as (categories, high_confidence,
# sources that detected HIGH, sources that detected LOW). Trace metadata is
# last-write-wins, so a turn whose first result was a HIGH-confidence attack and
# whose later one was a LOW correction letter ended recorded as 'low', and the
# alert, which pages on 'high', missed the attack (seventh review). Each write
# now merges: categories are unioned, confidence only ever rises, and every
# source is kept. Bounded, and locked because tool calls can run concurrently.
_SIGNALS_BY_TRACE: OrderedDict[
    str, tuple[tuple[str, ...], bool, tuple[str, ...], tuple[str, ...]]
] = OrderedDict()
_SIGNALS_MAX_TRACES = 1024
_SIGNALS_LOCK = threading.Lock()


def _merge_trace_signal(
    trace_id: str | None,
    categories: tuple[str, ...],
    high_confidence: bool,
    source: str,
) -> tuple[tuple[str, ...], bool, tuple[str, ...]]:
    """Union this detection into what the trace already recorded.

    Returns (categories, high_confidence, sources). Sources that produced a
    HIGH-confidence detection come first, so the first source an on-call
    engineer reads is the one that paged. It was last-write-wins: a later LOW
    detection from another tool renamed the source of a paged HIGH attack, and
    the wrong tool got investigated (eighth review).
    """
    if not trace_id:
        return tuple(categories), high_confidence, (source,)
    with _SIGNALS_LOCK:
        seen, was_high, high_src, low_src = _SIGNALS_BY_TRACE.pop(
            trace_id, ((), False, (), ())
        )
        if high_confidence:
            high_src = tuple(dict.fromkeys(high_src + (source,)))
        else:
            low_src = tuple(dict.fromkeys(low_src + (source,)))
        merged = (
            tuple(dict.fromkeys(seen + tuple(categories))),
            was_high or high_confidence,
            high_src,
            low_src,
        )
        _SIGNALS_BY_TRACE[trace_id] = merged
        while len(_SIGNALS_BY_TRACE) > _SIGNALS_MAX_TRACES:
            _SIGNALS_BY_TRACE.popitem(last=False)
    return merged[0], merged[1], tuple(dict.fromkeys(high_src + low_src))


def _record_injection_signal(
    categories: tuple[str, ...], source: str, *, high_confidence: bool = False
) -> None:
    """Record a detected injection on the active trace so it is ALERTABLE.

    A log line alone is not operational: nothing pages on it. Writing metadata
    lands the signal in agent_traces_trace_metadata, which the
    `lakercm_alert_injection_signal` DBSQL alert queries; tags make it
    filterable in the Trace UI (the same dual-write convention as
    services/observability.set_prompt_context).

    Only category NAMES, the tool sources and a confidence are recorded — never
    document text. The alert pages on HIGH confidence only: an override alone
    ("please disregard the previous instructions") is also ordinary
    correction-letter prose, so it is fenced and recorded as `low` but does not
    page (sixth review; see guards._HIGH_CONFIDENCE_CATEGORIES). Several
    detections in one turn MERGE (see _SIGNALS_BY_TRACE): categories and
    sources are unioned and confidence never drops. Fully isolated: telemetry
    must never break a tool call.
    """
    try:
        import mlflow

        span = mlflow.get_current_active_span()
        categories, high_confidence, sources = _merge_trace_signal(
            getattr(span, "trace_id", None), categories, high_confidence, source
        )
        cats = ",".join(categories)
        srcs = ",".join(sources)
        confidence = "high" if high_confidence else "low"
        mlflow.update_current_trace(
            tags={
                "injection_signal": cats,
                "injection_source": srcs,
                "injection_confidence": confidence,
            },
            metadata={
                "guard.injection_signal": cats,
                "guard.injection_source": srcs,
                "guard.injection_confidence": confidence,
            },
        )
    except Exception as e:  # pragma: no cover — telemetry must never break a tool
        logger.debug("injection signal trace tagging skipped: %s", e)


_workspace_client: Optional[WorkspaceClient] = None


def _get_workspace_client() -> WorkspaceClient:
    """Lazy-init a WorkspaceClient. In Databricks Apps the SDK picks up
    OAuth M2M credentials from the runtime env automatically."""
    global _workspace_client
    if _workspace_client is None:
        _workspace_client = WorkspaceClient()
    return _workspace_client


# ContextVar for audit — tracks who is making queries
_authorized_user_email: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_authorized_user_email", default=None
)


def set_authorized_user_email(email: str):
    """Set the authorized user email for the current request context."""
    _authorized_user_email.set(email)


def _get_authorized_user_email() -> str:
    """Get the authorized user email. Returns 'unknown' if not set."""
    return _authorized_user_email.get() or "unknown"


# ContextVar for the reviewer-pane "open document". The in-document assistant
# runs against ONE document; the reviewer app injects its id via AG-UI
# forwardedProps.document_id, which main.py parks here before the graph runs.
# The reviewer-action tools below read it to scope their reads and to stamp
# the staged actions they emit. When it is unset (the general chat, where no
# document is open) those tools decline to act.
_active_document_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_document_id", default=None
)


def set_active_document_id(document_id: str | None):
    """Set the reviewer-pane open document id for the current request context."""
    _active_document_id.set(document_id or None)


def _get_active_document_id() -> str | None:
    """Get the reviewer-pane open document id, or None outside the pane."""
    return _active_document_id.get()


# Returned (as an "error" field) by the reviewer-action tools when they are
# called with no open document — i.e. from the general chat rather than the
# in-document assistant. Phrased so the LLM relays it usefully to the user.
_NO_ACTIVE_DOC_MSG = (
    "This action is only available in the in-document assistant, where a "
    "document is open. There is no document open in this conversation."
)

# The three review verdicts, mirrored from reviewer_app/schemas.py
# ExtractionVerdict. Keep in sync if that enum changes.
_VALID_VERDICTS = ("correct", "partially_correct", "incorrect")


_GOLD_UNAVAILABLE_MSG = (
    "Extraction results cannot be read in THIS environment: its Lakebase copy of "
    "the gold extraction table is missing, so no document tool can see "
    "extractions, statuses or review reasons. This is a setup gap, NOT an empty "
    "environment — documents may well have been uploaded and processed. Say so "
    "plainly instead of concluding that no documents exist, and do not report a "
    "count of zero as fact. Document metadata (names, uploads) still works."
)


# Disjoint status buckets exposed to the LLM. Mirror reviewer_app's
# get_status_counts() semantics exactly so the agent's numbers match the
# dashboard pills. Source of truth: reviewer_app/services/lakehouse_db.py
# (_STATUS_CASE_SQL + _build_list_query). If the predicates drift, the
# agent will silently disagree with the reviewer UI.
_VALID_STATUS_FILTERS = (
    "processing",
    "pending",
    "reviewed",
    "auto_verified",
    "failed",
)

# auto_verified is the pipeline's call (gold_extraction_labels.is_automated):
# confidence at or above the threshold AND no invalid or non-billable code and
# no missing member ID. The reviewer app reads the same column.
_STATUS_CASE_SQL = (
    "CASE "
    "  WHEN d.processing_status = 'failed' THEN 'failed' "
    "  WHEN g.document_path IS NULL THEN COALESCE(d.processing_status, 'processing') "
    "  WHEN g.is_automated THEN 'auto_verified' "
    "  ELSE 'pending' "
    "END"
)

# Why the pipeline sent a document to review, when the Lakebase copy of gold has
# caught up with the column (NULL for documents processed before it existed).
_REVIEW_REASONS_NOTE = (
    "review_reasons lists every reason the pipeline sent a document to review: "
    "invalid_code (a code missing from the terminology or malformed), "
    "non_billable_code (a code that is not billable at that level), "
    "missing_member_id (no member or subscriber ID), low_confidence (blended "
    "confidence below the auto-verify threshold), confidence_unavailable (no "
    "confidence score could be computed). A document is held exactly when this "
    "list is non-empty. An EMPTY list on a pending document therefore does not "
    "mean low confidence — it means no reason was recorded, which is a defect "
    "worth reporting rather than explaining away. Null means the document was "
    "processed before the pipeline recorded reasons at all."
)


def _gold_sync() -> str:
    """The Lakebase copy of gold_extraction_labels, in this deploy's schema.

    A synced table takes its Postgres schema from its Unity Catalog schema, so
    dev reads lakercm_dev.* and prod lakercm.*; LAKERCM_SCHEMA
    carries it (bundles/apps/resources/agent_app.yml).
    """
    return f"{settings.schema_name}.gold_extraction_labels_sync"


def _review_reasons_column(db) -> str:
    """g.review_reasons once the synced table has it, else a NULL stand-in."""
    if db.gold_sync_has_column("review_reasons"):
        return "g.review_reasons"
    return "NULL::JSONB AS review_reasons"


@tool
def search_documents(
    search_text: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Look up documents by name (substring match). Returns document names, file sizes, uploader email, upload time. Use this ONLY for name-based lookup — for status counts or status-bucket listings use get_documents_by_status instead (search_documents is paginated and orders by recency, so it cannot answer 'how many X' questions reliably)."""
    name_filter = ""
    params: list = []
    if search_text:
        name_filter = "AND LOWER(d.document_name) LIKE %s"
        params.append(f"%{search_text.lower()}%")

    db = get_db()
    if db.gold_sync_available():
        rows = db.execute_query(
            f"""
            SELECT d.document_name,
                   d.file_path          AS document_path,
                   d.file_size,
                   d.user_email,
                   d.upload_timestamp,
                   d.processing_status,
                   g.extracted_at,
                   g.label
            FROM public.medical_documents d
            LEFT JOIN {_gold_sync()} g
              ON g.document_path = d.file_path
            WHERE d.deleted_at IS NULL {name_filter}
            ORDER BY d.upload_timestamp DESC
            LIMIT %s
            """,
            (*params, limit),
        )
    else:
        rows = db.execute_query(
            f"""
            SELECT d.document_name,
                   d.file_path          AS document_path,
                   d.file_size,
                   d.user_email,
                   d.upload_timestamp,
                   d.processing_status,
                   NULL::TIMESTAMP AS extracted_at,
                   NULL::TEXT AS label
            FROM public.medical_documents d
            WHERE d.deleted_at IS NULL {name_filter}
            ORDER BY d.upload_timestamp DESC
            LIMIT %s
            """,
            (*params, limit),
        )
    return json.dumps({"count": len(rows), "documents": rows}, indent=2, default=str)


@tool
def get_documents_by_status(
    status: Optional[str] = None,
    limit: int = 25,
) -> str:
    """Count or list medical documents by their processing status.

    Status buckets (disjoint — sum equals total documents):
      - processing: pipeline still extracting
      - pending: extraction done, awaiting human review
      - reviewed: a human has submitted a verdict
      - auto_verified: the pipeline auto-verified it — AI extraction confidence
        is at or above the auto-verify threshold AND it has no invalid or
        non-billable code and no missing member ID — and no human review has
        been submitted. These documents bypass the review queue and have NO
        row in the reviews table — they will not appear in review-statistics
        or recent-reviews tools.
      - failed: pipeline failed

    Without a status arg, returns counts for every bucket.
    With a status arg, returns the documents in that bucket (newest first, up
    to `limit`), each with its `review_reasons`: why the pipeline sent it to
    review. Use this tool when asked "how many are auto-verified", "what's the
    status breakdown", "why is this waiting for review", or to list documents
    in a specific status.
    """
    db = get_db()
    threshold = settings.auto_verdict_threshold

    if not db.gold_sync_available():
        # Without the gold sync table, auto_verified can't be derived. Fall
        # back to processing_status only — same shape, fewer buckets populated.
        if status is None:
            rows = db.execute_query("""
                SELECT COALESCE(processing_status, 'processing') AS bucket,
                       COUNT(*) AS cnt
                FROM public.medical_documents
                WHERE deleted_at IS NULL
                GROUP BY bucket
                """)
            counts = {b: 0 for b in _VALID_STATUS_FILTERS}
            for r in rows:
                bucket = r.get("bucket") or "processing"
                if bucket in counts:
                    counts[bucket] = int(r["cnt"])
            counts["total"] = sum(counts.values())
            counts["auto_verify_threshold"] = threshold
            counts["note"] = _GOLD_UNAVAILABLE_MSG
            return json.dumps(counts, indent=2, default=str)
        return json.dumps(
            {
                "status": status,
                "count": 0,
                "documents": [],
                "auto_verify_threshold": threshold,
                "note": _GOLD_UNAVAILABLE_MSG,
            },
            indent=2,
            default=str,
        )

    classified_cte = f"""
        WITH classified AS (
            SELECT d.id,
                   d.document_name,
                   d.file_path,
                   d.user_email,
                   d.upload_timestamp,
                   d.processing_timestamp,
                   g.confidence_score,
                   g.label AS extracted_label,
                   {_review_reasons_column(db)},
                   {_STATUS_CASE_SQL} AS effective_status,
                   EXISTS (
                       SELECT 1 FROM public.document_extraction_reviews r
                       WHERE r.document_id = d.id
                   ) AS has_review
            FROM public.medical_documents d
            LEFT JOIN {_gold_sync()} g
              ON g.document_path = d.file_path
            WHERE d.deleted_at IS NULL
        )
    """

    if status is None:
        # All-counts mode: one round-trip, mirrors reviewer's get_status_counts
        # disjoint partition (a reviewed doc lands in 'reviewed' regardless of
        # its effective_status).
        rows = db.execute_query(
            classified_cte + """
            SELECT
                SUM(CASE WHEN effective_status = 'processing' THEN 1 ELSE 0 END)::INT AS processing,
                SUM(CASE WHEN effective_status = 'pending' AND has_review = false THEN 1 ELSE 0 END)::INT AS pending,
                SUM(CASE WHEN has_review = true THEN 1 ELSE 0 END)::INT AS reviewed,
                SUM(CASE WHEN effective_status = 'auto_verified' AND has_review = false THEN 1 ELSE 0 END)::INT AS auto_verified,
                SUM(CASE WHEN effective_status = 'failed' THEN 1 ELSE 0 END)::INT AS failed
            FROM classified
            """,
        )
        row = rows[0] if rows else {}
        counts = {b: int(row.get(b) or 0) for b in _VALID_STATUS_FILTERS}
        counts["total"] = sum(counts.values())
        counts["auto_verify_threshold"] = threshold
        return json.dumps(counts, indent=2, default=str)

    if status not in _VALID_STATUS_FILTERS:
        return json.dumps(
            {
                "error": f"Unknown status '{status}'. Valid: {list(_VALID_STATUS_FILTERS)}"
            }
        )

    bucket_predicate = {
        "processing": "effective_status = 'processing'",
        "pending": "effective_status = 'pending' AND has_review = false",
        "reviewed": "has_review = true",
        "auto_verified": "effective_status = 'auto_verified' AND has_review = false",
        "failed": "effective_status = 'failed'",
    }[status]

    rows = db.execute_query(
        classified_cte + f"""
        SELECT id, document_name, file_path, user_email,
               upload_timestamp, processing_timestamp,
               confidence_score, extracted_label, review_reasons,
               effective_status, has_review
        FROM classified
        WHERE {bucket_predicate}
        ORDER BY COALESCE(upload_timestamp, processing_timestamp) DESC NULLS LAST
        LIMIT %s
        """,
        (limit,),
    )
    return json.dumps(
        {
            "status": status,
            "count": len(rows),
            "auto_verify_threshold": threshold,
            "review_reasons_note": _REVIEW_REASONS_NOTE,
            "documents": rows,
        },
        indent=2,
        default=str,
    )


@tool
def get_document_details(document_name: str) -> str:
    """Get detailed information about a specific document by its name — metadata, document type label, extracted identifiers (structured key-value pairs), elements, extraction confidence, and whether the pipeline auto-verified it or sent it to review (is_automated, review_reasons). All fields come from the curated gold layer; no raw OCR text is exposed."""
    db = get_db()
    if db.gold_sync_available():
        rows = db.execute_query(
            f"""
            SELECT d.document_name,
                   d.file_path            AS document_path,
                   d.file_size,
                   d.user_email,
                   d.upload_timestamp,
                   d.document_type,
                   d.notes,
                   d.processing_status,
                   g.label,
                   g.identifiers,
                   g.elements,
                   g.confidence_score,
                   g.is_automated,
                   {_review_reasons_column(db)},
                   g.extracted_at
            FROM public.medical_documents d
            LEFT JOIN {_gold_sync()} g
              ON g.document_path = d.file_path
            WHERE d.deleted_at IS NULL
              AND LOWER(d.document_name) LIKE %s
            ORDER BY d.upload_timestamp DESC
            LIMIT 1
            """,
            (f"%{document_name.lower()}%",),
        )
    else:
        rows = db.execute_query(
            """
            SELECT d.document_name,
                   d.file_path            AS document_path,
                   d.file_size,
                   d.user_email,
                   d.upload_timestamp,
                   d.document_type,
                   d.notes,
                   d.processing_status,
                   NULL::TEXT AS label,
                   NULL::JSONB AS identifiers,
                   NULL::JSONB AS elements,
                   NULL::DOUBLE PRECISION AS confidence_score,
                   NULL::TIMESTAMP AS extracted_at
            FROM public.medical_documents d
            WHERE d.deleted_at IS NULL
              AND LOWER(d.document_name) LIKE %s
            ORDER BY d.upload_timestamp DESC
            LIMIT 1
            """,
            (f"%{document_name.lower()}%",),
        )
    if not rows:
        return json.dumps({"error": f"Document matching '{document_name}' not found"})
    doc = dict(rows[0])
    if "is_automated" in doc:
        doc["review_reasons_note"] = _REVIEW_REASONS_NOTE
    return json.dumps(doc, indent=2, default=str)


@tool
def get_extraction_results(document_name: str) -> str:
    """Get AI extraction results for a specific document by name. Returns the extracted label (document type classification) and identified key-value pairs from the curated gold layer."""
    db = get_db()
    if not db.gold_sync_available():
        return json.dumps(
            {
                "search": document_name,
                "extraction_count": 0,
                "extractions": [],
                "note": _GOLD_UNAVAILABLE_MSG,
            }
        )
    rows = db.execute_query(
        f"""
        SELECT document_name, document_path, label, identifiers, extracted_at
        FROM {_gold_sync()}
        WHERE LOWER(document_name) LIKE %s
        ORDER BY extracted_at DESC
        """,
        (f"%{document_name.lower()}%",),
    )
    return json.dumps(
        {
            "search": document_name,
            "extraction_count": len(rows),
            "extractions": rows,
        },
        indent=2,
        default=str,
    )


@tool
def get_review_statistics(reviewer_email: Optional[str] = None) -> str:
    """Get extraction review accuracy statistics for HUMAN reviews.

    Returns total reviews, correct/partially_correct/incorrect counts, and
    accuracy percentage — all derived from rows in the reviews table. Also
    returns `auto_verified_count`: documents the pipeline auto-verified
    (extraction confidence at or above the auto-verify threshold, and no
    invalid or non-billable code or missing member ID) which have NOT been
    human-reviewed (these have NO row in the reviews table, so they don't
    affect the accuracy headline). Optionally filter to a specific reviewer.

    Reads the review_metrics metric view the reviewer dashboard shows, current
    as of the returned `as_of` (the lakehouse trails live reviews by minutes).
    """
    try:
        return _review_statistics_reyden(reviewer_email)
    except Exception as e:  # noqa: BLE001 — degrade to the live source
        logger.warning("review statistics on Reyden failed, using Lakebase: %s", e)
        return _review_statistics_lakebase(reviewer_email)


def _review_statistics_lakebase(reviewer_email: Optional[str]) -> str:
    """get_review_statistics computed from live Lakebase rows (the fallback)."""
    db = get_db()
    where = ""
    params: list = []
    if reviewer_email:
        where = "WHERE reviewer_email = %s"
        params.append(reviewer_email)

    rows = db.execute_query(
        f"""
        SELECT verdict,
               COUNT(*)                      AS cnt,
               COUNT(DISTINCT document_id)   AS docs
        FROM public.document_extraction_reviews
        {where}
        GROUP BY verdict
        """,
        tuple(params),
    )

    correct = partially_correct = incorrect = 0
    total_reviews = total_docs = 0
    for row in rows:
        v = row.get("verdict", "")
        c = int(row.get("cnt", 0))
        d = int(row.get("docs", 0))
        total_reviews += c
        total_docs += d
        if v == "correct":
            correct = c
        elif v == "partially_correct":
            partially_correct = c
        elif v == "incorrect":
            incorrect = c

    accuracy = (correct / total_reviews * 100) if total_reviews > 0 else 0.0

    # Disjoint auto-verified bucket: the pipeline auto-verified it AND not
    # human-reviewed. Same predicate as reviewer_app's get_status_counts
    # (auto_verified case).
    auto_verified_count: Optional[int] = None
    threshold = settings.auto_verdict_threshold
    if db.gold_sync_available():
        try:
            auto_rows = db.execute_query(f"""
                SELECT COUNT(*)::INT AS cnt
                FROM public.medical_documents d
                JOIN {_gold_sync()} g
                  ON g.document_path = d.file_path
                WHERE d.deleted_at IS NULL
                  AND d.processing_status <> 'failed'
                  AND g.is_automated
                  AND NOT EXISTS (
                      SELECT 1 FROM public.document_extraction_reviews r
                      WHERE r.document_id = d.id
                  )
                """)
            auto_verified_count = int(auto_rows[0]["cnt"]) if auto_rows else 0
        except Exception:
            auto_verified_count = None

    payload = {
        "total_reviews": total_reviews,
        "total_documents_reviewed": total_docs,
        "correct_count": correct,
        "partially_correct_count": partially_correct,
        "incorrect_count": incorrect,
        "accuracy_pct": round(accuracy, 1),
        "auto_verified_count": auto_verified_count,
        "auto_verify_threshold": threshold,
        "source": "Lakebase (live)",
    }
    return json.dumps(payload, indent=2, default=str)


@tool
def get_recent_reviews(
    limit: int = 10,
    reviewer_email: Optional[str] = None,
    verdict: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
) -> str:
    """Get recent extraction review activity. Shows which documents were reviewed, by whom, and the verdict (correct, partially_correct, incorrect). Supports filtering by reviewer, verdict, date range, and document name search."""
    conds = ["d.deleted_at IS NULL"]
    params: list = []
    if reviewer_email:
        conds.append("r.reviewer_email = %s")
        params.append(reviewer_email)
    if verdict:
        conds.append("r.verdict = %s")
        params.append(verdict)
    if date_from:
        conds.append("r.created_at >= %s")
        params.append(date_from)
    if date_to:
        conds.append("r.created_at < (%s::date + INTERVAL '1 day')")
        params.append(date_to)
    if search:
        conds.append("LOWER(d.document_name) LIKE %s")
        params.append(f"%{search.lower()}%")

    rows = get_db().execute_query(
        f"""
        SELECT r.id,
               r.document_id,
               d.document_name,
               r.reviewer_email,
               r.verdict,
               r.reasoning,
               r.is_automated,
               r.created_at
        FROM public.document_extraction_reviews r
        JOIN public.medical_documents d ON d.id = r.document_id
        WHERE {' AND '.join(conds)}
        ORDER BY r.created_at DESC
        LIMIT %s
        """,
        (*params, limit),
    )
    return json.dumps({"count": len(rows), "reviews": rows}, indent=2, default=str)


@tool
def search_extractions_by_label(label_search: str, limit: int = 20) -> str:
    """Search documents by their AI-extracted label type. Labels include: denial_management, referral_workqueue, invoice, prior_authorization, explanation_of_benefits, clinical_notes, lab_results, and others. Partial match supported."""
    db = get_db()
    if not db.gold_sync_available():
        return json.dumps(
            {
                "search": label_search,
                "count": 0,
                "results": [],
                "note": _GOLD_UNAVAILABLE_MSG,
            }
        )
    rows = db.execute_query(
        f"""
        SELECT document_path,
               document_name,
               user_email,
               label,
               identifiers,
               extracted_at
        FROM {_gold_sync()}
        WHERE LOWER(label) LIKE %s
        ORDER BY extracted_at DESC
        LIMIT %s
        """,
        (f"%{label_search.lower()}%", limit),
    )
    return json.dumps(
        {"search": label_search, "count": len(rows), "results": rows},
        indent=2,
        default=str,
    )


@tool
def get_pipeline_latency_stats(hours: int = 24) -> str:
    """Return aggregate pipeline latency stats for documents processed in the last N hours.

    Per-doc latency = `processing_timestamp - upload_timestamp` (end-to-end
    extraction duration, straight from Lakebase). Emits {count, avg, min, max,
    median} seconds.

    Use when asked about: pipeline latency, extraction speed, how long docs
    take to process, upload-to-extraction time, gold completion time.

    Reads fact_document_processing in the lakehouse gold layer (the figures the
    reviewer dashboard shows), current as of the returned `as_of`.
    """
    hours = max(1, min(hours, 24 * 30))
    try:
        return _pipeline_latency_reyden(hours)
    except Exception as e:  # noqa: BLE001 — degrade to the live source
        logger.warning("pipeline latency on Reyden failed, using Lakebase: %s", e)
        return _pipeline_latency_lakebase(hours)


def _pipeline_latency_lakebase(hours: int) -> str:
    """get_pipeline_latency_stats computed from live Lakebase rows (the fallback)."""
    db = get_db()
    rows = db.execute_query(
        """
        SELECT
            COUNT(*) AS total,
            ROUND(AVG(EXTRACT(EPOCH FROM (processing_timestamp - upload_timestamp)))::numeric, 1) AS avg_seconds,
            ROUND(MIN(EXTRACT(EPOCH FROM (processing_timestamp - upload_timestamp)))::numeric, 1) AS min_seconds,
            ROUND(MAX(EXTRACT(EPOCH FROM (processing_timestamp - upload_timestamp)))::numeric, 1) AS max_seconds,
            ROUND(
                PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (processing_timestamp - upload_timestamp))
                )::numeric, 1
            ) AS median_seconds
        FROM public.medical_documents
        WHERE upload_timestamp IS NOT NULL
          AND processing_timestamp IS NOT NULL
          AND processing_timestamp >= upload_timestamp
          AND upload_timestamp >= NOW() - (%s || ' hours')::interval
          AND deleted_at IS NULL
        """,
        (str(hours),),
    )
    row = rows[0] if rows else {}
    total = int(row.get("total") or 0)
    if total == 0:
        return json.dumps(
            {
                "hours": hours,
                "count": 0,
                "message": "No eligible documents in window.",
                "source": "Lakebase (live)",
            }
        )

    return json.dumps(
        {
            "hours": hours,
            "count": total,
            "avg_seconds": float(row["avg_seconds"]),
            "min_seconds": float(row["min_seconds"]),
            "max_seconds": float(row["max_seconds"]),
            "median_seconds": float(row["median_seconds"]),
            "source": "Lakebase (live)",
        },
        default=str,
    )


@tool
def get_recent_pipeline_events(limit: int = 20, flow_name_contains: str = "") -> str:
    """Return the most recent SDP pipeline events for the medical documents pipeline.

    Useful for debugging ("is the pipeline stuck?", "when did gold last finish?",
    "were there extraction failures?"). Returns raw event records with timestamp,
    event_type, flow_name, status, and message. Optional `flow_name_contains`
    filters to a specific flow (e.g. "gold_extraction_labels").
    """
    if not settings.pipeline_id:
        return json.dumps({"error": "PIPELINE_ID not configured for this agent."})

    limit = max(1, min(limit, 100))
    w = _get_workspace_client()

    resp = w.api_client.do(
        "GET",
        f"/api/2.0/pipelines/{settings.pipeline_id}/events",
        query={"max_results": limit * 3, "order_by": "timestamp desc"},
    )
    events = (resp or {}).get("events") or []

    results = []
    needle = flow_name_contains.lower() if flow_name_contains else ""
    for e in events:
        flow = (e.get("origin") or {}).get("flow_name") or ""
        if needle and needle not in flow.lower():
            continue
        status = ((e.get("details") or {}).get("flow_progress") or {}).get("status")
        results.append(
            {
                "timestamp": e.get("timestamp"),
                "event_type": e.get("event_type"),
                "flow_name": flow.split(".")[-1] if flow else None,
                "status": status,
                "message": e.get("message"),
                "level": e.get("level"),
            }
        )
        if len(results) >= limit:
            break

    # Guarded: a failure message can quote an uploaded file's NAME, which the
    # uploader chooses (fifth review).
    return json.dumps(
        {
            "pipeline_id": settings.pipeline_id,
            "count": len(results),
            "events": results,
        },
        indent=2,
        default=str,
    )


# =============================================================================
# SEMANTIC + KEYWORD SEARCH — Lakebase-native (pgvector + Postgres full-text)
# =============================================================================
# Both search tools read ONE table: public.document_chunks, the retrieval chunks
# ai_prep_search produces in the pipeline's third parallel branch
# (pipelines/silver/silver_doc_chunks.sql), embedded into Lakebase by
# jobs/load_document_chunks.py. Semantic arm = pgvector cosine (HNSW index);
# keyword arm = Postgres full-text (tsvector/GIN); hybrid = reciprocal-rank
# fusion of both. Entirely in Lakebase — only the query-embedding call hits the
# FM endpoint.
#
# Two granularities over that one table:
#   semantic_search_documents  — collapses chunk hits to their best chunk per
#                                document and ranks DOCUMENTS ("which documents
#                                are about X").
#   search_document_chunks     — returns the chunks themselves, optionally
#                                scoped to one document ("what does THIS
#                                document say about X").
#
# The text is RAW parsed document content, which is what lets these answer
# questions the curated gold summary cannot. Curated fields (label,
# confidence_score) are NOT stored here — they are looked up from the Lakebase
# gold replica for the final result set only, so they can never go stale and a
# missing synced table degrades to null instead of failing the search.

_RRF_K = 60  # standard reciprocal-rank-fusion damping constant

# Columns every arm projects, so chunk rows collapse and render uniformly.
_CHUNK_COLS = (
    "document_path, chunk_id, document_id, document_name, user_email, "
    "chunk_position, chunk_to_retrieve, page_id, image_uri"
)

# Lakebase Search BM25 index (migration 000035). `to_bm25query` takes the index
# NAME as an argument, so this constant is load-bearing: renaming the index
# without changing it breaks the keyword arm at runtime, not at deploy.
_BM25_INDEX = "idx_document_chunks_bm25"

# Detected once per process. Migration 000035 only creates the BM25 index when
# Lakebase Search is enabled on the project — a one-way UI toggle with no DAB
# field and no API — so both backends are live states, not a migration window.
_bm25_available: bool | None = None


def _bm25_ready(db) -> bool:
    """True when the Lakebase Search BM25 index exists on document_chunks."""
    global _bm25_available
    if _bm25_available is None:
        try:
            rows = db.execute_query(
                "SELECT 1 FROM pg_class WHERE relname = %s AND relkind = 'i'",
                (_BM25_INDEX,),
            )
            _bm25_available = bool(rows)
            logger.info(
                "keyword search backend: %s",
                "Lakebase Search BM25" if _bm25_available else "Postgres FTS (ts_rank)",
            )
        except Exception as e:  # pragma: no cover - probe is best-effort
            logger.debug("BM25 index probe failed, assuming Postgres FTS: %s", e)
            _bm25_available = False
    return _bm25_available


def _rrf_merge(
    vector_paths: list[str], keyword_paths: list[str], limit: int
) -> list[str]:
    """Reciprocal-rank-fuse two ranked key lists → fused order.

    Keys are document_paths for document-level search and
    "<document_path>\\x00<chunk_id>" composites for chunk-level search.
    """
    scores: dict[str, float] = {}
    for ranked in (vector_paths, keyword_paths):
        for rank, path in enumerate(ranked):
            scores[path] = scores.get(path, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda p: scores[p], reverse=True)[:limit]


def _chunk_key(row) -> str:
    """Composite identity for a chunk row (chunk_id is unique per document)."""
    return f"{row.get('document_path')}\x00{row.get('chunk_id')}"


def _collapse_to_documents(chunk_rows: list) -> tuple[list[str], dict]:
    """Chunk rows → (ordered unique document_paths, best chunk row per path).

    Arm queries come back already ordered best-first, so the first chunk seen
    for a document IS its best-matching chunk.
    """
    order: list[str] = []
    best: dict = {}
    for r in chunk_rows:
        path = r.get("document_path")
        if path is None or path in best:
            continue
        best[path] = r
        order.append(path)
    return order, best


def _curated_by_path(db, paths) -> dict:
    """label + confidence_score from the Lakebase gold replica, keyed by path.

    Scoped to the final result set, so this is a small lookup. Returns {} on any
    failure: the synced table is absent on a fresh workspace, and losing the
    enrichment must not lose the search results.
    """
    if not paths:
        return {}
    try:
        rows = db.execute_query(
            f"SELECT document_path, label, confidence_score FROM {_gold_sync()} "
            "WHERE document_path = ANY(%s)",
            (list(paths),),
        )
        return {r["document_path"]: r for r in rows}
    except Exception as e:  # pragma: no cover - optional enrichment
        logger.debug("gold enrichment skipped: %s", e)
        return {}


_EMPTY_INDEX_NOTE = (
    "The document chunk index is empty in this environment: either the "
    "load_chunks task has not run, or its source — the pipeline's "
    "silver_doc_chunks table — has no rows yet. A setup gap, not proof that no "
    "documents exist. Try the keyword tools (search_documents / "
    "search_extractions_by_label) meanwhile."
)

_EMBED_DOWN_NOTE = (
    "The embedding endpoint is unavailable, so semantic matching was skipped "
    "for this query. Results below (if any) are exact-keyword matches only, and "
    "a conceptually-worded question may miss documents it should find."
)

_EMBED_DOWN_ONLY_NOTE = (
    "The embedding endpoint is unavailable, so semantic search could not run. "
    "This is an outage, NOT evidence that no document matches. Retry, or use "
    "mode='keyword' / the keyword tools (search_documents / "
    "search_extractions_by_label)."
)


def _query_vector(q: str) -> str | None:
    """Embed the query for the vector arm; None when the endpoint is down.

    services.embeddings.embed_one never raises — on any failure it returns a
    zero vector. A zero vector MUST NOT reach the SQL: pgvector's cosine
    distance against it is NaN, so `ORDER BY embedding <=> %s::vector` ties every
    row and the arm returns an arbitrary `pool` of chunks, which the caller would
    then present as the top matches. jobs/load_document_chunks.py guards the same
    failure on the write path; this is the read-path half.
    """
    from services.embeddings import embed_one, to_pgvector_literal

    raw = embed_one(q)
    if not raw or not any(float(x) != 0.0 for x in raw):
        logger.warning("query embedding unavailable (zero vector); skipping vector arm")
        return None
    return to_pgvector_literal(raw)


def _chunk_scope(document_path: str | None) -> tuple[str, tuple]:
    """SQL fragment + params that scope a chunk search to one document.

    public.document_chunks keys on the FULL source path
    (dbfs:/Volumes/<catalog>/<schema>/documents_input/<name>.pdf), so a plain
    equality test silently matches nothing whenever the caller only has the file
    name. That is not hypothetical: get_active_review_context did not return
    document_path at all, so every "what does THIS document say" search arrived
    with a bare name, matched zero rows, and fell back to the extracted fields.

    A full path (what the search tools hand back) stays an exact equality, which
    is the index-friendly form and the overwhelmingly common case. A bare name
    also accepts the last path segment, so a caller holding only the name scopes
    correctly instead of getting a confident empty answer.
    """
    if not document_path:
        return "", ()
    path = document_path.strip()
    if not path:
        return "", ()
    if "/" in path:
        return " AND document_path = %s", (path,)
    return " AND (document_path = %s OR document_path LIKE %s)", (path, f"%/{path}")


def _chunk_arms(db, q: str, mode: str, pool: int, document_path: str | None = None):
    """Run the vector and/or keyword arm over public.document_chunks.

    Returns (vector_rows, keyword_rows, degraded_note). The note is set when the
    embedding endpoint is down: hybrid degrades to keyword-only rather than
    returning nothing, and a semantic-only search reports the outage instead of
    an empty result the model would read as "no such document". Raises on a query
    failure so callers can emit the graceful empty-index note.
    """
    scope_sql, scope_params = _chunk_scope(document_path)
    vector_rows: list = []
    keyword_rows: list = []
    degraded: str | None = None

    if mode in ("hybrid", "semantic"):
        qvec = _query_vector(q)
        if qvec is None:
            if mode == "semantic":
                return [], [], _EMBED_DOWN_ONLY_NOTE
            # hybrid: fall through to the keyword arm alone.
            degraded = _EMBED_DOWN_NOTE
            mode = "keyword"

    if mode in ("hybrid", "semantic"):
        params: tuple = (qvec,) + scope_params + (qvec, pool)
        vector_rows = db.execute_query(
            f"""
            SELECT {_CHUNK_COLS},
                   1 - (embedding <=> %s::vector) AS score
            FROM public.document_chunks
            WHERE TRUE{scope_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            params,
        )

    if mode in ("hybrid", "keyword"):
        if _bm25_ready(db):
            # Lakebase Search BM25, matching the documented query shape: rank by
            # the operator, no match predicate.
            #
            # `<@>` is DISTANCE-like, not a relevance score — the docs' own RRF
            # example ranks it with a plain ascending `ORDER BY score`, so DESC
            # here would return the WORST matches.
            #
            # And NO `@@ plainto_tsquery` filter, deliberately. plainto_tsquery
            # ANDs every term, so on a natural-language question ("what does this
            # letter say about the appeal window") it demands all six words in one
            # chunk, returns nothing, and silently deletes the keyword arm from the
            # hybrid — defeating the exact partial-match ranking BM25 exists to do.
            # The docs' keyword and hybrid examples both rank unfiltered.
            score_expr = (
                "content_tsv <@> "
                f"to_bm25query(to_tsvector('english', %s), '{_BM25_INDEX}')"
            )
            order_by = "score ASC"
            match_sql = ""
            params: tuple = (q,)
        else:
            # ts_rank scores 0 for a non-match, so the `@@` predicate is required
            # here to keep zero-score rows out. It inherits plainto_tsquery's
            # all-terms semantics, which is why this is the fallback backend.
            score_expr = "ts_rank(content_tsv, plainto_tsquery('english', %s))"
            order_by = "score DESC"
            match_sql = " AND content_tsv @@ plainto_tsquery('english', %s)"
            params = (q, q)

        params += scope_params + (pool,)
        keyword_rows = db.execute_query(
            f"""
            SELECT {_CHUNK_COLS},
                   {score_expr} AS score
            FROM public.document_chunks
            WHERE TRUE{match_sql}{scope_sql}
            ORDER BY {order_by}
            LIMIT %s
            """,
            params,
        )

    return vector_rows, keyword_rows, degraded


@tool
def semantic_search_documents(query: str, limit: int = 10, mode: str = "hybrid") -> str:
    """Find documents by MEANING over their extracted content, not just literal
    keywords. Use for conceptual/fuzzy asks ("denials about missing pre-auth",
    "appeals near their deadline", "documents similar to this one"); the keyword
    arm still catches exact codes, IDs, and names. `mode` is 'hybrid' (default —
    semantic + keyword fused), 'semantic', or 'keyword'. Returns the top matches
    with their document type label and the passage that matched."""
    q = (query or "").strip()
    if not q:
        return json.dumps({"query": query, "count": 0, "results": []})
    mode = (mode or "hybrid").lower()
    if mode not in ("hybrid", "semantic", "keyword"):
        mode = "hybrid"
    limit = max(1, min(limit, 50))
    db = get_db()
    # Wider pool than the chunk-level tool uses: many chunks collapse into one
    # document, so the arms must return well more than `limit` rows to yield
    # `limit` distinct documents.
    pool = max(limit * 20, 100)

    try:
        vector_rows, keyword_rows, degraded = _chunk_arms(db, q, mode, pool)
    except Exception as e:
        logger.warning("semantic_search_documents query failed: %s", e)
        return json.dumps(
            {"query": q, "count": 0, "results": [], "note": _EMPTY_INDEX_NOTE}
        )

    # Collapse each arm to documents FIRST, then fuse: RRF must rank documents,
    # not chunks, or a document with many mediocre chunks would outrank one with
    # a single excellent chunk.
    vector_docs, vector_best = _collapse_to_documents(vector_rows)
    keyword_docs, keyword_best = _collapse_to_documents(keyword_rows)

    # Show the passage from whichever arm ranked the document BETTER. Preferring
    # the vector arm unconditionally would surface a semantically-near passage for
    # a document that actually ranked on an exact keyword hit — so a "CO-197"
    # search could display, and the agent then quote, a passage with no CO-197 in
    # it. Whichever arm won is the arm that explains the match.
    _v_rank = {p: i for i, p in enumerate(vector_docs)}
    _k_rank = {p: i for i, p in enumerate(keyword_docs)}

    def _explaining_row(path):
        vr, kr = _v_rank.get(path), _k_rank.get(path)
        if vr is None:
            return keyword_best[path]
        if kr is None:
            return vector_best[path]
        return vector_best[path] if vr <= kr else keyword_best[path]

    by_path = {p: _explaining_row(p) for p in set(vector_best) | set(keyword_best)}

    if mode == "semantic":
        ordered = vector_docs[:limit]
    elif mode == "keyword":
        ordered = keyword_docs[:limit]
    else:
        ordered = _rrf_merge(vector_docs, keyword_docs, limit)

    curated = _curated_by_path(db, [p for p in ordered if p in by_path])

    results = [
        {
            "document_path": p,
            "document_id": by_path[p].get("document_id"),
            "document_name": by_path[p].get("document_name"),
            "user_email": by_path[p].get("user_email"),
            # label + confidence come from the Lakebase gold replica, not from
            # the chunk row, so they reflect the current curated state.
            "label": (curated.get(p) or {}).get("label"),
            "confidence_score": (curated.get(p) or {}).get("confidence_score"),
            # The document's best-matching passage — what made it rank.
            "matched_text": by_path[p].get("chunk_to_retrieve"),
            "page_id": by_path[p].get("page_id"),
        }
        for p in ordered
        if p in by_path
    ]
    payload = {"query": q, "mode": mode, "count": len(results), "results": results}
    if degraded:
        payload["note"] = degraded
    return json.dumps(payload, indent=2, default=str)


@tool
def search_document_chunks(
    query: str,
    document_path: str = "",
    limit: int = 10,
    mode: str = "hybrid",
) -> str:
    """Answer questions from the BODY TEXT of documents — the passages
    themselves, not a summary of what was extracted. Use this for "what does
    this letter say about the appeal window", "quote the reason they gave", or
    any question whose answer is wording inside the document. Pass
    `document_path` to search within ONE document (the "chat with this document"
    case); leave it blank to search every document's text. `mode` is 'hybrid'
    (default), 'semantic', or 'keyword'. Returns the matching passages with the
    page each came from, so answers can cite a page."""
    q = (query or "").strip()
    if not q:
        return json.dumps({"query": query, "count": 0, "results": []})
    mode = (mode or "hybrid").lower()
    if mode not in ("hybrid", "semantic", "keyword"):
        mode = "hybrid"
    limit = max(1, min(limit, 50))
    scope = (document_path or "").strip() or None
    db = get_db()
    pool = max(limit * 5, 25)  # per-arm candidate pool, wider than the final cut

    try:
        vector_rows, keyword_rows, degraded = _chunk_arms(
            db, q, mode, pool, document_path=scope
        )
    except Exception as e:
        logger.warning("search_document_chunks query failed: %s", e)
        return json.dumps(
            {"query": q, "count": 0, "results": [], "note": _EMPTY_INDEX_NOTE}
        )

    by_key = {_chunk_key(r): r for r in (vector_rows + keyword_rows)}
    if mode == "semantic":
        ordered = [_chunk_key(r) for r in vector_rows][:limit]
    elif mode == "keyword":
        ordered = [_chunk_key(r) for r in keyword_rows][:limit]
    else:
        ordered = _rrf_merge(
            [_chunk_key(r) for r in vector_rows],
            [_chunk_key(r) for r in keyword_rows],
            limit,
        )

    results = [
        {
            "document_path": by_key[k].get("document_path"),
            "document_id": by_key[k].get("document_id"),
            "document_name": by_key[k].get("document_name"),
            "chunk_position": by_key[k].get("chunk_position"),
            "text": by_key[k].get("chunk_to_retrieve"),
            "page_id": by_key[k].get("page_id"),
            "image_uri": by_key[k].get("image_uri"),
        }
        for k in ordered
        if k in by_key
    ]
    payload = {
        "query": q,
        "mode": mode,
        "document_path": scope,
        "count": len(results),
        "results": results,
    }
    if degraded:
        payload["note"] = degraded
    elif scope and not results:
        # Zero hits inside a named document is ambiguous: the document may have no
        # matching passage, or the path may simply be wrong / not indexed yet.
        # Reported as the same empty list, the model confidently answers "this
        # document does not mention X" — a false negative. One cheap existence
        # probe separates the two.
        try:
            present = db.execute_query(
                "SELECT 1 FROM public.document_chunks WHERE document_path = %s LIMIT 1",
                (scope,),
            )
        except Exception:  # pragma: no cover - diagnostic only
            present = None
        if present is not None and not present:
            payload["note"] = (
                f"No chunks are indexed for document_path {scope!r} at all, so this "
                "is NOT evidence that the document lacks the answer — the path may "
                "be wrong, or the document may not be indexed yet. Confirm the "
                "path with search_documents before concluding anything."
            )
    return json.dumps(payload, indent=2, default=str)


# =============================================================================
# REVIEWER-PANE ACTION TOOLS
#
# These power the in-document assistant embedded in the reviewer pane. They are
# document-scoped: each reads the open document from the `_active_document_id`
# contextvar (set from AG-UI forwardedProps.document_id) and declines when it is
# unset. Following the app's stage-and-confirm review posture, the write tools
# NEVER touch Lakebase directly — they return a compact `_frontend_action`
# payload that the reviewer pane renders as a staged card. Extraction edits and
# verdicts require the reviewer to Approve (which populates the existing review
# form / corrections and submits through the reviewer app's own APIs); notes,
# being scratch context rather than the official verdict, are applied
# immediately. The only Lakebase access here is READ-only, via
# get_active_review_context. Keeping all writes on the reviewer-app path means
# one source of truth, one audit identity, and no new write grants for the
# agent's Postgres role.
# =============================================================================


@tool
def get_active_review_context() -> str:
    """Read everything about the document currently open in the reviewer pane:
    its metadata, the AI-extracted fields (each identifier with the stable
    ``id:<n>`` correction key the reviewer UI uses, its value and confidence),
    the current human review verdict/reasoning and any field corrections, and
    the reviewer's private notepad.

    Call this FIRST, before proposing any edit or verdict, so your proposals
    cite real extracted values and their correct ``id:<n>`` keys. Only works in
    the in-document assistant (a document must be open)."""
    doc_id = _get_active_document_id()
    if not doc_id:
        return json.dumps({"error": _NO_ACTIVE_DOC_MSG})

    db = get_db()
    if db.gold_sync_available():
        doc_rows = db.execute_query(
            f"""
            SELECT d.document_name, d.document_type, d.processing_status,
                   d.file_path AS document_path,
                   g.label, g.identifiers, g.confidence_score
            FROM public.medical_documents d
            LEFT JOIN {_gold_sync()} g
              ON g.document_path = d.file_path
            WHERE d.id = %s AND d.deleted_at IS NULL
            LIMIT 1
            """,
            (doc_id,),
        )
    else:
        doc_rows = db.execute_query(
            """
            SELECT d.document_name, d.document_type, d.processing_status,
                   d.file_path AS document_path,
                   NULL::TEXT AS label, NULL::JSONB AS identifiers,
                   NULL::DOUBLE PRECISION AS confidence_score
            FROM public.medical_documents d
            WHERE d.id = %s AND d.deleted_at IS NULL
            LIMIT 1
            """,
            (doc_id,),
        )
    if not doc_rows:
        return json.dumps({"error": f"Open document {doc_id} not found."})
    doc = doc_rows[0]

    review_rows = db.execute_query(
        """
        SELECT verdict, reasoning, corrections, is_automated,
               reviewer_email, updated_at
        FROM public.document_extraction_reviews
        WHERE document_id = %s
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (doc_id,),
    )
    review = review_rows[0] if review_rows else None
    corrections = (review or {}).get("corrections") or {}

    # Expose identifiers with the same id:<idx> keys the reviewer UI derives
    # (parseIdentifiers), and surface any already-applied correction so the
    # agent sees the current effective value rather than a stale extraction.
    identifiers = doc.get("identifiers") or []
    fields = []
    if isinstance(identifiers, list):
        for idx, ident in enumerate(identifiers):
            if not isinstance(ident, dict):
                continue
            key = f"id:{idx}"
            original = ident.get("value")
            fields.append(
                {
                    "correction_key": key,
                    "name": ident.get("name"),
                    "extracted_value": original,
                    "corrected_value": corrections.get(key),
                    "current_value": corrections.get(key, original),
                    "confidence": ident.get("confidence"),
                }
            )

    # Notepad is optional context — degrade gracefully if the table/grant is
    # not present yet (fresh workspace before the notes migration runs).
    notepad = None
    try:
        note_rows = db.execute_query(
            "SELECT note_text FROM public.document_notes "
            "WHERE document_id = %s AND user_email = %s",
            (doc_id, _get_authorized_user_email()),
        )
        if note_rows:
            notepad = note_rows[0].get("note_text")
    except Exception as e:  # pragma: no cover - optional read
        logger.debug("notepad read skipped: %s", e)

    return json.dumps(
        {
            "document_id": doc_id,
            "document_name": doc.get("document_name"),
            # The KEY search_document_chunks scopes on, not a display field. The
            # query above already selected it (d.file_path AS document_path) and
            # this dict simply dropped it, so the model's only handle on the open
            # document was its file NAME -- which it then passed as
            # document_path. The chunk scope is an exact match against the full
            # dbfs:/Volumes/... path stored in public.document_chunks, so every
            # "what does THIS document say" search matched zero rows and fell
            # back to the extracted fields. Verified in the browser before fixing.
            "document_path": doc.get("document_path"),
            "document_type": doc.get("document_type"),
            "processing_status": doc.get("processing_status"),
            "extraction_label": doc.get("label"),
            "extraction_confidence": doc.get("confidence_score"),
            "fields": fields,
            "current_review": (
                {
                    "verdict": review.get("verdict"),
                    "reasoning": review.get("reasoning"),
                    "is_automated": review.get("is_automated"),
                    "reviewer_email": review.get("reviewer_email"),
                }
                if review
                else None
            ),
            "notepad": notepad,
        },
        indent=2,
        default=str,
    )


@tool
def get_review_remediation() -> str:
    """Read WHY the open document was held for review, and the candidate fixes
    already worked out for it.

    Returns the pipeline's ``review_reasons`` for the document plus, for each
    flagged item, the shortlist of terminology codes that could replace it and
    how much judgment is left:

    - ``deterministic`` — the terminology left exactly one option. Confirm it
      against the document, then propose it.
    - ``needs_judgment`` — several codes are valid and only the document can
      decide (site, laterality, severity). Read the document, propose ONE, and
      say which evidence decided it.
    - ``not_resolvable`` — there is nothing to propose. Do NOT invent a value;
      relay the guidance and say what the document needs instead.

    Call this BEFORE proposing anything on a held document: proposing a code
    that is not on the shortlist means proposing one the terminology rejects.
    Only works in the in-document assistant."""
    doc_id = _get_active_document_id()
    if not doc_id:
        return json.dumps({"error": _NO_ACTIVE_DOC_MSG})

    db = get_db()

    # Why it was held. The routing decision and its reasons live on the gold
    # row, not on the Lakebase document.
    reasons: list = []
    is_automated = None
    if db.gold_sync_available():
        rows = db.execute_query(
            f"""
            SELECT g.is_automated, {_review_reasons_column(db)}
            FROM public.medical_documents d
            LEFT JOIN {_gold_sync()} g ON g.document_path = d.file_path
            WHERE d.id = %s AND d.deleted_at IS NULL
            LIMIT 1
            """,
            (doc_id,),
        )
        if rows:
            is_automated = rows[0].get("is_automated")
            raw = rows[0].get("review_reasons")
            if isinstance(raw, list):
                reasons = [r for r in raw if r]
            elif isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                    reasons = (
                        [r for r in parsed if r] if isinstance(parsed, list) else []
                    )
                except (ValueError, TypeError):
                    reasons = []

    # What has already been worked out. Written by the reviewer app (triage or
    # the pane); this tool only reads. Degrades to "not computed yet" rather
    # than failing when the table or grant is not present.
    items: list = []
    proposals_available = True
    try:
        proposal_rows = db.execute_query(
            """
            SELECT review_reason, resolution, field_name, correction_key,
                   observed_value, proposed_value, rationale, candidates,
                   disposition
            FROM public.document_review_proposals
            WHERE document_id = %s AND withheld = FALSE
            ORDER BY proposed_at DESC
            """,
            (doc_id,),
        )
        for row in proposal_rows or []:
            candidates = row.get("candidates")
            if isinstance(candidates, str):
                try:
                    candidates = json.loads(candidates)
                except (ValueError, TypeError):
                    candidates = []
            items.append(
                {
                    "review_reason": row.get("review_reason"),
                    "resolution": row.get("resolution"),
                    "field_name": row.get("field_name"),
                    "correction_key": row.get("correction_key"),
                    "observed_value": row.get("observed_value"),
                    "already_proposed": row.get("proposed_value"),
                    "guidance": row.get("rationale"),
                    "candidates": candidates or [],
                    "disposition": row.get("disposition"),
                }
            )
    except Exception as e:  # pragma: no cover - optional read
        logger.debug("review proposals read skipped: %s", e)
        proposals_available = False

    if is_automated is True and not reasons:
        note = (
            "This document was auto-verified: it met the confidence threshold "
            "with no review reasons. There is nothing to remediate."
        )
    elif not reasons:
        note = (
            "No review reasons are recorded for this document. It may not have "
            "reached the gold layer yet — say so rather than inferring a problem."
        )
    elif not items:
        note = (
            "The document is held, but no candidate shortlist has been computed "
            "for it yet. Explain the reason from `review_reasons` and read the "
            "document before proposing anything; do not propose a code you "
            "cannot check against the terminology."
            if proposals_available
            else "The proposal store is unavailable, so no shortlist can be read."
        )
    else:
        note = (
            "Propose ONLY codes that appear in a candidate list below. A "
            "`not_resolvable` item must not receive a proposed value."
        )

    return json.dumps(
        {
            "document_id": doc_id,
            "is_automated": is_automated,
            "review_reasons": reasons,
            "items": items,
            "note": note,
        },
        indent=2,
        default=str,
    )


@tool
def propose_extraction_edit(
    correction_key: str,
    proposed_value: str,
    rationale: str,
    review_reason: str = "",
) -> str:
    """Propose a correction to ONE extracted field on the open document.

    This does NOT write anything. It stages a suggested edit as a card in the
    reviewer pane for the human to Approve or Reject; on approval the value is
    written into the review's field corrections (not submitted on its own).
    ``correction_key`` is the ``id:<n>`` key from get_active_review_context,
    ``proposed_value`` is the new value, and ``rationale`` briefly explains the
    change.

    ``review_reason`` is the reason this field was flagged, copied from
    get_review_remediation when the document is held (``invalid_code``,
    ``non_billable_code``). Pass it — acceptance is reported per reason, and a
    proposal that arrives without one cannot be attributed to the problem it
    was meant to fix. Leave it empty only for an edit you are suggesting on
    your own initiative, where the pipeline flagged nothing.

    Only works in the in-document assistant."""
    doc_id = _get_active_document_id()
    if not doc_id:
        return json.dumps({"error": _NO_ACTIVE_DOC_MSG})
    key = (correction_key or "").strip()
    if not key.startswith("id:"):
        return json.dumps(
            {
                "error": "correction_key must be an 'id:<n>' key from "
                "get_active_review_context (e.g. 'id:3')."
            }
        )
    return json.dumps(
        {
            "staged": True,
            "message": f"Proposed edit to {key} staged for reviewer approval.",
            "_frontend_action": {
                "type": "extraction_edit",
                "document_id": doc_id,
                "correction_key": key,
                "proposed_value": proposed_value,
                "rationale": rationale,
                # Carried so the pane can attribute the proposal to the reason
                # it addresses. Empty means "not tied to a pipeline finding".
                "review_reason": (review_reason or "").strip(),
            },
        }
    )


@tool
def propose_review_verdict(verdict: str, reasoning: str) -> str:
    """Propose a review verdict for the open document.

    This does NOT submit the review. It fills the reviewer pane's review form
    with the proposed verdict and reasoning as a staged card; the human
    reviews, Approves, and submits through the normal review flow. ``verdict``
    is one of 'correct', 'partially_correct', 'incorrect'. ``reasoning`` is
    required for anything other than 'correct'. Only works in the in-document
    assistant."""
    doc_id = _get_active_document_id()
    if not doc_id:
        return json.dumps({"error": _NO_ACTIVE_DOC_MSG})
    v = (verdict or "").strip().lower()
    if v not in _VALID_VERDICTS:
        return json.dumps({"error": f"verdict must be one of {list(_VALID_VERDICTS)}."})
    if v != "correct" and not (reasoning or "").strip():
        return json.dumps(
            {"error": "reasoning is required when verdict is not 'correct'."}
        )
    return json.dumps(
        {
            "staged": True,
            "message": f"Proposed verdict '{v}' staged; the reviewer will "
            "confirm and submit.",
            "_frontend_action": {
                "type": "verdict",
                "document_id": doc_id,
                "verdict": v,
                "reasoning": reasoning or "",
            },
        }
    )


@tool
def add_review_note(note_text: str) -> str:
    """Append a note to the reviewer's private notepad for the open document.

    Unlike verdicts and field edits, notes are scratch context (not part of the
    official review) and are applied immediately — no approval card. Returns
    confirmation; the note appears in the pane's notepad. Only works in the
    in-document assistant."""
    doc_id = _get_active_document_id()
    if not doc_id:
        return json.dumps({"error": _NO_ACTIVE_DOC_MSG})
    text = (note_text or "").strip()
    if not text:
        return json.dumps({"error": "note_text is empty."})
    return json.dumps(
        {
            "saved": True,
            "message": "Note added to the notepad.",
            "_frontend_action": {
                "type": "note_append",
                "document_id": doc_id,
                "note_text": text,
            },
        }
    )


# =============================================================================
# Payer-policy retrieval (Vector Search)
# =============================================================================
# This is the agent's only RETRIEVER-span tool. Two deliberate design points:
#
#   1. The inner `_retrieve_policy_documents` is traced as span_type=RETRIEVER
#      and returns mlflow `Document` objects. That output schema is what
#      `RetrievalGroundedness` reads to judge whether the answer is supported by
#      retrieved context. Returning plain strings/dicts here would log the span
#      but give the judge nothing to ground against.
#   2. Every failure degrades to a plain-language message instead of raising.
#      Vector Search may not be provisioned on a given demo workspace (see the
#      best-effort note in scripts/create_vector_index.py); when it is not, the
#      agent must still answer everything else rather than erroring the turn.


def _policy_index_name() -> str:
    """Fully-qualified index name, derived unless explicitly configured."""
    if settings.vector_search_index:
        return settings.vector_search_index
    return f"{settings.catalog}.{settings.schema_name}.payer_policy_index"


_POLICY_COLUMNS = [
    "policy_id",
    "payer",
    "policy_type",
    "title",
    "content",
    "citation_label",
    "related_codes",
]

_POLICY_UNAVAILABLE_MSG = (
    "Payer-policy retrieval is not available in this environment, so I cannot "
    "cite policy language for this question. I can still report what was "
    "extracted and reviewed."
)


def _retrieve_policy_documents(query: str, num_results: int) -> list:
    """Query the policy index and return mlflow Documents.

    Traced as a RETRIEVER span (decorated at call time in `search_payer_policy`
    so an import-time mlflow failure can never break tool registration).
    Raises on transport errors; the caller converts them to a message.
    """
    from mlflow.entities import Document

    w = _get_workspace_client()
    resp = w.vector_search_indexes.query_index(
        index_name=_policy_index_name(),
        columns=_POLICY_COLUMNS,
        query_text=query,
        num_results=num_results,
    )
    rows = getattr(getattr(resp, "result", None), "data_array", None) or []

    # Bind values by the manifest's COLUMN NAMES, not by the order we requested.
    # The response carries `manifest.columns` precisely because positional order
    # is not contractual; zipping against our own list would, on any reordering,
    # silently mislabel every field (content read as title, the wrong
    # citation_label attached) and the agent would cite the WRONG policy with
    # full confidence instead of failing. Fall back to the requested order only
    # if the manifest is absent.
    manifest = getattr(resp, "manifest", None)
    manifest_cols = getattr(manifest, "columns", None) or []
    col_names = [
        getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
        for c in manifest_cols
    ]
    col_names = [c for c in col_names if c]
    if not col_names:
        col_names = list(_POLICY_COLUMNS)

    docs = []
    for row in rows:
        values = dict(zip(col_names, row))
        # Vector Search reports the similarity score as a `score` column IN the
        # manifest, so read it by name. The positional fallback only covers a
        # response whose manifest omits it (an earlier version relied on the
        # fallback alone, and since the manifest does list `score`, the score was
        # always None).
        score = values.get("score")
        if score is None and len(row) > len(col_names):
            score = row[len(col_names)]
        docs.append(
            Document(
                page_content=values.get("content") or "",
                metadata={
                    "policy_id": values.get("policy_id"),
                    "payer": values.get("payer"),
                    "policy_type": values.get("policy_type"),
                    "title": values.get("title"),
                    "citation_label": values.get("citation_label"),
                    "related_codes": values.get("related_codes"),
                    "similarity_score": score,
                },
            )
        )
    return docs


@tool
def search_payer_policy(query: str, num_results: int = 0) -> str:
    """Search payer medical policy for the rules governing a claim.

    Use this whenever the user asks WHY a claim would be paid, denied, or need
    prior authorization — medical-necessity criteria, prior-authorization
    requirements, coding/billing rules, frequency limits, or appeal deadlines.
    Also use it to explain a denial reason or to check what a payer requires
    for a specific procedure or diagnosis code.

    Always attribute what you report to the returned citation label; do not
    state a policy rule without naming the policy it came from.

    Args:
        query: Natural-language description of the rule you need — include the
            payer, procedure/diagnosis code, or denial reason when known
            (e.g. "Veridane prior authorization for lumbar MRI 72148").
        num_results: How many policies to retrieve; 0 uses the configured
            default.
    """
    import mlflow
    from mlflow.entities import SpanType

    query = (query or "").strip()
    if not query:
        return "Provide a description of the policy rule to look up."

    limit = (
        num_results
        if num_results and num_results > 0
        else settings.policy_retrieval_num_results
    )
    limit = max(1, min(limit, 10))

    traced = mlflow.trace(
        _retrieve_policy_documents,
        name="retrieve_payer_policy",
        span_type=SpanType.RETRIEVER,
    )
    try:
        docs = traced(query, limit)
    except Exception as e:  # noqa: BLE001 — degrade, never error the turn
        logger.warning("payer-policy retrieval failed: %s", e)
        return _POLICY_UNAVAILABLE_MSG

    if not docs:
        return (
            "No payer policy matched that query. Try naming the payer, the "
            "procedure or diagnosis code, or the denial reason."
        )

    out = []
    for d in docs:
        meta = d.metadata or {}
        out.append(
            {
                "citation": meta.get("citation_label"),
                "payer": meta.get("payer"),
                "policy_type": meta.get("policy_type"),
                "title": meta.get("title"),
                "governs_codes": meta.get("related_codes") or None,
                "policy_text": d.page_content,
            }
        )
    return json.dumps(
        {
            "query": query,
            "count": len(out),
            "instruction": (
                "Answer only from these policy excerpts and cite the 'citation' "
                "value for every rule you state."
            ),
            "policies": out,
        },
        indent=2,
        default=str,
    )


# =============================================================================
# Reyden analytics tier (Lakehouse//RT via the system.ai.dbsql MCP service)
# =============================================================================
# Point lookups stay on Lakebase above: live, ~5 ms. Rates, trends and breakdowns
# read the governed gold layer on the Reyden warehouse instead: the same
# review_metrics / fact_* objects the reviewer dashboard reads, so the agent and
# the dashboard share one definition. Gold trails Lakebase by the CDF sync plus
# the MV refresh (minutes), which is why the fixed tools report `as_of` and fall
# back to Lakebase when this tier is unavailable.
#
# Transport: JSON-RPC to the Unity AI Gateway MCP service system.ai.dbsql
# (governed: EXECUTE grants, usage tracking), with `_meta.warehouse_id` pinning
# every call to the Reyden warehouse. Plain httpx plus the SDK's auth headers
# keeps the tools synchronous and adds no dependency.
#
# Read-only is enforced HERE, not by the warehouse. The gateway service exposes
# a read/write execute_sql, the agent SP holds MODIFY on the schema, and
# Lakehouse//RT itself executed a CREATE TABLE AS SELECT through this path while
# rejecting DROP TABLE (2026-09). So a statement is sent only if
# _qualify_read_only accepts it, and a missing or misdirected pin fails closed.

_DBSQL_MCP_PATH = "/ai-gateway/mcp-services/system.ai.dbsql"
_REYDEN_POLL_SECONDS = 1.0
_REYDEN_DEADLINE_SECONDS = 25.0
_REYDEN_MAX_ROWS = 200

# The only objects the analytics tier reads. Claim codes come only through the
# PHI-masked view, never the raw gold_claim_codes table (Genie's rule too).
_REYDEN_OBJECTS = frozenset(
    {
        "review_metrics",
        "document_ops_metrics",
        "claims_coding_metrics",
        "fact_review",
        "fact_document_processing",
        "dim_document",
        "gold_claim_codes_secure",
    }
)

_SQL_LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.)*\"")
_TABLE_REF = re.compile(r"\b(FROM|JOIN)(\s+)([`\w.]+)", re.IGNORECASE)
_CTE_NAME = re.compile(r"(?:\bWITH|,)\s*`?(\w+)`?\s+AS\s*\(", re.IGNORECASE)
_WRITE_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|COPY"
    r"|OPTIMIZE|VACUUM|REFRESH|CALL|MSCK|RESTORE|UNDROP|CACHE|UNCACHE)\b",
    re.IGNORECASE,
)
_UNSAFE_FUNCTIONS = re.compile(
    r"\b(http_request|secret|read_files|remote_query|reflect|java_method|ai_\w+)\s*\(",
    re.IGNORECASE,
)
# FROM inside these functions is an operand, not a table: EXTRACT(YEAR FROM x).
_FROM_IN_FUNCTIONS = frozenset({"extract", "trim", "substring", "substr", "overlay"})
# `FROM a, b` (optionally aliased): b would never pass through _TABLE_REF.
_COMMA_JOIN = re.compile(r"\s*(?:(?:AS\s+)?\w+\s*)?,", re.IGNORECASE)
_rpc_ids = itertools.count(1)
_dbsql_http = None


class ReydenUnavailable(RuntimeError):
    """The analytics tier cannot run here: no pin, or the pin is misdirected."""


class ReadOnlyViolation(ValueError):
    """The statement is not ONE read over the analytics objects."""


def _enclosing_function(code: str, pos: int) -> str:
    """Lower-cased name of the function whose parentheses enclose `pos`, or ''."""
    depth = 0
    for i in range(pos - 1, -1, -1):
        if code[i] == ")":
            depth += 1
        elif code[i] == "(":
            if depth == 0:
                name = re.search(r"(\w+)\s*$", code[:i])
                return name.group(1).lower() if name else ""
            depth -= 1
    return ""


def _qualify_read_only(
    statement: str, schema: str = "", objects: frozenset | None = None
) -> str:
    """Return `statement` with the named objects fully qualified.

    `schema` / `objects` default to the analytics scope. The KG traversal
    tool passes the ontobricks registry schema and its own single-view
    allowlist, so the scopes stay isolated: query_lakehouse still cannot
    read the triplestore, and the KG tool still cannot read gold. A previous
    cut duplicated this whole function instead, and the copy diverged.

    Raises ReadOnlyViolation unless it is a single SELECT (or WITH ... SELECT)
    that reads only _REYDEN_OBJECTS, with no comments and no side-effecting
    functions. Literals are masked before any check, so a string can neither
    hide a write nor trip a false positive.
    """
    sql = statement.strip().rstrip(";").rstrip()
    parts = []  # (is_literal, text)
    pos = 0
    for m in _SQL_LITERAL.finditer(sql):
        parts.append((False, sql[pos : m.start()]))
        parts.append((True, m.group(0)))
        pos = m.end()
    parts.append((False, sql[pos:]))
    code = " ".join(text for is_literal, text in parts if not is_literal)

    if "--" in code or "/*" in code:
        raise ReadOnlyViolation("remove SQL comments")
    if ";" in code:
        raise ReadOnlyViolation("send exactly one statement")
    if not re.match(r"\s*\(*\s*(SELECT|WITH)\b", code, re.IGNORECASE):
        raise ReadOnlyViolation("only SELECT (or WITH ... SELECT) is allowed")
    word = _WRITE_WORDS.search(code)
    if word:
        raise ReadOnlyViolation(f"{word.group(1).upper()} is not allowed")
    func = _UNSAFE_FUNCTIONS.search(code)
    if func:
        raise ReadOnlyViolation(f"{func.group(1)}() is not allowed")

    catalog = str(getattr(settings, "catalog", "") or "")
    if not schema:
        schema = str(getattr(settings, "schema_name", "") or "")
    if objects is None:
        objects = _REYDEN_OBJECTS
    ctes = {name.lower() for name in _CTE_NAME.findall(code)}
    allowed = ", ".join(sorted(objects))

    def qualify(m: re.Match, text: str) -> str:
        if _enclosing_function(text, m.start()) in _FROM_IN_FUNCTIONS:
            return m.group(0)
        if _COMMA_JOIN.match(text, m.end()):
            raise ReadOnlyViolation(
                "use an explicit JOIN instead of a comma-separated FROM"
            )
        ref = m.group(3).replace("`", "")
        names = ref.lower().split(".")
        if len(names) == 1 and names[0] in ctes:
            return m.group(0)
        obj = names[-1]
        scope = names[:-1]
        if obj not in objects or scope not in (
            [],
            [schema.lower()],
            [catalog.lower(), schema.lower()],
        ):
            raise ReadOnlyViolation(
                f"{m.group(3)} is not readable; use one of: {allowed}"
            )
        return f"{m.group(1)}{m.group(2)}`{catalog}`.`{schema}`.`{obj}`"

    return "".join(
        text if is_literal else _TABLE_REF.sub(lambda m, t=text: qualify(m, t), text)
        for is_literal, text in parts
    )


def _pinned_warehouse_id() -> str:
    """The Reyden warehouse id every call pins to; fails closed when unset."""
    warehouse_id = str(getattr(settings, "dbsql_mcp_warehouse_id", "") or "")
    if not warehouse_id:
        raise ReydenUnavailable("DBSQL_MCP_WAREHOUSE_ID is not set")
    if warehouse_id == os.environ.get("MLFLOW_TRACING_SQL_WAREHOUSE_ID"):
        # The standard (writable) warehouse is MLflow's; the pin must not be it.
        raise ReydenUnavailable(
            "DBSQL_MCP_WAREHOUSE_ID points at the standard warehouse"
        )
    return warehouse_id


def _parse_rpc(text: str) -> dict:
    """A JSON-RPC message from a JSON body or the last SSE `data:` line."""
    body = text.strip()
    if not body.startswith("{"):
        data = [ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")]
        body = data[-1] if data else ""
    return json.loads(body) if body else {}


def _mcp_call(tool_name: str, arguments: dict, warehouse_id: str) -> dict:
    """One system.ai.dbsql tools/call, pinned to `warehouse_id`; its structuredContent."""
    global _dbsql_http
    if _dbsql_http is None:
        import httpx

        _dbsql_http = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
    w = _get_workspace_client()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    headers.update(w.config.authenticate())
    resp = _dbsql_http.post(
        w.config.host.rstrip("/") + _DBSQL_MCP_PATH,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": next(_rpc_ids),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
                "_meta": {"warehouse_id": warehouse_id},
            },
        },
    )
    resp.raise_for_status()
    message = _parse_rpc(resp.text)
    if message.get("error"):
        raise RuntimeError(message["error"].get("message") or "MCP error")
    result = message.get("result") or {}
    if result.get("isError"):
        text = " ".join(c.get("text", "") for c in result.get("content") or [])
        raise RuntimeError(text.strip() or "MCP tool error")
    return result.get("structuredContent") or {}


def _span(name: str):
    """An MLflow span context, or a no-op one: tracing must never break a tool."""
    try:
        import mlflow

        return mlflow.start_span(name=name)
    except Exception:  # pragma: no cover
        return contextlib.nullcontext()


def _reyden_sql(
    statement: str, schema: str = "", objects: frozenset | None = None
) -> tuple[list[dict], bool]:
    """Run ONE read-only statement on the Reyden warehouse.

    Returns (rows as dicts of strings, truncated). Traced as `dbsql_mcp_call`,
    a plain span (not TOOL, so tool_call_budget doesn't count it twice) with no
    http.url / db.system, which no_sql_warehouse_regression reads as the
    retired direct-warehouse path.
    """
    warehouse_id = _pinned_warehouse_id()
    sql = _qualify_read_only(statement, schema=schema, objects=objects)
    with _span("dbsql_mcp_call") as span:
        content = _mcp_call("execute_sql", {"query": sql}, warehouse_id)
        deadline = time.monotonic() + _REYDEN_DEADLINE_SECONDS
        while (content.get("status") or {}).get("state") in ("PENDING", "RUNNING"):
            if time.monotonic() > deadline:
                raise TimeoutError("analytics query timed out")
            time.sleep(_REYDEN_POLL_SECONDS)
            content = _mcp_call(
                "poll_sql_result",
                {"statement_id": content.get("statement_id")},
                warehouse_id,
            )
        status = content.get("status") or {}
        if status.get("state") != "SUCCEEDED":
            error = (status.get("error") or {}).get("message") or status.get("state")
            raise RuntimeError(f"analytics query failed: {error}")
        columns = [
            c.get("name")
            for c in ((content.get("manifest") or {}).get("schema") or {}).get(
                "columns"
            )
            or []
        ]
        data = (content.get("result") or {}).get("data_array") or []
        rows = [
            dict(zip(columns, [v.get("string_value") for v in (r.get("values") or [])]))
            for r in data[:_REYDEN_MAX_ROWS]
        ]
        truncated = len(data) > _REYDEN_MAX_ROWS or bool(content.get("truncated"))
        if span is not None and hasattr(span, "set_attributes"):
            try:
                span.set_attributes(
                    {
                        "dbsql_mcp.warehouse_id": warehouse_id,
                        "dbsql_mcp.statement_id": content.get("statement_id") or "",
                        "dbsql_mcp.row_count": len(rows),
                    }
                )
            except Exception:  # pragma: no cover
                pass
    return rows, truncated


def _sql_string(value: str) -> str:
    """A SQL string literal (system.ai.dbsql takes no parameter markers)."""
    return "'" + str(value).replace("\\", "\\\\").replace("'", "''") + "'"


def _to_int(value) -> int:
    return int(float(value)) if value not in (None, "") else 0


def _to_float(value) -> Optional[float]:
    return float(value) if value not in (None, "") else None


def _review_statistics_reyden(reviewer_email: Optional[str]) -> str:
    """get_review_statistics from the review_metrics metric view on Reyden."""
    # A reviewer filter scopes the HUMAN rows only; the auto-verified count stays
    # workspace-wide, as it is in the Lakebase version.
    where = (
        f"WHERE is_automated OR reviewer_email = {_sql_string(reviewer_email)}"
        if reviewer_email
        else ""
    )
    rows, _ = _reyden_sql(f"""
        WITH m AS (
            SELECT verdict, is_automated,
                   MEASURE(review_count) AS cnt,
                   MEASURE(documents_reviewed) AS docs
            FROM review_metrics
            {where}
            GROUP BY verdict, is_automated
        ),
        f AS (SELECT MAX(reviewed_at) AS as_of FROM fact_review)
        SELECT m.verdict, m.is_automated, m.cnt, m.docs, f.as_of
        FROM f LEFT JOIN m ON TRUE
        """)
    counts = {"correct": 0, "partially_correct": 0, "incorrect": 0}
    total_reviews = total_docs = auto_verified = 0
    as_of = None
    for row in rows:
        as_of = as_of or row.get("as_of")
        if row.get("verdict") is None:
            continue
        if str(row.get("is_automated")).lower() == "true":
            auto_verified += _to_int(row.get("cnt"))
            continue
        cnt = _to_int(row.get("cnt"))
        total_reviews += cnt
        total_docs += _to_int(row.get("docs"))
        if row["verdict"] in counts:
            counts[row["verdict"]] += cnt
    accuracy = (counts["correct"] / total_reviews * 100) if total_reviews > 0 else 0.0
    return json.dumps(
        {
            "total_reviews": total_reviews,
            "total_documents_reviewed": total_docs,
            "correct_count": counts["correct"],
            "partially_correct_count": counts["partially_correct"],
            "incorrect_count": counts["incorrect"],
            "accuracy_pct": round(accuracy, 1),
            "auto_verified_count": auto_verified,
            "auto_verify_threshold": settings.auto_verdict_threshold,
            "as_of": as_of,
            "source": "lakehouse gold (review_metrics), via Reyden",
        },
        indent=2,
        default=str,
    )


def _pipeline_latency_reyden(hours: int) -> str:
    """get_pipeline_latency_stats from fact_document_processing on Reyden."""
    rows, _ = _reyden_sql(f"""
        WITH s AS (
            SELECT COUNT(pipeline_seconds) AS total,
                   ROUND(AVG(pipeline_seconds), 1) AS avg_seconds,
                   ROUND(MIN(pipeline_seconds), 1) AS min_seconds,
                   ROUND(MAX(pipeline_seconds), 1) AS max_seconds,
                   ROUND(MEDIAN(pipeline_seconds), 1) AS median_seconds
            FROM fact_document_processing
            WHERE pipeline_seconds >= 0
              AND upload_timestamp >= current_timestamp() - INTERVAL {int(hours)} HOURS
        ),
        f AS (SELECT MAX(processing_timestamp) AS as_of FROM fact_document_processing)
        SELECT s.*, f.as_of FROM s CROSS JOIN f
        """)
    row = rows[0] if rows else {}
    total = _to_int(row.get("total"))
    source = "lakehouse gold (fact_document_processing), via Reyden"
    if total == 0:
        return json.dumps(
            {
                "hours": hours,
                "count": 0,
                "message": "No eligible documents in window.",
                "as_of": row.get("as_of"),
                "source": source,
            }
        )
    return json.dumps(
        {
            "hours": hours,
            "count": total,
            "avg_seconds": _to_float(row.get("avg_seconds")),
            "min_seconds": _to_float(row.get("min_seconds")),
            "max_seconds": _to_float(row.get("max_seconds")),
            "median_seconds": _to_float(row.get("median_seconds")),
            "as_of": row.get("as_of"),
            "source": source,
        },
        default=str,
    )


# =============================================================================
# Knowledge graph traversal (via Reyden)
# =============================================================================
# Single multi-hop graph-traversal tool over the OntoBricks claims knowledge
# graph. Reuses the Reyden SQL infrastructure so it shares the same injection
# guard and warehouse pin. Feature-flagged OFF by default; the dev target
# enables it via config.kg_enabled.

# Boot outcome of the KG feature, surfaced on /health via kg_status(). Populated
# by get_all_tools(), which owns the registration decision. Same rationale as
# _ROUTING_BOOT in main.py: a flag nobody can observe from outside costs a
# debugging cycle, which is exactly how the routing clone() bug was found. The
# default is the "get_all_tools never ran" state, distinguishable from an
# explicit off.
_KG_BOOT: dict = {
    "enabled": False,
    "schema": "",
    "tool_registered": False,
    "reason": "not initialized",
}


def kg_status() -> dict:
    """Return the KG feature's boot outcome (a copy, so callers cannot mutate it)."""
    return dict(_KG_BOOT)


_KG_EDGE_DEFS = {
    # The patient edge. Traversed forward it answers "who is this document
    # about"; reversed (the `~` prefix the tool supports) it answers "what else
    # is on this patient", which is the question a reviewer actually asks and
    # which nothing in the schema could answer before. The Patient node is a
    # bare surrogate key -- see scripts/lakercm_domain_content.py.
    "documentsPatient": ("Document", "Patient"),
    "hasDiagnosis": ("Document", "DiagnosisCode"),
    "hasProcedure": ("Document", "ProcedureCode"),
    "billedTo": ("Document", "Payer"),
    "deniedFor": ("Document", "DenialReason"),
    "issuedBy": ("PayerPolicy", "Payer"),
    "governsDiagnosis": ("PayerPolicy", "DiagnosisCode"),
    "governsProcedure": ("PayerPolicy", "ProcedureCode"),
}


def _kg_unavailable() -> str:
    """Graceful degradation message when KG is not available."""
    return json.dumps(
        {
            "error": (
                "The knowledge graph is not available. I can still answer "
                "from the extraction, review and analytics data."
            )
        }
    )


@tool
def traverse_claims_graph(
    path: str = "",
    start: str = "",
    limit: int = 25,
) -> str:
    """Traverse the OntoBricks knowledge graph of claims relationships.

    Use this to follow connections across 2+ hops in the claims knowledge graph —
    from documents to diagnoses to policies, or documents to denial reasons to
    payer rules. This tool is only available when the knowledge graph is enabled
    (dev environment).

    Args:
        path: Comma-separated edge chain. Each edge connects one entity type to
            another. Prefix an edge with ~ to follow it backwards (e.g.
            "~deniedFor" goes from DenialReason back to Document).
            Valid edges: hasDiagnosis, hasProcedure, billedTo, deniedFor,
            issuedBy, governsDiagnosis, governsProcedure.
            Example: "~deniedFor,hasDiagnosis,~governsDiagnosis" traces from a
            policy back to denial reasons, forward to diagnoses they govern, then
            back to the policies that govern those diagnoses.
        start: Starting entity identifier, or its full URI (required for chains
            >3 hops). Either form works: only the last path segment is compared,
            so "synthetic-1-0001-referral.pdf" and
            "https://lakercm.example/ontology/Document/synthetic-1-0001-referral.pdf"
            are equivalent. Example: "POL-OK-PT-021".
        limit: Maximum endpoints to return (1-200, default 25).

    Returns: JSON with (endpoints_count, endpoints) listing unique destination
        entities by (type, id) with path counts and example paths. For policy
        endpoints, includes citation_label for citing payer rules.
    """
    # getattr, not attribute access: four pre-existing suites build their own
    # SimpleNamespace config and cannot be expected to declare every future
    # flag. Production settings is pydantic, where this field always exists.
    if not getattr(settings, "kg_enabled", False):
        return _kg_unavailable()

    kg_schema = str(getattr(settings, "kg_schema", "") or "")
    if not kg_schema:
        return _kg_unavailable()

    path_str = (path or "").strip()
    if not path_str:
        return json.dumps({"error": "path is empty"})

    # Tokenize and validate the path
    try:
        edges = [e.strip() for e in path_str.split(",")]
        edges = [e for e in edges if e]
        if not edges:
            return json.dumps({"error": "path contains no valid edges"})

        # Check >3 hops without start
        if len(edges) > 3 and not start.strip():
            return json.dumps(
                {
                    "error": f"path has {len(edges)} hops (max 3 without start). "
                    "Pass start to begin a longer traversal."
                }
            )

        # Validate and build the path SQL, tracking types through the chain
        edge_objs = []
        current_type = None
        for edge_name in edges:
            reverse = edge_name.startswith("~")
            clean_name = edge_name[1:] if reverse else edge_name
            if clean_name not in _KG_EDGE_DEFS:
                valid = ", ".join(sorted(_KG_EDGE_DEFS.keys()))
                return json.dumps(
                    {"error": f"unknown edge '{clean_name}'. Valid edges: {valid}"}
                )
            domain, range_ = _KG_EDGE_DEFS[clean_name]
            if reverse:
                domain, range_ = range_, domain

            # Check type continuity
            if current_type is not None and current_type != domain:
                return json.dumps(
                    {
                        "error": f"edge '{edge_name}' expects {domain} but current "
                        f"type is {current_type}"
                    }
                )
            current_type = range_
            edge_objs.append((edge_name, clean_name, reverse, domain, range_))

        # Build SQL with one CTE per hop.
        limit = max(1, min(int(limit) if limit else 25, 200))
        # Normalise the anchor to a LOCAL NAME, because the store holds full
        # URIs: `https://lakercm.example/ontology/Document/<name>`. The docstring
        # has always advertised "URI or identifier", but the comparison below was
        # exact equality, so a bare identifier matched NOTHING and the traversal
        # returned zero endpoints -- silently, and indistinguishably from a real
        # dead end. Taking the last path segment of whatever is passed makes both
        # documented forms work, and comparing against the store's own last
        # segment (below) is what makes the match land.
        start_local = start.strip().rstrip("/").rsplit("/", 1)[-1]
        start_sql = f"'{_sql_string(start_local)[1:-1]}'" if start_local else ""

        # Local name, never a URI prefix. The store's grammar is inconsistent:
        # predicates are `.../ontology/hasDiagnosis` (slash) while rdf:type
        # objects are `...#Document` (hash), and BASE_URI in
        # scripts/lakercm_domain_content.py is the hash form. Verified live
        # 2026-09-30 that every predicate in the triplestore uses the SLASH
        # form, so a hash-built URI matched nothing and every edge returned zero
        # rows. Splitting on [/#] is right for either, and needs no rdf:type join.
        pred = "ELEMENT_AT(SPLIT(t.predicate, '[/#]'), -1)"
        pred0 = "ELEMENT_AT(SPLIT(predicate, '[/#]'), -1)"

        cte_bodies = []
        for i, (_edge, clean_name, reverse, _domain, _range) in enumerate(edge_objs):
            # Direction is a property of the EDGE and it has to reach the SQL: a
            # `~` edge walks object -> subject. Swapping only the type check's
            # domain/range left every reverse traversal silently running forward.
            near, far = ("object", "subject") if reverse else ("subject", "object")
            if i == 0:
                anchor = (
                    f"    AND ELEMENT_AT(SPLIT({near}, '/'), -1) = {start_sql} "
                    if start_sql
                    else ""
                )
                cte_bodies.append(
                    f"  hop_0 AS ("
                    f"    SELECT {far} AS endpoint, COUNT(*) AS cnt, "
                    f"           ANY_VALUE(CONCAT({near}, '->', {pred0}, '->', {far})) AS path "
                    f"    FROM `{settings.catalog}`.`{kg_schema}`.triplestore_lakercm_v1 "
                    f"    WHERE {pred0} = '{clean_name}' "
                    f"{anchor}"
                    f"    GROUP BY {far} "
                    f"  )"
                )
            else:
                # The new endpoint is t.<far>, NOT p.endpoint. Carrying
                # p.endpoint forward pinned every hop to hop_0's result, so a
                # 3-hop path silently returned its 1-hop answer.
                cte_bodies.append(
                    f"  hop_{i} AS ("
                    f"    SELECT t.{far} AS endpoint, SUM(p.cnt) AS cnt, "
                    f"           ANY_VALUE(CONCAT(p.path, '->', t.{far})) AS path "
                    f"    FROM hop_{i - 1} p "
                    f"    JOIN `{settings.catalog}`.`{kg_schema}`.triplestore_lakercm_v1 t "
                    f"      ON p.endpoint = t.{near} "
                    f"    WHERE {pred} = '{clean_name}' "
                    f"    GROUP BY t.{far} "
                    f"  )"
                )

        # Comma-separated. `WITH a AS (...) b AS (...)` is a syntax error, and it
        # hid itself: _CTE_NAME only matches a name after WITH or a comma, so
        # hop_1+ were never recognised as CTEs and the read-only guard rejected
        # them as unauthorized tables. The guard was right.
        sql_parts = ["WITH", ",\n".join(cte_bodies)]

        # end_type comes free from the URI's second-to-last segment, so no
        # rdf:type join is needed. citation_label stays NULL: resolving it needs
        # a join per endpoint and no caller reads it yet.
        sql_parts.append(
            f"SELECT "
            f"  ELEMENT_AT(SPLIT(endpoint, '[/#]'), -2) AS end_type, "
            f"  ELEMENT_AT(SPLIT(endpoint, '[/#]'), -1) AS end_id, "
            f"  SUM(cnt) AS path_count, "
            f"  ANY_VALUE(path) AS example_path, "
            f"  ELEMENT_AT(SPLIT(endpoint, '[/#]'), -1) AS rdfs_label, "
            f"  CAST(NULL AS STRING) AS citation_label "
            f"FROM hop_{len(edge_objs) - 1} "
            f"GROUP BY end_type, end_id "
            f"ORDER BY path_count DESC "
            f"LIMIT {limit} "
        )

        sql = "\n".join(sql_parts)

        # Run the query
        try:
            rows, truncated = _reyden_sql(
                sql, schema=kg_schema, objects=frozenset({"triplestore_lakercm_v1"})
            )
        except ReadOnlyViolation as e:
            return json.dumps({"error": str(e)})
        except ReydenUnavailable:
            return _kg_unavailable()
        except Exception as e:
            logger.warning("KG traversal failed: %s", e)
            return _kg_unavailable()

        # Format results
        endpoints = []
        for row in rows:
            endpoints.append(
                {
                    "end_type": row.get("end_type"),
                    "end_id": row.get("end_id"),
                    "path_count": int(row.get("path_count") or 0),
                    "example_path": row.get("example_path"),
                    "label": row.get("rdfs_label"),
                    "citation": row.get("citation_label"),
                }
            )

        return json.dumps(
            {
                "path": path_str,
                "start": start,
                "endpoints_count": len(endpoints),
                "endpoints": endpoints,
                "truncated": truncated,
            },
            indent=2,
            default=str,
        )

    except Exception as e:  # pragma: no cover - comprehensive error handling above
        logger.warning("KG traversal exception: %s", e)
        return _kg_unavailable()


@tool
def query_lakehouse(sql: str) -> str:
    """Run ONE read-only analytics query over the lakehouse gold layer.

    Use for rates, trends, breakdowns, distributions and comparisons that
    get_review_statistics and get_pipeline_latency_stats don't answer, e.g.
    accuracy by document type per month, turnaround percentiles by reviewer,
    denial rates by payer. For a specific document, reviewer or queue item, or
    anything that must be live, use the other tools: this layer trails them by
    minutes.

    Rules: a single SELECT (WITH ... SELECT is fine), no comments, no writes.
    Name only these objects, unqualified (the tool qualifies them):
    Metric views: select dimensions, wrap measures in MEASURE(), GROUP BY the
    dimensions.
      review_metrics: dims review_date, review_month, verdict, is_automated,
        reviewer_email, document_type. Measures review_count,
        documents_reviewed, correct_count, partially_correct_count,
        incorrect_count, accuracy_pct, human_review_count, auto_review_count,
        human_accuracy_pct, auto_accuracy_pct, override_rate_pct,
        avg_fields_corrected, correction_rate_pct, avg_turnaround_seconds,
        median_turnaround_seconds, p90_turnaround_seconds.
      document_ops_metrics: dims processed_date, current_status,
        document_type, uploader_email. Measures document_count,
        auto_verification_rate_pct, failure_rate_pct, pipeline_total,
        pipeline_avg_seconds, pipeline_median_seconds, pipeline_p90_seconds,
        pending_backlog_count, oldest_pending_age_hours.
      claims_coding_metrics: dims processed_date, document_type, payer,
        auth_status, coding_outcome, denial_category, denial_carc_code.
        Measures document_count, clean_claim_rate_pct, invalid_code_rate_pct,
        denial_count, denial_rate_pct, prior_auth_count,
        prior_auth_approval_rate_pct, billed_amount_total, auto_verified_count.
    Tables:
      fact_review(review_id, document_id, document_name, document_type,
        reviewer_email, verdict, reasoning, is_automated, reviewed_at,
        turnaround_seconds, fields_corrected_count, was_overridden)
      fact_document_processing(document_id, document_type, uploader_email,
        current_status, upload_timestamp, processing_timestamp,
        first_human_reviewed_at, pipeline_seconds, review_turnaround_seconds)
      dim_document(document_id, document_name, document_type, uploader_email,
        processing_status, num_pages, upload_timestamp, processing_timestamp,
        is_deleted)
      gold_claim_codes_secure(document_name, document_type, payer, claim_id,
        auth_status, billed_amount, codes_total, codes_invalid, is_clean_claim,
        confidence_score, is_automated, extracted_at, denial_carc_code,
        denial_category, denial_reason)
    Returns up to 200 rows as {"columns", "rows"}.
    """
    try:
        rows, truncated = _reyden_sql(sql)
    except ReadOnlyViolation as e:
        return f"Query refused: {e}. Rewrite it as one SELECT over the listed objects."
    except ReydenUnavailable as e:
        logger.warning("query_lakehouse unavailable: %s", e)
        return (
            "Lakehouse analytics are not available in this environment. Answer "
            "with get_review_statistics, get_pipeline_latency_stats or the "
            "document tools instead."
        )
    except Exception as e:  # noqa: BLE001 — surface the error, never error the turn
        logger.warning("query_lakehouse failed: %s", e)
        return f"Query failed: {e}"
    columns = list(rows[0].keys()) if rows else []
    return json.dumps(
        {
            "row_count": len(rows),
            "truncated": truncated,
            "columns": columns,
            "rows": [[r.get(c) for c in columns] for r in rows],
        },
        default=str,
    )


# =============================================================================
# Injection guard over EVERY tool result
# =============================================================================
# Guarding was opt-in per tool, through a serializer each tool had to remember
# to call, and get_user_memory never did: a document-borne injection the model
# was talked into saving came back unfenced, and unalerted, in every later
# session (eighth review). get_all_tools() now wraps every tool it returns, so a
# new tool is guarded by default. These are the deliberate exceptions.
_UNGUARDED_TOOLS = frozenset(
    {
        # The reviewer pane JSON.parses these results for `_frontend_action`
        # (reviewer_app/frontend/src/api/chatApi.js), so a fenced result would
        # silently drop the staged card. Their only free text is the model's
        # own arguments, which fencing cannot make any safer.
        "propose_extraction_edit",
        "propose_review_verdict",
        "add_review_note",
    }
)


def _tool_name(tool_obj) -> str:
    return getattr(tool_obj, "name", None) or getattr(tool_obj, "__name__", "")


def _guarded(tool_obj):
    """A copy of `tool_obj` whose every result passes through `_guard_tool_output`.

    The module-level tool is left untouched (the eval replay tools copy its
    name and schema). A plain function, which is what `@tool` yields where a
    test stubs langchain, is wrapped directly.
    """
    name = _tool_name(tool_obj)

    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            return _guard_tool_output(fn(*args, **kwargs), name)

        run.injection_guarded = True
        return run

    def wrap_async(fn):
        @functools.wraps(fn)
        async def run(*args, **kwargs):
            return _guard_tool_output(await fn(*args, **kwargs), name)

        run.injection_guarded = True
        return run

    if not hasattr(tool_obj, "model_copy"):
        return wrap(tool_obj)
    update = {}
    if getattr(tool_obj, "func", None) is not None:
        update["func"] = wrap(tool_obj.func)
    if getattr(tool_obj, "coroutine", None) is not None:
        update["coroutine"] = wrap_async(tool_obj.coroutine)
    return tool_obj.model_copy(update=update)


def get_all_tools() -> list:
    """Return all agent tools — document-retrieval + long-term memory.

    Every tool except `_UNGUARDED_TOOLS` is returned wrapped by the injection
    guard, so its result is fenced as data when it carries an override attempt.
    """
    from agent.memory_tools import get_memory_tools

    # Order matters: LLMs scan tool lists top-down and bias toward earlier
    # entries. get_documents_by_status leads so status questions don't get
    # mis-routed to search_documents.
    tools = [
        get_documents_by_status,
        get_review_statistics,
        get_recent_reviews,
        search_documents,
        get_document_details,
        get_extraction_results,
        search_extractions_by_label,
        semantic_search_documents,
        search_document_chunks,
        search_payer_policy,
        get_pipeline_latency_stats,
        get_recent_pipeline_events,
        # Ad-hoc lakehouse analytics on the Reyden warehouse. After the fixed
        # tools, so the model reaches for those first.
        query_lakehouse,
        # Knowledge graph traversal (feature-flagged; only registered when enabled).
        # After analytics tools so the model reaches for fixed analytics first.
    ]
    kg_on = bool(getattr(settings, "kg_enabled", False))
    kg_schema = str(getattr(settings, "kg_schema", "") or "")
    if kg_on:
        tools.append(traverse_claims_graph)
    _KG_BOOT.update(
        enabled=kg_on,
        schema=kg_schema,
        tool_registered=kg_on,
        reason=(
            "LAKERCM_KG_ENABLED is false"
            if not kg_on
            else (
                # The gate above is the flag alone, but traverse_claims_graph
                # ALSO requires a schema and returns unavailable without one. So
                # this pair is reachable: registered, and refusing every call.
                "LAKERCM_KG_SCHEMA is empty: tool registered but every call "
                "returns unavailable"
                if not kg_schema
                else "ok"
            )
        ),
    )

    tools.extend(
        [
            # Reviewer-pane action tools. Harmless in the general chat (they
            # self-gate on an open document and decline otherwise); active when the
            # in-document assistant sets forwardedProps.document_id.
            get_active_review_context,
            # Why the document was held + the terminology-derived shortlist. Listed
            # before the propose_* tools so the model reads the constraints before
            # it suggests a code.
            get_review_remediation,
            propose_extraction_edit,
            propose_review_verdict,
            add_review_note,
            *get_memory_tools(),
        ]
    )
    return [t if _tool_name(t) in _UNGUARDED_TOOLS else _guarded(t) for t in tools]
