"""
LakeRCM Analytics Routes

Physician-facing endpoints that query the Kimball gold layer built from the
Lakebase CDC mirrors (fact_review, fact_document_processing, silver_reviews)
and the review_metrics metric view, via the Statement Execution API (SQL
warehouse). All analytics run on the warehouse — Lakebase is operational-only
(document routes, review submission).

Statement Execution returns every value as a STRING: coerce booleans with
`str(v).lower() == "true"` (bool("false") is True!) and numbers with
int()/float(). Raw TIMESTAMP columns serialize ISO-8601 with a Z offset,
which pydantic parses tz-aware — never date_format() timestamps without an
offset (naive strings parse as browser-local time in the frontend).
"""

import logging
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    Disposition,
    Format,
    StatementParameterListItem,
    StatementState,
)

from schemas import (
    AnalyticsSummaryResponse,
    AnalyticsTrendResponse,
    MonthlyTrendItem,
    ProcessingMetricsResponse,
    RecentReviewsResponse,
    RecentReviewItem,
    ReviewerListResponse,
    ReviewerOption,
)
from dependencies import get_workspace_client, resolve_user_identity
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics")

_POLL_INTERVAL_S = 1.0
_POLL_DEADLINE_S = 90.0
_REVIEWERS_TTL_S = 60.0

# Distinct-reviewer cache: _get_reviewer_raw_ids fires on every filtered
# summary/trend/recent-reviews call, so a dashboard interaction would
# otherwise re-run the same DISTINCT query several times per second.
_reviewers_cache: Dict[str, object] = {"ts": 0.0, "values": []}


def _qualified(table: str) -> str:
    return f"{settings.catalog}.{settings.lakercm_schema}.{table}"


def _execute_warehouse_query(
    workspace_client: WorkspaceClient,
    query: str,
    params: Optional[Dict[str, str]] = None,
) -> list:
    """Execute a query via Statement Execution API and return rows as dicts.

    User-supplied values MUST be bound via `params` (named :markers) — never
    interpolated into the statement. Polls PENDING/RUNNING to a terminal
    state instead of silently returning [] on a slow (cold-start) statement.
    """
    warehouse_id = settings.get_warehouse_id()

    parameters = None
    if params:
        parameters = [
            StatementParameterListItem(name=k, value=v, type="STRING")
            for k, v in params.items()
        ]

    # format/disposition are pinned, NOT left to the warehouse default. A
    # Lakehouse//RT (REALTIME) warehouse defaults to an ARROW_STREAM
    # `result.attachment`, where `result.data_array` is None — so the parse
    # below silently returned [] and every analytics endpoint reported zeros
    # while the tables held data (verified live on the dev RT warehouse:
    # row_count=1, data_array=None). A PRO warehouse defaulted to inline JSON,
    # which is why this only broke after the analytics reads moved to RT.
    response = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
        parameters=parameters,
        wait_timeout="30s",
        format=Format.JSON_ARRAY,
        disposition=Disposition.INLINE,
    )

    deadline = time.monotonic() + _POLL_DEADLINE_S
    while response.status and response.status.state in (
        StatementState.PENDING,
        StatementState.RUNNING,
    ):
        if time.monotonic() > deadline:
            raise HTTPException(status_code=504, detail="Warehouse query timed out")
        time.sleep(_POLL_INTERVAL_S)
        response = workspace_client.statement_execution.get_statement(
            response.statement_id
        )

    if not response.status or response.status.state != StatementState.SUCCEEDED:
        error_msg = (
            response.status.error.message
            if response.status and response.status.error
            else f"state={response.status.state if response.status else 'unknown'}"
        )
        raise HTTPException(
            status_code=500, detail=f"Warehouse query failed: {error_msg}"
        )

    if not response.result:
        return []

    if response.result.data_array is None:
        # An empty result set legitimately has no data_array. A result that
        # reports rows but carries no inline array does NOT: the format above
        # failed to take effect, and returning [] here is what turned a
        # populated gold layer into an all-zeros dashboard. Fail loudly.
        row_count = response.result.row_count or 0
        if row_count:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Warehouse returned {row_count} row(s) with no inline "
                    "data_array — unexpected result disposition"
                ),
            )
        return []

    columns = [col.name for col in response.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in response.result.data_array]


def _is_true(value) -> bool:
    """Statement Execution returns booleans as 'true'/'false' strings."""
    return str(value).lower() == "true"


def _warehouse_distinct_reviewers(workspace_client: WorkspaceClient) -> List[str]:
    """Distinct human reviewer identifiers (raw values: legacy numeric SCIM
    ids and emails). Sourced from silver_reviews — no document join — so
    reviewers whose only reviews are on soft-deleted docs still appear
    (parity with the old Lakebase get_distinct_reviewers)."""
    now = time.monotonic()
    if (
        now - float(_reviewers_cache["ts"]) < _REVIEWERS_TTL_S
        and _reviewers_cache["values"]
    ):
        return list(_reviewers_cache["values"])
    rows = _execute_warehouse_query(
        workspace_client,
        f"""
        SELECT DISTINCT reviewer_email
        FROM {_qualified("silver_reviews")}
        WHERE NOT is_automated
        ORDER BY reviewer_email
        """,
    )
    values = [r["reviewer_email"] for r in rows if r.get("reviewer_email")]
    _reviewers_cache["ts"] = now
    _reviewers_cache["values"] = values
    return values


def _get_reviewer_raw_ids(
    workspace_client: WorkspaceClient, reviewer_email: str
) -> List[str]:
    """All raw reviewer_email values that map to this email.

    A user may have reviews stored under a numeric SCIM ID (old) and their
    email (new). This returns both so queries match all their reviews.
    """
    all_raw = _warehouse_distinct_reviewers(workspace_client)
    ids = []
    for raw_id in all_raw:
        identity = resolve_user_identity(raw_id)
        if identity["email"] == reviewer_email or raw_id == reviewer_email:
            ids.append(raw_id)
    if reviewer_email not in ids:
        ids.append(reviewer_email)
    return ids


def _reviewer_in_clause(
    raw_ids: List[str], params: Dict[str, str], column: str = "reviewer_email"
) -> str:
    """Bind an IN-list as numbered named markers; returns the SQL fragment."""
    markers = []
    for i, raw_id in enumerate(raw_ids):
        key = f"r{i}"
        params[key] = raw_id
        markers.append(f":{key}")
    return f"{column} IN ({', '.join(markers)})"


@router.get("/reviewers", response_model=ReviewerListResponse)
async def get_reviewer_list(
    workspace_client: WorkspaceClient = Depends(get_workspace_client),
):
    """List distinct reviewers, resolved to display names."""
    try:
        raw_ids = _warehouse_distinct_reviewers(workspace_client)
        # Resolve and deduplicate: multiple raw IDs (numeric + email) may
        # map to the same person. Group by resolved email.
        seen_emails: dict = {}  # email -> {raw_ids: [...], display_name}
        for raw_id in raw_ids:
            identity = resolve_user_identity(raw_id)
            email = identity["email"]
            if email not in seen_emails:
                seen_emails[email] = {
                    "raw_ids": [raw_id],
                    "display_name": identity["display_name"],
                }
            else:
                seen_emails[email]["raw_ids"].append(raw_id)

        options = []
        for email, info in seen_emails.items():
            label = info["display_name"] or email
            options.append(ReviewerOption(value=email, label=label))
        options.sort(key=lambda o: o.label.lower())
        return ReviewerListResponse(reviewers=options)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get reviewer list: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get reviewer list: {str(e)}"
        )


@router.get("/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(
    reviewer: Optional[str] = Query(None, description="Filter by reviewer"),
    workspace_client: WorkspaceClient = Depends(get_workspace_client),
):
    """Aggregate accuracy summary from the review_metrics metric view."""
    try:
        params: Dict[str, str] = {}
        where = ""
        if reviewer:
            raw_ids = _get_reviewer_raw_ids(workspace_client, reviewer)
            where = f"WHERE {_reviewer_in_clause(raw_ids, params)}"

        query = f"""
            SELECT
                verdict,
                is_automated,
                MEASURE(review_count) AS total_reviews,
                MEASURE(documents_reviewed) AS total_docs
            FROM {_qualified("review_metrics")}
            {where}
            GROUP BY verdict, is_automated
        """
        rows = _execute_warehouse_query(workspace_client, query, params or None)

        correct = partially_correct = incorrect = 0
        total_reviews = total_docs = 0
        auto_reviewed = human_reviewed = 0
        auto_correct = human_correct = 0
        human_docs = auto_docs = 0
        human_partial = human_incorrect = 0

        for row in rows:
            verdict = row.get("verdict", "")
            count = int(row.get("total_reviews", 0))
            docs = int(row.get("total_docs", 0))
            is_auto = _is_true(row.get("is_automated", "false"))
            total_reviews += count
            total_docs += docs

            if is_auto:
                auto_reviewed += count
                auto_docs += docs
                if verdict == "correct":
                    auto_correct += count
            else:
                human_reviewed += count
                human_docs += docs
                if verdict == "correct":
                    human_correct += count
                elif verdict == "partially_correct":
                    human_partial += count
                elif verdict == "incorrect":
                    human_incorrect += count

            if verdict == "correct":
                correct += count
            elif verdict == "partially_correct":
                partially_correct += count
            elif verdict == "incorrect":
                incorrect += count

        accuracy_pct = (correct / total_reviews * 100) if total_reviews > 0 else 0.0
        human_acc = (
            (human_correct / human_reviewed * 100) if human_reviewed > 0 else 0.0
        )
        auto_acc = (auto_correct / auto_reviewed * 100) if auto_reviewed > 0 else 0.0

        return AnalyticsSummaryResponse(
            total_reviews=total_reviews,
            total_documents_reviewed=total_docs,
            correct_count=correct,
            partially_correct_count=partially_correct,
            incorrect_count=incorrect,
            accuracy_pct=round(accuracy_pct, 1),
            auto_reviewed=auto_reviewed,
            human_reviewed=human_reviewed,
            auto_accuracy_pct=round(auto_acc, 1),
            human_accuracy_pct=round(human_acc, 1),
            human_documents_reviewed=human_docs,
            auto_documents_reviewed=auto_docs,
            human_correct_count=human_correct,
            human_partially_correct_count=human_partial,
            human_incorrect_count=human_incorrect,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Failed to get analytics summary: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get analytics summary: {str(e)}"
        )


def _duration_stats(row: Optional[dict]) -> dict:
    """Coerce one stats row (strings from Statement Execution) into the
    DurationStats shape. Empty/zero ⇒ total=0 + all-None (old contract)."""
    if not row or int(row.get("total", 0) or 0) == 0:
        return {
            "total": 0,
            "avg_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "median_seconds": None,
        }
    return {
        "total": int(row["total"]),
        "avg_seconds": (
            float(row["avg_seconds"]) if row.get("avg_seconds") is not None else None
        ),
        "min_seconds": (
            float(row["min_seconds"]) if row.get("min_seconds") is not None else None
        ),
        "max_seconds": (
            float(row["max_seconds"]) if row.get("max_seconds") is not None else None
        ),
        "median_seconds": (
            float(row["median_seconds"])
            if row.get("median_seconds") is not None
            else None
        ),
    }


@router.get("/processing-metrics", response_model=ProcessingMetricsResponse)
async def get_processing_metrics(
    workspace_client: WorkspaceClient = Depends(get_workspace_client),
):
    """Per-document processing metrics: pipeline latency + review turnaround.

    Pipeline latency comes from fact_document_processing (upload → extraction
    complete), review turnaround from fact_review (extraction complete →
    first human review). One round-trip; both facts already exclude
    soft-deleted documents.
    """
    try:
        query = f"""
            SELECT
                'pipeline' AS kind,
                COUNT(pipeline_seconds) AS total,
                ROUND(AVG(pipeline_seconds), 1) AS avg_seconds,
                ROUND(MIN(pipeline_seconds), 1) AS min_seconds,
                ROUND(MAX(pipeline_seconds), 1) AS max_seconds,
                ROUND(MEDIAN(pipeline_seconds), 1) AS median_seconds
            FROM {_qualified("fact_document_processing")}
            UNION ALL
            SELECT
                'review',
                COUNT(turnaround_seconds),
                ROUND(AVG(turnaround_seconds), 1),
                ROUND(MIN(turnaround_seconds), 1),
                ROUND(MAX(turnaround_seconds), 1),
                ROUND(MEDIAN(turnaround_seconds), 1)
            FROM {_qualified("fact_review")}
            WHERE NOT is_automated
        """
        rows = _execute_warehouse_query(workspace_client, query)
        by_kind = {row.get("kind"): row for row in rows}
        return ProcessingMetricsResponse(
            pipeline=_duration_stats(by_kind.get("pipeline")),
            review=_duration_stats(by_kind.get("review")),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get processing metrics: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get processing metrics: {str(e)}"
        )


@router.get("/trend", response_model=AnalyticsTrendResponse)
async def get_analytics_trend(
    reviewer: Optional[str] = Query(None, description="Filter by reviewer"),
    workspace_client: WorkspaceClient = Depends(get_workspace_client),
):
    """Monthly accuracy trend (human verdicts only) from review_metrics.

    Month buckets are computed at query time in the metric view — nothing is
    materialized at month grain.
    """
    try:
        params: Dict[str, str] = {}
        reviewer_clause = ""
        if reviewer:
            raw_ids = _get_reviewer_raw_ids(workspace_client, reviewer)
            reviewer_clause = f"AND {_reviewer_in_clause(raw_ids, params)}"

        query = f"""
            SELECT
                review_month,
                verdict,
                MEASURE(review_count) AS cnt
            FROM {_qualified("review_metrics")}
            WHERE NOT is_automated
            {reviewer_clause}
            GROUP BY review_month, verdict
            ORDER BY review_month
        """
        rows = _execute_warehouse_query(workspace_client, query, params or None)

        months: dict = {}
        for row in rows:
            month = row.get("review_month", "")[:7]
            verdict = row.get("verdict", "")
            count = int(row.get("cnt", 0))

            if month not in months:
                months[month] = {"correct": 0, "partially_correct": 0, "incorrect": 0}

            if verdict in months[month]:
                months[month][verdict] += count

        trend = []
        for month, counts in sorted(months.items()):
            total = (
                counts["correct"] + counts["partially_correct"] + counts["incorrect"]
            )
            accuracy = (counts["correct"] / total * 100) if total > 0 else 0.0
            trend.append(
                MonthlyTrendItem(
                    review_month=month,
                    correct=counts["correct"],
                    partially_correct=counts["partially_correct"],
                    incorrect=counts["incorrect"],
                    total=total,
                    accuracy_pct=round(accuracy, 1),
                )
            )

        return AnalyticsTrendResponse(trend=trend)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Failed to get analytics trend: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get analytics trend: {str(e)}"
        )


@router.get("/recent-reviews", response_model=RecentReviewsResponse)
async def get_recent_reviews(
    reviewer: Optional[str] = Query(None, description="Filter by reviewer email"),
    verdict: Optional[str] = Query(None, description="Filter by verdict"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    search: Optional[str] = Query(None, description="Search document name"),
    include_automated: bool = Query(False, description="Include auto-verified rows"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    workspace_client: WorkspaceClient = Depends(get_workspace_client),
):
    """Most recent reviews with document info, filtering, and pagination.

    Row-level query on fact_review (already excludes soft-deleted docs; auto
    rows carry their contract shape: 'auto-<doc>' id, '<automated>' reviewer,
    'correct' verdict).
    """
    try:
        params: Dict[str, str] = {}
        conditions: List[str] = []

        if not include_automated:
            conditions.append("NOT is_automated")
        if reviewer:
            raw_ids = _get_reviewer_raw_ids(workspace_client, reviewer)
            conditions.append(_reviewer_in_clause(raw_ids, params))
        if verdict:
            conditions.append("verdict = :verdict")
            params["verdict"] = verdict
        if date_from:
            conditions.append("reviewed_at >= CAST(:date_from AS DATE)")
            params["date_from"] = date_from
        if date_to:
            conditions.append("reviewed_at < DATE_ADD(CAST(:date_to AS DATE), 1)")
            params["date_to"] = date_to
        if search:
            conditions.append("document_name ILIKE '%' || :search || '%'")
            params["search"] = search

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        # limit/offset are FastAPI-validated ints — safe to interpolate.
        # review_id DESC tiebreak: auto rows share batch timestamps, and
        # LIMIT/OFFSET pagination needs a deterministic order.
        query = f"""
            SELECT
                review_id AS id,
                document_id,
                document_name,
                reviewer_email,
                verdict,
                reasoning,
                is_automated,
                reviewed_at AS created_at,
                COUNT(*) OVER () AS total_count
            FROM {_qualified("fact_review")}
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT {limit} OFFSET {offset}
        """
        rows = _execute_warehouse_query(workspace_client, query, params or None)

        if rows:
            total_count = int(rows[0].get("total_count", 0))
        elif offset > 0:
            # Page past the end: the window count disappears with the rows.
            count_rows = _execute_warehouse_query(
                workspace_client,
                f"SELECT COUNT(*) AS cnt FROM {_qualified('fact_review')} {where}",
                params or None,
            )
            total_count = int(count_rows[0]["cnt"]) if count_rows else 0
        else:
            total_count = 0

        reviews = []
        for row in rows:
            raw_reviewer = row["reviewer_email"]
            identity = resolve_user_identity(raw_reviewer)
            reviews.append(
                RecentReviewItem(
                    id=str(row["id"]),
                    document_id=str(row["document_id"]),
                    document_name=row["document_name"],
                    reviewer_email=raw_reviewer,
                    reviewer_display_name=identity["display_name"] or identity["email"],
                    verdict=row["verdict"],
                    reasoning=row.get("reasoning"),
                    is_automated=_is_true(row.get("is_automated", "false")),
                    created_at=row["created_at"],
                )
            )
        return RecentReviewsResponse(
            reviews=reviews,
            total_count=total_count,
            limit=limit,
            offset=offset,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get recent reviews: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get recent reviews: {str(e)}"
        )
