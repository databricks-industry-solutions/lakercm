"""
LakeRCM Lakebase Database Layer

Manages PostgreSQL database operations for document extraction review.
Uses Lakebase Autoscaling with postgres.generate_database_credential()
for automatic credential rotation.

Connection Pattern (Databricks Apps + Lakebase Autoscaling):
- PG* env vars set in app.yaml based on Lakebase endpoint
- Password: workspace_client.postgres.generate_database_credential()
- psycopg (v3) + psycopg_pool.ConnectionPool for connection management
- Auto-refreshes credential every 45 minutes (tokens expire after 1 hour)
"""

import json
import os
import time
import logging
from typing import List, Optional, Dict, Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from databricks.sdk import WorkspaceClient

from config import settings
from services import hold_reasons


def _as_reason_list(value) -> List[str]:
    """Coerce the synced review_reasons column to a list of strings.

    The column is JSONB, and psycopg hands it back as a Python list in normal
    operation — but a JSON round trip elsewhere can present it as a string, and
    the pre-gold fallback substitutes NULL. Anything unreadable becomes an empty
    list, which is safe here ONLY because "no reasons" no longer implies a cause:
    hold_reasons refuses to explain a hold it cannot account for instead of
    assuming low confidence.
    """
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        if isinstance(parsed, list):
            return [str(v) for v in parsed if v]
    return []


logger = logging.getLogger(__name__)

TOKEN_REFRESH_INTERVAL_SECONDS = 2700  # 45 min (tokens expire after 1 hour)
_OTEL_PSYCOPG_INSTRUMENTED = False


def _instrument_psycopg_safely() -> None:
    """Enable psycopg OTel spans without recording SQL text or bind values.

    The stock DB-API instrumentor records the executed statement by default.
    LakeRCM handles medical claims data, so redact statements at the
    instrumentation boundary and keep parameter capture/sqlcommenter disabled.
    `opentelemetry-instrument` is configured to skip psycopg auto-loading; this
    function installs the sanitized instrumentation before the first pool opens.
    """
    global _OTEL_PSYCOPG_INSTRUMENTED
    if _OTEL_PSYCOPG_INSTRUMENTED:
        return

    try:
        from opentelemetry.instrumentation import dbapi
        from opentelemetry.instrumentation import psycopg as otel_psycopg
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
        from psycopg.sql import Composable
    except ImportError:
        logger.warning("OpenTelemetry psycopg instrumentation package not installed")
        return

    try:

        def _redacted_statement(self, cursor, args):
            return ""

        def _safe_operation_name(self, cursor, args):
            # opentelemetry-instrumentation-psycopg renders Composable queries
            # via as_string() for operation-name extraction. Avoid rendering SQL
            # objects entirely; a generic DB span is preferable to PHI exposure.
            if args and isinstance(args[0], Composable):
                return "SQL"
            return dbapi.CursorTracer.get_operation_name(self, cursor, args)

        for cursor_tracer in (
            dbapi.CursorTracer,
            otel_psycopg.CursorTracer,
        ):
            if not cursor_tracer.__dict__.get("_lakercm_statement_redacted", False):
                cursor_tracer.get_statement = _redacted_statement
                cursor_tracer._lakercm_statement_redacted = True

        if not otel_psycopg.CursorTracer.__dict__.get(
            "_lakercm_operation_name_redacted", False
        ):
            otel_psycopg.CursorTracer.get_operation_name = _safe_operation_name
            otel_psycopg.CursorTracer._lakercm_operation_name_redacted = True

        PsycopgInstrumentor().instrument(
            capture_parameters=False,
            enable_commenter=False,
            enable_attribute_commenter=False,
        )
        _OTEL_PSYCOPG_INSTRUMENTED = True
        logger.info("OpenTelemetry psycopg instrumentation enabled with SQL redaction")
    except Exception as e:
        logger.warning("OpenTelemetry psycopg instrumentation failed: %s", e)


class LakeRCMDatabase:
    """
    Manages Lakebase PostgreSQL database for LakeRCM document review.
    """

    def __init__(self, workspace_client: Optional[WorkspaceClient] = None):
        self.workspace_client = workspace_client or WorkspaceClient()
        self._postgres_password: Optional[str] = None
        self._last_password_refresh: float = 0
        self._connection_pool: Optional[ConnectionPool] = None
        # Answers to gold_sync_has_column. Only changes when a sync lands, and
        # the document list asks on every request.
        self._column_cache: Dict[str, bool] = {}

        logger.info("Initializing LakeRCM Lakebase connection...")

        try:
            self._refresh_credential()
            self._create_connection_pool()
            logger.info(
                "Connected to Lakebase: %s:%s/%s",
                os.getenv("PGHOST", "?"),
                os.getenv("PGPORT", "5432"),
                os.getenv("PGDATABASE", "?"),
            )
            # Schema/privilege DDL (synced-schema GRANTs and checkpoint
            # group-role GRANTs) used to be re-asserted here on every pod
            # boot. That's now declared in alembic migrations and applied
            # via `_run_alembic_upgrade()` in main.py. Keeping it out of
            # the runtime data layer removes MLflow trace pollution and
            # avoids the 12-factor anti-pattern of apps mutating their
            # own data layer at startup.
        except Exception as e:
            logger.error("Failed to connect to Lakebase: %s", e, exc_info=True)
            logger.warning("Lazy initialization will be attempted on first request.")
            self._connection_pool = None

    @property
    def pg_host(self):
        return os.getenv("PGHOST", "")

    @property
    def pg_user(self):
        # PGUSER falls back to DATABRICKS_CLIENT_ID (auto-injected by the
        # Apps runtime) — the running SP's client_id IS the Postgres role
        # name for Lakebase OAuth.
        return os.getenv("PGUSER") or os.getenv("DATABRICKS_CLIENT_ID", "")

    @property
    def pool(self):
        return self._connection_pool

    def _refresh_credential(self) -> bool:
        if (
            self._postgres_password is not None
            and time.time() - self._last_password_refresh
            < TOKEN_REFRESH_INTERVAL_SECONDS
        ):
            return True

        logger.info("Generating Lakebase database credential...")
        try:
            credential = self.workspace_client.postgres.generate_database_credential(
                endpoint=settings.endpoint_name
            )
            self._postgres_password = credential.token
            self._last_password_refresh = time.time()
            logger.info("Lakebase credential generated")
            return True
        except Exception as e:
            logger.warning(
                "Failed to generate Lakebase credential via Autoscaling API: %s", e
            )
            # Fallback: try OAuth token (works with Provisioned Lakebase)
            try:
                self._postgres_password = (
                    self.workspace_client.config.oauth_token().access_token
                )
                self._last_password_refresh = time.time()
                logger.info("Fell back to OAuth token for Lakebase auth")
                return True
            except Exception as e2:
                logger.error("All credential methods failed: %s", e2, exc_info=True)
                return False

    def _create_connection_pool(self):
        if self._connection_pool:
            try:
                self._connection_pool.close()
            except Exception as e:
                logger.warning("Error closing existing connection pool: %s", e)
            self._connection_pool = None

        conn_string = (
            f"dbname={os.getenv('PGDATABASE', 'databricks_postgres')} "
            f"user={self.pg_user} "
            f"password={self._postgres_password} "
            f"host={os.getenv('PGHOST', '')} "
            f"port={os.getenv('PGPORT', '5432')} "
            f"sslmode={os.getenv('PGSSLMODE', 'require')} "
            f"application_name={os.getenv('PGAPPNAME', 'lakercm')}"
        )

        _instrument_psycopg_safely()
        self._connection_pool = ConnectionPool(
            conn_string,
            min_size=settings.db_pool_min_connections,
            max_size=settings.db_pool_max_connections,
        )

    def _get_connection(self):
        if (
            self._postgres_password is None
            or time.time() - self._last_password_refresh
            >= TOKEN_REFRESH_INTERVAL_SECONDS
        ):
            if self._connection_pool:
                self._connection_pool.close()
                self._connection_pool = None
            self._refresh_credential()
            self._create_connection_pool()

        if not self._connection_pool:
            raise RuntimeError("Connection pool not initialized")

        return self._connection_pool.connection()

    def _execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        if not self._connection_pool:
            self._refresh_credential()
            self._create_connection_pool()

        if not self._connection_pool:
            raise RuntimeError(
                "Database connection not initialized. "
                "Ensure Lakebase project is linked to this Databricks App."
            )

        retry_attempted = False

        while True:
            try:
                with self._get_connection() as conn:
                    with conn.cursor(row_factory=dict_row) as cursor:
                        cursor.execute(query, params or ())
                        if fetch:
                            return cursor.fetchall()
                        else:
                            conn.commit()
                            return cursor.rowcount
            except psycopg.IntegrityError:
                raise
            except psycopg.OperationalError as e:
                error_msg = str(e).lower()
                is_auth_error = any(
                    kw in error_msg
                    for kw in ["authentication", "password", "credentials", "token"]
                )
                if is_auth_error and not retry_attempted:
                    logger.warning("Auth error, refreshing credential: %s", e)
                    self._refresh_credential()
                    self._create_connection_pool()
                    retry_attempted = True
                    continue
                else:
                    logger.error("Database operational error: %s", e, exc_info=True)
                    raise
            except Exception as e:
                logger.error("Database query failed: %s", e, exc_info=True)
                raise

    def close(self):
        if self._connection_pool:
            try:
                self._connection_pool.close()
                logger.info("Database connection pool closed")
                self._connection_pool = None
            except Exception as e:
                logger.error("Error closing connection pool: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()

    # Public tables the agent's LangGraph tools SELECT from. Source of
    # truth: agent_app/agent/tools.py. Update this list if a new tool
    # joins a new reviewer-owned table.
    _AGENT_READ_TABLES = (
        "medical_documents",
        "document_extraction_reviews",
        "conversations",
        "lakebase_events",
    )

    def health_check(self) -> bool:
        try:
            if not self._connection_pool:
                return False
            self._execute_query("SELECT 1")
            return True
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    # =========================================================================
    # DOCUMENT OPERATIONS
    # =========================================================================

    def find_document_by_content_hash(
        self, content_hash: str
    ) -> Optional[Dict[str, Any]]:
        query = """
            SELECT id, document_name, file_path, upload_timestamp
            FROM public.medical_documents
            WHERE content_hash = %s AND deleted_at IS NULL
        """
        results = self._execute_query(query, (content_hash,))
        return dict(results[0]) if results else None

    def create_document_record(self, document_data: Dict[str, Any]) -> str:
        query = """
            INSERT INTO public.medical_documents (
                user_email, document_name, file_path,
                file_size, document_type, notes,
                processing_status, content_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (
            document_data["user_email"],
            document_data["document_name"],
            document_data["file_path"],
            document_data["file_size"],
            document_data.get("document_type"),
            document_data.get("notes"),
            document_data.get("processing_status", "processing"),
            document_data.get("content_hash"),
        )
        result = self._execute_query(query, params)
        return str(result[0]["id"])

    # Live status is derived from a LEFT JOIN against the Lakebase synced copy
    # of gold_extraction_labels. The stored `processing_status` column is only
    # authoritative for 'processing' (pre-extraction) and 'failed'. Everything
    # else (pending / auto_verified) comes from the gold row: its presence, and
    # the pipeline's routing decision `is_automated` (confidence at or above
    # auto_verdict_threshold AND no invalid or non-billable code, no missing
    # member ID; see pipelines/gold/gold_extraction_labels.sql). The app used to
    # re-derive it from confidence alone, so a confident document with a code a
    # payer would reject skipped the review queue.
    # The Lakebase copy of gold_extraction_labels. A synced table takes its
    # Postgres schema from its Unity Catalog schema, so dev reads
    # lakercm_dev.* and prod lakercm.* — the app must never hardcode
    # one. Set per deploy from LAKERCM_SCHEMA (bundles/apps/resources/app.yml).
    GOLD_SYNC = f"{settings.lakercm_schema}.gold_extraction_labels_sync"
    GOLD_SYNC_SCHEMA = settings.lakercm_schema

    _STATUS_CASE_SQL = (
        "CASE "
        "  WHEN d.processing_status = 'failed' THEN 'failed' "
        "  WHEN g.document_path IS NULL THEN COALESCE(d.processing_status, 'processing') "
        "  WHEN g.is_automated THEN 'auto_verified' "
        "  ELSE 'pending' "
        "END"
    )

    def _gold_sync_available(self) -> bool:
        """Check whether the SDP-synced gold table exists in Lakebase.

        Returns False when the pipeline hasn't materialized the sync table yet —
        callers fall back to a simpler query so the UI still renders.
        """
        rows = self._execute_query(f"SELECT to_regclass('{self.GOLD_SYNC}') AS oid")
        return bool(rows and rows[0].get("oid"))

    def gold_sync_has_column(self, column: str) -> bool:
        """Whether the Lakebase copy of gold_extraction_labels has `column`.

        A column the pipeline adds reaches the synced table only after the
        pipeline's next update AND the sync after it, so there is always a window
        where this app is newer than the table it reads. Selecting a column that
        has not arrived yet fails the whole query, which would empty the document
        list rather than degrade one badge on it.

        Ported from agent_app/services/lakehouse_db.py, which has needed this
        since review_reasons was added there. Cached per column because the
        document list calls it on every request and the answer only changes when
        a sync lands.
        """
        # setdefault rather than a plain attribute read: the routing tests build
        # this object without running __init__, and a schema probe must not be
        # the thing that makes a test double unusable.
        cache = self.__dict__.setdefault("_column_cache", {})
        if column in cache:
            return cache[column]
        try:
            rows = self._execute_query(
                """
                SELECT 1 AS present
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = 'gold_extraction_labels_sync'
                  AND column_name = %s
                """,
                (self.GOLD_SYNC_SCHEMA, column),
            )
            present = bool(rows)
        except Exception as e:  # noqa: BLE001 - degrade, never break the list
            logger.warning("gold_sync_has_column(%s) check failed: %s", column, e)
            present = False
        cache[column] = present
        return present

    def gold_sync_available(self) -> bool:
        """Whether this deploy's Lakebase copy of gold exists.

        Public for /health: without it the document list, the review queue and
        the agent's document tools are empty however well the pipeline ran, and
        nothing else in the UI says why.
        """
        return self._gold_sync_available()

    def backfill_streamed_documents(self) -> None:
        """Public entry point for the streamed-document backfill.

        Anything that reads medical_documents OUTSIDE an HTTP request has to call
        this itself. Until it existed the only callers were list_all_documents,
        count_all_documents and get_status_counts — all reviewer-app read
        handlers — so a streamed document entered Lakebase only once somebody
        opened the app. That is fine for the UI, which does the backfill on its
        way to rendering, and wrong for a batch job: the triage job precomputes
        proposals so they are WAITING when a reviewer arrives, and it was reading
        a table that nothing had populated yet.

        Measured on dev: after ingesting 100 documents, gold held 2,097 and
        Lakebase 1,997. Triage found 36 newly-held documents in gold, could not
        resolve any of them, and wrote 0 proposals. One GET of
        /api/documents/status-counts closed the gap, and the next identical run
        wrote 36.
        """
        self._backfill_streamed_documents()

    def _backfill_streamed_documents(self) -> None:
        """Upsert a medical_documents row for every gold_sync.document_path
        that doesn't have one yet.

        The UC volume can be written to directly (streaming / batch copy),
        bypassing /api/documents/upload. The SDP gold pipeline extracts those
        files but never touches Lakebase, so without this backfill they have
        no medical_documents row and disappear from the reviewer UI (which
        LEFT JOINs medical_documents), from the analytics pipeline (which
        resolves document_path -> document_id via medical_documents), and from
        the triage job (which attaches proposals by document_id).

        Idempotent via the UNIQUE constraint on file_path. Runs every list
        call — trivial for <10k docs; add a high-water predicate if it gets
        expensive.
        """
        if not self._gold_sync_available():
            return
        # Streamed docs skip /api/documents/upload entirely — we only learn
        # about them after SDP has already extracted. upload_timestamp is
        # NULL because we don't know when the file landed in the volume;
        # processing_timestamp is extracted_at, which IS meaningful (it's
        # when SDP finished processing the doc).
        query = f"""
            INSERT INTO public.medical_documents
                (user_email, document_name, file_path, file_size,
                 processing_status, upload_timestamp, processing_timestamp)
            SELECT
                COALESCE(g.user_email, 'streamed'),
                COALESCE(g.document_name, regexp_replace(g.document_path, '.*/', '')),
                g.document_path,
                0,
                CASE WHEN g.is_automated THEN 'auto_verified' ELSE 'pending' END,
                NULL,
                g.extracted_at
            FROM {self.GOLD_SYNC} g
            WHERE NOT EXISTS (
                SELECT 1 FROM public.medical_documents d
                WHERE d.file_path = g.document_path
            )
            ON CONFLICT (file_path) DO NOTHING
        """
        try:
            self._execute_query(query, fetch=False)
        except Exception as e:
            logger.warning("Streamed-doc backfill skipped: %s", e)

    _VALID_STATUS_FILTERS = (
        "processing",
        "pending",
        "reviewed",
        "auto_verified",
        "failed",
    )

    def _build_list_query(
        self,
        *,
        select_clause: str,
        include_auto_verified: bool,
        status_filter: Optional[str],
        search: Optional[str] = None,
    ) -> tuple:
        """Shared query builder for list_all_documents / count_all_documents.

        Wraps the per-doc effective_status + has_review derivation in a
        subquery so the status filter can reference the derived columns.
        """
        gold_available = self._gold_sync_available()
        if gold_available:
            # review_reasons is what turns a flat "Pending" badge into the actual
            # reason on the card. Guarded because the column reaches the synced
            # table only after the pipeline's next update and the sync after it —
            # selecting it too early would empty the whole list.
            reasons_col = (
                "g.review_reasons"
                if self.gold_sync_has_column("review_reasons")
                # NULL::TEXT, not ::JSONB: test_review_routing.py executes
                # this exact SQL on DuckDB, which has no JSONB type. Nothing in
                # SQL touches the value -- _as_reason_list parses it in Python --
                # so the placeholder's type only has to be portable.
                else "CAST(NULL AS TEXT)"
            )
            inner_sql = f"""
                SELECT
                    d.*,
                    g.confidence_score AS extraction_confidence,
                    g.extracted_at AS extracted_at,
                    g.label AS extracted_label,
                    {reasons_col} AS review_reasons,
                    g.is_automated AS gold_is_automated,
                    {self._STATUS_CASE_SQL} AS effective_status,
                    EXISTS (
                        SELECT 1 FROM public.document_extraction_reviews r
                        WHERE r.document_id = d.id
                    ) AS has_review
                FROM public.medical_documents d
                LEFT JOIN {self.GOLD_SYNC} g
                    ON g.document_path = d.file_path
                WHERE d.deleted_at IS NULL
            """
            params: list = []
        else:
            inner_sql = """
                SELECT
                    d.*,
                    NULL::DOUBLE PRECISION AS extraction_confidence,
                    NULL::TIMESTAMP AS extracted_at,
                    NULL::TEXT AS extracted_label,
                    CAST(NULL AS TEXT) AS review_reasons,
                    NULL::BOOLEAN AS gold_is_automated,
                    COALESCE(d.processing_status, 'processing') AS effective_status,
                    EXISTS (
                        SELECT 1 FROM public.document_extraction_reviews r
                        WHERE r.document_id = d.id
                    ) AS has_review
                FROM public.medical_documents d
                WHERE d.deleted_at IS NULL
            """
            params = []

        outer_conditions: list = []
        if status_filter == "processing":
            outer_conditions.append("effective_status = 'processing'")
        elif status_filter == "pending":
            outer_conditions.append("effective_status = 'pending'")
            outer_conditions.append("has_review = false")
        elif status_filter == "reviewed":
            if include_auto_verified:
                outer_conditions.append(
                    "(has_review = true OR effective_status = 'auto_verified')"
                )
            else:
                outer_conditions.append("has_review = true")
        elif status_filter == "auto_verified":
            # Disjoint from reviewed: a doc with a human review belongs in
            # the 'reviewed' bucket regardless of confidence. Without
            # has_review = false, the dashboard total double-counts the 37
            # human-reviewed docs (all of which have confidence >= threshold).
            outer_conditions.append("effective_status = 'auto_verified'")
            outer_conditions.append("has_review = false")
        elif status_filter == "failed":
            outer_conditions.append("effective_status = 'failed'")
        else:
            if not include_auto_verified:
                # Hide unreviewed-auto-verified docs; keep human-reviewed
                # docs visible regardless of confidence. The 37 reviewed
                # docs all have confidence >= threshold; without the
                # has_review escape, the All tab collapses to zero.
                outer_conditions.append(
                    "(effective_status <> 'auto_verified' OR has_review = true)"
                )

        if search:
            outer_conditions.append(
                "(document_name ILIKE %s OR file_path ILIKE %s OR notes ILIKE %s)"
            )
            wildcard = f"%{search}%"
            params.extend([wildcard, wildcard, wildcard])

        where = (" WHERE " + " AND ".join(outer_conditions)) if outer_conditions else ""
        query = f"SELECT {select_clause} FROM ({inner_sql}) AS sub{where}"
        return query, params

    def list_all_documents(
        self,
        include_auto_verified: bool = False,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List documents with live status derived from the gold sync table.

        status_filter: 'processing' | 'pending' | 'reviewed' | None (all).
        search: case-insensitive substring match on document_name, file_path, notes.
        Excludes auto_verified docs unless include_auto_verified=True.
        """
        self._backfill_streamed_documents()
        if status_filter not in (None,) + self._VALID_STATUS_FILTERS:
            status_filter = None

        query, params = self._build_list_query(
            select_clause="*",
            include_auto_verified=include_auto_verified,
            status_filter=status_filter,
            search=search,
        )
        # Streamed docs have no true upload_timestamp (they bypass /upload);
        # fall back to processing_timestamp (SDP completion) or created_at so
        # they still sort chronologically with user-uploaded docs.
        query += " ORDER BY COALESCE(upload_timestamp, processing_timestamp, created_at) DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        results = self._execute_query(query, tuple(params))
        out = []
        for row in results:
            d = dict(row)
            d["processing_status"] = d.pop("effective_status")
            d.pop("has_review", None)
            # What the card shows instead of a flat "Pending". Derived through
            # the SAME function the detail page uses, so a card and the document
            # it opens can never name different reasons.
            primary = hold_reasons.primary_for_list(
                effective_status=d["processing_status"],
                is_automated=d.get("gold_is_automated"),
                confidence_score=d.get("extraction_confidence"),
                review_reasons=_as_reason_list(d.pop("review_reasons", None)),
                threshold=settings.auto_verdict_threshold,
            )
            d.pop("gold_is_automated", None)
            d["hold_primary_code"] = primary.code if primary else None
            d["hold_primary_label"] = primary.title if primary else None
            d["hold_unexplained"] = bool(
                primary and primary.code == hold_reasons.REASON_NOT_RECORDED
            )
            out.append(d)
        return out

    def count_all_documents(
        self,
        include_auto_verified: bool = False,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> int:
        self._backfill_streamed_documents()
        if status_filter not in (None,) + self._VALID_STATUS_FILTERS:
            status_filter = None

        query, params = self._build_list_query(
            select_clause="COUNT(*) AS count",
            include_auto_verified=include_auto_verified,
            status_filter=status_filter,
            search=search,
        )
        result = self._execute_query(query, tuple(params) if params else None)
        return int(result[0]["count"]) if result else 0

    def get_status_counts(self, include_auto_verified: bool = False) -> Dict[str, int]:
        """Counts of documents in each UI-facing status bucket.

        Buckets are disjoint so total = sum(buckets) without double-counting:
          processing    -> extraction in flight
          pending       -> extraction done, awaiting review (has_review=false)
          reviewed      -> has a human review (regardless of confidence)
          auto_verified -> the pipeline auto-verified it AND not human-reviewed
          failed        -> processing_status='failed'

        Previously `auto_verified` and `reviewed` overlapped because all 37
        human-reviewed docs also had confidence >= threshold, inflating
        the dashboard total to 510 instead of the real 473.
        """
        self._backfill_streamed_documents()
        counts = {
            "processing": self.count_all_documents(
                include_auto_verified=include_auto_verified,
                status_filter="processing",
            ),
            "pending": self.count_all_documents(
                include_auto_verified=include_auto_verified,
                status_filter="pending",
            ),
            "reviewed": self.count_all_documents(
                include_auto_verified=include_auto_verified,
                status_filter="reviewed",
            ),
            "failed": self.count_all_documents(
                include_auto_verified=include_auto_verified,
                status_filter="failed",
            ),
        }
        # auto_verified is the disjoint "auto-verified AND not reviewed"
        # bucket — see _build_list_query. Always count with
        # include_auto_verified=True so the toggle off doesn't suppress
        # the legend tile on /overview.
        counts["auto_verified"] = self.count_all_documents(
            include_auto_verified=True,
            status_filter="auto_verified",
        )
        counts["total"] = (
            counts["processing"]
            + counts["pending"]
            + counts["reviewed"]
            + counts["auto_verified"]
            + counts["failed"]
        )
        return counts

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        if not self._gold_sync_available():
            rows = self._execute_query(
                """
                SELECT
                    d.*,
                    NULL::DOUBLE PRECISION AS extraction_confidence,
                    NULL::TIMESTAMP AS extracted_at,
                    NULL::TEXT AS extracted_label,
                    COALESCE(d.processing_status, 'processing') AS effective_status
                FROM public.medical_documents d
                WHERE d.id = %s AND d.deleted_at IS NULL
                """,
                (document_id,),
            )
            if not rows:
                return None
            d = dict(rows[0])
            d["processing_status"] = d.pop("effective_status")
            return d

        status_case = self._STATUS_CASE_SQL
        query = f"""
            SELECT
                d.*,
                g.confidence_score AS extraction_confidence,
                g.extracted_at AS extracted_at,
                g.label AS extracted_label,
                {status_case} AS effective_status
            FROM public.medical_documents d
            LEFT JOIN {self.GOLD_SYNC} g
                ON g.document_path = d.file_path
            WHERE d.id = %s AND d.deleted_at IS NULL
        """
        results = self._execute_query(query, (document_id,))
        if not results:
            return None
        d = dict(results[0])
        d["processing_status"] = d.pop("effective_status")
        return d

    def get_documents_by_names(self, names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Reviewer id, status and type for documents named by file basename.

        The knowledge graph names a document by basename(file_path) -- its
        Document id -- and knows nothing of medical_documents.id, which is what
        the review page routes on. This is the bridge, so a graph node can link
        straight to a document instead of searching for it by name in whatever
        page of the list the browser cached (which missed every auto-verified
        document: the default list excludes them).

        Matching on the basename, not document_name: document_name is the
        ORIGINAL upload filename, while the graph's id comes from the volume path
        the upload prefixed (see routes/kg.py). A basename can recur across
        re-uploads, so the most recently updated live row wins.
        """
        if not names:
            return {}
        if self._gold_sync_available():
            inner = f"""
                SELECT regexp_replace(d.file_path, '.*/', '') AS base,
                       d.id,
                       g.label AS label,
                       {self._STATUS_CASE_SQL} AS status,
                       d.updated_at
                FROM public.medical_documents d
                LEFT JOIN {self.GOLD_SYNC} g
                    ON g.document_path = d.file_path
                WHERE d.deleted_at IS NULL
            """
        else:
            inner = """
                SELECT regexp_replace(d.file_path, '.*/', '') AS base,
                       d.id,
                       NULL::TEXT AS label,
                       COALESCE(d.processing_status, 'processing') AS status,
                       d.updated_at
                FROM public.medical_documents d
                WHERE d.deleted_at IS NULL
            """
        # The basename is computed once in the subquery so DISTINCT ON and ORDER
        # BY name a plain column rather than leaning on alias resolution.
        query = f"""
            SELECT DISTINCT ON (base) base, id, label, status
            FROM ({inner}) m
            WHERE base = ANY(%s)
            ORDER BY base, updated_at DESC NULLS LAST
        """
        rows = self._execute_query(query, (list(names),))
        return {
            r["base"]: {
                "id": str(r["id"]),
                "status": r.get("status"),
                "label": r.get("label"),
            }
            for r in rows or []
        }

    def update_document_status(
        self,
        file_path: str,
        processing_status: str,
        num_pages: Optional[int] = None,
        element_count: Optional[int] = None,
        processing_error: Optional[str] = None,
    ) -> bool:
        query = """
            UPDATE public.medical_documents
            SET processing_status = %s,
                num_pages = COALESCE(%s, num_pages),
                element_count = COALESCE(%s, element_count),
                processing_error = %s,
                processing_timestamp = CASE
                    WHEN %s IN ('pending', 'auto_verified', 'failed') THEN NOW()
                    ELSE processing_timestamp
                END,
                updated_at = NOW()
            WHERE file_path = %s AND deleted_at IS NULL
        """
        params = (
            processing_status,
            num_pages,
            element_count,
            processing_error,
            processing_status,
            file_path,
        )
        rowcount = self._execute_query(query, params, fetch=False)
        return rowcount > 0

    def update_document_status_with_timestamp(
        self,
        file_path: str,
        processing_status: str,
        processing_timestamp: str,
    ) -> bool:
        """Update status and set processing_timestamp to a specific value."""
        query = """
            UPDATE public.medical_documents
            SET processing_status = %s,
                processing_timestamp = %s::timestamptz,
                updated_at = NOW()
            WHERE file_path = %s AND deleted_at IS NULL
        """
        rowcount = self._execute_query(
            query, (processing_status, processing_timestamp, file_path), fetch=False
        )
        return rowcount > 0

    def update_document_status_by_id(
        self, document_id: str, processing_status: str
    ) -> bool:
        query = """
            UPDATE public.medical_documents
            SET processing_status = %s, updated_at = NOW()
            WHERE id = %s AND deleted_at IS NULL
        """
        rowcount = self._execute_query(
            query, (processing_status, document_id), fetch=False
        )
        return rowcount > 0

    def get_pending_documents(self) -> List[Dict[str, Any]]:
        """Return all documents still being processed (pre-extraction)."""
        query = """
            SELECT id, file_path, upload_timestamp
            FROM public.medical_documents
            WHERE processing_status = 'processing'
              AND deleted_at IS NULL
        """
        return [dict(r) for r in self._execute_query(query)]

    def get_documents_needing_timestamp_resync(self) -> List[Dict[str, Any]]:
        """Return documents where processing_timestamp was set to NOW() by a
        previous sync instead of the real pipeline extracted_at value.

        Heuristic: processing_timestamp is within the last 24 hours but
        upload_timestamp is more than 24 hours old — meaning the sync
        backfilled with NOW() rather than the real time.
        """
        query = """
            SELECT id, file_path, upload_timestamp, processing_timestamp
            FROM public.medical_documents
            WHERE processing_status = 'pending'
              AND processing_timestamp IS NOT NULL
              AND processing_timestamp > NOW() - INTERVAL '24 hours'
              AND upload_timestamp < NOW() - INTERVAL '24 hours'
              AND deleted_at IS NULL
        """
        return [dict(r) for r in self._execute_query(query)]

    def get_unreviewed_ready_documents(self) -> List[Dict[str, Any]]:
        """Return 'pending' (extraction-done) docs with no human review yet.
        Candidates for promotion to 'auto_verified' if the gold MV now
        contains the file_path — closes a race where the initial sync marked
        the doc 'pending' before the gold MV caught up.
        """
        query = """
            SELECT d.id, d.file_path, d.upload_timestamp
            FROM public.medical_documents d
            LEFT JOIN public.document_extraction_reviews r
              ON r.document_id = d.id
            WHERE d.processing_status = 'pending'
              AND d.deleted_at IS NULL
              AND r.id IS NULL
        """
        return [dict(r) for r in self._execute_query(query)]

    def get_latest_review_per_document(
        self, document_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Return {document_id: {verdict, reviewer_email, updated_at}} —
        one row per document.

        Shared-pool model: UNIQUE(document_id) guarantees at most one review
        row per doc. Upserts from any reviewer overwrite the canonical row
        (see migration 20260421_000022).
        """
        if not document_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(document_ids))
        query = f"""
            SELECT document_id, verdict, reviewer_email, updated_at
            FROM public.document_extraction_reviews
            WHERE document_id IN ({placeholders})
        """
        results = self._execute_query(query, tuple(document_ids))
        return {
            str(row["document_id"]): {
                "verdict": row["verdict"],
                "reviewer_email": row["reviewer_email"],
                "updated_at": row["updated_at"],
            }
            for row in results
        }

    def soft_delete_document(self, document_id: str) -> bool:
        query = """
            UPDATE public.medical_documents
            SET deleted_at = NOW(), updated_at = NOW()
            WHERE id = %s AND deleted_at IS NULL
        """
        rowcount = self._execute_query(query, (document_id,), fetch=False)
        return rowcount > 0

    def get_document_elements(self, file_path: str) -> List[Dict[str, Any]]:
        logger.warning(
            "get_document_elements: stub - no synced element table yet (file_path=%s)",
            file_path,
        )
        return []

    def get_document_extractions(self, file_path: str) -> Dict[str, Any]:
        logger.warning(
            "get_document_extractions: stub - no synced extraction table yet (file_path=%s)",
            file_path,
        )
        return {
            "dates": [],
            "medical_entities": [],
            "patient_info": {},
            "document_metadata": {},
        }

    def get_gold_labels_by_paths(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Fetch gold extraction metadata for a set of paths from the
        Lakebase-synced table. Used by sync-status to avoid a warehouse
        round-trip — gold_extraction_labels_sync carries the same columns."""
        if not file_paths:
            return []
        placeholders = ",".join(["%s"] * len(file_paths))
        query = f"""
            SELECT
                document_path,
                confidence_score,
                is_automated,
                CAST(extracted_at AS TEXT) AS extracted_at
            FROM {self.GOLD_SYNC}
            WHERE document_path IN ({placeholders})
        """
        return [dict(r) for r in self._execute_query(query, tuple(file_paths))]

    def get_document_page_images(self, file_path: str) -> List[Optional[str]]:
        """Rendered per-page image URIs (ai_parse_document imageOutputPath),
        ordered by page id, from the gold sync table. Returns [] when the doc
        predates page rendering OR the column hasn't synced yet — the caller
        falls back to the original upload. Fail-soft: never raises."""
        query = f"""
            SELECT page_images
            FROM {self.GOLD_SYNC}
            WHERE document_path = %s
            LIMIT 1
        """
        try:
            results = self._execute_query(query, (file_path,))
            if not results:
                return []
            raw = results[0].get("page_images")
            if raw is None:
                return []
            if isinstance(raw, str):  # some drivers return text[] as a JSON string
                try:
                    raw = json.loads(raw)
                except (ValueError, TypeError):
                    return []
            return [u for u in raw if u]
        except Exception as e:
            # Column may not exist yet on this target, or the row is absent —
            # both are fine; the image endpoint falls back to the original.
            logger.debug(
                "get_document_page_images: no page images for %r (%s)", file_path, e
            )
            return []

    def get_extraction_comparisons(self, file_path: str) -> List[Dict[str, Any]]:
        """Gold extraction row(s) for one document, for the reviewer UI.

        The SQL below must contain no percent sign ANYWHERE, comments included.
        psycopg reads that character as the start of a parameter placeholder
        before it parses SQL comments, and this query is parameterised, so a
        single one in a comment fails the whole call with
        "incomplete placeholder" and the endpoint answers 500.
        """
        logger.info(
            "get_extraction_comparisons: querying gold for file_path=%r", file_path
        )
        # ai_classify v2.1's confidence and rationale reach the synced table only
        # after the pipeline's next update and the sync after it, so they are
        # selected once they exist and stood in for as NULL until then. Selecting
        # them unconditionally would 500 this endpoint for that whole window and
        # take the extraction panel down with it.
        classify_cols = (
            "classify_confidence, classify_rationale"
            if self.gold_sync_has_column("classify_confidence")
            else (
                "NULL::DOUBLE PRECISION AS classify_confidence, "
                "NULL::TEXT AS classify_rationale"
            )
        )
        query = f"""
            SELECT
                document_path, document_name, user_email,
                label, identifiers, elements, confidence_score, extracted_at,
                page_images,
                {classify_cols},
                -- Why the pipeline did or did not auto-verify this document.
                -- Confidence alone does NOT decide it: auto-verification needs
                -- confidence >= threshold AND no review reasons, so a document
                -- read with near-perfect confidence is still routed to a human
                -- when a code is invalid or a member id is missing. Without
                -- these two the UI had only the confidence and told reviewers
                -- every high-confidence document had been auto-accepted,
                -- including the ones it was showing them BECAUSE the pipeline
                -- refused to.
                -- Keep this statement free of percent signs; see the
                -- method docstring for why.
                is_automated, review_reasons
            FROM {self.GOLD_SYNC}
            WHERE document_path = %s
        """
        try:
            results = self._execute_query(query, (file_path,))
            logger.info(
                "get_extraction_comparisons: got %d rows for file_path=%r",
                len(results),
                file_path,
            )
            if results:
                logger.debug(
                    "get_extraction_comparisons: first row document_path=%r label=%r",
                    results[0].get("document_path"),
                    results[0].get("label"),
                )
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(
                "get_extraction_comparisons: query failed for file_path=%r: %s",
                file_path,
                e,
                exc_info=True,
            )
            raise

    # =========================================================================
    # EXTRACTION REVIEW OPERATIONS
    # =========================================================================

    def upsert_extraction_review(
        self,
        document_id: str,
        reviewer_email: str,
        verdict: str,
        reasoning: Optional[str] = None,
        is_automated: bool = False,
        corrections: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        # One row per document (shared-review model): any reviewer's latest
        # verdict overwrites the prior one. `id` is preserved across
        # ON CONFLICT UPDATE so CDC emits an UPDATE on the same PK.
        # `corrections` is the JSONB map of inline edits the reviewer made
        # to extracted field values, keyed by parseIdentifiers id:<idx>.
        corrections_json = json.dumps(corrections) if corrections else None
        query = """
            INSERT INTO public.document_extraction_reviews
                (document_id, reviewer_email, verdict, reasoning,
                 is_automated, corrections)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (document_id) DO UPDATE
                SET reviewer_email = EXCLUDED.reviewer_email,
                    verdict        = EXCLUDED.verdict,
                    reasoning      = EXCLUDED.reasoning,
                    is_automated   = EXCLUDED.is_automated,
                    corrections    = EXCLUDED.corrections
            RETURNING id, document_id, reviewer_email,
                      verdict, reasoning, is_automated, corrections,
                      created_at, updated_at
        """
        results = self._execute_query(
            query,
            (
                document_id,
                reviewer_email,
                verdict,
                reasoning,
                is_automated,
                corrections_json,
            ),
        )
        return dict(results[0])

    def get_extraction_review(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Return the latest review on a document, regardless of reviewer.

        Shared-pool model: the most recent verdict wins. Used by the review
        panel to preload whatever the last reviewer decided.
        """
        query = """
            SELECT id, document_id, reviewer_email,
                   verdict, reasoning, is_automated, corrections,
                   created_at, updated_at
            FROM public.document_extraction_reviews
            WHERE document_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
        """
        results = self._execute_query(query, (document_id,))
        return dict(results[0]) if results else None

    # =========================================================================
    # REVIEWER NOTEPAD OPERATIONS
    # =========================================================================

    def get_document_notes(
        self, document_id: str, user_email: str
    ) -> Optional[Dict[str, Any]]:
        """Return the reviewer's private notepad for one document, or None.

        Notes are per (document, reviewer): the notepad is a personal scratch
        space, distinct from the shared, single-row review verdict.
        """
        query = """
            SELECT document_id, user_email, note_text, updated_at
            FROM public.document_notes
            WHERE document_id = %s AND user_email = %s
        """
        results = self._execute_query(query, (document_id, user_email))
        return dict(results[0]) if results else None

    def upsert_document_notes(
        self, document_id: str, user_email: str, note_text: str
    ) -> Dict[str, Any]:
        """Create or replace the reviewer's notepad text for one document.

        One row per (document_id, user_email). The client sends the full text
        (debounced autosave), so this is a straight replace, not an append.
        """
        query = """
            INSERT INTO public.document_notes
                (document_id, user_email, note_text)
            VALUES (%s, %s, %s)
            ON CONFLICT (document_id, user_email) DO UPDATE
                SET note_text  = EXCLUDED.note_text,
                    updated_at = NOW()
            RETURNING document_id, user_email, note_text, updated_at
        """
        results = self._execute_query(query, (document_id, user_email, note_text))
        return dict(results[0])

    # =========================================================================
    # REVIEW DRAFT OPERATIONS (unsubmitted autosave)
    #
    # One row per (document, reviewer) holding the in-progress review: verdict,
    # reasoning and the inline field corrections. Deliberately NOT a status
    # column on document_extraction_reviews -- that table feeds the change feed
    # into silver/gold, where silver_reviews.sql constrains the verdict
    # vocabulary that a half-filled draft violates. See migration
    # 20260930_000037.
    #
    # Deleted on submit, so a draft only ever represents work that has not been
    # committed to the shared review row.
    # =========================================================================

    def get_review_draft(
        self, document_id: str, user_email: str
    ) -> Optional[Dict[str, Any]]:
        """Return the reviewer's unsubmitted draft for one document, or None."""
        query = """
            SELECT document_id, user_email, verdict, reasoning,
                   corrections, updated_at
            FROM public.review_drafts
            WHERE document_id = %s AND user_email = %s
        """
        results = self._execute_query(query, (document_id, user_email))
        return dict(results[0]) if results else None

    def upsert_review_draft(
        self,
        document_id: str,
        user_email: str,
        verdict: Optional[str] = None,
        reasoning: Optional[str] = None,
        corrections: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create or replace the reviewer's draft for one document.

        A straight replace, not a merge: the client always sends the whole draft
        (debounced autosave of the full form), exactly like the notepad. Last
        write wins, which is proportionate -- the row is keyed per reviewer, so
        the only realistic conflict is one reviewer in two tabs.

        `corrections` defaults to an empty object rather than NULL so a caller
        never has to distinguish "no corrections" from "column unset".
        """
        corrections_json = json.dumps(corrections or {})
        query = """
            INSERT INTO public.review_drafts
                (document_id, user_email, verdict, reasoning, corrections)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (document_id, user_email) DO UPDATE
                SET verdict     = EXCLUDED.verdict,
                    reasoning   = EXCLUDED.reasoning,
                    corrections = EXCLUDED.corrections,
                    updated_at  = NOW()
            RETURNING document_id, user_email, verdict, reasoning,
                      corrections, updated_at
        """
        results = self._execute_query(
            query,
            (document_id, user_email, verdict, reasoning, corrections_json),
        )
        return dict(results[0])

    def list_review_draft_ids(self, user_email: str) -> list:
        """Document ids this reviewer has an unsubmitted draft for.

        Ids only, not the drafts themselves: this backs a badge in the document
        list, which needs to know THAT a draft exists, not what is in it. One
        small query beats N per-row fetches.
        """
        query = """
            SELECT document_id
            FROM public.review_drafts
            WHERE user_email = %s
        """
        results = self._execute_query(query, (user_email,))
        return [str(r["document_id"]) for r in results]

    def delete_review_draft(self, document_id: str, user_email: str) -> bool:
        """Drop the draft for one (document, reviewer). True if a row went.

        Called after a successful submit: once the verdict is on the shared
        review row, a lingering draft would take precedence over it on reload
        and resurrect the pre-submit state.
        """
        query = """
            DELETE FROM public.review_drafts
            WHERE document_id = %s AND user_email = %s
            RETURNING document_id
        """
        results = self._execute_query(query, (document_id, user_email))
        return bool(results)

    # =========================================================================
    # AGENT REVIEW PROPOSAL OPERATIONS
    #
    # Append-only: every proposal the agent made on a held document, and what
    # the reviewer did with it. Writes live here (the reviewer app owns the
    # review path and the audit identity); the agent app only reads.
    # =========================================================================

    def insert_review_proposal(
        self,
        document_id: str,
        review_reason: str,
        resolution: str,
        source: str,
        field_name: Optional[str] = None,
        correction_key: Optional[str] = None,
        observed_value: Optional[str] = None,
        proposed_value: Optional[str] = None,
        rationale: Optional[str] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        withheld: bool = False,
    ) -> Dict[str, Any]:
        """Record one proposal. Never updates an existing row.

        A ``not_resolvable`` proposal is the agent declining, so it carries no
        value — the table's ck_review_proposals_no_value_when_unresolvable
        constraint enforces that, and this drops any value a caller passed
        rather than letting the insert fail. That keeps a bad prompt from
        turning into a 500 on the reviewer's screen while still making it
        impossible to store a guessed member ID.
        """
        if resolution == "not_resolvable":
            proposed_value = None
        candidates_json = json.dumps(candidates) if candidates else None
        query = """
            INSERT INTO public.document_review_proposals
                (document_id, review_reason, resolution, source, field_name,
                 correction_key, observed_value, proposed_value, rationale,
                 candidates, model, withheld, disposition)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            RETURNING id, document_id, review_reason, resolution, source,
                      field_name, correction_key, observed_value,
                      proposed_value, rationale, candidates, model, withheld,
                      disposition, proposed_at
        """
        results = self._execute_query(
            query,
            (
                document_id,
                review_reason,
                resolution,
                source,
                field_name,
                correction_key,
                observed_value,
                proposed_value,
                rationale,
                candidates_json,
                model,
                withheld,
                # A refusal is already dispositioned: there is nothing for the
                # reviewer to accept or reject, so it is not left 'pending'
                # cluttering the queue.
                "declined" if resolution == "not_resolvable" else "pending",
            ),
        )
        return dict(results[0])

    def list_review_proposals(
        self,
        document_id: str,
        include_withheld: bool = False,
        pending_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Proposals on one document, newest first.

        ``include_withheld`` defaults to False so the reviewer-facing paths
        cannot accidentally show a control-slice proposal and invalidate the
        comparison it exists for. Analytics passes True.
        """
        conditions = ["document_id = %s"]
        params: List[Any] = [document_id]
        if not include_withheld:
            conditions.append("withheld = FALSE")
        if pending_only:
            conditions.append("disposition = 'pending'")
        query = f"""
            SELECT id, document_id, review_reason, resolution, source,
                   field_name, correction_key, observed_value, proposed_value,
                   rationale, candidates, model, withheld, disposition,
                   disposition_at, disposition_by, human_value, proposed_at
            FROM public.document_review_proposals
            WHERE {" AND ".join(conditions)}
            ORDER BY proposed_at DESC
        """
        return [dict(r) for r in (self._execute_query(query, tuple(params)) or [])]

    def set_proposal_disposition(
        self,
        proposal_id: str,
        disposition: str,
        disposition_by: str,
        human_value: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record what the reviewer did with a proposal.

        Only a still-pending proposal can be dispositioned, so a double-click
        or a replayed request cannot overwrite an 'accepted' with a 'rejected'
        (or restamp the timestamp the turnaround metric is built on). Returns
        None when nothing was pending, which callers surface as 409.
        """
        query = """
            UPDATE public.document_review_proposals
               SET disposition    = %s,
                   disposition_at = NOW(),
                   disposition_by = %s,
                   human_value    = %s
             WHERE id = %s AND disposition = 'pending'
            RETURNING id, document_id, review_reason, resolution,
                      correction_key, proposed_value, disposition,
                      disposition_at, disposition_by, human_value
        """
        results = self._execute_query(
            query, (disposition, disposition_by, human_value, proposal_id)
        )
        return dict(results[0]) if results else None

    def supersede_pending_proposals(
        self, document_id: str, correction_key: Optional[str]
    ) -> int:
        """Mark still-pending proposals for the same target as superseded.

        A second proposal for the same field should not leave two pending rows
        competing for one Approve button. Scoped to the correction_key when
        there is one; a keyless proposal (a verdict, or a refusal) supersedes
        only other keyless ones.
        """
        if correction_key:
            query = """
                UPDATE public.document_review_proposals
                   SET disposition = 'superseded', disposition_at = NOW()
                 WHERE document_id = %s AND correction_key = %s
                   AND disposition = 'pending'
            """
            params: tuple = (document_id, correction_key)
        else:
            query = """
                UPDATE public.document_review_proposals
                   SET disposition = 'superseded', disposition_at = NOW()
                 WHERE document_id = %s AND correction_key IS NULL
                   AND disposition = 'pending'
            """
            params = (document_id,)
        self._execute_query(query, params, fetch=False)
        return 0

    def reconcile_accepted_proposals(
        self, document_id: str, corrections: Optional[Dict[str, str]]
    ) -> int:
        """Reclassify accepted proposals the reviewer then changed.

        Approving a card writes the proposed value into the correction form, but
        the reviewer can still edit it before submitting. Counting that as a
        plain 'accepted' overstates agreement — the interesting signal is
        exactly that the human kept the agent's finding but not its answer.

        Called at review submit, when the final values are known. Compares each
        accepted proposal's value against what was actually submitted for its
        correction_key and flips the differing ones to 'modified', recording
        what the reviewer used. Not pending-gated (unlike
        set_proposal_disposition) because this is a deliberate
        accepted -> modified transition.
        """
        if not corrections:
            return 0
        rows = self._execute_query(
            """
            SELECT id, correction_key, proposed_value
            FROM public.document_review_proposals
            WHERE document_id = %s
              AND disposition = 'accepted'
              AND correction_key IS NOT NULL
            """,
            (document_id,),
        )
        changed = 0
        for row in rows or []:
            key = row.get("correction_key")
            if key not in corrections:
                continue
            final = corrections.get(key)
            if final == row.get("proposed_value"):
                continue
            self._execute_query(
                """
                UPDATE public.document_review_proposals
                   SET disposition = 'modified', human_value = %s
                 WHERE id = %s AND disposition = 'accepted'
                """,
                (final, row.get("id")),
                fetch=False,
            )
            changed += 1
        return changed

    def lookup_documents_for_triage(
        self, file_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Resolve the given gold document_paths to live Lakebase documents.

        Drives the triage job. Returns one row per path that has a live
        (not-deleted) document, carrying `has_proposal` so the caller can tell
        "already triaged" apart from "no Lakebase row at all" — a path missing
        from the result is the latter.

        Scoped to the paths the caller actually cares about, rather than the
        N newest documents: the previous version took a bare `limit` and the job
        passed `max_docs * 4`, so it returned a WINDOW of recent documents and
        intersected it with the held set by path. Once the corpus outgrew that
        window the two sets stopped overlapping, every held document fell through
        the not-found branch, and the run reported "skipped_already_triaged" for
        all of them and wrote zero proposals — silently, with a SUCCESS state.
        Measured on a 1,997-document dev corpus: max_docs=50 gave a 200-row
        window and triaged 0 of 50.
        """
        if not file_paths:
            return []
        placeholders = ",".join(["%s"] * len(file_paths))
        query = f"""
            SELECT d.id AS document_id, d.document_name, d.file_path,
                   EXISTS (
                       SELECT 1 FROM public.document_review_proposals p
                        WHERE p.document_id = d.id
                   ) AS has_proposal
            FROM public.medical_documents d
            WHERE d.deleted_at IS NULL
              AND d.file_path IN ({placeholders})
            ORDER BY d.created_at DESC
        """
        return [dict(r) for r in (self._execute_query(query, tuple(file_paths)) or [])]

    # =========================================================================
    # CONVERSATION OPERATIONS (Chat Agent Memory)
    # =========================================================================

    def insert_conversation_message(
        self,
        conversation_id: str,
        user_email: str,
        document_id: Optional[str],
        title: Optional[str],
        role: str,
        content: str,
        tool_calls: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        query = """
            INSERT INTO public.conversations (
                conversation_id, user_email, document_id, title,
                message_role, message_content, tool_calls, trace_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (
            conversation_id,
            user_email,
            document_id,
            title,
            role,
            content,
            tool_calls,
            trace_id,
        )
        result = self._execute_query(query, params)
        return str(result[0]["id"])

    def get_conversation_history(
        self, conversation_id: str, user_email: str
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT message_role, message_content, tool_calls, trace_id, created_at
            FROM public.conversations
            WHERE conversation_id = %s AND user_email = %s
            ORDER BY created_at ASC
        """
        results = self._execute_query(query, (conversation_id, user_email))
        return [dict(row) for row in results]

    def list_conversations(
        self, user_email: str, document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT
                conversation_id,
                MAX(title) AS title,
                MAX(document_id::text)::uuid AS document_id,
                COUNT(*) AS message_count,
                MAX(created_at) AS updated_at
            FROM public.conversations
            WHERE user_email = %s
        """
        params: list = [user_email]

        if document_id:
            query += " AND document_id = %s"
            params.append(document_id)

        query += " GROUP BY conversation_id ORDER BY updated_at DESC"
        results = self._execute_query(query, tuple(params))
        return [dict(row) for row in results]

    def delete_conversation(self, conversation_id: str, user_email: str) -> int:
        query = """
            DELETE FROM public.conversations
            WHERE conversation_id = %s AND user_email = %s
        """
        return self._execute_query(query, (conversation_id, user_email), fetch=False)

    def log_conversation_deletion(
        self,
        conversation_id: str,
        user_email: str,
        agent_cleanup_status: str,
        reviewer_rowcount: int = 0,
        error_detail: Optional[str] = None,
    ) -> None:
        """Insert an audit row for a cascade-delete attempt.

        Writes both success and failure paths so the table doubles as a HIPAA
        §164.312(b) audit trail and a retry surface (a 'failed' row means the
        agent-side cleanup errored and the reviewer row was intentionally
        preserved for retry).
        """
        query = """
            INSERT INTO public.conversation_deletions
              (conversation_id, user_email, agent_cleanup_status,
               reviewer_rowcount, error_detail)
            VALUES (%s, %s, %s, %s, %s)
        """
        self._execute_query(
            query,
            (
                conversation_id,
                user_email,
                agent_cleanup_status,
                reviewer_rowcount,
                error_detail,
            ),
            fetch=False,
        )
