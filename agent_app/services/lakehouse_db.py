"""
LakeRCM Agent — Lakebase client (single long-lived pool).

The pool is constructed once at startup and reused for the lifetime of the
process. Lakebase OAuth tokens (60-min TTL) are rotated *per physical
connection* via psycopg_pool 3.3's callable ``conninfo`` parameter, and
``max_lifetime`` (50 min) guarantees every physical connection is retired
before its token expires. The pool object itself never closes, which keeps
the LangGraph singletons (PostgresStore, PostgresSaver) that capture the
pool reference at init time working indefinitely.

Previous implementation closed-and-recreated the pool every 45 min to rotate
tokens, which silently invalidated the pool references held by those
singletons — producing ``PoolClosed`` errors on ``store.aput()`` and
``saver.delete_thread()`` after ~45 min of uptime.
"""

import os
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, AsyncConnectionPool
from databricks.sdk import WorkspaceClient

from services.tracing import trace_if_active

from config import settings

logger = logging.getLogger(__name__)

# Physical connections are retired before the 60-min Lakebase token expiry.
CONNECTION_MAX_LIFETIME_SECONDS = 50 * 60
POOL_WAIT_WARN_MS = 25


def _query_operation(query: str) -> str:
    """Return only the leading SQL verb; never include SQL text in telemetry."""
    stripped = (query or "").lstrip()
    if not stripped:
        return "UNKNOWN"
    return stripped.split(None, 1)[0].upper()


def _set_span_attr(span, key: str, value) -> None:
    if span is None or value is None:
        return
    try:
        span.set_attribute(key, value)
    except Exception:
        pass


def _set_pool_stats(span, pool: Any, prefix: str) -> None:
    if span is None or pool is None:
        return
    try:
        stats = pool.get_stats()
    except Exception:
        return
    for key, value in (stats or {}).items():
        if isinstance(value, (int, float, bool)):
            _set_span_attr(span, f"{prefix}.{key}", value)


@contextmanager
def lakebase_operation_span(name: str, span_type: str = "RETRIEVER", **attrs):
    """Create a child-only MLflow span for Lakebase operations without SQL text."""
    started = time.monotonic()
    with trace_if_active(name, span_type=span_type) as span:
        for key, value in attrs.items():
            _set_span_attr(span, key, value)
        try:
            yield span
            _set_span_attr(span, "lakebase.operation.status", "ok")
        except Exception as e:
            _set_span_attr(span, "lakebase.operation.status", "error")
            _set_span_attr(span, "error.type", type(e).__name__)
            raise
        finally:
            _set_span_attr(
                span,
                "lakebase.operation.duration_ms",
                int((time.monotonic() - started) * 1000),
            )


class AgentLakehouseDB:
    """Read-only Lakebase client for the LakeRCM chat agent."""

    def __init__(self, workspace_client: Optional[WorkspaceClient] = None):
        self.workspace_client = workspace_client or WorkspaceClient()
        self._connection_pool: Optional[ConnectionPool] = None
        # Parallel async pool for LangGraph's AsyncPostgresSaver
        # (AG-UI invokes the graph via async methods; the sync
        # PostgresSaver does not implement aget_tuple). Same conninfo
        # callable so credential rotation behaves identically.
        self._async_connection_pool: Optional[AsyncConnectionPool] = None

        logger.info("Initializing LakeRCM agent Lakebase connection...")
        try:
            self._create_connection_pool()
            logger.info(
                "Connected to Lakebase: %s:%s/%s",
                os.getenv("PGHOST", "?"),
                os.getenv("PGPORT", "5432"),
                os.getenv("PGDATABASE", "?"),
            )
            # Schema/privilege DDL was previously re-asserted here on every
            # pod boot via `_grant_synced_schema_access()`. That logic is
            # now declared in alembic migrations 20260224_000006 and
            # 20260420_000013 and applied at deploy/reviewer-app-boot — see
            # reviewer_app/migrations/versions/. The runtime duplication
            # polluted the MLflow experiment with GRANT/ALTER root traces
            # and is an anti-pattern (12-factor: apps should not mutate
            # their own data layer at boot).
        except Exception as e:
            logger.error("Failed to connect to Lakebase: %s", e, exc_info=True)
            logger.warning("Lazy initialization will be attempted on first query.")
            self._connection_pool = None

    def _fresh_conninfo(self) -> str:
        """Mint a fresh Lakebase conninfo string. Called per new physical connection.

        psycopg_pool 3.3 invokes this callable on every new physical
        connection, so we return a conn string with a fresh OAuth token each
        time. Combined with ``max_lifetime``, every connection is born with a
        valid token and retired before that token expires.
        """
        try:
            credential = self.workspace_client.postgres.generate_database_credential(
                endpoint=settings.endpoint_name
            )
            token = credential.token
        except Exception as e:
            logger.warning(
                "Autoscaling credential failed, falling back to OAuth token: %s", e
            )
            token = self.workspace_client.config.oauth_token().access_token

        # PGUSER falls back to DATABRICKS_CLIENT_ID (auto-injected by the
        # Apps runtime) — the running SP's client_id IS the Postgres role
        # name for Lakebase OAuth.
        pg_user = os.getenv("PGUSER") or os.getenv("DATABRICKS_CLIENT_ID", "")
        return make_conninfo(
            dbname=os.getenv("PGDATABASE", "databricks_postgres"),
            user=pg_user,
            password=token,
            host=os.getenv("PGHOST", ""),
            port=os.getenv("PGPORT", "5432"),
            sslmode=os.getenv("PGSSLMODE", "require"),
            application_name=os.getenv("PGAPPNAME", "lakercm-agent"),
        )

    def _create_connection_pool(self):
        """Build the long-lived pool. Called exactly once per process under normal operation."""
        if self._connection_pool is not None:
            return

        def _configure(conn):
            # autocommit=True is required for LangGraph's PostgresSaver.setup()
            # (DDL runs outside a transaction). The shared pool is used by
            # both the tool queries and the checkpointer/store.
            conn.autocommit = True

        self._connection_pool = ConnectionPool(
            conninfo=self._fresh_conninfo,
            min_size=1,
            max_size=8,
            max_lifetime=CONNECTION_MAX_LIFETIME_SECONDS,
            max_idle=600,
            # Validate every connection on getconn() so an idle conn the
            # server killed (Lakebase autoscale / version bump / admin
            # shutdown) is evicted before it reaches the caller. Without
            # this, the bad conn raises AdminShutdown inside
            # PostgresSaver.get_tuple() and we lose a chat turn.
            check=ConnectionPool.check_connection,
            configure=_configure,
            open=True,
        )

    def _get_connection(self):
        if not self._connection_pool:
            self._create_connection_pool()
        if not self._connection_pool:
            raise RuntimeError("Connection pool not initialized")
        return self._connection_pool.connection()

    def execute_query(
        self, query: str, params: Optional[tuple] = None, fetch: bool = True
    ) -> List[Dict[str, Any]]:
        if not self._connection_pool:
            self._create_connection_pool()
        if not self._connection_pool:
            raise RuntimeError(
                "Database connection not initialized. "
                "Ensure Lakebase endpoint is bound to this app's SP."
            )

        # Open a child span only when an agent_turn / agent_turn_agui is
        # already in flight. Boot-time probes / lazy initializers without a
        # parent yield None and produce no MLflow output — see
        # services/tracing.py for the rationale.
        with lakebase_operation_span(
            "lakebase.execute_query",
            span_type="RETRIEVER",
            **{
                "db.system": "postgresql",
                "db.name": os.getenv("PGDATABASE", "databricks_postgres"),
                "db.operation.name": _query_operation(query),
                "db.params_count": len(params) if params else 0,
            },
        ) as span:
            if span is not None:
                _set_pool_stats(span, self._connection_pool, "db.client.pool.before")

            return self._execute_query_inner(query, params, fetch, span=span)

    def _execute_query_inner(
        self, query: str, params: Optional[tuple], fetch: bool, span=None
    ) -> List[Dict[str, Any]]:
        query_operation = _query_operation(query)
        t0 = time.monotonic()
        retry_attempted = False
        while True:
            try:
                wait_started = time.monotonic()
                with self._get_connection() as conn:
                    wait_ms = int((time.monotonic() - wait_started) * 1000)
                    _set_span_attr(span, "db.client.connection.wait_ms", wait_ms)
                    _set_span_attr(
                        span,
                        "db.client.connection.waited",
                        wait_ms >= POOL_WAIT_WARN_MS,
                    )
                    with conn.cursor(row_factory=dict_row) as cursor:
                        cursor.execute(query, params or ())
                        if fetch:
                            rows = [dict(r) for r in cursor.fetchall()]
                            _set_span_attr(span, "db.row_count", len(rows))
                            _set_pool_stats(
                                span, self._connection_pool, "db.client.pool.after"
                            )
                            elapsed_ms = int((time.monotonic() - t0) * 1000)
                            _set_span_attr(span, "db.query.duration_ms", elapsed_ms)
                            _set_span_attr(span, "db.rows_affected", len(rows))
                            logger.info(
                                "lakebase_query ok latency_ms=%d rows=%d op=%s",
                                elapsed_ms,
                                len(rows),
                                query_operation,
                            )
                            return rows
                        conn.commit()
                        elapsed_ms = int((time.monotonic() - t0) * 1000)
                        rowcount = cursor.rowcount if cursor.rowcount is not None else 0
                        _set_pool_stats(
                            span, self._connection_pool, "db.client.pool.after"
                        )
                        _set_span_attr(span, "db.query.duration_ms", elapsed_ms)
                        _set_span_attr(span, "db.rows_affected", rowcount)
                        logger.info(
                            "lakebase_query ok latency_ms=%d rows=0 op=%s",
                            elapsed_ms,
                            query_operation,
                        )
                        return []
            except psycopg.errors.AdminShutdown as e:
                if not retry_attempted:
                    logger.warning(
                        "lakebase_query admin_shutdown retrying_once op=%s",
                        query_operation,
                    )
                    retry_attempted = True
                    continue
                self._log_pool_stats_on_error("admin_shutdown_after_retry", e)
                raise
            except psycopg.OperationalError as e:
                error_msg = str(e).lower()
                is_auth_error = any(
                    kw in error_msg
                    for kw in ["authentication", "password", "credentials", "token"]
                )
                if is_auth_error and not retry_attempted:
                    # A stale connection with an expired token slipped through
                    # (race with max_lifetime, or token revoked mid-flight).
                    # Retry once — the pool will mint a fresh connection with
                    # a fresh token on the next checkout.
                    logger.warning("Auth error, retrying with fresh connection: %s", e)
                    retry_attempted = True
                    continue
                # Connection-closed errors (server killed conn between
                # check_connection and execute) — retry once.
                is_conn_closed = any(
                    kw in error_msg
                    for kw in [
                        "connection is bad",
                        "server closed the connection",
                        "consuming input failed",
                        "ssl connection has been closed",
                    ]
                )
                if is_conn_closed and not retry_attempted:
                    logger.warning(
                        "lakebase_query connection_closed retrying_once op=%s exc=%s",
                        query_operation,
                        str(e)[:200],
                    )
                    retry_attempted = True
                    continue
                self._log_pool_stats_on_error("operational_error", e)
                logger.error(
                    "lakebase_query failed latency_ms=%d op=%s exc=%s",
                    int((time.monotonic() - t0) * 1000),
                    query_operation,
                    type(e).__name__,
                    exc_info=True,
                )
                raise
            except Exception as e:
                self._log_pool_stats_on_error("query_failed", e)
                logger.error(
                    "lakebase_query failed latency_ms=%d op=%s exc=%s",
                    int((time.monotonic() - t0) * 1000),
                    query_operation,
                    type(e).__name__,
                    exc_info=True,
                )
                raise

    def _log_pool_stats_on_error(self, label: str, exc: BaseException) -> None:
        try:
            stats = self._connection_pool.get_stats() if self._connection_pool else {}
        except Exception:
            stats = {}
        logger.warning(
            "pool_stats label=%s exc=%s pool=%s",
            label,
            type(exc).__name__,
            stats,
        )

    def gold_sync_available(self) -> bool:
        """Check whether the SDP-synced gold table exists in Lakebase.

        Returns False when the pipeline hasn't materialized the sync table yet —
        tool queries fall back to a friendly "not ready" message instead of
        bubbling a raw postgres UndefinedTable error to the LLM.
        """
        try:
            rows = self.execute_query(
                f"SELECT to_regclass('{settings.schema_name}.gold_extraction_labels_sync') AS oid"
            )
            return bool(rows and rows[0].get("oid"))
        except Exception as e:
            logger.warning("gold_sync_available check failed: %s", e)
            return False

    def gold_sync_has_column(self, column: str) -> bool:
        """Whether the Lakebase copy of gold_extraction_labels has `column`.

        A column the pipeline adds reaches the synced table only after the
        pipeline's next update and the sync after it. Tools select such a column
        once it has arrived, rather than fail every query until then.
        """
        try:
            rows = self.execute_query(
                """
                SELECT 1 AS present
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = 'gold_extraction_labels_sync'
                  AND column_name = %s
                """,
                (settings.schema_name, column),
            )
            return bool(rows)
        except Exception as e:
            logger.warning("gold_sync_has_column(%s) check failed: %s", column, e)
            return False

    def get_pool(self) -> ConnectionPool:
        """Return the long-lived psycopg_pool for reuse by checkpointer/store.

        LangGraph singletons capture this reference once and rely on it
        staying open for the life of the process. Token rotation happens
        per physical connection inside the pool, not by rebuilding the pool.
        """
        if not self._connection_pool:
            self._create_connection_pool()
        if not self._connection_pool:
            raise RuntimeError("Connection pool not initialized")
        return self._connection_pool

    def get_async_pool(self) -> AsyncConnectionPool:
        """Return the long-lived async psycopg_pool for AsyncPostgresSaver.

        Created lazily on first call. The pool object is constructed with
        `open=False` so we can safely build it from sync code; the caller
        (async checkpointer init) must `await pool.open()` once before
        first use. After that, the pool is shared across all async
        consumers in the process.
        """
        if self._async_connection_pool is None:

            async def _configure(conn):
                await conn.set_autocommit(True)

            self._async_connection_pool = AsyncConnectionPool(
                conninfo=self._fresh_conninfo,
                min_size=1,
                max_size=4,
                max_lifetime=CONNECTION_MAX_LIFETIME_SECONDS,
                max_idle=600,
                check=AsyncConnectionPool.check_connection,
                configure=_configure,
                open=False,
            )
        return self._async_connection_pool

    def close(self):
        if self._connection_pool:
            try:
                self._connection_pool.close()
                self._connection_pool = None
            except Exception as e:
                logger.error("Error closing connection pool: %s", e)


_db_singleton: Optional[AgentLakehouseDB] = None


def get_db() -> AgentLakehouseDB:
    """Return the process-wide Lakebase client (lazy-init)."""
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = AgentLakehouseDB()
    return _db_singleton
