"""
Postgres diagnostic SQL for the Admin page.

Each helper runs a defensive query and returns a structured result. Queries
that may fail under reduced privileges (pg_stat_replication, pg_stat_wal,
pg_stat_statements) wrap exceptions and return `{available: False,
reason: ...}` so the UI can render an inline hint instead of erroring.

All queries are read-only against pg_catalog / pg_stat_* views.
"""

import logging
from typing import Any, Dict, List, Optional

from services.lakehouse_db import LakeRCMDatabase

logger = logging.getLogger(__name__)


# Capture the most recent _safe_query failure so callers can surface the
# real reason in their {available: false, reason: ...} responses instead of
# generic "permission denied" messaging.
_LAST_QUERY_ERROR: Dict[str, str] = {}


def _safe_query(
    db: LakeRCMDatabase, sql: str, params: tuple = ()
) -> Optional[List[Dict[str, Any]]]:
    """Run a read-only query, return rows or None on error."""
    try:
        rows = db._execute_query(sql, params)
        return [dict(r) for r in rows] if rows else []
    except Exception as e:
        # Log at WARN once so admin queries that fail unexpectedly leave a
        # trace at default log level. Strip newlines from the SQL for the log.
        sql_oneliner = " ".join(sql.split())[:120]
        logger.warning("admin query failed: %s | sql: %s", e, sql_oneliner)
        _LAST_QUERY_ERROR["reason"] = f"{type(e).__name__}: {e}"[:200]
        return None


def _last_error_reason(default: str) -> str:
    return _LAST_QUERY_ERROR.pop("reason", default)


def _scalar(db: LakeRCMDatabase, sql: str) -> Optional[Any]:
    rows = _safe_query(db, sql)
    if not rows:
        return None
    first = rows[0]
    return next(iter(first.values()), None)


def get_pg_version(db: LakeRCMDatabase) -> Optional[str]:
    """Returns the Postgres server_version string (e.g. "17.4")."""
    return _scalar(db, "SHOW server_version")


# -----------------------------------------------------------------------------
# Sessions (pg_stat_activity) — top 10 longest-running
# -----------------------------------------------------------------------------

_SESSIONS_BASE_SQL = """
    SELECT
        pid,
        COALESCE(usename, '?') AS usename,
        COALESCE(application_name, '') AS application_name,
        COALESCE(state, '') AS state,
        COALESCE(wait_event_type, '') AS wait_event_type,
        COALESCE(wait_event, '') AS wait_event,
        query_start,
        state_change,
        EXTRACT(EPOCH FROM (NOW() - query_start)) AS duration_seconds,
        EXTRACT(EPOCH FROM (NOW() - state_change)) AS state_age_seconds,
        LEFT(COALESCE(query, ''), 240) AS query
    FROM pg_stat_activity
    WHERE pid <> pg_backend_pid()
      AND datname = current_database()
      AND state IS NOT NULL
"""

# "Interesting" = sessions actually doing something. Excludes plain `idle`
# pool connections, whose `query_start` is the time of the last query they
# ran (which gets misread as "running for X minutes").
#
# Note: avoid LIKE here. psycopg interprets `%` as a parameter placeholder
# when execute() receives a (possibly empty) params tuple, which made the
# previous `state LIKE 'idle in transaction%'` raise a parsing error.
_SESSIONS_INTERESTING_FILTER = (
    "  AND (state = 'active' "
    "    OR state = 'idle in transaction' "
    "    OR state = 'idle in transaction (aborted)' "
    "    OR wait_event_type = 'Lock')"
)
_SESSIONS_ORDER = " ORDER BY query_start NULLS LAST LIMIT 25"


def get_sessions(db: LakeRCMDatabase, interesting: bool = True) -> Dict[str, Any]:
    sql = (
        _SESSIONS_BASE_SQL
        + (_SESSIONS_INTERESTING_FILTER if interesting else "")
        + _SESSIONS_ORDER
    )
    rows = _safe_query(db, sql)
    if rows is None:
        return {
            "available": False,
            "reason": _last_error_reason("permission denied"),
            "sessions": [],
        }
    out = []
    for r in rows:
        out.append(
            {
                "pid": int(r.get("pid") or 0),
                "user": r.get("usename"),
                "app": r.get("application_name") or None,
                "state": r.get("state") or None,
                "wait": (
                    f"{r['wait_event_type']}:{r['wait_event']}"
                    if r.get("wait_event_type")
                    else None
                ),
                "wait_event_type": r.get("wait_event_type") or None,
                "query_start": (
                    r["query_start"].isoformat() if r.get("query_start") else None
                ),
                "state_change": (
                    r["state_change"].isoformat() if r.get("state_change") else None
                ),
                # `duration_seconds` is NOW() - query_start. For `idle` sessions
                # this is "time since the last query finished" — a stale value.
                # `state_age_seconds` is NOW() - state_change — time in the
                # current state — which is the right value to color rows by.
                "duration_seconds": (
                    float(r["duration_seconds"])
                    if r.get("duration_seconds") is not None
                    else None
                ),
                "state_age_seconds": (
                    float(r["state_age_seconds"])
                    if r.get("state_age_seconds") is not None
                    else None
                ),
                "query": r.get("query") or None,
            }
        )
    return {"available": True, "sessions": out}


# -----------------------------------------------------------------------------
# Slow queries (pg_stat_statements) — top 10 by total time
# -----------------------------------------------------------------------------

SLOW_QUERIES_SQL = """
    SELECT
        queryid::text AS queryid,
        calls,
        total_exec_time,
        mean_exec_time,
        rows AS row_count,
        LEFT(query, 240) AS query
    FROM pg_stat_statements
    ORDER BY total_exec_time DESC
    LIMIT 10
"""


def get_slow_queries(db: LakeRCMDatabase) -> Dict[str, Any]:
    rows = _safe_query(db, SLOW_QUERIES_SQL)
    if rows is None:
        return {
            "available": False,
            "reason": "pg_stat_statements not enabled or no permission",
            "queries": [],
        }
    out = []
    for r in rows:
        out.append(
            {
                "queryid": r.get("queryid"),
                "calls": int(r.get("calls") or 0),
                "total_exec_ms": float(r.get("total_exec_time") or 0.0),
                "mean_exec_ms": float(r.get("mean_exec_time") or 0.0),
                "rows": int(r.get("row_count") or 0),
                "query": r.get("query"),
            }
        )
    return {"available": True, "queries": out}


# -----------------------------------------------------------------------------
# Top tables by total relation size
# -----------------------------------------------------------------------------

TOP_TABLES_SQL = """
    SELECT
        schemaname,
        relname,
        n_live_tup,
        n_dead_tup,
        last_vacuum,
        last_autovacuum,
        pg_total_relation_size(
            quote_ident(schemaname) || '.' || quote_ident(relname)
        ) AS total_bytes
    FROM pg_stat_user_tables
    ORDER BY total_bytes DESC NULLS LAST
    LIMIT 10
"""


def get_top_tables(db: LakeRCMDatabase) -> Dict[str, Any]:
    rows = _safe_query(db, TOP_TABLES_SQL)
    if rows is None:
        return {"available": False, "reason": "permission denied", "tables": []}
    out = []
    for r in rows:
        live = int(r.get("n_live_tup") or 0)
        dead = int(r.get("n_dead_tup") or 0)
        out.append(
            {
                "schema": r.get("schemaname"),
                "table": r.get("relname"),
                "live_rows": live,
                "dead_rows": dead,
                "dead_ratio_pct": (
                    round(dead / (live + dead) * 100, 1) if (live + dead) > 0 else 0.0
                ),
                "total_bytes": int(r.get("total_bytes") or 0),
                "last_vacuum": (
                    r["last_vacuum"].isoformat() if r.get("last_vacuum") else None
                ),
                "last_autovacuum": (
                    r["last_autovacuum"].isoformat()
                    if r.get("last_autovacuum")
                    else None
                ),
            }
        )
    return {"available": True, "tables": out}


# -----------------------------------------------------------------------------
# Replication state
# -----------------------------------------------------------------------------

REPLICATION_SQL = """
    SELECT
        pid,
        COALESCE(application_name, '') AS application_name,
        COALESCE(state, '') AS state,
        COALESCE(sync_state, '') AS sync_state,
        EXTRACT(EPOCH FROM write_lag) AS write_lag_seconds,
        EXTRACT(EPOCH FROM flush_lag) AS flush_lag_seconds,
        EXTRACT(EPOCH FROM replay_lag) AS replay_lag_seconds,
        pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes
    FROM pg_stat_replication
"""


def get_replication(db: LakeRCMDatabase) -> Dict[str, Any]:
    rows = _safe_query(db, REPLICATION_SQL)
    if rows is None:
        return {"available": False, "reason": "permission denied", "replicas": []}
    out = []
    for r in rows:
        out.append(
            {
                "pid": int(r.get("pid") or 0),
                "app": r.get("application_name") or None,
                "state": r.get("state") or None,
                "sync_state": r.get("sync_state") or None,
                "write_lag_seconds": (
                    float(r["write_lag_seconds"])
                    if r.get("write_lag_seconds") is not None
                    else None
                ),
                "flush_lag_seconds": (
                    float(r["flush_lag_seconds"])
                    if r.get("flush_lag_seconds") is not None
                    else None
                ),
                "replay_lag_seconds": (
                    float(r["replay_lag_seconds"])
                    if r.get("replay_lag_seconds") is not None
                    else None
                ),
                "lag_bytes": (
                    int(r["lag_bytes"]) if r.get("lag_bytes") is not None else None
                ),
            }
        )
    return {"available": True, "replicas": out}


# -----------------------------------------------------------------------------
# WAL / bgwriter / locks (augment /metrics)
# -----------------------------------------------------------------------------


def get_wal_stats(db: LakeRCMDatabase) -> Optional[Dict[str, Any]]:
    rows = _safe_query(
        db,
        """
        SELECT wal_records, wal_fpi, wal_bytes, wal_buffers_full,
               wal_write, wal_sync, stats_reset
        FROM pg_stat_wal
        """,
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "wal_records": int(r.get("wal_records") or 0),
        "wal_fpi": int(r.get("wal_fpi") or 0),
        "wal_bytes": int(r.get("wal_bytes") or 0),
        "wal_buffers_full": int(r.get("wal_buffers_full") or 0),
        "wal_write": int(r.get("wal_write") or 0),
        "wal_sync": int(r.get("wal_sync") or 0),
        "stats_reset": r["stats_reset"].isoformat() if r.get("stats_reset") else None,
    }


def get_bgwriter_stats(db: LakeRCMDatabase) -> Optional[Dict[str, Any]]:
    """Background-writer + checkpointer stats.

    Postgres 17 split pg_stat_bgwriter: checkpoint counters moved to a new
    pg_stat_checkpointer view (`num_timed`, `num_requested`, `buffers_written`).
    Try the new view first, fall back to the legacy view.
    """
    chk = (
        _safe_query(
            db,
            """
        SELECT num_timed AS checkpoints_timed,
               num_requested AS checkpoints_req,
               buffers_written AS buffers_checkpoint,
               'pg_stat_checkpointer' AS source
        FROM pg_stat_checkpointer
        """,
        )
        or _safe_query(
            db,
            """
        SELECT checkpoints_timed, checkpoints_req,
               buffers_checkpoint,
               'pg_stat_bgwriter' AS source
        FROM pg_stat_bgwriter
        """,
        )
    )
    if not chk:
        return None
    c = chk[0]
    # buffers_clean / buffers_backend / buffers_alloc still live on
    # pg_stat_bgwriter on PG17 — best-effort, may be empty on older PGs.
    bg = (
        _safe_query(
            db,
            """
        SELECT buffers_clean, buffers_backend, buffers_alloc, maxwritten_clean
        FROM pg_stat_bgwriter
        """,
        )
        or [{}]
    )
    b = bg[0] if bg else {}
    return {
        "checkpoints_timed": int(c.get("checkpoints_timed") or 0),
        "checkpoints_req": int(c.get("checkpoints_req") or 0),
        "buffers_checkpoint": int(c.get("buffers_checkpoint") or 0),
        "buffers_clean": int(b.get("buffers_clean") or 0),
        "buffers_backend": int(b.get("buffers_backend") or 0),
        "buffers_alloc": int(b.get("buffers_alloc") or 0),
        "maxwritten_clean": int(b.get("maxwritten_clean") or 0),
        "source": c.get("source"),
    }


def get_longest_idle_in_tx(db: LakeRCMDatabase) -> Optional[float]:
    """Longest 'idle in transaction' duration, in seconds. None if none.

    Idle-in-transaction connections hold locks and block autovacuum, so the
    longest-running one is a real ops signal.
    """
    rows = _safe_query(
        db,
        """
        SELECT MAX(EXTRACT(EPOCH FROM (NOW() - state_change))) AS longest
        FROM pg_stat_activity
        WHERE state IN ('idle in transaction', 'idle in transaction (aborted)')
          AND datname = current_database()
        """,
    )
    if not rows:
        return None
    longest = rows[0].get("longest")
    return float(longest) if longest is not None else None


def get_on_lock_count(db: LakeRCMDatabase) -> Optional[int]:
    """Count of sessions currently waiting on a lock (real contention).

    Distinct from sessions waiting on Client:ClientRead (idle pool).
    """
    rows = _safe_query(
        db,
        """
        SELECT COUNT(*) AS n
        FROM pg_stat_activity
        WHERE wait_event_type = 'Lock'
          AND datname = current_database()
        """,
    )
    if not rows:
        return None
    return int(rows[0].get("n") or 0)


# -----------------------------------------------------------------------------
# Index health
# -----------------------------------------------------------------------------

INDEX_STATS_SQL = """
    SELECT
        s.schemaname,
        s.relname AS table_name,
        s.indexrelname AS index_name,
        s.idx_scan,
        s.idx_tup_read,
        s.idx_tup_fetch,
        pg_relation_size(s.indexrelid) AS bytes,
        ix.indisunique AS is_unique,
        ix.indisprimary AS is_primary
    FROM pg_stat_user_indexes s
    JOIN pg_index ix ON ix.indexrelid = s.indexrelid
    ORDER BY pg_relation_size(s.indexrelid) DESC NULLS LAST
    LIMIT 15
"""


def get_index_stats(db: LakeRCMDatabase) -> Dict[str, Any]:
    """Top user indexes by size with usage stats.

    `idx_scan = 0` flags unused indexes (dead weight on the write path).
    """
    rows = _safe_query(db, INDEX_STATS_SQL)
    if rows is None:
        return {"available": False, "reason": "permission denied", "indexes": []}
    out = []
    for r in rows:
        out.append(
            {
                "schema": r.get("schemaname"),
                "table": r.get("table_name"),
                "index": r.get("index_name"),
                "idx_scan": int(r.get("idx_scan") or 0),
                "idx_tup_read": int(r.get("idx_tup_read") or 0),
                "idx_tup_fetch": int(r.get("idx_tup_fetch") or 0),
                "bytes": int(r.get("bytes") or 0),
                "is_unique": bool(r.get("is_unique")),
                "is_primary": bool(r.get("is_primary")),
            }
        )
    return {"available": True, "indexes": out}


def get_lock_summary(db: LakeRCMDatabase) -> Optional[Dict[str, Any]]:
    rows = _safe_query(
        db,
        """
        SELECT mode, COUNT(*) AS waiting
        FROM pg_locks
        WHERE NOT granted
        GROUP BY mode
        ORDER BY waiting DESC
        """,
    )
    if rows is None:
        return None
    return {
        "waiting_total": sum(int(r.get("waiting") or 0) for r in rows),
        "by_mode": [
            {"mode": r.get("mode"), "waiting": int(r.get("waiting") or 0)} for r in rows
        ],
    }
