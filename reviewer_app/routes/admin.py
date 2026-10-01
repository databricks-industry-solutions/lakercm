"""
LakeRCM Admin Diagnostics Routes

All endpoints under /api/admin/* require workspace `admins` group membership,
verified server-side via require_admin (SCIM lookup, 5min TTL cache).

Surfaces Lakebase Autoscaling endpoint state, live Postgres metrics, recent
state transitions, active sessions, top tables, replication, and app runtime
telemetry. Each handler runs its own try/except so a single broken query
doesn't take down the page.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, Depends, HTTPException, Query

import dependencies
from config import settings
from dependencies import get_lakercm_db, get_workspace_client
from services.admin_auth import require_admin
from services.lakehouse_db import LakeRCMDatabase
from services import admin_queries

logger = logging.getLogger(__name__)

router = APIRouter()

# Captured at import time — used by /api/admin/runtime to compute uptime.
_PROCESS_START_TS = time.time()


@router.get("/lakebase/instance")
async def get_lakebase_instance(
    _admin_email: str = Depends(require_admin),
    workspace: WorkspaceClient = Depends(get_workspace_client),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Lakebase Autoscaling endpoint state, capacity, and timestamps.

    Reads endpoint_type / disabled / suspend_timeout from `status` because
    the GET /api/2.0/postgres/endpoints/{name} response populates `status`
    but leaves `spec` null. (Earlier code read these from `spec`, which is
    why "Type" rendered as "—".)
    """
    try:
        endpoint = workspace.postgres.get_endpoint(settings.endpoint_name)
    except Exception as e:
        logger.error("get_endpoint failed for %s: %s", settings.endpoint_name, e)
        raise HTTPException(
            status_code=502, detail=f"Lakebase endpoint lookup failed: {e}"
        )

    status = endpoint.status

    # pg_version is best-effort — runs against the same Postgres pool.
    pg_version = None
    try:
        pg_version = admin_queries.get_pg_version(db)
    except Exception as e:
        logger.debug("pg_version probe failed: %s", e)

    return {
        "name": endpoint.name,
        "parent": endpoint.parent,
        "uid": getattr(endpoint, "uid", None),
        "endpoint_type": (
            status.endpoint_type.value
            if status and getattr(status, "endpoint_type", None)
            else None
        ),
        "current_state": (
            status.current_state.value if status and status.current_state else None
        ),
        "pending_state": (
            status.pending_state.value if status and status.pending_state else None
        ),
        "min_cu": status.autoscaling_limit_min_cu if status else None,
        "max_cu": status.autoscaling_limit_max_cu if status else None,
        "disabled": status.disabled if status else None,
        "suspend_timeout_seconds": _duration_to_seconds(
            getattr(status, "suspend_timeout_duration", None) if status else None
        ),
        "host": status.hosts.host if status and status.hosts else None,
        "region": (
            _parse_region_from_host(status.hosts.host)
            if status and status.hosts
            else None
        ),
        "pg_version": pg_version,
        "pg_settings": (
            dict(status.settings.pg_settings)
            if status and status.settings and status.settings.pg_settings
            else None
        ),
        "create_time": _ts_iso(getattr(endpoint, "create_time", None)),
        "update_time": _ts_iso(getattr(endpoint, "update_time", None)),
    }


@router.get("/lakebase/metrics")
async def get_lakebase_metrics(
    _admin_email: str = Depends(require_admin),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Live Postgres metrics from pg_stat_database / pg_stat_activity.

    All values are cumulative since stats_reset (or instance start). The
    frontend computes per-second rates from successive samples.

    WAL / bgwriter / lock-summary are best-effort — included if the SP has
    pg_monitor or equivalent, omitted otherwise.
    """
    try:
        stat_db_rows = db._execute_query("""
            SELECT
                xact_commit,
                xact_rollback,
                blks_read,
                blks_hit,
                tup_returned,
                tup_fetched,
                tup_inserted,
                tup_updated,
                tup_deleted,
                deadlocks,
                temp_files,
                temp_bytes,
                pg_database_size(datname) AS db_size_bytes,
                stats_reset
            FROM pg_stat_database
            WHERE datname = current_database()
            """)
        activity_rows = db._execute_query("""
            SELECT
                COUNT(*) FILTER (WHERE state = 'active') AS active_connections,
                COUNT(*) FILTER (WHERE state = 'idle') AS idle_connections,
                COUNT(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_tx,
                COUNT(*) AS total_connections,
                COUNT(*) FILTER (WHERE wait_event IS NOT NULL) AS waiting
            FROM pg_stat_activity
            WHERE datname = current_database()
            """)
        max_conn_rows = db._execute_query("SHOW max_connections")
    except Exception as e:
        logger.error("metrics query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Postgres metrics failed: {e}")

    s = dict(stat_db_rows[0]) if stat_db_rows else {}
    a = dict(activity_rows[0]) if activity_rows else {}

    blks_hit = int(s.get("blks_hit") or 0)
    blks_read = int(s.get("blks_read") or 0)
    cache_hit_ratio = (
        (blks_hit / (blks_hit + blks_read) * 100)
        if (blks_hit + blks_read) > 0
        else None
    )

    max_conn = None
    if max_conn_rows:
        try:
            max_conn = int(list(max_conn_rows[0].values())[0])
        except Exception:
            pass

    # Best-effort augmentations.
    wal = None
    bgwriter = None
    locks = None
    longest_idle_in_tx_seconds = None
    try:
        wal = admin_queries.get_wal_stats(db)
    except Exception:
        pass
    try:
        bgwriter = admin_queries.get_bgwriter_stats(db)
    except Exception:
        pass
    try:
        locks = admin_queries.get_lock_summary(db)
    except Exception:
        pass
    try:
        longest_idle_in_tx_seconds = admin_queries.get_longest_idle_in_tx(db)
    except Exception:
        pass

    return {
        "sampled_at": datetime.utcnow().isoformat() + "Z",
        "transactions": {
            "commit": int(s.get("xact_commit") or 0),
            "rollback": int(s.get("xact_rollback") or 0),
        },
        "rows": {
            "inserted": int(s.get("tup_inserted") or 0),
            "updated": int(s.get("tup_updated") or 0),
            "deleted": int(s.get("tup_deleted") or 0),
            "returned": int(s.get("tup_returned") or 0),
            "fetched": int(s.get("tup_fetched") or 0),
        },
        "cache": {
            "blks_hit": blks_hit,
            "blks_read": blks_read,
            "hit_ratio_pct": (
                round(cache_hit_ratio, 2) if cache_hit_ratio is not None else None
            ),
        },
        "deadlocks": int(s.get("deadlocks") or 0),
        "temp_files": int(s.get("temp_files") or 0),
        "temp_bytes": int(s.get("temp_bytes") or 0),
        "db_size_bytes": int(s.get("db_size_bytes") or 0),
        "connections": {
            "active": int(a.get("active_connections") or 0),
            "idle": int(a.get("idle_connections") or 0),
            "idle_in_transaction": int(a.get("idle_in_tx") or 0),
            # Number of sessions actually blocked on a lock — distinct from
            # idle pool connections waiting on Client:ClientRead. Was named
            # `waiting` previously and counted both, which was misleading.
            "on_lock": int(a.get("on_lock") or 0),
            "total": int(a.get("total_connections") or 0),
            "max": max_conn,
            "longest_idle_in_tx_seconds": longest_idle_in_tx_seconds,
        },
        "wal": wal,
        "bgwriter": bgwriter,
        "locks": locks,
        "stats_reset": _ts_iso(s.get("stats_reset")),
    }


@router.get("/lakebase/sessions")
async def get_lakebase_sessions(
    interesting: bool = Query(
        True,
        description="When true (default), filter out plain `idle` pool "
        "connections — keep only sessions actually doing something "
        "(active / idle in transaction / waiting on lock). Set false to "
        "see the full pool too.",
    ),
    _admin_email: str = Depends(require_admin),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Sessions from pg_stat_activity. Defaults to operationally-interesting
    sessions only — idle pool connections are filtered out (idle pool
    connections report a stale `query_start` that misleadingly looks like
    'running for X minutes')."""
    return admin_queries.get_sessions(db, interesting=interesting)


@router.get("/lakebase/slow_queries")
async def get_lakebase_slow_queries(
    _admin_email: str = Depends(require_admin),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Top 10 queries by total exec time from pg_stat_statements.

    Returns {available: false} if the extension isn't enabled.
    """
    return admin_queries.get_slow_queries(db)


@router.get("/lakebase/top_tables")
async def get_lakebase_top_tables(
    _admin_email: str = Depends(require_admin),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Top 10 user tables by total relation size, with vacuum hygiene."""
    return admin_queries.get_top_tables(db)


@router.get("/lakebase/replication")
async def get_lakebase_replication(
    _admin_email: str = Depends(require_admin),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Replication state from pg_stat_replication. Empty list = no replicas."""
    return admin_queries.get_replication(db)


@router.get("/lakebase/indexes")
async def get_lakebase_indexes(
    _admin_email: str = Depends(require_admin),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Top user indexes by size with usage stats. `idx_scan == 0` flags
    unused indexes (dead weight on the write path)."""
    return admin_queries.get_index_stats(db)


@router.get("/lakebase/events")
async def get_lakebase_events(
    hours: int = Query(24, ge=1, le=720),
    _admin_email: str = Depends(require_admin),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Recent Lakebase endpoint state transitions captured by the
    background lakebase_monitor task. Window is 1-720 hours (default 24).
    """
    try:
        rows = db._execute_query(
            """
            SELECT id, ts, endpoint_name, prev_state, new_state,
                   cold_start_seconds, metadata
            FROM public.lakebase_events
            WHERE ts > NOW() - (%s || ' hours')::interval
              AND endpoint_name = %s
            ORDER BY ts DESC
            """,
            (str(hours), settings.endpoint_name),
        )
    except Exception as e:
        logger.error("events query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Events query failed: {e}")

    events: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        events.append(
            {
                "id": str(d["id"]),
                "ts": _ts_iso(d.get("ts")),
                "endpoint_name": d.get("endpoint_name"),
                "prev_state": d.get("prev_state"),
                "new_state": d.get("new_state"),
                "cold_start_seconds": (
                    float(d["cold_start_seconds"])
                    if d.get("cold_start_seconds") is not None
                    else None
                ),
                "metadata": d.get("metadata"),
            }
        )

    cold_starts = [
        e["cold_start_seconds"] for e in events if e["cold_start_seconds"] is not None
    ]
    summary = {
        "count": len(events),
        "window_hours": hours,
        "cold_start_count": len(cold_starts),
        "cold_start_max_seconds": max(cold_starts) if cold_starts else None,
        "cold_start_avg_seconds": (
            round(sum(cold_starts) / len(cold_starts), 2) if cold_starts else None
        ),
    }
    return {"summary": summary, "events": events}


@router.get("/runtime")
async def get_runtime(
    _admin_email: str = Depends(require_admin),
) -> Dict[str, Any]:
    """App-process telemetry: uptime, memory, monitor task health, alembic head."""
    proc = psutil.Process(os.getpid())
    try:
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        threads = proc.num_threads()
        cpu_pct = proc.cpu_percent(interval=None)
    except Exception:
        rss_mb = None
        threads = None
        cpu_pct = None

    monitor_status = _monitor_status()
    alembic_head = _alembic_head()

    return {
        "uptime_seconds": time.time() - _PROCESS_START_TS,
        "rss_mb": round(rss_mb, 1) if rss_mb is not None else None,
        "threads": threads,
        "cpu_percent": cpu_pct,
        "pid": os.getpid(),
        "monitor": monitor_status,
        "alembic_head": alembic_head,
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/health")
async def get_admin_health(
    _admin_email: str = Depends(require_admin),
    workspace: WorkspaceClient = Depends(get_workspace_client),
    db: LakeRCMDatabase = Depends(get_lakercm_db),
) -> Dict[str, Any]:
    """Per-component health: workspace API, Lakebase Postgres, app process,
    background monitor, and alembic head."""
    components: List[Dict[str, Any]] = []

    t0 = time.monotonic()
    try:
        workspace.current_user.me()
        components.append(_ok_component("workspace_api", t0))
    except Exception as e:
        components.append(_fail_component("workspace_api", t0, e))

    t0 = time.monotonic()
    try:
        ok = db.health_check()
        components.append(
            {
                "name": "lakebase_postgres",
                "ok": ok,
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                "detail": None if ok else "health_check returned False",
            }
        )
    except Exception as e:
        components.append(_fail_component("lakebase_postgres", t0, e))

    monitor = _monitor_status()
    components.append(
        {
            "name": "background_monitor",
            "ok": monitor["healthy"],
            "latency_ms": 0.0,
            "detail": monitor["detail"],
        }
    )

    alembic = _alembic_head()
    components.append(
        {
            "name": "alembic_head",
            "ok": bool(alembic.get("revision")),
            "latency_ms": 0.0,
            "detail": alembic.get("revision") or alembic.get("error"),
        }
    )

    components.append(
        {
            "name": "app_process",
            "ok": True,
            "latency_ms": 0.0,
            "detail": (
                f"uptime {int(time.time() - _PROCESS_START_TS)}s, "
                f"endpoint={settings.endpoint_name}"
            ),
        }
    )

    overall = (
        "healthy"
        if all(c["ok"] for c in components)
        else ("degraded" if any(c["ok"] for c in components) else "unhealthy")
    )
    return {
        "status": overall,
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "components": components,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ok_component(name: str, t0: float) -> Dict[str, Any]:
    return {
        "name": name,
        "ok": True,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        "detail": None,
    }


def _fail_component(name: str, t0: float, e: Exception) -> Dict[str, Any]:
    return {
        "name": name,
        "ok": False,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        "detail": str(e)[:200],
    }


def _monitor_status() -> Dict[str, Any]:
    """Reach into the lakebase_monitor singleton for last poll info."""
    monitor = getattr(dependencies, "lakebase_monitor", None)
    if monitor is None:
        return {"healthy": False, "detail": "monitor not initialized"}
    task = getattr(monitor, "_task", None)
    if task is None or task.done():
        return {"healthy": False, "detail": "monitor task not running"}
    last_state = getattr(monitor, "_last_state", None)
    return {
        "healthy": True,
        "endpoint_name": getattr(monitor, "endpoint_name", None),
        "last_state": last_state,
        "poll_interval_seconds": getattr(monitor, "poll_interval_seconds", None),
        "detail": f"running, last_state={last_state}",
    }


def _alembic_head() -> Dict[str, Any]:
    """Read the current alembic revision from public.alembic_version."""
    try:
        db = dependencies.lakercm_db
        if db is None:
            return {"revision": None, "error": "db not initialized"}
        rows = db._execute_query("SELECT version_num FROM public.alembic_version")
        rev = rows[0]["version_num"] if rows else None
        return {"revision": rev}
    except Exception as e:
        return {"revision": None, "error": str(e)[:200]}


def _duration_to_seconds(value: Any) -> Optional[float]:
    """Convert a google.protobuf.Duration shape to a float seconds value.

    The SDK exposes Duration as either a string ("60s") or an object with
    .seconds / .nanos. Be defensive about both.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().rstrip("s")
        try:
            return float(s)
        except ValueError:
            return None
    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", 0) or 0
    if seconds is None:
        return None
    try:
        return float(seconds) + float(nanos) / 1e9
    except (TypeError, ValueError):
        return None


def _parse_region_from_host(host: Optional[str]) -> Optional[str]:
    """Pull the region out of a Lakebase host string.

    Example: ep-example-endpoint-a1b2c3d4.database.us-east-1.cloud.databricks.com
                                            ^^^^^^^^^
    """
    if not host:
        return None
    # Look for ".database.<region>." pattern.
    parts = host.split(".database.")
    if len(parts) < 2:
        return None
    tail = parts[1]
    # tail looks like "us-east-1.cloud.databricks.com" — region is up to first dot.
    return tail.split(".")[0] or None


def _ts_iso(value: Any) -> Optional[str]:
    """Best-effort serializer for SDK Timestamp / datetime / pg row values."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_datetime"):
        try:
            return value.to_datetime().isoformat()
        except Exception:
            pass
    if hasattr(value, "seconds"):
        try:
            return datetime.utcfromtimestamp(int(value.seconds)).isoformat() + "Z"
        except Exception:
            pass
    return str(value)
