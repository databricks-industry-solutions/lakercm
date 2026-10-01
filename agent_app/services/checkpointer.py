"""
LangGraph PostgresSaver wired to the shared Lakebase pool.

The checkpointer persists agent state (messages, tool_calls, etc.) across
invocations of the same thread_id. Crash-safe mid-turn state and
interrupt/resume semantics depend on this.

Two flavors live here, backed by the SAME `public.checkpoint_*` tables:

  - `get_checkpointer()` → sync `PostgresSaver`, used by the legacy
    `/responses` SSE path which runs `agent.invoke(...)` in a worker
    thread. Sync methods (`get_tuple`, `put`, `delete_thread`).
  - `get_async_checkpointer()` → async `AsyncPostgresSaver`, used by
    the AG-UI `/copilotkit` path. The AG-UI runtime awaits
    `graph.aget_state(...)` → `checkpointer.aget_tuple(...)`; the sync
    saver's base `aget_tuple` raises NotImplementedError, hence the
    split.

Both share the same group-backed Postgres role
(`lakercm-checkpoint-runtime`) — see scripts/add_lakebase_roles.py.
Setup is one-time and idempotent; the async setup runs from the FastAPI
lifespan startup handler so the AsyncConnectionPool is bound to the
running event loop.
"""

import logging
from typing import Any, Dict, Optional

from langgraph.checkpoint.postgres import PostgresSaver

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:  # pragma: no cover - test stubs / older package shapes
    AsyncPostgresSaver = None  # type: ignore[assignment]

from services.lakehouse_db import get_db

try:
    from services.lakehouse_db import lakebase_operation_span, _set_pool_stats
except ImportError:  # pragma: no cover - unit-test stubs install get_db only
    from contextlib import contextmanager

    @contextmanager
    def lakebase_operation_span(*args, **kwargs):
        yield None

    def _set_pool_stats(*args, **kwargs) -> None:
        return None


logger = logging.getLogger(__name__)

_checkpointer_singleton: Optional[PostgresSaver] = None
_checkpointer_resolved: bool = False

_async_checkpointer_singleton: Optional[Any] = None
_async_pool_opened: bool = False
_async_setup_done: bool = False


def get_checkpointer() -> Optional[PostgresSaver]:
    """Return the process-wide sync PostgresSaver, bound to the shared pool.

    Returns None if setup() fails AND the expected tables don't already exist
    — so the agent can run stateless instead of raising on every request.
    """
    global _checkpointer_singleton, _checkpointer_resolved
    if _checkpointer_resolved:
        return _checkpointer_singleton

    # Run setup + probe with tracing disabled — PostgresSaver.setup()
    # issues CREATE TABLE IF NOT EXISTS DDL on first call, which autolog
    # would otherwise emit as an orphan root trace into the experiment.
    from services.tracing import tracing_disabled

    pool = get_db().get_pool()
    saver = PostgresSaver(pool)
    with tracing_disabled():
        try:
            saver.setup()
            _checkpointer_singleton = saver
            _checkpointer_resolved = True
            return _checkpointer_singleton
        except Exception as setup_err:
            logger.warning("PostgresSaver.setup() failed: %s", setup_err)

        # Setup failed — verify the tables already exist (someone else ran the
        # DDL). If they don't, cache None so create_react_agent runs stateless
        # instead of blowing up on every invoke.
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM public.checkpoint_migrations LIMIT 1")
            _checkpointer_singleton = saver
        except Exception as probe_err:
            logger.warning(
                "Checkpointer tables unavailable (%s) — running agent stateless",
                probe_err,
            )
            _checkpointer_singleton = None

    _checkpointer_resolved = True
    return _checkpointer_singleton


def get_async_checkpointer() -> Optional[AsyncPostgresSaver]:
    """Return the process-wide async AsyncPostgresSaver.

    The instance is bound to the lazy AsyncConnectionPool returned by
    `lakehouse_db.get_async_pool()` — the pool itself is created with
    `open=False` so this function can be called at module-import time
    (where no event loop is running) without blowing up. The pool is
    opened and setup() is awaited later by `init_async_checkpointer()`,
    which the FastAPI lifespan runs once at startup.
    """
    global _async_checkpointer_singleton
    if _async_checkpointer_singleton is not None:
        return _async_checkpointer_singleton
    if AsyncPostgresSaver is None:
        logger.warning("AsyncPostgresSaver unavailable; AG-UI checkpointer disabled")
        return None
    try:
        pool = get_db().get_async_pool()
    except Exception as e:
        logger.warning("AsyncConnectionPool unavailable at import: %s", e)
        return None
    _async_checkpointer_singleton = AsyncPostgresSaver(pool)
    return _async_checkpointer_singleton


def checkpointer_status() -> dict:
    """Read-only snapshot of checkpointer readiness for /health.

    No side effects: reads the module-level flags only — never constructs
    pools or triggers setup() (health probes must not mutate state).
    """
    return {
        "async_pool_opened": _async_pool_opened,
        "async_setup_done": _async_setup_done,
        "sync_resolved": _checkpointer_resolved,
        "sync_available": _checkpointer_singleton is not None,
    }


async def init_async_checkpointer() -> bool:
    """Open the async pool and run AsyncPostgresSaver.setup() exactly once.

    Called from the FastAPI startup hook so all async operations run on
    the live event loop. Idempotent — short-circuits on the second call.
    Returns True on success, False on failure (caller may fall back to
    MemorySaver).
    """
    global _async_pool_opened, _async_setup_done
    if _async_setup_done:
        return True

    saver = get_async_checkpointer()
    if saver is None:
        logger.warning("Async checkpointer not constructed; skipping setup")
        return False

    pool = get_db().get_async_pool()
    with lakebase_operation_span(
        "lakebase.async_checkpointer.open_pool", span_type="UNKNOWN"
    ):
        try:
            if not _async_pool_opened:
                await pool.open()
                _async_pool_opened = True
                logger.info("Async Lakebase pool opened")
        except Exception as e:
            logger.warning("AsyncConnectionPool.open() failed: %s", e)
            return False

    with lakebase_operation_span(
        "lakebase.async_checkpointer.setup", span_type="UNKNOWN"
    ):
        try:
            await saver.setup()
            _async_setup_done = True
            logger.info(
                "AsyncPostgresSaver.setup() succeeded — AG-UI checkpointer ready"
            )
            return True
        except Exception as e:
            logger.warning("AsyncPostgresSaver.setup() failed: %s", e)
            return False


def delete_thread(thread_id: str) -> Dict[str, Any]:
    """Remove all checkpointer rows for a thread_id.

    Prefers PostgresSaver.delete_thread (added in langgraph-checkpoint-postgres
    2.0.10) so future schema changes stay behind a stable API. Falls back to
    raw DELETE against the three known tables if the method is missing on an
    older install. Returns a status dict; does not raise on "not found" —
    deletes for a missing thread_id are a no-op by design (idempotent).
    """
    if not thread_id:
        return {"status": "skipped", "reason": "empty thread_id"}

    saver = get_checkpointer()
    if saver is None:
        # No checkpointer bound (e.g. tables missing at boot) means there is
        # nothing to delete on the agent side.
        return {"status": "skipped", "reason": "checkpointer unavailable"}

    delete_fn = getattr(saver, "delete_thread", None)
    if callable(delete_fn):
        with lakebase_operation_span(
            "lakebase.checkpointer.delete_thread", span_type="UNKNOWN"
        ) as span:
            try:
                _set_pool_stats(span, get_db().get_pool(), "db.client.pool.before")
                delete_fn(thread_id)
                _set_pool_stats(span, get_db().get_pool(), "db.client.pool.after")
                return {"status": "deleted", "method": "PostgresSaver.delete_thread"}
            except Exception as e:
                logger.error(
                    "PostgresSaver.delete_thread(%s) failed: %s",
                    thread_id,
                    e,
                    exc_info=True,
                )
                raise

    # Fallback — emulate the delete by hand. Guarded because the schema could
    # change; we still want the operation to succeed against known tables.
    logger.warning(
        "PostgresSaver.delete_thread missing; using raw-SQL fallback. "
        "Consider upgrading langgraph-checkpoint-postgres."
    )
    pool = get_db().get_pool()
    with lakebase_operation_span(
        "lakebase.checkpointer.delete_thread_raw", span_type="UNKNOWN"
    ) as span:
        try:
            _set_pool_stats(span, pool, "db.client.pool.before")
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM public.checkpoint_writes WHERE thread_id = %s",
                        (thread_id,),
                    )
                    cur.execute(
                        "DELETE FROM public.checkpoint_blobs WHERE thread_id = %s",
                        (thread_id,),
                    )
                    cur.execute(
                        "DELETE FROM public.checkpoints WHERE thread_id = %s",
                        (thread_id,),
                    )
            _set_pool_stats(span, pool, "db.client.pool.after")
            return {"status": "deleted", "method": "raw-sql-fallback"}
        except Exception as e:
            logger.error("Raw-SQL thread delete failed: %s", e, exc_info=True)
            raise
