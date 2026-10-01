"""
Lakebase-backed long-term memory store for the LakeRCM agent.

Wraps LangGraph's PostgresStore on the shared psycopg_pool from lakehouse_db,
with pgvector-backed semantic indexing over an embedding column served by
the Databricks-hosted GTE model. A single process-wide instance is reused by
the agent so token refresh + pool management stays unified.

Namespacing: every memory lives under `(user_email, "memories")`. The
helper tools in agent/memory_tools.py enforce this prefix at the tool
boundary — the system prompt is NOT the security layer.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext

import mlflow
from databricks.sdk import WorkspaceClient

from services.lakehouse_db import get_db
from services.lakehouse_db import lakebase_operation_span, _set_pool_stats

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("LAKERCM_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = int(os.getenv("LAKERCM_EMBEDDING_DIMS", "1024"))


def _embed_fn(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the Databricks Foundation Model API.

    Uses the WorkspaceClient's serving endpoint wrapper so auth + host
    resolution match the rest of the app. Returns a list of 1024-dim vectors.
    """
    w = WorkspaceClient()
    try:
        resp = w.serving_endpoints.query(
            name=EMBEDDING_MODEL,
            input=texts,
        )
        data = getattr(resp, "data", None) or []
        vectors: list[list[float]] = []
        for item in data:
            vec = getattr(item, "embedding", None)
            if vec is None and isinstance(item, dict):
                vec = item.get("embedding")
            if vec is not None:
                # Coerce every element to float — psycopg's list adapter rejects
                # lists with mixed int/float elements ("cannot dump lists of mixed
                # types"), which the GTE endpoint occasionally produces when a
                # dimension lands exactly on 0 or 1.
                vectors.append([float(x) for x in vec])
        if len(vectors) != len(texts):
            logger.warning(
                "Embedding batch size mismatch: got %d vectors for %d texts",
                len(vectors),
                len(texts),
            )
        return vectors
    except Exception as e:
        logger.error("Embedding call failed: %s", e, exc_info=True)
        return [[0.0] * EMBEDDING_DIMS for _ in texts]


_store_singleton = None


def get_store():
    """Return the shared PostgresStore instance (lazy-init).

    The store is constructed from the same psycopg_pool that backs the
    checkpointer, so Lakebase credential refresh stays centralized in
    AgentLakehouseDB.
    """
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    from langgraph.store.postgres import PostgresStore

    pool = get_db().get_pool()

    # Suppress boot-time setup noise, but allow this Lakebase span to attach
    # when lazy store initialization happens inside an existing agent turn.
    from services.tracing import tracing_disabled

    setup_trace_context = (
        nullcontext()
        if mlflow.get_current_active_span() is not None
        else tracing_disabled()
    )

    store = PostgresStore(
        pool,
        index={
            "dims": EMBEDDING_DIMS,
            "embed": _embed_fn,
            "fields": ["content"],
        },
    )
    with setup_trace_context:
        try:
            with lakebase_operation_span(
                "lakebase.postgres_store.setup", span_type="UNKNOWN"
            ) as span:
                _set_pool_stats(span, pool, "db.client.pool.before")
                store.setup()
                _set_pool_stats(span, pool, "db.client.pool.after")
        except Exception as e:
            logger.warning(
                "PostgresStore.setup() raised (may already be initialized): %s", e
            )

    _store_singleton = store
    logger.info("PostgresStore initialized with %s embeddings", EMBEDDING_MODEL)
    return store
