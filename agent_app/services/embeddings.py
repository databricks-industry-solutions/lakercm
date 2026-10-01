"""Shared Foundation-Model embedding helper for document semantic search.

One embedding path for every consumer (the semantic-search agent tool, the
embed-documents job, and a future reviewer endpoint): the Databricks FM
serving endpoint `databricks-gte-large-en` (1024-dim), queried via the
WorkspaceClient so auth + host resolution match the rest of the app.

This mirrors the proven call shape in services/store.py:_embed_fn, but stands
alone so importers don't pull in the LangGraph PostgresStore (store.py imports
langgraph + the checkpointer). Never raises: on any failure it returns
zero-vectors so callers degrade to "no semantic hits" rather than erroring.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# Same env knobs + defaults as services/store.py so the memory store and the
# document corpus embed with the identical model + dimensionality.
EMBEDDING_MODEL = os.getenv("LAKERCM_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = int(os.getenv("LAKERCM_EMBEDDING_DIMS", "1024"))

_workspace_client: Optional[WorkspaceClient] = None


def _client() -> WorkspaceClient:
    global _workspace_client
    if _workspace_client is None:
        _workspace_client = WorkspaceClient()
    return _workspace_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts → one EMBEDDING_DIMS-length float vector each.

    Elements are coerced to float (psycopg's list adapter rejects mixed
    int/float lists, which the GTE endpoint occasionally emits when a dimension
    lands exactly on 0/1). On any error, returns zero-vectors of the right
    shape so the caller still gets len(texts) vectors and never has to
    exception-handle around embedding.
    """
    if not texts:
        return []
    try:
        resp = _client().serving_endpoints.query(name=EMBEDDING_MODEL, input=texts)
        data = getattr(resp, "data", None) or []
        vectors: list[list[float]] = []
        for item in data:
            vec = getattr(item, "embedding", None)
            if vec is None and isinstance(item, dict):
                vec = item.get("embedding")
            if vec is not None:
                vectors.append([float(x) for x in vec])
        if len(vectors) < len(texts):
            logger.warning(
                "embed batch short: %d vectors for %d texts; zero-padding",
                len(vectors),
                len(texts),
            )
            vectors += [
                [0.0] * EMBEDDING_DIMS for _ in range(len(texts) - len(vectors))
            ]
        return vectors[: len(texts)]
    except Exception as e:
        logger.error("embedding call failed: %s", e, exc_info=True)
        return [[0.0] * EMBEDDING_DIMS for _ in texts]


def embed_one(text: str) -> list[float]:
    """Embed a single text → one vector."""
    return embed_texts([text])[0]


def to_pgvector_literal(vec: list[float]) -> str:
    """Render a float vector as a pgvector text literal ('[v1,v2,...]').

    We bind this string with an explicit `%s::vector` cast rather than passing
    a Python list: psycopg renders a list as a Postgres array literal (`{...}`),
    while pgvector's input syntax is bracketed (`[...]`). The literal + cast is
    deterministic and needs no pgvector-python registration / extra dependency.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
