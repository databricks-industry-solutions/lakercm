"""
Long-term memory tools for the LakeRCM agent.

Three tools are exposed to the model:
  - save_user_memory(content) — write a fact/preference worth remembering
  - get_user_memory(query)     — semantic top-k recall over past memories
  - delete_user_memory(memory_id) — explicit forget

Security model: the user_email comes from the request's ContextVar (set by
main.py before each invoke). The tool wrappers enforce the namespace tuple
(user_email, "memories") — the LLM cannot specify another user's namespace,
and the system prompt is explicitly NOT the security layer here.

Write strategy: LLM-in-the-loop. The system prompt instructs the agent to
call save_user_memory when it learns something durable about the reviewer
(preferred formatting, recurring insurers, ongoing investigations). A
background auto-extraction path is intentionally deferred to Step 6.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from langchain_core.tools import tool

from agent.tools import _get_authorized_user_email
from services.store import get_store

logger = logging.getLogger(__name__)


def _user_ns() -> tuple[str, str]:
    """Return the (user_email, 'memories') namespace for the active request.

    ContextVar is set in main.py per request — if unset, memory operations
    are a no-op rather than leaking across users. Periods and slashes are
    replaced with underscores because PostgresStore rejects them as
    namespace-label separators.
    """
    email = _get_authorized_user_email()
    if not email or email == "unknown":
        return ("anonymous", "memories")
    safe = email.replace(".", "_").replace("/", "_")
    return (safe, "memories")


@tool
def save_user_memory(content: str) -> str:
    """Save a durable fact or preference about the user for recall in future conversations.

    Use this when you learn something that would help you work with this
    reviewer in later sessions — e.g. formatting preferences, specific
    insurers or document types they frequently work with, ongoing
    investigations, named projects they care about. Do NOT save transient
    question/answer pairs or facts already obvious from the data.

    Returns a memory id the user can reference to delete the memory later.
    """
    if not content or not content.strip():
        return json.dumps({"error": "content required"})

    memory_id = str(uuid4())
    try:
        store = get_store()
        store.put(
            namespace=_user_ns(),
            key=memory_id,
            value={"content": content.strip()},
        )
        return json.dumps({"memory_id": memory_id, "saved": True})
    except Exception as e:
        logger.warning("save_user_memory failed: %s", e)
        return json.dumps({"error": f"failed to save memory: {type(e).__name__}"})


@tool
def get_user_memory(query: str, limit: int = 3) -> str:
    """Semantically search the user's long-term memories.

    Use this at the start of a conversation or when the user's question
    references prior context ("the one I was looking at yesterday", "same
    format as last time"). Returns the top-k most relevant memories.
    """
    if not query or not query.strip():
        return json.dumps({"results": []})

    limit = max(1, min(int(limit or 3), 10))
    try:
        store = get_store()
        items = store.search(_user_ns(), query=query.strip(), limit=limit)
        results = []
        for item in items:
            value = item.value if isinstance(item.value, dict) else {}
            results.append(
                {
                    "memory_id": item.key,
                    "content": value.get("content", ""),
                    "score": getattr(item, "score", None),
                }
            )
        return json.dumps({"query": query, "count": len(results), "results": results})
    except Exception as e:
        logger.warning("get_user_memory failed: %s", e)
        return json.dumps({"results": [], "error": f"{type(e).__name__}"})


@tool
def delete_user_memory(memory_id: str) -> str:
    """Delete a specific long-term memory by id. Use when the user asks you to forget something."""
    if not memory_id or not memory_id.strip():
        return json.dumps({"error": "memory_id required"})
    try:
        store = get_store()
        store.delete(namespace=_user_ns(), key=memory_id.strip())
        return json.dumps({"memory_id": memory_id, "deleted": True})
    except Exception as e:
        logger.warning("delete_user_memory failed: %s", e)
        return json.dumps({"error": f"{type(e).__name__}"})


def get_memory_tools() -> list:
    """Return the three memory tools as a list."""
    return [save_user_memory, get_user_memory, delete_user_memory]
