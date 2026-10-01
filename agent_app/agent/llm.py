"""
LakeRCM agent — canonical LLM client factory.

Single source of truth for constructing the OpenAI-compatible chat client the
agent talks to (the Unity AI Gateway model service). EVERY ``ChatOpenAI`` in the
app must come from :func:`build_chat_openai` so auth is refreshed per request and
the gateway base path is applied in exactly one place.

Why this module exists — the bug it prevents
---------------------------------------------
``langchain_openai.ChatOpenAI`` bakes ``api_key`` into a *static* Authorization
header at construction. Long-lived clients therefore send a FIXED bearer:

* the AG-UI graph, compiled once at boot in ``graph.get_agui_graph`` (serves
  ``/copilotkit``), and
* the process-wide summarizer cache in ``hooks._get_summarizer``.

The app's service-principal OAuth token expires ~1h after boot, so those baked
clients then return ``401 Invalid Token`` — the AG-UI SSE stream aborts and the
browser shows a network error. The per-request ``/responses`` path was immune
only because it rebuilt the agent (and re-minted the token) every turn.

The OpenAI SDK exposes no token-refresh callback (it takes only ``api_key`` +
``http_client``), so we hand every client a shared ``httpx`` client whose
``request`` event hook OVERWRITES Authorization with a freshly-resolved bearer on
every call. The token comes from the Databricks SDK's config header factory,
which refreshes the SP's OAuth token as it nears expiry — the same per-borrow
refresh principle the Lakebase pool already uses
(``services/lakehouse_db.py`` ``_fresh_conninfo``).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os

import httpx
from databricks.sdk import WorkspaceClient
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)

_auth_ws: WorkspaceClient | None = None


def _ws() -> WorkspaceClient:
    """Return a process-cached WorkspaceClient (holds the SDK's token cache)."""
    global _auth_ws
    if _auth_ws is None:
        _auth_ws = WorkspaceClient()
    return _auth_ws


def get_bearer() -> str:
    """Resolve a CURRENT auth bearer from the cached WorkspaceClient.

    The SDK's header factory refreshes the SP's OAuth token as it nears expiry,
    so repeated calls return a live token. The httpx hooks below re-inject this
    per request, so a client that baked a stale api_key at construction never
    actually sends it.
    """
    w = _ws()
    return w.config.token or dict(w.config._header_factory()).get(
        "Authorization", ""
    ).replace("Bearer ", "")


# Per-conversation session id → Unity AI Gateway session-affinity header.
# When a stable session header is present, the gateway pins a whole
# conversation to ONE destination model (weighted-rendezvous hashing) instead
# of re-rolling the weighted traffic split per request. That gives prefix-cache
# reuse across turns and one consistent model (voice) per conversation.
# Header name is `x-session-affinity` — one of the gateway's SAFE-flag header
# list — NOT `x-databricks-session-id`, which collided with a platform browser
# session header and pinned all browser traffic to one model (incident
# ES-2042964). The id is set into this contextvar in the SAME execution context
# that makes the LLM call (the /responses worker thread and the AG-UI async
# task) because contextvars do not propagate across the worker-thread boundary.
_llm_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_llm_session_id", default=""
)


def set_llm_session_id(session_id: str) -> None:
    """Pin subsequent LLM calls in THIS execution context to a gateway session.

    Call inside the context that issues the model call — the /responses worker
    thread (`_run_agent` in main.py) and the AG-UI `run()` async task — since a
    contextvar set in the request handler does not cross the worker thread.
    """
    if session_id:
        _llm_session_id.set(session_id)


def _apply_session_affinity(request: httpx.Request) -> None:
    """Inject the gateway session-affinity header if a session id is set."""
    sid = _llm_session_id.get()
    if sid:
        request.headers["x-session-affinity"] = sid


def _auth_hook_sync(request: httpx.Request) -> None:
    request.headers["Authorization"] = f"Bearer {get_bearer()}"
    _apply_session_affinity(request)


async def _auth_hook_async(request: httpx.Request) -> None:
    # Resolve the token OFF the event loop: get_bearer() calls the SDK's
    # synchronous header factory, which does a blocking network refresh when the
    # token nears expiry (~once/hour). Running it inline in this async hook would
    # stall the whole event loop during that refresh.
    token = await asyncio.to_thread(get_bearer)
    request.headers["Authorization"] = f"Bearer {token}"
    _apply_session_affinity(request)


# Shared httpx clients handed to every ChatOpenAI. The `openai` client bakes the
# api_key into a static Authorization header at construction; these request hooks
# OVERWRITE it with a freshly-resolved bearer on every call, so a boot-built
# client can't send an expired token. Created once at import and reused across
# all ChatOpenAI instances — never per-call, which would leak connection pools.
# The generous client timeout is only a ceiling; ChatOpenAI's own per-request
# `timeout` still applies.
HTTP_SYNC_CLIENT = httpx.Client(
    timeout=httpx.Timeout(300.0),
    event_hooks={"request": [_auth_hook_sync]},
)
HTTP_ASYNC_CLIENT = httpx.AsyncClient(
    timeout=httpx.Timeout(300.0),
    event_hooks={"request": [_auth_hook_async]},
)


def _resolve_host() -> str:
    """Resolve the Databricks host at call time.

    SDK-first: the Apps runtime injects ``DATABRICKS_HOST`` and the SDK
    normalizes it to a well-formed URL. Fall back to env / settings for local
    dev without a profile. (graph.py keeps its own richer variant for callers
    that import it; this local copy avoids a circular import back into graph.)
    """
    try:
        host = _ws().config.host or ""
    except Exception:
        host = ""
    host = host or os.getenv("DATABRICKS_HOST", "") or settings.databricks_host or ""
    return host.rstrip("/")


def build_chat_openai(*, base_path: str | None = None, **overrides) -> ChatOpenAI:
    """Construct a ChatOpenAI wired for per-request Databricks auth refresh.

    All callers go through here so token refresh + the gateway base path are
    guaranteed in one place. ``model`` / ``base_url`` / ``api_key`` /
    ``http_client`` / ``http_async_client`` are set here; ``**overrides`` passes
    through everything else (``max_tokens``, ``timeout``, ``max_retries``,
    ``streaming``, …) and an explicit override wins. ``base_path`` overrides the
    default gateway path for the rare non-gateway caller.
    """
    host = _resolve_host()
    path = base_path if base_path is not None else settings.llm_base_path
    params = dict(
        model=settings.llm_endpoint,
        base_url=f"{host}{path}",
        # Seed only: the httpx request hooks above overwrite Authorization with a
        # fresh bearer on every call. The `or` guard keeps construction (and app
        # boot) from crashing on a transient empty token — a real auth failure
        # then surfaces as a clear 401 at request time, not a boot crash.
        api_key=get_bearer() or "placeholder-token-refreshed-per-request",
        http_client=HTTP_SYNC_CLIENT,
        http_async_client=HTTP_ASYNC_CLIENT,
    )
    params.update(overrides)
    return ChatOpenAI(**params)
