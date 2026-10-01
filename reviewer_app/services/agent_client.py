"""
Client for streaming responses from the standalone LakeRCM Agent App.

Uses httpx async streaming to connect to the agent app's /responses endpoint
and forward SSE events to the UI.
"""

import json
from typing import AsyncGenerator

import httpx
from databricks.sdk import WorkspaceClient

from config import settings


def _get_agent_app_url() -> str:
    """Get the agent app URL from settings."""
    return settings.agent_app_url


def _get_auth_headers() -> dict:
    """Get auth headers from WorkspaceClient."""
    w = WorkspaceClient()
    headers = dict(w.config._header_factory())
    headers["Content-Type"] = "application/json"
    return headers


async def stream_from_agent(
    messages: list[dict],
    user_context: dict,
    conversation_id: str = None,
    timeout: float = 120.0,
) -> AsyncGenerator[dict, None]:
    """Stream SSE events from the agent app.

    Yields parsed event dicts: {"type": "token"|"tool_call"|"tool_result"|"done"|"error", ...}
    """
    agent_url = _get_agent_app_url()
    if not agent_url:
        yield {
            "type": "error",
            "content": "Agent app URL not configured. Set LAKERCM_AGENT_APP_URL.",
        }
        return

    headers = _get_auth_headers()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10.0)
    ) as client:
        async with client.stream(
            "POST",
            f"{agent_url}/responses",
            headers=headers,
            json={
                "messages": messages,
                "user_context": user_context,
                "conversation_id": conversation_id,
            },
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                yield {
                    "type": "error",
                    "content": f"Agent returned {response.status_code}: {body.decode()[:200]}",
                }
                return

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                payload = line[6:]
                if payload == "[DONE]":
                    return

                try:
                    event = json.loads(payload)
                    yield event
                except json.JSONDecodeError:
                    continue


async def get_feedback_for_traces(
    trace_ids: list[str],
    user_email: str,
    timeout: float = 10.0,
) -> dict[str, bool | None]:
    """Fetch the latest user_satisfaction Assessment value per trace_id.

    Returns {trace_id: True | False | None}. None means no Assessment exists
    for this user on that trace. Reviewer app uses this on conversation
    history load so the thumbs selection state survives reload.
    """
    if not trace_ids:
        return {}

    agent_url = _get_agent_app_url()
    if not agent_url:
        return {tid: None for tid in trace_ids}

    headers = _get_auth_headers()
    headers["X-Forwarded-Email"] = user_email

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0)
        ) as client:
            resp = await client.get(
                f"{agent_url}/feedback",
                headers=headers,
                params={"trace_ids": ",".join(trace_ids)},
            )
            if resp.status_code >= 300:
                return {tid: None for tid in trace_ids}
            data = resp.json()
            # Normalize to bool|None for every requested id.
            return {
                tid: data.get(tid) if isinstance(data.get(tid), bool) else None
                for tid in trace_ids
            }
    except Exception:
        # Never fail history load over feedback hydration.
        return {tid: None for tid in trace_ids}


async def delete_thread(
    thread_id: str,
    user_email: str,
    timeout: float = 30.0,
) -> dict:
    """Ask the agent app to drop all checkpointer state for a thread.

    Returns the agent's JSON payload on 2xx. Raises on non-2xx so the caller
    (reviewer_app delete endpoint) can fail closed and leave its own
    conversations row intact as a retry re-entry point.
    """
    agent_url = _get_agent_app_url()
    if not agent_url:
        raise RuntimeError("agent_app_url not configured")

    headers = _get_auth_headers()
    # Forward identity so the agent app tags its MLflow span with the reviewer.
    headers["X-Forwarded-Email"] = user_email

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10.0)
    ) as client:
        resp = await client.delete(
            f"{agent_url}/threads/{thread_id}",
            headers=headers,
        )
        if resp.status_code >= 300:
            raise RuntimeError(
                f"agent /threads/{thread_id} returned {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        return resp.json()
