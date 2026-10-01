"""
LakeRCM Chat Route — SSE streaming endpoint proxying to the agent app.

Chat streams from the standalone agent app. Conversations are stored
in Lakebase (PostgreSQL). All conversations are scoped to the authenticated user.
"""

import asyncio
import json

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from config import settings
from services.agent_client import (
    delete_thread as agent_delete_thread,
    get_feedback_for_traces,
)
from dependencies import get_lakercm_db, get_current_user_email
from services.lakehouse_db import LakeRCMDatabase

router = APIRouter()


# -- CopilotKit / AG-UI proxy --
#
# The frontend talks to /api/copilotkit; we forward to the agent app's
# /copilotkit endpoint (mounted via ag_ui_langgraph).
#
# Beyond raw byte forwarding, the proxy *observes* the AG-UI event stream
# in flight to maintain parity with the legacy /api/chat/stream:
#
#   - User message → inserted into public.conversations BEFORE streaming
#     so the row exists even if the agent crashes mid-turn.
#   - TEXT_MESSAGE_CONTENT events → text accumulated per assistantId so
#     a future multi-message turn produces one DB row per messageId,
#     not one giant concat.
#   - TOOL_CALL_START / TOOL_CALL_RESULT → captured per messageId as a
#     JSON list. The legacy chat_stream never persisted these; the new
#     proxy fixes that gap.
#   - RAW events with event.event == "on_custom_event" and
#     event.name == "trace_marker" → trace_id captured for the assistant
#     rows (binding the thumbs-up/down UI to the right MLflow trace).
#     The event is emitted by post_model_hook via dispatch_custom_event;
#     ag_ui_langgraph wraps it as a RAW event in the wire stream.
#   - RUN_FINISHED → flush per-messageId rows to public.conversations
#     using asyncio.to_thread so the sync psycopg call doesn't block the
#     event loop.
#
# Bytes are forwarded to the client verbatim — the parser is a pure
# side observer. DB failures are logged and swallowed, never abort the
# stream. Catch-all path keeps future AG-UI sub-routes working.


def _parse_sse_event(line: str) -> dict | None:
    """Parse a single SSE `data: ...` line into a dict. Returns None for
    non-data lines, blank lines, or invalid JSON."""
    if not line.startswith("data: "):
        return None
    payload = line[6:]
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return None


def _looks_like_trace_marker(event: dict) -> bool:
    """Recognize the post_model_hook `trace_marker` custom event wrapped
    as ag_ui_langgraph RAW.

    Wire shape (verified empirically against ag-ui-langgraph 0.0.35):

        {
          "type": "RAW",
          "event": {
            "event": "on_custom_event",
            "name": "trace_marker",
            "data": {"trace_id": "trace:/..."}
          }
        }

    The outer field name is `event` (NOT `rawEvent` / `raw_event` — those
    keys appear elsewhere in the ag-ui SDK and confused an earlier draft).
    """
    if event.get("type") != "RAW":
        return False
    raw = event.get("event") or {}
    if raw.get("event") != "on_custom_event":
        return False
    return raw.get("name") == "trace_marker"


def _extract_trace_marker_value(event: dict) -> str | None:
    raw = event.get("event") or {}
    data = raw.get("data") or {}
    if isinstance(data, dict):
        return data.get("trace_id")
    return None


# No `routing_marker` helper here on purpose. This proxy forwards every SSE line
# VERBATIM (see the stream loop below) and only side-observes for persistence, so
# the browser receives the RAW routing_marker event directly and reads it with the
# AG-UI client's onRawEvent subscriber. The trace marker is parsed here only
# because its id has to be PERSISTED onto the conversation rows; the tier badge is
# per-turn UI state with nothing to store, so adding a second parser would be
# unused code pretending to be a transport.


@router.api_route(
    "/copilotkit{rest:path}",
    methods=["GET", "POST", "OPTIONS"],
)
async def copilotkit_proxy(
    rest: str,
    request: Request,
    db: LakeRCMDatabase = Depends(get_lakercm_db),
):
    from databricks.sdk import WorkspaceClient

    agent_url = settings.agent_app_url
    if not agent_url:
        raise HTTPException(
            status_code=503,
            detail="agent_app_url not configured",
        )

    user_email = get_current_user_email(request)

    w = WorkspaceClient()
    headers = dict(w.config._header_factory())
    # Strip hop-by-hop and host-rewriting headers; preserve the rest so
    # AG-UI sees the same Content-Type / Accept the client sent.
    skip_lower = {
        "host",
        "content-length",
        "connection",
        "accept-encoding",
        "authorization",  # replaced by SDK header factory below
    }
    for k, v in request.headers.items():
        if k.lower() in skip_lower:
            continue
        headers.setdefault(k, v)
    # Forward identity so the agent app tags trace + assessments with the
    # reviewer's email rather than the agent SP that signs the bearer.
    headers["X-Forwarded-Email"] = user_email

    upstream = f"{agent_url}/copilotkit{rest}"
    body = await request.body()

    # Pull persistence metadata out of the AG-UI run-input body, AND
    # inject the authenticated user_email into forwardedProps before
    # forwarding upstream.
    #
    # Why body-injection vs headers: Databricks Apps app-to-app calls
    # *replace* `x-forwarded-email` with the calling app's SP UUID, so
    # the agent-app can't read the original user from headers — it
    # sees the reviewer-app SP. The legacy /responses path solves this
    # by passing user_context.user_email in the JSON body; the AG-UI
    # path uses the same idiom via `forwardedProps.user_email`.
    thread_id: str | None = None
    user_msg_content: str = ""
    document_id: str | None = None
    rewritten_body: bytes | None = None
    if request.method == "POST" and body:
        try:
            parsed = json.loads(body)
            thread_id = parsed.get("threadId") or parsed.get("thread_id")
            msgs = parsed.get("messages") or []
            for msg in reversed(msgs):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msg_content = msg.get("content") or ""
                    break

            # Inject user_email so the agent-app can populate
            # mlflow.trace.user (and any future user-scoped logic).
            fp_camel = parsed.get("forwardedProps")
            fp_snake = parsed.get("forwarded_props")
            target = fp_camel if isinstance(fp_camel, dict) else None
            if target is None and isinstance(fp_snake, dict):
                target = fp_snake
            if target is None:
                target = {}
                parsed["forwardedProps"] = target
            target.setdefault("user_email", user_email)
            # The in-document assistant sends the open document id here; keep
            # it (do not clobber) so the agent's reviewer-action tools scope to
            # it and the persisted conversation row is document-linked.
            doc_id_val = target.get("document_id")
            if isinstance(doc_id_val, str) and doc_id_val:
                document_id = doc_id_val
            rewritten_body = json.dumps(parsed).encode()
        except (ValueError, TypeError):
            pass

    forwarded_body = rewritten_body if rewritten_body is not None else body

    # Persist the user message before we start streaming so its row exists
    # even if the agent crashes mid-turn. asyncio.to_thread keeps the sync
    # psycopg call off the event loop.
    if thread_id and user_msg_content:
        title = (
            user_msg_content[:50] + "..."
            if len(user_msg_content) > 50
            else user_msg_content
        )
        try:
            await asyncio.to_thread(
                db.insert_conversation_message,
                conversation_id=thread_id,
                user_email=user_email,
                document_id=document_id,
                title=title,
                role="user",
                content=user_msg_content,
            )
        except Exception as persist_err:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "user message persist failed (continuing): %s", persist_err
            )

    async def stream_upstream():
        # Per-request accumulators. messageId-keyed so a multi-message
        # turn produces one DB row per assistant message — matching what
        # the user sees rendered, not a concat.
        text_by_msg: dict[str, str] = {}
        tools_by_msg: dict[str, list[dict]] = {}
        # Maps toolCallId → messageId so TOOL_CALL_RESULT can be merged
        # back into the right assistant message's tool list.
        tool_msg_for_call: dict[str, str] = {}
        # Preserve order so we insert rows chronologically.
        message_order: list[str] = []
        trace_id: str | None = None
        # Buffer raw bytes across lines because httpx.aiter_lines strips
        # newlines but SSE clients expect the `\n\n` separator between
        # events. We rebuild the framing as we forward.
        sse_buffer: list[str] = []

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0)
        ) as client:
            async with client.stream(
                request.method,
                upstream,
                headers=headers,
                params=dict(request.query_params),
                content=forwarded_body,
            ) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode(errors="replace")[:300]
                    yield (
                        f"event: error\ndata: "
                        f"{json.dumps({'status': resp.status_code, 'detail': detail})}\n\n"
                    ).encode()
                    return

                async for line in resp.aiter_lines():
                    # Forward verbatim. Blank line = SSE event terminator.
                    if line == "":
                        yield "\n".encode()
                        sse_buffer.clear()
                        continue
                    yield (line + "\n").encode()

                    # Side-observe `data: …` lines. SSE in practice
                    # always has the JSON on one `data:` line per event
                    # for ag_ui_langgraph's encoder; the buffer is here
                    # only to be robust to multi-line `data:` in the
                    # future.
                    if not line.startswith("data: "):
                        continue
                    event = _parse_sse_event(line)
                    if not event:
                        continue

                    et = event.get("type")

                    if et == "TEXT_MESSAGE_START":
                        mid = event.get("messageId") or event.get("message_id")
                        if mid and mid not in text_by_msg:
                            text_by_msg[mid] = ""
                            # Adopt any pre-message tools: AG-UI emits
                            # TOOL_CALL_* BEFORE the assistant text in a
                            # ReAct turn, so anything we buffered under
                            # "__pre__" belongs to this incoming message.
                            adopted = tools_by_msg.pop("__pre__", [])
                            tools_by_msg.setdefault(mid, []).extend(adopted)
                            # Re-key any tool_msg_for_call entries that
                            # pointed at the sentinel.
                            for k, v in list(tool_msg_for_call.items()):
                                if v == "__pre__":
                                    tool_msg_for_call[k] = mid
                            message_order.append(mid)

                    elif et == "TEXT_MESSAGE_CONTENT":
                        mid = event.get("messageId") or event.get("message_id")
                        if mid is None:
                            continue
                        if mid not in text_by_msg:
                            text_by_msg[mid] = ""
                            adopted = tools_by_msg.pop("__pre__", [])
                            tools_by_msg.setdefault(mid, []).extend(adopted)
                            for k, v in list(tool_msg_for_call.items()):
                                if v == "__pre__":
                                    tool_msg_for_call[k] = mid
                            message_order.append(mid)
                        text_by_msg[mid] += event.get("delta") or ""

                    elif et == "TOOL_CALL_START":
                        tc_id = event.get("toolCallId") or event.get("tool_call_id")
                        # ReAct ordering puts TOOL_CALL_START *before* the
                        # assistant TEXT_MESSAGE_START that explains the
                        # tool result — so when no message exists yet, we
                        # park tools under "__pre__" and adopt them at the
                        # next TEXT_MESSAGE_START. See above adoption logic.
                        attach_to = message_order[-1] if message_order else "__pre__"
                        tools_by_msg.setdefault(attach_to, [])
                        if tc_id:
                            tool_msg_for_call[tc_id] = attach_to
                        tools_by_msg[attach_to].append(
                            {
                                "tool": event.get("toolCallName")
                                or event.get("tool_call_name")
                                or "tool",
                                "input": {},
                                "status": "running",
                                "run_id": tc_id,
                            }
                        )

                    elif et == "TOOL_CALL_RESULT":
                        tc_id = event.get("toolCallId") or event.get("tool_call_id")
                        if not tc_id:
                            continue
                        mid = tool_msg_for_call.get(tc_id)
                        if not mid or mid not in tools_by_msg:
                            continue
                        out = event.get("content") or event.get("result") or ""
                        if not isinstance(out, str):
                            out = json.dumps(out, default=str)
                        for tc in tools_by_msg[mid]:
                            if tc.get("run_id") == tc_id:
                                tc["status"] = "completed"
                                tc["output_preview"] = out[:500]
                                break

                    elif _looks_like_trace_marker(event):
                        tid = _extract_trace_marker_value(event)
                        if tid:
                            trace_id = tid

                    elif et == "RUN_FINISHED":
                        # Defensive: if the agent ran tools without ever
                        # emitting an assistant text message, fold the
                        # "__pre__" tools into a synthetic assistant row
                        # so they don't get silently dropped.
                        pre_tools = tools_by_msg.pop("__pre__", [])
                        if pre_tools and not message_order:
                            synth_mid = "__synth_tools_only__"
                            text_by_msg[synth_mid] = ""
                            tools_by_msg[synth_mid] = pre_tools
                            message_order.append(synth_mid)
                        if thread_id:
                            await _persist_assistant_rows(
                                db=db,
                                user_email=user_email,
                                conversation_id=thread_id,
                                message_order=message_order,
                                text_by_msg=text_by_msg,
                                tools_by_msg=tools_by_msg,
                                trace_id=trace_id,
                            )

    # AG-UI uses Server-Sent Events; same no-buffer headers as legacy.
    return StreamingResponse(
        stream_upstream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _persist_assistant_rows(
    *,
    db: LakeRCMDatabase,
    user_email: str,
    conversation_id: str,
    message_order: list[str],
    text_by_msg: dict[str, str],
    tools_by_msg: dict[str, list[dict]],
    trace_id: str | None,
):
    """Write one row per assistant messageId to public.conversations.

    Called once per turn at RUN_FINISHED. Each row is independent — a
    failure on one doesn't prevent the others. Telemetry only: failures
    are logged but never abort the stream. asyncio.to_thread keeps the
    sync psycopg call off the event loop.
    """
    import logging as _logging

    log = _logging.getLogger(__name__)
    for mid in message_order:
        text = text_by_msg.get(mid) or ""
        tools = tools_by_msg.get(mid) or []
        if not text and not tools:
            # Empty assistant turns — skip so the rail doesn't show
            # blank rows.
            continue
        try:
            await asyncio.to_thread(
                db.insert_conversation_message,
                conversation_id=conversation_id,
                user_email=user_email,
                document_id=None,
                title=None,
                role="assistant",
                content=text,
                tool_calls=json.dumps(tools) if tools else None,
                trace_id=trace_id,
            )
        except Exception as e:
            log.warning("assistant row persist failed for messageId=%s: %s", mid, e)


# -- Feedback (proxied to agent app) --


@router.post("/chat/feedback")
async def chat_feedback(request: Request):
    """Proxy feedback from the UI to the agent app's /feedback endpoint.

    Body: {trace_id, value, rationale?, name?}
    The agent app logs the feedback as an MLflow Assessment bound to the trace.
    """
    body = await request.json()
    user_email = get_current_user_email(request)

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    headers = dict(w.config._header_factory())
    headers["Content-Type"] = "application/json"
    # Pass user identity both ways: the header is best-effort (Databricks
    # Apps app-to-app calls overwrite x-forwarded-email with the calling
    # app's SP UUID), but injecting `user_email` into the body bypasses
    # that and matches the pattern used by the AG-UI /copilotkit proxy
    # and the legacy /responses route's user_context.user_email.
    headers["X-Forwarded-Email"] = user_email
    if isinstance(body, dict):
        body.setdefault("user_email", user_email)

    agent_url = settings.agent_app_url
    if not agent_url:
        return {"status": "error", "detail": "agent_app_url not configured"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(
                f"{agent_url}/feedback", headers=headers, json=body
            )
            return resp.json()
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}


# -- Conversations CRUD --


@router.get("/conversations")
async def list_conversations(
    request: Request,
    db: LakeRCMDatabase = Depends(get_lakercm_db),
):
    user_email = get_current_user_email(request)
    rows = db.list_conversations(user_email)
    return [
        {
            "id": r["conversation_id"],
            "title": r["title"],
            "message_count": int(r["message_count"]),
            "updated_at": str(r["updated_at"]) if r.get("updated_at") else None,
        }
        for r in rows
    ]


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
    db: LakeRCMDatabase = Depends(get_lakercm_db),
):
    user_email = get_current_user_email(request)
    rows = db.get_conversation_history(conversation_id, user_email)

    # Hydrate feedback selection state per message so the thumbs UI survives
    # reload. One batched call to the agent app rather than N MLflow lookups.
    trace_ids = [row["trace_id"] for row in rows if row.get("trace_id")]
    feedback_by_trace = await get_feedback_for_traces(trace_ids, user_email)

    return {
        "id": conversation_id,
        "messages": [
            {
                "role": row["message_role"],
                "content": row["message_content"],
                "tool_calls": row.get("tool_calls"),
                "trace_id": row.get("trace_id"),
                "feedback": (
                    feedback_by_trace.get(row.get("trace_id"))
                    if row.get("trace_id")
                    else None
                ),
                "created_at": str(row["created_at"]) if row.get("created_at") else None,
            }
            for row in rows
        ],
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    db: LakeRCMDatabase = Depends(get_lakercm_db),
):
    """Cascade-delete a conversation: agent checkpointer first, reviewer second.

    Fails closed — if the agent cleanup errors, the reviewer's conversations
    row is intentionally preserved so the user can retry. An audit row is
    written to public.conversation_deletions on both success and failure paths.
    Long-term memories in the agent's PostgresStore are user-scoped (not
    conversation-scoped) and are intentionally preserved.
    """
    import logging

    log = logging.getLogger(__name__)
    user_email = get_current_user_email(request)

    # Ownership / existence check. Idempotent-missing: nothing to do.
    history = db.get_conversation_history(conversation_id, user_email)
    if not history:
        return {"status": "deleted", "rows": 0, "agent": "skipped"}

    try:
        agent_result = await agent_delete_thread(conversation_id, user_email)
    except Exception as exc:
        error_detail = f"{type(exc).__name__}: {str(exc)[:300]}"
        log.error("Agent cleanup failed for %s: %s", conversation_id, error_detail)
        try:
            db.log_conversation_deletion(
                conversation_id=conversation_id,
                user_email=user_email,
                agent_cleanup_status="failed",
                reviewer_rowcount=0,
                error_detail=error_detail,
            )
        except Exception as audit_exc:
            log.error("Audit insert failed on cascade error path: %s", audit_exc)
        raise HTTPException(
            status_code=502,
            detail="Agent cleanup failed; please retry.",
        )

    rowcount = db.delete_conversation(conversation_id, user_email)

    try:
        db.log_conversation_deletion(
            conversation_id=conversation_id,
            user_email=user_email,
            agent_cleanup_status="deleted",
            reviewer_rowcount=int(rowcount or 0),
        )
    except Exception as audit_exc:
        # The deletion already happened; an audit-write failure should not
        # surface as a 5xx to the user. Log and continue.
        log.error("Audit insert failed on cascade success path: %s", audit_exc)

    return {
        "status": "deleted",
        "rows": int(rowcount or 0),
        "agent": agent_result,
    }
