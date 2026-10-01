"""
Pre-model hook for the LakeRCM agent.

Runs before every LLM call and does two things:

1. Trims the message window to a bounded token budget so long conversations
   don't blow the context window or the cost envelope.
2. When the trim drops enough messages, runs a cheap summarization pass and
   stores the result in state["summary"]. The summary is prepended to the
   system prompt so the model still sees relevant history.

Design notes:
- The hook returns an `llm_input_messages` override, NOT a write to
  state["messages"]. This keeps the full checkpointed transcript intact
  (display layer + audit use it) while only the model sees the trimmed view.
- Summarization uses the same LLM endpoint as the main agent; keep
  max_tokens small (400) so the summarization call itself is cheap.
- The summarization trigger is "we had to drop > SUMMARIZE_WHEN_DROPPED
  messages this turn" rather than a turn-count threshold, because a single
  user turn can produce many tool messages and the token budget matters
  more than turn count.
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)
from langchain_openai import ChatOpenAI

from agent.llm import build_chat_openai
from config import settings

logger = logging.getLogger(__name__)

# Hard ceiling on prompt tokens passed to the model. Claude Opus 4.6/4.7
# has a 200k window. The previous default of 12k was too tight: a single
# tool result (e.g. search_documents with limit=20) plus the system prompt
# could blow it, causing trim_messages to over-prune and the agent to loop
# (call tool → result trimmed → re-call tool, +2 messages per turn until
# the recursion_limit). 60k holds a comfortable tool-heavy turn while
# still bounding cost.
MAX_TOKENS = int(os.getenv("LAKERCM_TRIM_MAX_TOKENS", "60000"))

# Trigger summarization when this turn's trim dropped more than N messages.
SUMMARIZE_WHEN_DROPPED = int(os.getenv("LAKERCM_SUMMARIZE_DROP_THRESHOLD", "6"))

# Bound the summary itself so it doesn't grow without limit across turns.
# Sizing (best-practice grounded): langmem's SummarizationNode defaults
# max_summary_tokens=256 for terse summaries, but our prompt targets "under
# 300 words" (~390 tokens) and on the /responses path this is max_output_tokens
# — it must also cover low-effort reasoning tokens (~50 observed) plus an
# overshoot margin, because a summary truncated mid-sentence corrupts the
# rolling memory and compounds across turns. 768 (~3x the langmem baseline)
# fits the 300-word target with comfortable headroom on both API paths.
MAX_SUMMARY_TOKENS = int(os.getenv("LAKERCM_MAX_SUMMARY_TOKENS", "768"))


_summarizer_cache: ChatOpenAI | None = None


def _get_summarizer() -> ChatOpenAI | None:
    """Lazy-build the summarization LLM via the shared factory.

    Using build_chat_openai (agent/llm.py) means the summarizer refreshes its
    auth token per request like every other client, and targets the same Unity
    AI Gateway path as the agent. Before this it baked a token at first use AND
    was pinned to the pre-gateway `/serving-endpoints` path, so once the agent
    moved to the gateway the summarizer both misrouted and 401'd after ~1h —
    silently, since summarization is an infrequent path. Cached process-wide
    (safe now that auth is per-request); returns None on build failure so
    _summarize degrades gracefully.
    """
    global _summarizer_cache
    if _summarizer_cache is not None:
        return _summarizer_cache
    try:
        if settings.agent_llm_use_responses_api:
            # Match the agent's /responses routing so the summarizer works no
            # matter which destination the conversation is pinned to — some
            # destinations only tool-call / accept a uniform reasoning param on
            # /responses, so a plain /chat/completions summarizer could break on
            # part of the blend. effort='low' keeps it cheap; _extract_text()
            # below keeps reasoning blocks out of the stored summary.
            _summarizer_cache = build_chat_openai(
                max_tokens=MAX_SUMMARY_TOKENS,
                use_responses_api=True,
                reasoning_effort="low",
            )
        else:
            _summarizer_cache = build_chat_openai(max_tokens=MAX_SUMMARY_TOKENS)
        return _summarizer_cache
    except Exception as e:
        logger.warning("Failed to build summarizer LLM: %s", e)
        return None


def _extract_text(content) -> str:
    """Flatten an AIMessage.content to plain text.

    On the Responses API a reasoning model returns content as a LIST of blocks
    (reasoning + text). ``str(list)`` would serialize the reasoning trace into
    the stored summary (a prior "summary leak"), so keep only text blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("text", "output_text") and block.get("text"):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


SUMMARIZATION_PROMPT = (
    "You are maintaining a running summary of a conversation between a medical "
    "claims reviewer and an AI assistant. Update the summary to incorporate the "
    "newly-dropped messages below. Keep it under 300 words, focus on:\n"
    "- What documents or extractions the user has been looking at\n"
    "- Questions the user has asked and answers the assistant gave\n"
    "- Any reviewer preferences that emerged (formatting, focus areas)\n"
    "- Open threads the user may return to\n"
    "Do NOT invent details. Output ONLY the updated summary text."
)


def _summarize(previous_summary: str, dropped: list[BaseMessage]) -> str | None:
    """Merge the just-dropped messages into a rolling summary."""
    summarizer = _get_summarizer()
    if summarizer is None or not dropped:
        return None

    transcript_lines: list[str] = []
    for m in dropped:
        if isinstance(m, HumanMessage):
            transcript_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content.strip():
                transcript_lines.append(f"Assistant: {content}")
        # Skip ToolMessages — they're implementation detail; the assistant's
        # subsequent text message already encodes the useful result.

    if not transcript_lines:
        return previous_summary or None

    prompt_messages: list[BaseMessage] = [
        SystemMessage(content=SUMMARIZATION_PROMPT),
        HumanMessage(
            content=(
                f"Previous summary:\n{previous_summary or '(none)'}\n\n"
                f"Newly dropped messages:\n" + "\n".join(transcript_lines)
            )
        ),
    ]
    try:
        resp = summarizer.invoke(prompt_messages)
        new_summary = _extract_text(resp.content)
        return new_summary.strip() or previous_summary
    except Exception as e:
        logger.warning("Summarization call failed: %s", e)
        return previous_summary or None


def _salvage_json_args(raw: str) -> str | None:
    """Return a canonical JSON string for a tool-call `arguments` value, or None
    if `raw` is already valid JSON (no repair needed).

    Root cause this fixes: on the /responses path, some models stream a no-arg
    tool call as an `{}` delta followed by a spurious empty-string `""` delta;
    langchain concatenates the raw deltas into `{}""`. That survives the first
    (lenient) call but fails STRICT upstream JSON parsing when the tool history
    is re-sent on the post-tool model call — 400 "Extra data: line 1 column 3
    (char 2)" — which aborts the whole turn. `json.raw_decode` parses the
    leading valid JSON value and reports where it ends, so we can drop the
    trailing garbage and re-serialize canonically.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        json.loads(raw)
        return None  # already valid — leave it untouched
    except Exception:
        pass
    try:
        obj, _end = json.JSONDecoder().raw_decode(raw.lstrip())
        return json.dumps(obj)
    except Exception:
        return None  # unsalvageable here; caller falls back to parsed args


def _normalize_tool_call_args(msgs: list[BaseMessage]) -> list[BaseMessage]:
    """Repair malformed tool-call `arguments` strings in AIMessages so the
    re-sent tool history parses cleanly upstream (see _salvage_json_args).

    Walks every AIMessage and rewrites malformed argument strings in BOTH
    representations langchain may serialize from: Responses-API content blocks
    (`content` is a list with `type == "function_call"` items) and the
    chat-completions `additional_kwargs["tool_calls"]` shape. When a raw string
    can't be salvaged on its own, falls back to the message's already-parsed
    `tool_calls` args (clean — the tool executed from them). Produces modified
    COPIES (model_copy) so the checkpointed message objects are never mutated.
    Fully defensive: any per-message failure leaves that message untouched.
    """
    # call_id/id -> canonical JSON of the parsed args (authoritative fallback).
    parsed_by_id: dict[str, str] = {}
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                tc_id = tc.get("id")
                if tc_id:
                    try:
                        parsed_by_id[tc_id] = json.dumps(tc.get("args") or {})
                    except Exception:
                        parsed_by_id[tc_id] = "{}"

    def fix_args(raw, call_id: str | None) -> str | None:
        salvaged = _salvage_json_args(raw) if isinstance(raw, str) else None
        if salvaged is not None:
            return salvaged
        # raw was valid (None from salvage on the valid path) OR unsalvageable.
        if isinstance(raw, str):
            try:
                json.loads(raw)
                return None  # valid, no change
            except Exception:
                pass
        # unsalvageable → authoritative parsed args, else empty object
        return parsed_by_id.get(call_id or "", "{}")

    out: list[BaseMessage] = []
    for m in msgs:
        if not isinstance(m, AIMessage):
            out.append(m)
            continue
        changed = False
        new_content = m.content
        # Responses-API content blocks.
        if isinstance(m.content, list):
            rebuilt = []
            for block in m.content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "function_call"
                    and "arguments" in block
                ):
                    fixed = fix_args(block.get("arguments"), block.get("call_id"))
                    if fixed is not None and fixed != block.get("arguments"):
                        block = {**block, "arguments": fixed}
                        changed = True
                rebuilt.append(block)
            if changed:
                new_content = rebuilt
        # chat-completions tool_calls in additional_kwargs.
        new_ak = m.additional_kwargs
        ak_tcs = (m.additional_kwargs or {}).get("tool_calls")
        if isinstance(ak_tcs, list):
            rebuilt_tcs = []
            ak_changed = False
            for tc in ak_tcs:
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                    fn = tc["function"]
                    fixed = fix_args(fn.get("arguments"), tc.get("id"))
                    if fixed is not None and fixed != fn.get("arguments"):
                        tc = {**tc, "function": {**fn, "arguments": fixed}}
                        ak_changed = True
                rebuilt_tcs.append(tc)
            if ak_changed:
                new_ak = {**m.additional_kwargs, "tool_calls": rebuilt_tcs}
                changed = True
        if changed:
            try:
                m = m.model_copy(
                    update={"content": new_content, "additional_kwargs": new_ak}
                )
            except Exception as e:
                logger.warning("tool-arg normalization copy failed: %s", e)
        out.append(m)
    return out


def _repair_tool_call_integrity(msgs: list[BaseMessage]) -> list[BaseMessage]:
    """Enforce the tool_call <-> ToolMessage invariant on a message list.

    create_react_agent validates (via _validate_chat_history) that every
    AIMessage.tool_call has a matching ToolMessage; a violation raises a
    ValueError that tears down the AG-UI SSE run ("network error" in the
    browser). A turn that errors mid-tool-execution leaves the CHECKPOINTED
    history with an AIMessage(tool_calls) whose ToolMessage(s) were never
    written — which then poisons EVERY later turn in that thread. Trimming can
    also, in edge cases, leave an orphan ToolMessage.

    Repair the model-input view so the conversation self-heals:
      * synthesize a placeholder ToolMessage for any unanswered tool_call
        (inserted right after the requesting AIMessage so the block stays
        contiguous, as the chat APIs require), and
      * drop orphan ToolMessages whose tool_call is not present in the list.

    Only the model-input list is repaired; the underlying checkpoint is left
    untouched. pre_model_hook runs before every model call, so the repair
    re-applies each turn — the poison never reaches the validator.
    """
    requested_ids: set[str] = set()
    for m in msgs:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            requested_ids.update(tc.get("id") for tc in m.tool_calls if tc.get("id"))
    answered_ids = {
        m.tool_call_id
        for m in msgs
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None)
    }

    repaired: list[BaseMessage] = []
    synthesized: set[str] = set()
    for m in msgs:
        if isinstance(m, ToolMessage):
            # Keep only ToolMessages that answer a tool_call present in the list.
            if getattr(m, "tool_call_id", None) in requested_ids:
                repaired.append(m)
            continue
        repaired.append(m)
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tc_id = tc.get("id")
                if tc_id and tc_id not in answered_ids and tc_id not in synthesized:
                    repaired.append(
                        ToolMessage(
                            content=(
                                "Tool call did not complete — it was interrupted "
                                "in an earlier turn. Continue without its result."
                            ),
                            tool_call_id=tc_id,
                            name=tc.get("name", "") or "",
                        )
                    )
                    synthesized.add(tc_id)
    return repaired


def pre_model_hook(state: dict) -> dict:
    """Trim messages + optionally refresh the rolling summary.

    Returns state updates:
      - llm_input_messages: the trimmed view the model will actually see
      - summary: refreshed rolling summary (only when trim dropped enough)
    """
    messages: list[BaseMessage] = list(state.get("messages") or [])
    previous_summary: str = state.get("summary") or ""

    if not messages:
        return {"llm_input_messages": messages}

    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )

    # Defensive fallback: trim_messages with start_on="human" + end_on=("human","tool")
    # + allow_partial=False can over-prune to [] when the trailing slice can't
    # satisfy both endpoint constraints (e.g. ends on AIMessage with tool_calls
    # waiting for ToolMessages that got dropped). An empty list passes through
    # to the LLM call and FMAPI returns "messages: at least one message is
    # required".
    #
    # Previously the fallback kept only the last HumanMessage, which stripped
    # the AIMessage(tool_calls) + ToolMessage pair the LLM needs to continue.
    # The model would re-call the same tool, the result was again too large
    # to fit, and the agent looped until LangGraph's recursion limit.
    #
    # New fallback: walk backward from the end, keep the most recent
    # HumanMessage and EVERY message after it (the in-flight agent-tool
    # exchange) so the LLM sees the question AND its own work-in-progress.
    if not trimmed:
        last_human_idx = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if isinstance(messages[i], HumanMessage)
            ),
            None,
        )
        if last_human_idx is not None:
            trimmed = list(messages[last_human_idx:])
        else:
            trimmed = list(messages[-1:])
        logger.warning(
            "trim_messages over-pruned to []; falling back to last user turn "
            "(input had %d messages, kept %d, MAX_TOKENS=%d)",
            len(messages),
            len(trimmed),
            MAX_TOKENS,
        )

    dropped_count = len(messages) - len(trimmed)
    logger.debug(
        "pre_model_hook trim: %d → %d (dropped %d, summary=%s)",
        len(messages),
        len(trimmed),
        dropped_count,
        "yes" if previous_summary else "no",
    )
    updates: dict = {"llm_input_messages": trimmed}

    if dropped_count > SUMMARIZE_WHEN_DROPPED:
        dropped = messages[:dropped_count]
        refreshed = _summarize(previous_summary, dropped)
        if refreshed:
            updates["summary"] = refreshed
            # Prepend the summary to the view the model sees so the context
            # survives the trim.
            summary_msg = SystemMessage(
                content=f"Conversation summary so far:\n{refreshed}"
            )
            updates["llm_input_messages"] = [summary_msg] + list(trimmed)

    elif previous_summary:
        # No new summarization this turn, but we still want the model to see
        # whatever summary we already had.
        summary_msg = SystemMessage(
            content=f"Conversation summary so far:\n{previous_summary}"
        )
        updates["llm_input_messages"] = [summary_msg] + list(trimmed)

    # Guarantee the tool_call <-> ToolMessage invariant on the final model input
    # so a thread with a dangling tool_call (from a turn that errored mid-tool)
    # self-heals instead of failing _validate_chat_history on every later turn.
    updates["llm_input_messages"] = _repair_tool_call_integrity(
        updates["llm_input_messages"]
    )
    # Repair malformed tool-call argument strings (e.g. a model streaming `{}""`
    # for a no-arg call) so the re-sent tool history parses cleanly upstream on
    # the /responses post-tool model call instead of 400-ing "Extra data".
    updates["llm_input_messages"] = _normalize_tool_call_args(
        updates["llm_input_messages"]
    )
    return updates


def _record_output_phi_signal(state: dict) -> None:
    """Tag the trace when the model's own output carries PHI-shaped text.

    Closes a real gap rather than duplicating the gateway: AI-Gateway OUTPUT
    guardrails do not apply to streaming responses, and this agent streams
    (`streaming=True`), so nothing else inspects what the model actually emits.
    This records the SHAPES found (kinds + count) on the trace: tags for
    Trace-UI filtering, metadata for the `agent_traces_trace_metadata` table the
    dashboards query. It is deliberately not alerted (see obs_alerts.yml).

    Three deliberate properties:
      * **Never blocks and never mutates the response.** Showing identifiers to
        an authorized reviewer is the product working correctly; this is
        measurement, not enforcement.
      * **Never records PHI values** — only the kind names and a count. Writing
        the matched values into telemetry would create the leak it is watching
        for.
      * **Fully isolated.** Its own try/except, and called BEFORE the trace-id
        capture below, so a failure here can never skip the `trace_marker`
        dispatch that reviewer-app feedback binding depends on.

    The decision logic lives in `guards.phi_signal_for_messages` (pure, and
    therefore unit-tested in tests/test_guards.py); this wrapper only turns its
    result into trace tags and metadata.
    """
    try:
        import mlflow

        from agent.guards import phi_signal_for_messages

        signal = phi_signal_for_messages(state.get("messages") or [])
        if signal is None:
            return
        kinds, count = signal
        # BOTH channels, per the convention in services/observability.py
        # (set_prompt_context): tags are indexed for Trace-UI filtering, while
        # METADATA is the durable record that lands in
        # agent_traces_trace_metadata.trace_metadata — the table the dashboard and
        # alert queries actually read. A tags-only write would be invisible to
        # every SQL consumer.
        mlflow.update_current_trace(
            tags={
                "phi_shapes_in_output": ",".join(kinds),
                "phi_shape_count": str(count),
            },
            metadata={
                "guard.phi_shapes_in_output": ",".join(kinds),
                "guard.phi_shape_count": str(count),
            },
        )
        logger.info(
            "output PHI shapes present: kinds=%s count=%d", ",".join(kinds), count
        )
    except Exception as e:  # pragma: no cover — monitoring must never break a turn
        logger.debug("output PHI scan skipped: %s", e)


def _record_user_injection_signal(state: dict) -> None:
    """Record a direct prompt-injection attempt in the user's own message.

    Detection only, like the PHI monitor above: hard-blocking on a regex would
    refuse legitimate reviewer questions, so the attempt is recorded on the trace
    and the agent's resistance is scored offline by the injection_resistance
    judge. Only category names are recorded, never the message. Separate from
    `guard.injection_signal` (document-borne, alerted) because a reviewer
    probing the agent is a different event from a poisoned document. Fully
    isolated, like `_record_output_phi_signal`.
    """
    try:
        import mlflow

        from agent.guards import injection_signal_for_messages

        categories = injection_signal_for_messages(state.get("messages") or [])
        if not categories:
            return
        joined = ",".join(categories)
        mlflow.update_current_trace(
            tags={"user_injection_signal": joined},
            metadata={"guard.user_injection_signal": joined},
        )
        logger.info("direct injection signal in user message: %s", joined)
    except Exception as e:  # pragma: no cover — monitoring must never break a turn
        logger.debug("user injection scan skipped: %s", e)


def _dispatch_routing_marker() -> None:
    """Tell the client which tier actually served this turn.

    The reviewer's tier selector needs this because safety escalation OUTRANKS an
    explicit choice: a reviewer can pick Low and still get High on a turn that
    mentions fraud. A control that is silently overridden is worse than no
    control, so the effective tier and the reason travel back to the UI.

    Dispatched here rather than from the request handler because
    dispatch_custom_event has to run inside a runnable's execution context, and
    the post-model hook is one. Deliberately called AFTER the trace_marker
    dispatch and in its own try/except: reviewer-app feedback binding depends on
    that trace id, and a cosmetic routing badge must never be what costs it.

    Silent when routing is off — get_routing_decision() returns None and there is
    nothing to say.
    """
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event

        from agent.routing import get_routing_decision

        decision = get_routing_decision()
        if not decision:
            return
        tier, source = decision
        dispatch_custom_event("routing_marker", {"tier": tier, "source": source})
        logger.info("routing_marker dispatched: tier=%s source=%s", tier, source)
    except Exception as e:  # pragma: no cover — a badge must never break a turn
        logger.debug("routing_marker dispatch skipped: %s", e)


def post_model_hook(state: dict) -> dict:
    """Capture the active MLflow trace id and surface it to the AG-UI client.

    Two channels:

    1. **State field `last_trace_id`** — written into LangGraph state.
       Kept for the future when ag_ui_langgraph emits custom state fields
       in `STATE_SNAPSHOT` (its 0.0.35 release filters the snapshot down
       to `{messages, tools}` so this field doesn't currently reach the
       client through that path; harmless to keep).
    2. **LangChain `on_custom_event` callback** — emitted via
       `dispatch_custom_event("trace_marker", {trace_id})`. ag_ui_langgraph
       forwards this through its astream_events bridge as a `RAW` AG-UI
       event with `rawEvent.event === "on_custom_event"` and
       `rawEvent.name === "trace_marker"`. The reviewer-app proxy listens
       for that exact shape to bind feedback to the right MLflow trace.

    Idempotent w.r.t. missing telemetry: no active span → return empty
    state delta and skip the custom event.
    """
    # Isolated, best-effort, and FIRST so they can never interfere with the
    # trace_marker dispatch below (see _record_output_phi_signal).
    _record_output_phi_signal(state)
    _record_user_injection_signal(state)
    try:
        import mlflow
        from langchain_core.callbacks.manager import dispatch_custom_event

        span = mlflow.get_current_active_span()
        tid = getattr(span, "trace_id", None) if span else None
        logger.info(
            "post_model_hook fired: span=%s tid=%s",
            "present" if span else "missing",
            tid[:24] + "…" if tid else None,
        )
        if not tid:
            return {}

        # Sync dispatch. dispatch_custom_event needs to be called from
        # inside a runnable's execution context — LangGraph's prebuilt
        # post_model_hook IS such a context. If the dispatcher raises
        # (e.g. wrong context), we log and continue: the state field
        # still records the trace_id for any future state-aware client.
        try:
            dispatch_custom_event("trace_marker", {"trace_id": tid})
            logger.info("trace_marker dispatched for tid=%s", tid[:24] + "…")
        except Exception as dispatch_err:
            logger.warning(
                "dispatch_custom_event failed (%s): %s",
                type(dispatch_err).__name__,
                dispatch_err,
            )
        _dispatch_routing_marker()
        return {"last_trace_id": tid}
    except Exception as e:  # pragma: no cover — never fail the agent over telemetry
        logger.warning("post_model_hook trace capture skipped: %s", e)
    return {}
