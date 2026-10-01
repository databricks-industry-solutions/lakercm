/**
 * AG-UI chat adapter.
 *
 * Bridges between the legacy ChatInterface's expected SSE event shape
 *   ({type: 'token'|'tool_call'|'tool_result'|'trace'|'error'|'done', ...})
 * and AG-UI's typed callback subscriber from @ag-ui/client.HttpAgent.
 *
 * The legacy ChatInterface remains the source of truth for chat UX — this
 * adapter lets us keep that UX while switching the transport to AG-UI
 * end-to-end (no custom Python event format, no CopilotKit Python runtime
 * with its broken LangGraphAGUIAgent wrapper).
 *
 * Conversation CRUD endpoints are reviewer-app backed and unchanged from
 * the legacy chat — we re-export the same axios wrappers so callers don't
 * need to know which transport is live.
 */

import axios from 'axios'
import { HttpAgent } from '@ag-ui/client'

// One HttpAgent per page session is plenty — it owns an AbortController
// for the in-flight run and serves as our connection to /api/copilotkit.
// Constructed lazily so SSR / static analysis don't trip on `crypto`.
let _agent = null

function getAgent() {
  if (_agent) return _agent
  _agent = new HttpAgent({ url: '/api/copilotkit' })
  return _agent
}

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
  withCredentials: true,
})

// -- Conversation CRUD (re-exported from /api, unchanged) -------------------

export async function getConversations() {
  const { data } = await api.get('/conversations')
  return data
}

export async function getConversation(conversationId) {
  const { data } = await api.get(`/conversations/${conversationId}`)
  return data
}

export async function deleteConversation(conversationId) {
  const { data } = await api.delete(`/conversations/${conversationId}`)
  return data
}

// -- Feedback (unchanged — still goes through reviewer_app proxy to
//    agent_app's /feedback, which writes an MLflow Assessment.)
export async function postFeedback(traceId, value, rationale) {
  const { data } = await api.post('/chat/feedback', {
    trace_id: traceId,
    value,
    rationale,
    name: 'user_satisfaction',
  })
  return data
}

// -- Streaming chat ---------------------------------------------------------

// The post_model_hook custom events, as ag_ui_langgraph wraps them:
//
//   { type: 'RAW', event: { event: 'on_custom_event', name: '<name>', data: {...} } }
//
// The outer key is `event`, NOT `rawEvent` — the same trap documented on the
// reviewer proxy's _looks_like_trace_marker, which an earlier draft got wrong.
//
// Read over RAW rather than state: agent/hooks.py records that ag_ui_langgraph
// 0.0.35 filters STATE_SNAPSHOT down to {messages, tools}, so a custom state field
// never reaches the browser. The state subscribers below are kept as a belt-and-
// braces path in case that filtering is relaxed.
const customEventData = (event, name) => {
  const raw = event?.event
  if (!raw || raw.event !== 'on_custom_event' || raw.name !== name) return null
  return raw.data && typeof raw.data === 'object' ? raw.data : null
}

/**
 * Stream a chat turn over AG-UI, yielding legacy-shape events the existing
 * ChatInterface consumes verbatim.
 *
 * Events yielded:
 *   { type: 'token', content }
 *   { type: 'tool_call', tool, input, status: 'running', run_id }
 *   { type: 'tool_result', tool, output_preview, status: 'completed', run_id }
 *   { type: 'trace', trace_id }
 *   { type: 'routing', tier, source }
 *   { type: 'error', content }
 *   { type: 'done' }
 *
 * The backend's PostgresSaver checkpointer (or in-memory fallback) is
 * threadId-scoped, so we send only the new user message — history is
 * reconstructed server-side from the checkpoint.
 */
export async function* streamChatAGUI(conversationId, message, signal, agentTier) {
  const agent = getAgent()

  // Per-call queue + signaling — converts AG-UI's callback-based subscriber
  // into an async iterator the consumer can `for await` over.
  const queue = []
  let resolveNext = null
  let done = false
  let runError = null

  const push = (evt) => {
    queue.push(evt)
    if (resolveNext) {
      const r = resolveNext
      resolveNext = null
      r()
    }
  }

  const finish = (err) => {
    if (done) return
    done = true
    if (err) runError = err
    if (resolveNext) {
      const r = resolveNext
      resolveNext = null
      r()
    }
  }

  // Track {toolCallId → name} so tool_result events (which only carry id)
  // can be matched back to the tool_call we emitted earlier.
  const toolNameById = new Map()
  // Track which trace_id we've already emitted to avoid duplicate 'trace'
  // events when state updates several times in one turn.
  let emittedTraceId = ''

  const subscriber = {
    onTextMessageContentEvent: ({ event }) => {
      // AG-UI naming uses snake_case in some events and camelCase in others
      // depending on the encoder; defend against both.
      const delta = event.delta ?? event.content ?? ''
      if (delta) push({ type: 'token', content: delta })
    },
    onToolCallStartEvent: ({ event }) => {
      const toolCallId = event.toolCallId ?? event.tool_call_id
      const toolCallName = event.toolCallName ?? event.tool_call_name ?? 'tool'
      if (toolCallId) toolNameById.set(toolCallId, toolCallName)
      push({
        type: 'tool_call',
        tool: toolCallName,
        input: {},
        status: 'running',
        run_id: toolCallId,
      })
    },
    onToolCallEndEvent: ({ event, toolCallName, toolCallArgs }) => {
      // No result yet — just record the args we accumulated. The result
      // arrives in onToolCallResultEvent; some agents skip the end event,
      // so emit a completion here too if there's a body to show.
      const toolCallId = event.toolCallId ?? event.tool_call_id
      const name = toolCallName ?? toolNameById.get(toolCallId) ?? 'tool'
      push({
        type: 'tool_result',
        tool: name,
        output_preview: toolCallArgs ? JSON.stringify(toolCallArgs).slice(0, 500) : null,
        status: 'completed',
        run_id: toolCallId,
      })
    },
    onToolCallResultEvent: ({ event }) => {
      const toolCallId = event.toolCallId ?? event.tool_call_id
      const name = toolNameById.get(toolCallId) ?? 'tool'
      const out = event.content ?? event.result ?? ''
      const preview = typeof out === 'string' ? out.slice(0, 500) : JSON.stringify(out).slice(0, 500)
      push({
        type: 'tool_result',
        tool: name,
        output_preview: preview,
        status: 'completed',
        run_id: toolCallId,
      })
    },
    // Which tier ACTUALLY served this turn. Needed because a high-risk cue
    // outranks an explicit tier selection, so the selector has to be able to say
    // it was overridden rather than show a choice that did not take effect.
    onRawEvent: ({ event }) => {
      const trace = customEventData(event, 'trace_marker')
      if (trace?.trace_id && trace.trace_id !== emittedTraceId) {
        emittedTraceId = trace.trace_id
        push({ type: 'trace', trace_id: trace.trace_id })
      }
      const routing = customEventData(event, 'routing_marker')
      if (routing?.tier) {
        push({ type: 'routing', tier: routing.tier, source: routing.source || '' })
      }
    },
    onStateSnapshotEvent: ({ state }) => {
      const tid = state?.last_trace_id
      if (tid && tid !== emittedTraceId) {
        emittedTraceId = tid
        push({ type: 'trace', trace_id: tid })
      }
    },
    onStateDeltaEvent: ({ state }) => {
      const tid = state?.last_trace_id
      if (tid && tid !== emittedTraceId) {
        emittedTraceId = tid
        push({ type: 'trace', trace_id: tid })
      }
    },
    onRunErrorEvent: ({ event }) => {
      push({
        type: 'error',
        content: event.message || event.error || 'The assistant failed to respond.',
      })
    },
    onRunFinalized: () => finish(),
    onRunFailed: ({ error }) => finish(error),
  }

  // Set the conversation context. The agent's checkpointer keys off threadId,
  // so history reconstruction happens server-side; we only send the new turn.
  agent.threadId = conversationId
  agent.messages = [
    {
      id: typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `u_${Date.now()}`,
      role: 'user',
      content: message,
    },
  ]
  agent.state = {}

  // Abort plumbing: tie the caller's signal to the agent's abort.
  if (signal) {
    if (signal.aborted) {
      yield { type: 'done' }
      return
    }
    signal.addEventListener(
      'abort',
      () => {
        try { agent.abortRun() } catch { /* ignore */ }
        finish()
      },
      { once: true },
    )
  }

  // Kick the run. Don't await — we want to consume events as they stream in.
  // forwardedProps, not an instance assignment — AbstractAgent builds the request
  // body in prepareRunAgentInput() from the CALL parameters, so an assignment on
  // the agent is silently dropped. Same trap documented on the reviewer stream.
  const forwardedProps =
    agentTier && agentTier !== 'auto' ? { agent_tier: agentTier } : {}
  agent.runAgent({ forwardedProps }, subscriber).catch((err) => finish(err))

  while (!done || queue.length) {
    if (signal?.aborted) {
      try { agent.abortRun() } catch { /* ignore */ }
      break
    }
    if (queue.length) {
      yield queue.shift()
      continue
    }
    if (done) break
    // Park until the next event arrives or the run finishes.
    await new Promise((r) => {
      resolveNext = r
    })
  }

  if (runError) {
    yield {
      type: 'error',
      content: runError.message || String(runError).slice(0, 200),
    }
  }
  yield { type: 'done' }
}

// A dedicated HttpAgent for the in-document reviewer assistant, kept separate
// from the general-chat singleton so document scoping (forwardedProps) can't
// leak between the two surfaces.
let _reviewerAgent = null

function getReviewerAgent() {
  if (_reviewerAgent) return _reviewerAgent
  _reviewerAgent = new HttpAgent({ url: '/api/copilotkit' })
  return _reviewerAgent
}

/**
 * Stream a turn for the in-document reviewer assistant.
 *
 * Same transport as streamChatAGUI, plus two things the pane needs:
 *  1. `documentId` is sent in AG-UI forwardedProps.document_id so the agent's
 *     reviewer-action tools scope to the open document (the reviewer-app proxy
 *     preserves it and the agent parks it in a contextvar).
 *  2. Staged UI actions ride back on the tool RESULT as a compact
 *     `_frontend_action` object (see agent_app/agent/tools.py). We parse the
 *     FULL tool-result content (not the truncated preview) and surface it as a
 *     `{ type: 'frontend_action', action }` event the pane renders as a card.
 *
 * Yields the same events as streamChatAGUI, plus:
 *   { type: 'frontend_action', action: { type, ... } }
 */
export async function* streamReviewerAgent(
  conversationId,
  message,
  documentId,
  signal,
  agentTier
) {
  const agent = getReviewerAgent()

  const queue = []
  let resolveNext = null
  let done = false
  let runError = null

  const push = (evt) => {
    queue.push(evt)
    if (resolveNext) {
      const r = resolveNext
      resolveNext = null
      r()
    }
  }
  const finish = (err) => {
    if (done) return
    done = true
    if (err) runError = err
    if (resolveNext) {
      const r = resolveNext
      resolveNext = null
      r()
    }
  }

  const toolNameById = new Map()
  let emittedTraceId = ''

  // Parse a tool result's full content for a staged _frontend_action and, if
  // present, emit it. Defensive: tolerate non-JSON / plain-string results.
  const maybeEmitAction = (raw) => {
    if (typeof raw !== 'string' || raw.indexOf('_frontend_action') === -1) return
    try {
      const parsed = JSON.parse(raw)
      const action = parsed && parsed._frontend_action
      if (action && typeof action === 'object' && action.type) {
        push({ type: 'frontend_action', action })
      }
    } catch {
      /* not JSON — nothing to stage */
    }
  }

  const subscriber = {
    onTextMessageContentEvent: ({ event }) => {
      const delta = event.delta ?? event.content ?? ''
      if (delta) push({ type: 'token', content: delta })
    },
    onToolCallStartEvent: ({ event }) => {
      const toolCallId = event.toolCallId ?? event.tool_call_id
      const toolCallName = event.toolCallName ?? event.tool_call_name ?? 'tool'
      if (toolCallId) toolNameById.set(toolCallId, toolCallName)
      push({
        type: 'tool_call',
        tool: toolCallName,
        input: {},
        status: 'running',
        run_id: toolCallId,
      })
    },
    onToolCallEndEvent: ({ event, toolCallName, toolCallArgs }) => {
      const toolCallId = event.toolCallId ?? event.tool_call_id
      const name = toolCallName ?? toolNameById.get(toolCallId) ?? 'tool'
      push({
        type: 'tool_result',
        tool: name,
        output_preview: toolCallArgs
          ? JSON.stringify(toolCallArgs).slice(0, 500)
          : null,
        status: 'completed',
        run_id: toolCallId,
      })
    },
    onToolCallResultEvent: ({ event }) => {
      const toolCallId = event.toolCallId ?? event.tool_call_id
      const name = toolNameById.get(toolCallId) ?? 'tool'
      const out = event.content ?? event.result ?? ''
      const rawStr = typeof out === 'string' ? out : JSON.stringify(out)
      // Stage any UI action BEFORE truncating for the display preview.
      maybeEmitAction(rawStr)
      push({
        type: 'tool_result',
        tool: name,
        output_preview: rawStr.slice(0, 500),
        status: 'completed',
        run_id: toolCallId,
      })
    },
    // Which tier ACTUALLY served this turn. Needed because a high-risk cue
    // outranks an explicit tier selection, so the selector has to be able to say
    // it was overridden rather than show a choice that did not take effect.
    onRawEvent: ({ event }) => {
      const trace = customEventData(event, 'trace_marker')
      if (trace?.trace_id && trace.trace_id !== emittedTraceId) {
        emittedTraceId = trace.trace_id
        push({ type: 'trace', trace_id: trace.trace_id })
      }
      const routing = customEventData(event, 'routing_marker')
      if (routing?.tier) {
        push({ type: 'routing', tier: routing.tier, source: routing.source || '' })
      }
    },
    onStateSnapshotEvent: ({ state }) => {
      const tid = state?.last_trace_id
      if (tid && tid !== emittedTraceId) {
        emittedTraceId = tid
        push({ type: 'trace', trace_id: tid })
      }
    },
    onStateDeltaEvent: ({ state }) => {
      const tid = state?.last_trace_id
      if (tid && tid !== emittedTraceId) {
        emittedTraceId = tid
        push({ type: 'trace', trace_id: tid })
      }
    },
    onRunErrorEvent: ({ event }) => {
      push({
        type: 'error',
        content: event.message || event.error || 'The assistant failed to respond.',
      })
    },
    onRunFinalized: () => finish(),
    onRunFailed: ({ error }) => finish(error),
  }

  agent.threadId = conversationId
  agent.messages = [
    {
      id:
        typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : `u_${Date.now()}`,
      role: 'user',
      content: message,
    },
  ]
  agent.state = {}
  // Document scoping for the agent's reviewer-action tools. This MUST be passed
  // to runAgent() below, not assigned to the agent: AbstractAgent builds the
  // request body in prepareRunAgentInput(), which takes `forwardedProps` (and
  // tools/context) from the CALL parameters and only `state`/`messages` from the
  // instance — so an assignment here is silently dropped and the agent receives
  // `forwardedProps: {}`. That is what left get_active_review_context with no
  // open document, disabling every reviewer-action tool in the pane.
  // agent_tier rides the same channel as document_id, for the same reason: it is
  // the part of the AG-UI body a client controls end to end. 'auto' is omitted so
  // the agent sees no request at all and behaves exactly as it did before there
  // was a selector.
  const forwardedProps = {
    ...(documentId ? { document_id: documentId } : {}),
    ...(agentTier && agentTier !== 'auto' ? { agent_tier: agentTier } : {}),
  }

  if (signal) {
    if (signal.aborted) {
      yield { type: 'done' }
      return
    }
    signal.addEventListener(
      'abort',
      () => {
        try { agent.abortRun() } catch { /* ignore */ }
        finish()
      },
      { once: true },
    )
  }

  agent.runAgent({ forwardedProps }, subscriber).catch((err) => finish(err))

  while (!done || queue.length) {
    if (signal?.aborted) {
      try { agent.abortRun() } catch { /* ignore */ }
      break
    }
    if (queue.length) {
      yield queue.shift()
      continue
    }
    if (done) break
    await new Promise((r) => {
      resolveNext = r
    })
  }

  if (runError) {
    yield {
      type: 'error',
      content: runError.message || String(runError).slice(0, 200),
    }
  }
  yield { type: 'done' }
}
