import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import MessageBubble from '../MessageBubble/MessageBubble'
import SuggestedPrompts from '../SuggestedPrompts/SuggestedPrompts'
import MicButton from '../MicButton/MicButton'
import { useLiveTranscription } from '../../hooks/useLiveTranscription'
// Transport is AG-UI end-to-end: the adapter yields a uniform event shape
// (`token`/`tool_call`/`tool_result`/`trace`/`error`/`done`) over AG-UI.
// `getConversation` history is served by reviewer_app from Lakebase.
import { streamChatAGUI as streamChat, getConversation } from '../../api/chatApi'
import AgentTierSelector, {
  readStoredTier,
  storeTier,
  tierToWire,
} from '../AgentTierSelector/AgentTierSelector'
import './ChatInterface.css'

let _counter = 0
const nextId = () => `msg_${++_counter}_${Date.now()}`

function ChatInterface({ conversationId, currentUser, onStreamingChange }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [streamingId, setStreamingId] = useState(null)
  // Tracks the GET /conversations/{id} round-trip so we don't flash
  // SuggestedPrompts ("What can I help you with?") for ~200ms when
  // switching to an existing conversation that's about to render messages.
  const [loadingHistory, setLoadingHistory] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const abortRef = useRef(null)

  // -- Live speech-to-text: the mic dictates into this same composer --
  const {
    isRecording,
    start: startMic,
    stop: stopMic,
    committedPhrases,
    interimText,
    error: micError,
    supported: micSupported,
  } = useLiveTranscription()
  const preMicTextRef = useRef('')
  const micUnavailable = currentUser?.transcription_available === false

  // Reasoning tier. Shared component and shared localStorage key with the
  // in-document pane, so a reviewer's choice follows them between the two
  // surfaces instead of being a per-screen setting they have to discover twice.
  const [tier, setTier] = useState(readStoredTier)
  const [effectiveRouting, setEffectiveRouting] = useState(null)
  const handleTierChange = (next) => {
    setTier(next)
    storeTier(next)
    setEffectiveRouting(null)
  }

  // While recording, the textarea shows a derived value (pre-mic text +
  // committed phrases + live interim). Memoized to blunt re-render thrash from
  // the ~2s interim setState.
  const micValue = useMemo(
    () =>
      [preMicTextRef.current, committedPhrases.join(' '), interimText]
        .filter(Boolean)
        .join(' '),
    [committedPhrases, interimText],
  )

  // Keep the composer scrolled to the newest transcribed text while recording.
  useEffect(() => {
    if (isRecording && inputRef.current) {
      inputRef.current.scrollTop = inputRef.current.scrollHeight
    }
  }, [micValue, isRecording])

  const handleMicToggle = async () => {
    if (isRecording) {
      const finalText = await stopMic()
      const merged = [preMicTextRef.current, finalText].filter(Boolean).join(' ')
      setInput(merged)
      inputRef.current?.focus()
    } else {
      preMicTextRef.current = input
      startMic()
    }
  }

  const scrollToBottom = useCallback(() => {
    const container = messagesEndRef.current?.closest('.chat-interface__messages')
    if (container) {
      container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
    }
  }, [])

  // Track total content length across all messages so auto-scroll fires
  // as text streams in, not just when a new message is appended.
  const totalContentLength = messages.reduce(
    (acc, m) => acc + (m.content?.length || 0),
    0,
  )

  useEffect(() => {
    const timer = setTimeout(scrollToBottom, 80)
    return () => clearTimeout(timer)
  }, [messages.length, totalContentLength, scrollToBottom])

  // Reset state and load history when conversation changes
  useEffect(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setIsLoading(false)
    setStreamingId(null)
    setInput('')

    const loadHistory = async () => {
      if (!conversationId) {
        setMessages([])
        return
      }
      setLoadingHistory(true)
      try {
        const conv = await getConversation(conversationId)
        if (conv?.messages?.length) {
          setMessages(
            conv.messages.map((m, i) => ({
              id: `hist_${i}_${Date.now()}`,
              role: m.role,
              content: m.content,
              trace_id: m.trace_id || null,
              feedback: typeof m.feedback === 'boolean' ? m.feedback : null,
              tool_calls: m.tool_calls
                ? typeof m.tool_calls === 'string'
                  ? JSON.parse(m.tool_calls)
                  : m.tool_calls
                : [],
            }))
          )
        } else {
          setMessages([])
        }
      } catch {
        setMessages([])
      } finally {
        setLoadingHistory(false)
      }
      inputRef.current?.focus()
    }

    loadHistory()
  }, [conversationId])

  const handleSend = async (text) => {
    if (isRecording) return // block send while dictating (B.5)
    const messageText = text || input.trim()
    if (!messageText || isLoading) return

    const userId = nextId()
    const agentId = nextId()
    const userMsg = { id: userId, role: 'user', content: messageText }
    const agentMsg = { id: agentId, role: 'assistant', content: '', tool_calls: [] }

    // Stash the prompt text on the user message so a "Try again" button can
    // re-send it if the agent errors.
    userMsg.text = messageText

    setMessages((prev) => [...prev, userMsg, agentMsg])
    setInput('')
    setIsLoading(true)
    setStreamingId(agentId)
    onStreamingChange?.(conversationId)

    const controller = new AbortController()
    abortRef.current = controller

    const updateAgent = (updater) => {
      setMessages((prev) => prev.map((m) => (m.id === agentId ? updater(m) : m)))
    }

    let timeoutId
    try {
      let fullContent = ''
      const toolCalls = []

      const timeoutPromise = new Promise((_, reject) => {
        timeoutId = setTimeout(() => reject(new Error('timeout')), 120000)
      })

      const streamPromise = (async () => {
        setEffectiveRouting(null)
        for await (const event of streamChat(
          conversationId,
          messageText,
          controller.signal,
          tierToWire(tier)
        )) {
          if (controller.signal.aborted) break

          if (event.type === 'token') {
            fullContent += event.content
            updateAgent((m) => ({ ...m, content: fullContent }))
          } else if (event.type === 'tool_call') {
            toolCalls.push({
              ...event,
              status: event.status || 'running',
              // Local startedAt feeds the live ticker until the server-
              // measured duration_ms arrives in tool_result.
              startedAt: Date.now(),
            })
            updateAgent((m) => ({ ...m, tool_calls: [...toolCalls] }))
          } else if (event.type === 'tool_result') {
            // Match on run_id (unique per tool invocation) so two calls of
            // the same tool in one turn don't race. Fall back to (tool,
            // running) for older agent builds without run_id.
            let idx = -1
            if (event.run_id) {
              idx = toolCalls.findIndex((t) => t.run_id === event.run_id)
            }
            if (idx === -1) {
              idx = toolCalls.findIndex(
                (t) => t.tool === event.tool && t.status === 'running'
              )
            }
            if (idx !== -1) {
              toolCalls[idx] = {
                ...toolCalls[idx],
                status: 'completed',
                output_preview: event.output_preview || null,
                // Prefer server-measured duration; fall back to SSE-arrival
                // diff for backward compat.
                duration_ms:
                  typeof event.duration_ms === 'number'
                    ? event.duration_ms
                    : Date.now() - toolCalls[idx].startedAt,
              }
            }
            updateAgent((m) => ({ ...m, tool_calls: [...toolCalls] }))
          } else if (event.type === 'trace') {
            updateAgent((m) => ({
              ...m,
              trace_id: event.trace_id,
              workspace_host: event.workspace_host || m.workspace_host || null,
              experiment_id: event.experiment_id || m.experiment_id || null,
            }))
          } else if (event.type === 'routing') {
            // Which tier actually ran. Only consequential when it differs from
            // the selection: a high-risk cue escalates regardless of the choice.
            setEffectiveRouting({ tier: event.tier, source: event.source })
          } else if (event.type === 'error') {
            updateAgent((m) => ({
              ...m,
              content: event.content || 'Sorry, something went wrong. Try again.',
              error: true,
              retry_text: messageText,
            }))
          } else if (event.type === 'done') {
            break
          }
        }
      })()

      await Promise.race([streamPromise, timeoutPromise])

      // Mark remaining tool calls as completed
      const finalTools = toolCalls.map((t) => ({ ...t, status: 'completed' }))
      updateAgent((m) => ({ ...m, tool_calls: finalTools }))

      // Ensure non-empty response
      updateAgent((m) => {
        if (!m.content?.trim() && (!m.tool_calls || m.tool_calls.length === 0)) {
          return { ...m, content: 'The assistant is processing your request. Please try again if no response appears.' }
        }
        return m
      })
    } catch (err) {
      if (controller.signal.aborted) return
      updateAgent((m) => {
        if (!m.content?.trim()) {
          return {
            ...m,
            content:
              err.message === 'timeout'
                ? 'The request took too long. Please try again.'
                : 'Unable to connect to the assistant. Please try again.',
            error: true,
            retry_text: messageText,
          }
        }
        return m
      })
    } finally {
      clearTimeout(timeoutId)
      setIsLoading(false)
      setStreamingId(null)
      onStreamingChange?.(null)
      abortRef.current = null
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="chat-interface">
      <div className="chat-interface__messages">
        {isEmpty && loadingHistory ? (
          // Existing conversation is about to load — render nothing rather
          // than flash SuggestedPrompts during the GET round-trip.
          <div className="chat-interface__loading" aria-hidden="true" />
        ) : isEmpty ? (
          <SuggestedPrompts onSelect={handleSend} currentUser={currentUser} />
        ) : (
          <div className="chat-interface__messages-list">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                // Workspace-stable trace-link context from /api/me. Passed at
                // render rather than stamped onto each message: the AG-UI stream
                // carries only the trace id, and a replayed conversation out of
                // Lakebase stores only the trace id too — so both paths get
                // working links from one source.
                workspaceHost={currentUser?.workspace_host}
                experimentId={currentUser?.experiment_id}
                isStreaming={streamingId === msg.id}
                onRetry={
                  msg.error && msg.retry_text
                    ? () => handleSend(msg.retry_text)
                    : undefined
                }
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="chat-interface__input-area">
        <div className="chat-interface__input-wrap">
          <div className={`chat-interface__input-container ${isRecording ? 'chat-interface__input-container--recording' : ''}`}>
            <textarea
              ref={inputRef}
              className="chat-interface__input"
              value={isRecording ? micValue : input}
              onChange={(e) => { if (!isRecording) setInput(e.target.value) }}
              onKeyDown={handleKeyDown}
              placeholder={isRecording ? 'Listening…' : 'Message LakeRCM...'}
              rows={1}
              disabled={isLoading}
              readOnly={isRecording}
            />
            {micSupported && (
              <MicButton
                isRecording={isRecording}
                onToggle={handleMicToggle}
                disabled={isLoading || micUnavailable}
                title={
                  micUnavailable
                    ? 'Speech-to-text is unavailable'
                    : micError || (isRecording ? 'Stop dictation' : 'Start dictation')
                }
              />
            )}
            <button
              className={`chat-interface__send ${input.trim() && !isLoading && !isRecording ? 'active' : ''}`}
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading || isRecording}
            >
              {isLoading ? (
                <div className="chat-interface__send-spinner" />
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="19" x2="12" y2="5" />
                  <polyline points="5 12 12 5 19 12" />
                </svg>
              )}
            </button>
          </div>
          <div className="chat-interface__footer">
            <AgentTierSelector
              value={tier}
              onChange={handleTierChange}
              effective={effectiveRouting}
              disabled={isLoading}
            />
            <p className="chat-interface__disclaimer">
              LakeRCM may produce inaccurate information. Verify extraction data.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatInterface
