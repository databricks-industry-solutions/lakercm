import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ToolCallCard from '../ToolCallCard/ToolCallCard'
import { postFeedback } from '../../api/chatApi'
import './MessageBubble.css'

function ThumbIcon({ up }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      {up ? (
        <path d="M7 10v12M15 5.88L14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H7V10a5 5 0 0 0 5-5V3a3 3 0 0 1 3 2.88z" />
      ) : (
        <path d="M17 14V2M9 18.12L10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H17v12a5 5 0 0 0-5 5v2a3 3 0 0 1-3-2.88z" />
      )}
    </svg>
  )
}

function CopyIcon({ copied }) {
  if (copied) {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12" />
      </svg>
    )
  }
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  )
}

const RATIONALE_PROMPTS = [
  { value: 'inaccurate', label: 'Inaccurate' },
  { value: 'unhelpful', label: 'Unhelpful' },
  { value: 'unsafe', label: 'Unsafe / inappropriate' },
  { value: 'wrong_tool', label: 'Wrong tool / data' },
  { value: 'other', label: 'Other' },
]

function MessageBubble({
  message,
  isStreaming,
  onRetry,
  workspaceHost,
  experimentId,
}) {
  const isUser = message.role === 'user'
  const toolCalls = message.tool_calls || []
  // Initialize from server-hydrated feedback so reload preserves selection.
  // The reviewer GET /conversations/{id} batch-fetches this from the agent
  // app's /feedback endpoint, which in turn reads MLflow trace assessments.
  const initialFeedback =
    typeof message.feedback === 'boolean' ? message.feedback : null
  const [feedback, setFeedback] = useState(initialFeedback)
  const [sending, setSending] = useState(false)
  const [showRationale, setShowRationale] = useState(false)
  const [rationaleTags, setRationaleTags] = useState([])
  const [rationaleText, setRationaleText] = useState('')
  const [rationaleSubmitted, setRationaleSubmitted] = useState(false)
  const [copied, setCopied] = useState(false)
  const rationaleTextareaRef = useRef(null)

  const submitBinary = async (value) => {
    if (!message.trace_id || sending) return
    if (feedback === value) return  // already at this value — no redundant POST
    const previous = feedback
    setSending(true)
    setFeedback(value)  // Optimistic; rolled back on error.
    // Switching FROM down TO up means the user changed their mind — collapse
    // the rationale form. The earlier rationale assessment (if submitted) is
    // preserved on the trace as audit history; the latest assessment value
    // wins for `feedback.user_satisfaction` filters.
    if (value === true) {
      setShowRationale(false)
      setRationaleSubmitted(false)
    }
    try {
      await postFeedback(message.trace_id, value)
    } catch {
      setFeedback(previous)
    } finally {
      setSending(false)
    }
  }

  const handleThumbDown = () => {
    if (feedback === false) {
      // Already thumbs-downed; treat the click as "let me leave a comment".
      setShowRationale(true)
      return
    }
    submitBinary(false)
    setShowRationale(true)
  }

  const submitRationale = async () => {
    if (!message.trace_id || rationaleSubmitted) return
    const tags = rationaleTags.length ? `[${rationaleTags.join(', ')}] ` : ''
    const rationale = `${tags}${rationaleText.trim()}`.trim()
    if (!rationale) {
      setShowRationale(false)
      return
    }
    setRationaleSubmitted(true)
    try {
      // Second assessment on the same trace; MLflow stacks these so the
      // curator sees both the binary signal and the labeled rationale.
      await postFeedback(message.trace_id, false, rationale)
    } catch {
      setRationaleSubmitted(false)
      return
    }
    setShowRationale(false)
  }

  const dismissRationale = () => setShowRationale(false)

  const toggleTag = (tag) => {
    setRationaleTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    )
  }

  const copyContent = async () => {
    try {
      await navigator.clipboard.writeText(message.content || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard API may be unavailable in non-https contexts */
    }
  }

  useEffect(() => {
    if (showRationale) {
      // Focus the textarea after the form opens so the user can type immediately.
      const t = setTimeout(() => rationaleTextareaRef.current?.focus(), 50)
      return () => clearTimeout(t)
    }
  }, [showRationale])

  const showActions =
    !isUser && !isStreaming && message.content?.trim()
  const showThumbs = showActions && !!message.trace_id

  return (
    <div className={`msg ${isUser ? 'msg--user' : 'msg--assistant'}`}>
      <div className="msg__avatar">
        {isUser ? (
          <div className="msg__avatar-circle msg__avatar-circle--user">Y</div>
        ) : (
          <div className="msg__avatar-circle msg__avatar-circle--assistant">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M9 12h6M9 16h6M9 8h6M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z" />
            </svg>
          </div>
        )}
      </div>
      <div className="msg__body">
        <span className="msg__name">{isUser ? 'You' : 'LakeRCM'}</span>
        {!isUser && toolCalls.length > 0 && (
          <div className="msg__tools">
            {toolCalls.map((tc, i) => (
              <ToolCallCard
                key={tc.run_id || `${tc.tool}-${i}`}
                toolCall={tc}
                traceId={message.trace_id}
                // Per-message value wins when the stream supplied one; otherwise
                // fall back to the workspace-stable values from /api/me.
                workspaceHost={message.workspace_host || workspaceHost}
                experimentId={message.experiment_id || experimentId}
              />
            ))}
          </div>
        )}
        <div className="msg__content">
          {isUser ? (
            <p className="msg__text">{message.content}</p>
          ) : (
            <div className="msg__markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content || ''}
              </ReactMarkdown>
              {isStreaming && !message.content && toolCalls.length === 0 && (
                <div className="msg__thinking">
                  <span className="msg__dot" />
                  <span className="msg__dot" />
                  <span className="msg__dot" />
                </div>
              )}
              {isStreaming && message.content && (
                <span className="msg__cursor" />
              )}
            </div>
          )}
        </div>
        {message.error && onRetry && (
          <div className="msg__retry">
            <button className="msg__retry-btn" onClick={onRetry}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
                <polyline points="3 3 3 8 8 8" />
              </svg>
              Try again
            </button>
          </div>
        )}
        {showActions && (
          <div className="msg__actions">
            <button
              className={`msg__action-btn ${copied ? 'msg__action-btn--success' : ''}`}
              onClick={copyContent}
              title={copied ? 'Copied' : 'Copy response'}
              aria-label="Copy response"
            >
              <CopyIcon copied={copied} />
            </button>
            {showThumbs && (
              <>
                <button
                  className={`msg__action-btn ${feedback === true ? 'msg__action-btn--up' : ''}`}
                  onClick={() => submitBinary(true)}
                  disabled={sending}
                  title="Helpful"
                  aria-label="Mark response helpful"
                >
                  <ThumbIcon up />
                </button>
                <button
                  className={`msg__action-btn ${feedback === false ? 'msg__action-btn--down' : ''}`}
                  onClick={handleThumbDown}
                  disabled={sending}
                  title="Not helpful"
                  aria-label="Mark response not helpful"
                >
                  <ThumbIcon />
                </button>
              </>
            )}
          </div>
        )}
        {showRationale && (
          <div className="msg__rationale">
            <div className="msg__rationale-header">
              <span className="msg__rationale-title">What went wrong?</span>
              <button
                className="msg__rationale-close"
                onClick={dismissRationale}
                aria-label="Dismiss feedback form"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <div className="msg__rationale-tags">
              {RATIONALE_PROMPTS.map((p) => (
                <button
                  key={p.value}
                  type="button"
                  className={`msg__rationale-tag ${rationaleTags.includes(p.value) ? 'msg__rationale-tag--active' : ''}`}
                  onClick={() => toggleTag(p.value)}
                  disabled={rationaleSubmitted}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <textarea
              ref={rationaleTextareaRef}
              className="msg__rationale-textarea"
              value={rationaleText}
              onChange={(e) => setRationaleText(e.target.value)}
              placeholder="Optional: tell us more so we can improve the prompt."
              rows={2}
              disabled={rationaleSubmitted}
            />
            <div className="msg__rationale-footer">
              <button
                className="msg__rationale-submit"
                onClick={submitRationale}
                disabled={rationaleSubmitted}
              >
                {rationaleSubmitted ? 'Sent' : 'Send feedback'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default MessageBubble
