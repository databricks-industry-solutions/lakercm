import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import MessageBubble from '../MessageBubble/MessageBubble'
import MicButton from '../MicButton/MicButton'
import AgentTierSelector, {
  readStoredTier,
  storeTier,
  tierToWire,
} from '../AgentTierSelector/AgentTierSelector'
import { useLiveTranscription } from '../../hooks/useLiveTranscription'
import { streamReviewerAgent } from '../../api/chatApi'
import { recordDocumentProposal, setProposalDisposition } from '../../api/lakeRcmApi'
import './ReviewerAssistantPane.css'

// The in-document assistant: the same LakeRCM assistant, embedded in the
// reviewer pane and scoped to the open document. It can PROPOSE extraction
// edits and a verdict (staged cards the reviewer approves — never auto-applied)
// and append to the reviewer's notes. Approvals flow through the page's existing
// correction / review-form state, so the reviewer still presses Submit to write
// the verdict.
//
// The pane used to host its own notepad. It has been removed: ReviewNotes at the
// bottom of the document page owns the same backend record (one per document +
// reviewer), and two editors on one record is last-writer-wins. Removing this one
// leaves a single editor rather than moving the hazard around. The agent can still
// append — it writes through the shared record and tells the page to reload.

let _c = 0
const nid = () => `ra_${++_c}_${Date.now()}`

const VERDICT_LABEL = {
  correct: 'Correct',
  partially_correct: 'Partially correct',
  incorrect: 'Incorrect',
}

const EXAMPLE_ASKS = [
  'Review this extraction and flag anything questionable',
  'Suggest a verdict for this document, with reasoning',
  'Summarize the low-confidence fields',
]

function StagedActionCard({ item, fieldLabelForKey, onApprove, onReject }) {
  const a = item.action || {}
  const done = item.status === 'applied'
  const rejected = item.status === 'rejected'

  let title = 'Proposed change'
  let body = null
  if (a.type === 'extraction_edit') {
    const label = fieldLabelForKey?.(a.correction_key) || a.correction_key
    title = 'Proposed field correction'
    body = (
      <>
        <div className="ra-card-row">
          <span className="ra-card-key">{label}</span>
        </div>
        <div className="ra-card-value">{a.proposed_value}</div>
        {a.rationale && <p className="ra-card-rationale">{a.rationale}</p>}
      </>
    )
  } else if (a.type === 'verdict') {
    title = 'Proposed verdict'
    body = (
      <>
        <div className="ra-card-row">
          <span className={`ra-verdict-chip ra-verdict-${a.verdict}`}>
            {VERDICT_LABEL[a.verdict] || a.verdict}
          </span>
        </div>
        {a.reasoning && <p className="ra-card-rationale">{a.reasoning}</p>}
      </>
    )
  } else if (a.type === 'note_append') {
    return (
      <div className="ra-action-card ra-action-note">
        <span className="ra-note-icon">✎</span>
        <span>Added a note to the notepad.</span>
      </div>
    )
  }

  return (
    <div
      className={`ra-action-card ${done ? 'is-applied' : ''} ${
        rejected ? 'is-rejected' : ''
      }`}
    >
      <div className="ra-card-title">{title}</div>
      <div className="ra-card-body">{body}</div>
      {done ? (
        <div className="ra-card-status ra-card-status--ok">
          ✓ Applied to the review form — review and Submit to save.
        </div>
      ) : rejected ? (
        <div className="ra-card-status ra-card-status--muted">Dismissed</div>
      ) : (
        <div className="ra-card-actions">
          <button className="ra-btn ra-btn-approve" onClick={onApprove}>
            Approve
          </button>
          <button className="ra-btn ra-btn-reject" onClick={onReject}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  )
}

export default function ReviewerAssistantPane({
  documentId,
  // Already passed by DocumentDetailPage; the pane simply never destructured it,
  // which is why trace links and the mic never appeared here.
  currentUser,
  onClose,
  onApplyCorrection,
  onApplyVerdict,
  fieldLabelForKey,
  // The assistant proposed a note. Handed up so ReviewNotes — the single owner
  // of that record — applies it; this pane deliberately does not write notes.
  onNoteAppended,
  // Drag-to-resize handle wiring, owned by DocumentDetailPage (which also owns
  // the --assistant-width it writes to).
  onResizeStart,
  onResizeKey,
  // Reported up so the review form can show an "Agent verification" strip while
  // a turn is streaming. This pane can be closed, collapsed or scrolled away
  // mid-turn, and the decision the agent informs is made over in the review
  // form -- so the indicator cannot live only in here.
  onActivityChange,
}) {
  const [items, setItems] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [streamingId, setStreamingId] = useState(null)
  const abortRef = useRef(null)
  const messagesEndRef = useRef(null)

  // One conversation (checkpointer thread) per pane mount.
  const convIdRef = useRef(null)
  if (!convIdRef.current) {
    convIdRef.current =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `conv_${Date.now()}`
  }

  // -- Tier selection -------------------------------------------------------
  const [tier, setTier] = useState(readStoredTier)
  // What the server said actually served the last turn. Only used to explain an
  // override: a high-risk cue outranks the selector, and the UI has to say so.
  const [effectiveRouting, setEffectiveRouting] = useState(null)

  const handleTierChange = (next) => {
    setTier(next)
    storeTier(next)
    // Clear the override notice: it described the previous turn, and leaving it up
    // next to a changed selection would read as a fresh escalation.
    setEffectiveRouting(null)
  }

  // -- Voice ----------------------------------------------------------------
  // Same hook, same merge recipe as ChatInterface — see the comment there. The
  // recipe is small and the failure mode of diverging from it is losing dictated
  // text, so it is copied deliberately rather than approximated.
  const {
    isRecording,
    start: startMic,
    stop: stopMic,
    error: micError,
    committedPhrases,
    interimText,
    supported: micSupported,
  } = useLiveTranscription()
  const preMicTextRef = useRef('')
  const micUnavailable = currentUser?.transcription_available === false
  const inputRef = useRef(null)

  const micValue = useMemo(
    () =>
      [preMicTextRef.current, committedPhrases.join(' '), interimText]
        .filter(Boolean)
        .join(' '),
    [committedPhrases, interimText]
  )

  useEffect(() => {
    if (isRecording && inputRef.current) {
      inputRef.current.scrollTop = inputRef.current.scrollHeight
    }
  }, [micValue, isRecording])

  const handleMicToggle = async () => {
    if (isRecording) {
      const finalText = await stopMic()
      setInput([preMicTextRef.current, finalText].filter(Boolean).join(' '))
      inputRef.current?.focus()
    } else {
      preMicTextRef.current = input
      startMic()
    }
  }

  // -- Notes (handed to the page; this pane never writes them) --------------
  // The pane used to be the ONLY writer for note_append: the agent's
  // add_review_note tool returns {"saved": true, "_frontend_action": ...} and
  // persists nothing, so the pane's autosave was what actually stored the note.
  // That made deleting the notepad a data-loss change unless the write moved
  // somewhere, and the obvious move — GET the record, append, POST it back — is
  // the worst option available: the reviewer's own debounced save is in flight at
  // that moment, so the read is stale and one of the two writes loses.
  //
  // So the append goes to ReviewNotes, which already holds the live buffer and a
  // debounced flush. One writer, no read-modify-write window, and the hazard its
  // own header calls out ("consolidating to one owner is the real fix") is closed
  // rather than relocated.
  const requestNoteAppend = useCallback(
    (text) => {
      if (!text || !onNoteAppended) return
      onNoteAppended(text)
    },
    [onNoteAppended]
  )

  // -- Auto-scroll chat -----------------------------------------------------
  const totalLen = items.reduce((n, it) => n + (it.content?.length || 0), 0)
  useEffect(() => {
    const t = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, 60)
    return () => clearTimeout(t)
  }, [items.length, totalLen])

  const updateItem = (id, updater) =>
    setItems((prev) => prev.map((it) => (it.id === id ? updater(it) : it)))

  // -- Staged actions -------------------------------------------------------
  const handleAction = (action) => {
    if (!action || !action.type) return
    if (action.type === 'note_append') {
      requestNoteAppend(action.note_text || '')
      setItems((prev) => [
        ...prev,
        { id: nid(), kind: 'action', status: 'applied', action },
      ])
      return
    }
    // extraction_edit / verdict → a pending card for the reviewer to approve.
    const itemId = nid()
    setItems((prev) => [
      ...prev,
      { id: itemId, kind: 'action', status: 'pending', action },
    ])
    // Record it now, while it is being SHOWN. Recording on approval instead
    // would make the acceptance rate 100% by construction — a proposal the
    // reviewer ignores or rejects has to be in the denominator.
    recordProposal(itemId, action)
  }

  // Attaches the stored proposal's id to its card, so approving or rejecting
  // can report a disposition against it. A failure here is deliberately silent
  // in the UI: the measurement must never block the review.
  const recordProposal = useCallback(
    async (itemId, action) => {
      if (!documentId) return
      const isEdit = action.type === 'extraction_edit'
      try {
        const saved = await recordDocumentProposal(documentId, {
          // Empty when the agent volunteered the edit rather than answering a
          // pipeline finding; kept distinct so per-reason acceptance is not
          // polluted by unprompted suggestions.
          review_reason: action.review_reason || 'agent_initiated',
          resolution: action.resolution || 'needs_judgment',
          field_name: action.field_name || null,
          correction_key: isEdit ? action.correction_key : null,
          observed_value: action.observed_value || null,
          proposed_value: isEdit ? action.proposed_value : action.verdict,
          rationale: isEdit ? action.rationale : action.reasoning,
          candidates: action.candidates || [],
          model: action.model || null,
        })
        setItems((prev) =>
          prev.map((it) =>
            it.id === itemId ? { ...it, proposalId: saved.id } : it
          )
        )
      } catch (e) {
        console.warn('[LakeRCM] proposal not recorded:', e.message)
      }
    },
    [documentId]
  )

  // 409 means it was already dispositioned (a double-click, a replayed
  // request). That is the expected outcome of the second attempt, not a fault,
  // so it is not surfaced.
  const reportDisposition = useCallback(
    async (proposalId, disposition, humanValue = null) => {
      if (!proposalId) return
      try {
        await setProposalDisposition(proposalId, disposition, humanValue)
      } catch (e) {
        console.warn('[LakeRCM] disposition not recorded:', e.message)
      }
    },
    []
  )

  const approveAction = (id) => {
    setItems((prev) =>
      prev.map((it) => {
        if (it.id !== id || it.status !== 'pending') return it
        const a = it.action
        if (a.type === 'extraction_edit') {
          onApplyCorrection?.(a.correction_key, a.proposed_value)
        } else if (a.type === 'verdict') {
          onApplyVerdict?.({ verdict: a.verdict, reasoning: a.reasoning })
        }
        reportDisposition(it.proposalId, 'accepted')
        return { ...it, status: 'applied' }
      })
    )
  }

  const rejectAction = (id) =>
    setItems((prev) =>
      prev.map((it) => {
        if (it.id !== id || it.status !== 'pending') return it
        reportDisposition(it.proposalId, 'rejected')
        return { ...it, status: 'rejected' }
      })
    )

  // -- Send -----------------------------------------------------------------
  const handleSend = async (text) => {
    const messageText = (text ?? input).trim()
    if (!messageText || isLoading) return
    // A new turn's routing supersedes whatever the last one reported.
    setEffectiveRouting(null)

    const userItem = { id: nid(), kind: 'user', role: 'user', content: messageText }
    const asstId = nid()
    const asstItem = {
      id: asstId,
      kind: 'assistant',
      role: 'assistant',
      content: '',
      tool_calls: [],
    }
    setItems((prev) => [...prev, userItem, asstItem])
    setInput('')
    setIsLoading(true)
    setStreamingId(asstId)
    onActivityChange?.({ busy: true, tool: null })

    const controller = new AbortController()
    abortRef.current = controller

    try {
      let full = ''
      const tools = []
      for await (const ev of streamReviewerAgent(
        convIdRef.current,
        messageText,
        documentId,
        controller.signal,
        tierToWire(tier)
      )) {
        if (controller.signal.aborted) break
        if (ev.type === 'token') {
          full += ev.content
          updateItem(asstId, (m) => ({ ...m, content: full }))
        } else if (ev.type === 'tool_call') {
          tools.push({ ...ev, status: ev.status || 'running', startedAt: Date.now() })
          updateItem(asstId, (m) => ({ ...m, tool_calls: [...tools] }))
          onActivityChange?.({ busy: true, tool: ev.tool || null })
        } else if (ev.type === 'tool_result') {
          let idx = -1
          if (ev.run_id) idx = tools.findIndex((t) => t.run_id === ev.run_id)
          if (idx === -1)
            idx = tools.findIndex((t) => t.tool === ev.tool && t.status === 'running')
          if (idx !== -1) {
            tools[idx] = {
              ...tools[idx],
              status: 'completed',
              output_preview: ev.output_preview || null,
            }
            updateItem(asstId, (m) => ({ ...m, tool_calls: [...tools] }))
          }
          // Still busy -- the model is composing after the tool returned. Drop
          // the tool name so the strip stops naming work that has finished.
          onActivityChange?.({ busy: true, tool: null })
        } else if (ev.type === 'frontend_action') {
          handleAction(ev.action)
        } else if (ev.type === 'trace') {
          updateItem(asstId, (m) => ({ ...m, trace_id: ev.trace_id }))
        } else if (ev.type === 'routing') {
          // Which tier actually ran. Matters when it is not the one selected:
          // high-risk wording escalates regardless of the choice.
          setEffectiveRouting({ tier: ev.tier, source: ev.source })
          // Also reported up, so the reasoning strength is visible beside the
          // REVIEW FORM. The selector lives in this pane's toolbar, which is
          // invisible whenever the pane is closed -- and the decision the tier
          // affects is made over there.
          onActivityChange?.({ busy: true, tool: null, tier: ev.tier, tierSource: ev.source })
        } else if (ev.type === 'error') {
          updateItem(asstId, (m) => ({
            ...m,
            content: ev.content || 'Sorry, something went wrong. Try again.',
            error: true,
            retry_text: messageText,
          }))
        } else if (ev.type === 'done') {
          break
        }
      }
      updateItem(asstId, (m) => {
        if (!m.content?.trim() && (!m.tool_calls || m.tool_calls.length === 0)) {
          return { ...m, content: 'No response — please try again.' }
        }
        return m
      })
    } catch (err) {
      if (!controller.signal.aborted) {
        updateItem(asstId, (m) =>
          m.content?.trim()
            ? m
            : {
                ...m,
                content: 'Unable to reach the assistant. Please try again.',
                error: true,
                retry_text: messageText,
              }
        )
      }
    } finally {
      setIsLoading(false)
      setStreamingId(null)
      abortRef.current = null
      // In `finally`, so an abort, a stream error and a normal completion all
      // clear it. A busy flag that only clears on success is a spinner that
      // never stops.
      onActivityChange?.({ busy: false, tool: null })
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  useEffect(() => () => abortRef.current?.abort(), [])

  const isEmpty = items.length === 0

  return (
    <aside className="reviewer-assistant" aria-label="In-document assistant">
      {/* Drag to widen. The handle lives on the drawer's left edge but the width
          itself is owned by DocumentDetailPage, which writes --assistant-width —
          the drawer's own width AND the page's right gutter both already read
          that property, so one value moves both. */}
      <div
        className="ra-resizer"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize assistant panel"
        tabIndex={0}
        onMouseDown={onResizeStart}
        onKeyDown={onResizeKey}
      />

      <div className="ra-header">
        <div className="ra-header-title">
          <span className="ra-spark">✦</span> Assistant
        </div>
        <button className="ra-close" onClick={onClose} title="Close assistant" aria-label="Close">
          ✕
        </button>
      </div>

      {/* Replaces the Assistant/Notepad tab strip: the notepad is gone, so a
          two-tab control with one tab left would be chrome for nothing. The space
          goes to the tier selector instead. */}
      <div className="ra-toolbar">
        <AgentTierSelector
          compact
          value={tier}
          onChange={handleTierChange}
          effective={effectiveRouting}
          disabled={isLoading}
        />
      </div>

      <>
          <div className="ra-messages">
            {isEmpty ? (
              <div className="ra-empty">
                <p className="ra-empty-title">Ask about this document</p>
                <p className="ra-empty-sub">
                  I can review the extraction, propose corrections and a verdict for
                  your approval, and take notes.
                </p>
                <div className="ra-suggestions">
                  {EXAMPLE_ASKS.map((q) => (
                    <button key={q} className="ra-suggestion" onClick={() => handleSend(q)}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="ra-messages-list">
                {items.map((it) =>
                  it.kind === 'action' ? (
                    <StagedActionCard
                      key={it.id}
                      item={it}
                      fieldLabelForKey={fieldLabelForKey}
                      onApprove={() => approveAction(it.id)}
                      onReject={() => rejectAction(it.id)}
                    />
                  ) : (
                    <MessageBubble
                      key={it.id}
                      message={it}
                      isStreaming={streamingId === it.id}
                      workspaceHost={currentUser?.workspace_host}
                      experimentId={currentUser?.experiment_id}
                      onRetry={
                        it.error && it.retry_text
                          ? () => handleSend(it.retry_text)
                          : undefined
                      }
                    />
                  )
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          <div className="ra-input-area">
            <div className="ra-input-wrap">
              <textarea
                ref={inputRef}
                className="ra-input"
                value={isRecording ? micValue : input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about this document…"
                rows={1}
                disabled={isLoading}
                readOnly={isRecording}
              />
              {micSupported && !micUnavailable && (
                <MicButton
                  isRecording={isRecording}
                  onClick={handleMicToggle}
                  disabled={isLoading}
                  // Same title precedence as ChatInterface: a transcription error
                  // belongs on the control that produced it, not swallowed.
                  title={
                    micError || (isRecording ? 'Stop dictation' : 'Start dictation')
                  }
                />
              )}
              <button
                className={`ra-send ${
                  (isRecording ? micValue : input).trim() && !isLoading ? 'active' : ''
                }`}
                onClick={() => handleSend()}
                disabled={!(isRecording ? micValue : input).trim() || isLoading}
                aria-label="Send"
              >
                {isLoading ? <span className="ra-send-spinner" /> : '↑'}
              </button>
            </div>
            <p className="ra-disclaimer">
              Proposals are staged for your approval. Verify before submitting.
            </p>
          </div>
      </>
    </aside>
  )
}
