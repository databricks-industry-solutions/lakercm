import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getDocumentNotes, saveDocumentNotes } from '../api/lakeRcmApi'
import './ReviewNotes.css'

// Reviewer notes, beside the verdict rather than buried in the assistant drawer.
//
// Markdown, because notes on a clinical document are structured in practice —
// "checked against p2", a list of codes queried, a link to the payer policy — and
// a plain textarea renders all of that as one grey paragraph. Same renderer the
// chat transcript uses (react-markdown + remark-gfm), so tables and task lists
// work and no new dependency is involved.
//
// Per (document, reviewer) and autosaved: the note is scratch reasoning attached
// to THIS document, and a reviewer who has typed three lines of justification
// should never be the one responsible for pressing save. Persistence and the
// debounce interval are the same as the assistant notepad's, deliberately — this
// reads and writes the same /documents/{id}/notes record, so the two surfaces show
// the same text.
//
// KNOWN HAZARD, and why the reload-on-focus exists. That shared record currently
// has TWO editors: this pane and the assistant's notepad tab, which the assistant
// can also append to via a note_append action. Two debounced autosavers over one
// record is last-writer-wins, so an append made while this pane holds stale text
// would be overwritten by this pane's next flush. Reloading whenever the pane
// regains focus closes the common ordering (edit there, then come back here)
// without a refactor. Consolidating to one owner is the real fix and wants its own
// change — this component is deliberately written so that owner can be this one.

const AUTOSAVE_MS = 1200

const PLACEHOLDER = `What did you check, and against what?

- Markdown works: **bold**, lists, \`codes\`
- Saved automatically, private to you
- Kept with this document`

/**
 * @param {string} documentId
 * @param {{id: number, text: string}|null} appendRequest
 *   An append the assistant proposed, handed down by DocumentDetailPage. Applied
 *   ON TOP of the live buffer rather than fetched-and-merged, because this
 *   component is the only writer of the record and already holds whatever the
 *   reviewer has typed but not yet saved. `id` makes the request idempotent
 *   across the re-renders a parent state change causes.
 */
function ReviewNotes({ documentId, appendRequest = null }) {
  const [text, setText] = useState('')
  const [status, setStatus] = useState('idle') // idle|saving|saved|error
  const [preview, setPreview] = useState(false)
  const timer = useRef(null)
  const latest = useRef('')
  const loadedFor = useRef(null)
  const appliedAppendId = useRef(null)

  const flush = useCallback(
    (value) => {
      if (!documentId) return
      setStatus('saving')
      saveDocumentNotes(documentId, value)
        .then(() => setStatus('saved'))
        .catch(() => setStatus('error'))
    },
    [documentId]
  )

  const schedule = useCallback(
    (value) => {
      latest.current = value
      if (timer.current) clearTimeout(timer.current)
      timer.current = setTimeout(() => flush(value), AUTOSAVE_MS)
    },
    [flush]
  )

  const load = useCallback(() => {
    if (!documentId) return
    getDocumentNotes(documentId)
      .then((r) => {
        const loaded = r?.note_text || ''
        // Never clobber unsaved local edits with a reload: if what we have
        // locally differs from the server AND a save is still pending, the local
        // copy is the newer one.
        if (timer.current) return
        setText(loaded)
        latest.current = loaded
        loadedFor.current = documentId
      })
      .catch(() => {
        /* GET always 200s with {note_text: ''}; a transport failure leaves the
           pane empty rather than erroring, and the next focus retries. */
      })
  }, [documentId])

  // Apply an assistant-proposed append. Deliberately additive to `latest.current`
  // (the live buffer) and routed through the existing debounce, so a note that
  // arrives while the reviewer is mid-sentence is merged rather than racing them
  // — which is exactly what a GET-append-POST in the assistant pane would have
  // done. Guarded by id: a parent re-render must not append twice.
  useEffect(() => {
    if (!appendRequest || !appendRequest.text) return
    if (appliedAppendId.current === appendRequest.id) return
    appliedAppendId.current = appendRequest.id
    const current = latest.current || ''
    const next = current ? `${current}\n${appendRequest.text}` : appendRequest.text
    setText(next)
    schedule(next)
  }, [appendRequest, schedule])

  // Load on document change, and flush anything pending before switching away so
  // a verdict-then-next-document keystroke sequence cannot lose the last edit.
  useEffect(() => {
    setText('')
    setStatus('idle')
    latest.current = ''
    load()
    return () => {
      if (timer.current) {
        clearTimeout(timer.current)
        timer.current = null
        if (documentId) saveDocumentNotes(documentId, latest.current).catch(() => {})
      }
    }
  }, [documentId, load])

  // See the KNOWN HAZARD note above.
  useEffect(() => {
    const onFocus = () => load()
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [load])

  const onChange = (e) => {
    const value = e.target.value
    setText(value)
    setStatus('idle')
    schedule(value)
  }

  const statusLabel = {
    idle: '',
    saving: 'Saving…',
    saved: 'Saved',
    error: 'Not saved — retrying on the next edit',
  }[status]

  return (
    <section className="rn-panel" aria-label="Reviewer notes">
      <header className="rn-head">
        <h3 className="rn-title">Notes</h3>
        <span className="rn-scope">private to you</span>

        <div className="rn-head-right">
          <span
            className={`rn-status rn-status--${status}`}
            role="status"
            aria-live="polite"
          >
            {statusLabel}
          </span>
          <div className="rn-toggle" role="group" aria-label="Notes view">
            <button
              type="button"
              className={`rn-toggle-btn ${!preview ? 'is-active' : ''}`}
              onClick={() => setPreview(false)}
            >
              Write
            </button>
            <button
              type="button"
              className={`rn-toggle-btn ${preview ? 'is-active' : ''}`}
              onClick={() => setPreview(true)}
              disabled={!text.trim()}
              title={!text.trim() ? 'Nothing to preview yet' : 'Render markdown'}
            >
              Preview
            </button>
          </div>
        </div>
      </header>

      {preview ? (
        <div className="rn-preview">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        </div>
      ) : (
        <textarea
          className="rn-textarea"
          value={text}
          onChange={onChange}
          placeholder={PLACEHOLDER}
          spellCheck
          aria-label="Reviewer notes, markdown"
        />
      )}
    </section>
  )
}

export default ReviewNotes
