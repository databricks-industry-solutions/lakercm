import { useCallback, useEffect, useRef, useState } from 'react'

import {
  beaconReviewDraft,
  discardReviewDraft,
  getReviewDraft,
  saveReviewDraft,
} from '../api/lakeRcmApi'

// Same cadence as the reviewer notepad (ReviewNotes.jsx). Deliberately NOT the
// 150-300ms figure quoted for input debouncing: that guidance is about
// search-as-you-type and resize handlers, not about debouncing a network write.
// At 300ms a reasoning textarea issues roughly 4x the writes for no visible
// gain, and a panel that saved on a different rhythm from the notepad sitting
// directly below it would just read as inconsistent.
//
// Responsiveness comes from flushing DISCRETE changes immediately instead
// (save(next, { immediate: true }) — see ReviewPanel's verdict handler): a radio
// selection is one event, not a keystroke stream, so there is nothing to
// coalesce and debouncing it only widens the window where it can be lost.
export const AUTOSAVE_MS = 1200

/** Stable serialisation, so key insertion order cannot look like a change. */
const canonical = (draft) => {
  const corrections = draft?.corrections || {}
  const keys = Object.keys(corrections).sort()
  return JSON.stringify({
    verdict: draft?.verdict || null,
    reasoning: draft?.reasoning || null,
    corrections: keys.map((k) => [k, corrections[k]]),
  })
}

const isEmpty = (draft) =>
  !draft?.verdict &&
  !(draft?.reasoning || '').trim() &&
  Object.keys(draft?.corrections || {}).length === 0

/**
 * Autosave for the UNSUBMITTED review: verdict, reasoning and field corrections.
 *
 * A draft is explicitly not a review. Saving one submits nothing, and it is
 * deleted server-side the moment a real review is submitted.
 *
 * @param {string} documentId
 * @returns {{
 *   status: 'idle'|'saving'|'saved'|'error',
 *   draft: object|null,   // what loaded for this document, or null
 *   loaded: boolean,      // has the initial GET resolved for THIS document
 *   save: (draft: object, opts?: {immediate?: boolean}) => void,
 *   suspend: () => void,  // stop autosaving (hold during submit)
 *   discard: () => Promise<void>,
 * }}
 */
export function useReviewDraft(documentId) {
  const [status, setStatus] = useState('idle')
  const [draft, setDraft] = useState(null)
  const [loaded, setLoaded] = useState(false)

  const timer = useRef(null)
  const latest = useRef(null)
  // Which document the initial GET has resolved for. Autosave is gated on this:
  // see the guard in save().
  const loadedFor = useRef(null)
  // Last payload the server has (loaded or saved). Skipping an identical write
  // stops the load itself from echoing straight back as a save.
  const lastSaved = useRef(null)
  const suspended = useRef(false)

  const flush = useCallback(
    (value) => {
      if (!documentId) return
      setStatus('saving')
      lastSaved.current = canonical(value)
      saveReviewDraft(documentId, value)
        .then(() => setStatus('saved'))
        .catch(() => {
          // Let the next edit retry: a draft the server rejected must not be
          // remembered as saved, or the reviewer sees "Saved" over lost work.
          lastSaved.current = null
          setStatus('error')
        })
    },
    [documentId]
  )

  const save = useCallback(
    (next, { immediate = false } = {}) => {
      if (!documentId) return
      if (suspended.current) return
      // THE EMPTY-CLOBBER GUARD. ReviewPanel resets verdict/reasoning to '' at
      // the top of its load effect, before the async GET resolves. Without this,
      // that reset autosaves an empty draft over a good one and the reviewer's
      // work is destroyed by the very feature meant to preserve it.
      if (loadedFor.current !== documentId) return
      // Nothing changed (commonly: the load pushing its own values back down).
      if (canonical(next) === lastSaved.current) return
      // Never create a row just to say "nothing here". An empty draft with no
      // server-side counterpart is pure noise in the queue's draft badges.
      if (isEmpty(next) && !lastSaved.current) return

      latest.current = next
      if (timer.current) {
        clearTimeout(timer.current)
        timer.current = null
      }
      if (immediate) {
        flush(next)
        return
      }
      timer.current = setTimeout(() => {
        timer.current = null
        flush(next)
      }, AUTOSAVE_MS)
    },
    [documentId, flush]
  )

  const load = useCallback(() => {
    if (!documentId) return
    getReviewDraft(documentId)
      .then((r) => {
        // Do not clobber unsaved local edits with a reload: a pending timer
        // means what we hold locally is the newer copy.
        if (timer.current) return
        const next = r?.exists
          ? {
              verdict: r.verdict || '',
              reasoning: r.reasoning || '',
              corrections: r.corrections || {},
            }
          : null
        setDraft(next)
        latest.current = next
        lastSaved.current = next ? canonical(next) : null
        loadedFor.current = documentId
        setLoaded(true)
      })
      .catch(() => {
        // The GET always 200s when the app is reachable, so this is a transport
        // failure. Mark the document loaded anyway: refusing to ever autosave is
        // worse than autosaving against an unknown baseline, and a genuinely
        // empty form still cannot create a row (see isEmpty above).
        loadedFor.current = documentId
        setLoaded(true)
      })
  }, [documentId])

  const suspend = useCallback(() => {
    suspended.current = true
    if (timer.current) {
      clearTimeout(timer.current)
      timer.current = null
    }
  }, [])

  const discard = useCallback(async () => {
    suspend()
    setDraft(null)
    latest.current = null
    lastSaved.current = null
    setStatus('idle')
    try {
      await discardReviewDraft(documentId)
    } catch {
      /* The submit already succeeded; a stale draft row is cosmetic. */
    }
  }, [documentId, suspend])

  // Load on document change, and flush anything pending before switching away,
  // so pressing `j` mid-sentence cannot lose the last edit.
  useEffect(() => {
    setStatus('idle')
    setDraft(null)
    setLoaded(false)
    latest.current = null
    lastSaved.current = null
    loadedFor.current = null
    suspended.current = false
    load()
    const id = documentId
    return () => {
      if (timer.current) {
        clearTimeout(timer.current)
        timer.current = null
        if (id && latest.current) {
          saveReviewDraft(id, latest.current).catch(() => {})
        }
      }
    }
  }, [documentId, load])

  // Flush on the page being hidden — tab switch, app switch, navigation away.
  //
  // visibilitychange + pagehide, never beforeunload: beforeunload disqualifies
  // the page from the back/forward cache and no longer fires reliably, and an
  // in-flight XHR is cancelled when the page goes away. sendBeacon is queued by
  // the browser and survives.
  useEffect(() => {
    const flushPending = () => {
      if (!timer.current || !documentId || !latest.current) return
      clearTimeout(timer.current)
      timer.current = null
      const value = latest.current
      lastSaved.current = canonical(value)
      if (!beaconReviewDraft(documentId, value)) {
        // Beacon refused (unsupported, or over its size cap). A keepalive-less
        // request may still be cancelled, but trying beats dropping the edit.
        saveReviewDraft(documentId, value).catch(() => {})
      }
    }
    const onVisibility = () => {
      if (document.visibilityState === 'hidden') flushPending()
    }
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('pagehide', flushPending)
    return () => {
      document.removeEventListener('visibilitychange', onVisibility)
      window.removeEventListener('pagehide', flushPending)
    }
  }, [documentId])

  // Reload when the window regains focus, so a draft edited in another tab is
  // picked up. Same hazard and same mitigation as ReviewNotes.
  useEffect(() => {
    const onFocus = () => load()
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [load])

  return { status, draft, loaded, save, suspend, discard }
}

export default useReviewDraft
