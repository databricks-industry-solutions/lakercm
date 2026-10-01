import { useCallback, useEffect, useRef, useState } from 'react'

import './ReviewSideTabs.css'

const LS_TAB = 'lakercm.reviewSideTab'
const LS_HEIGHT = 'lakercm.reviewSideHeight'

export const MIN_HEIGHT = 160
export const MAX_HEIGHT = 720
export const DEFAULT_HEIGHT = 300

const clamp = (h) => Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, Math.round(h)))

const readTab = (tabs) => {
  if (typeof window === 'undefined') return tabs[0].id
  const raw = window.localStorage.getItem(LS_TAB)
  return tabs.some((t) => t.id === raw) ? raw : tabs[0].id
}

const readHeight = () => {
  if (typeof window === 'undefined') return DEFAULT_HEIGHT
  const raw = Number(window.localStorage.getItem(LS_HEIGHT))
  return Number.isFinite(raw) && raw > 0 ? clamp(raw) : DEFAULT_HEIGHT
}

/**
 * Tabbed, resizable home for the reviewer's notepad and the knowledge graph.
 *
 * WHY TABS. Both were stacked in the detail panel, below the review form, which
 * put the graph permanently below the fold on a laptop -- it had to be scrolled
 * to and then collapsed again to reach the notes. They are alternatives, not a
 * sequence: a reviewer is either writing up what they checked or looking at what
 * the document connects to.
 *
 * WHY RESIZABLE. A radial diagram and a markdown notepad want very different
 * amounts of room, and which one matters is per-reviewer and per-document. The
 * height is persisted, like the assistant's width.
 *
 * Children are kept MOUNTED and hidden with `hidden`, not unmounted: the
 * notepad's debounced autosave lives in component state, and unmounting it on a
 * tab switch would drop whatever had not flushed yet.
 *
 * @param {{id: string, label: string, node: React.ReactNode}[]} tabs
 */
export default function ReviewSideTabs({ tabs }) {
  const [activeId, setActiveId] = useState(() => readTab(tabs))
  const [height, setHeight] = useState(readHeight)
  const [dragging, setDragging] = useState(false)
  const bodyRef = useRef(null)

  useEffect(() => {
    try {
      window.localStorage.setItem(LS_TAB, activeId)
    } catch {
      /* private mode must not break the panel */
    }
  }, [activeId])

  useEffect(() => {
    try {
      window.localStorage.setItem(LS_HEIGHT, String(height))
    } catch {
      /* as above */
    }
  }, [height])

  // Dragging UP grows the section, which is why the delta is inverted: the handle
  // sits on its TOP edge.
  const onDragStart = useCallback(
    (e) => {
      e.preventDefault()
      const startY = e.clientY
      const startH = height
      setDragging(true)

      const onMove = (ev) => setHeight(clamp(startH + (startY - ev.clientY)))
      const onUp = () => {
        window.removeEventListener('mousemove', onMove)
        window.removeEventListener('mouseup', onUp)
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
        setDragging(false)
      }
      document.body.style.cursor = 'row-resize'
      document.body.style.userSelect = 'none'
      window.addEventListener('mousemove', onMove)
      window.addEventListener('mouseup', onUp)
    },
    [height]
  )

  // Keyboard-operable, like the assistant's handle: a pane you cannot size
  // without a pointer is a pane some reviewers cannot size at all.
  const onDragKey = useCallback((e) => {
    const step = e.shiftKey ? 80 : 20
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHeight((h) => clamp(h + step))
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHeight((h) => clamp(h - step))
    } else if (e.key === 'Home') {
      e.preventDefault()
      setHeight(MAX_HEIGHT)
    } else if (e.key === 'End') {
      e.preventDefault()
      setHeight(MIN_HEIGHT)
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      setHeight(DEFAULT_HEIGHT)
    }
  }, [])

  // Roving arrow keys across the tab strip, per the ARIA tabs pattern.
  const onTabKey = useCallback(
    (e, idx) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      e.preventDefault()
      const dir = e.key === 'ArrowRight' ? 1 : -1
      const next = tabs[(idx + dir + tabs.length) % tabs.length]
      setActiveId(next.id)
      document.getElementById(`rst-tab-${next.id}`)?.focus()
    },
    [tabs]
  )

  return (
    <section className={`rst${dragging ? ' is-dragging' : ''}`}>
      <div
        className="rst-handle"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize notes and knowledge graph"
        aria-valuenow={height}
        aria-valuemin={MIN_HEIGHT}
        aria-valuemax={MAX_HEIGHT}
        tabIndex={0}
        onMouseDown={onDragStart}
        onKeyDown={onDragKey}
        title="Drag to resize — arrow keys also work"
      />

      <div className="rst-tabs" role="tablist" aria-label="Notes and knowledge graph">
        {tabs.map((t, i) => (
          <button
            key={t.id}
            id={`rst-tab-${t.id}`}
            role="tab"
            type="button"
            aria-selected={activeId === t.id}
            aria-controls={`rst-panel-${t.id}`}
            tabIndex={activeId === t.id ? 0 : -1}
            className={`rst-tab${activeId === t.id ? ' is-active' : ''}`}
            onClick={() => setActiveId(t.id)}
            onKeyDown={(e) => onTabKey(e, i)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="rst-body" ref={bodyRef} style={{ height: `${height}px` }}>
        {tabs.map((t) => (
          <div
            key={t.id}
            id={`rst-panel-${t.id}`}
            role="tabpanel"
            aria-labelledby={`rst-tab-${t.id}`}
            className="rst-panel"
            hidden={activeId !== t.id}
          >
            {t.node}
          </div>
        ))}
      </div>
    </section>
  )
}
