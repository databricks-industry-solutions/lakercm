import { beforeEach, describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import ReviewSideTabs, { DEFAULT_HEIGHT, MAX_HEIGHT, MIN_HEIGHT } from './ReviewSideTabs'

// Notes and the knowledge graph used to be STACKED in the detail panel, which put
// the graph permanently below the fold on a laptop. They are alternatives, not a
// sequence, so they became tabs — and the section is resizable because a radial
// diagram and a markdown notepad want very different amounts of room.

const TABS = [
  { id: 'notes', label: 'Notes', node: <textarea aria-label="notepad" /> },
  { id: 'graph', label: 'Knowledge graph', node: <p>graph body</p> },
]

beforeEach(() => {
  window.localStorage.clear()
})

describe('ReviewSideTabs', () => {
  it('exposes a real ARIA tablist', () => {
    render(<ReviewSideTabs tabs={TABS} />)
    expect(screen.getByRole('tablist')).toBeTruthy()
    expect(screen.getAllByRole('tab')).toHaveLength(2)
    expect(screen.getByRole('tab', { name: 'Notes' }).getAttribute('aria-selected')).toBe('true')
  })

  it('switches panels on click', () => {
    render(<ReviewSideTabs tabs={TABS} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Knowledge graph' }))
    expect(screen.getByRole('tab', { name: 'Knowledge graph' }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByRole('tab', { name: 'Notes' }).getAttribute('aria-selected')).toBe('false')
  })

  it('keeps BOTH panels mounted, hiding the inactive one', () => {
    // Load-bearing: the notepad's debounced autosave lives in component state, so
    // unmounting it on a tab switch would drop whatever had not flushed yet.
    render(<ReviewSideTabs tabs={TABS} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Knowledge graph' }))
    const notepad = screen.getByLabelText('notepad')
    expect(notepad).toBeTruthy()
    expect(notepad.closest('[role="tabpanel"]').hasAttribute('hidden')).toBe(true)
  })

  it('moves between tabs with arrow keys', () => {
    render(<ReviewSideTabs tabs={TABS} />)
    const first = screen.getByRole('tab', { name: 'Notes' })
    fireEvent.keyDown(first, { key: 'ArrowRight' })
    expect(screen.getByRole('tab', { name: 'Knowledge graph' }).getAttribute('aria-selected')).toBe('true')
    // Wraps.
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Knowledge graph' }), { key: 'ArrowRight' })
    expect(first.getAttribute('aria-selected')).toBe('true')
  })

  it('remembers the chosen tab', () => {
    const { unmount } = render(<ReviewSideTabs tabs={TABS} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Knowledge graph' }))
    unmount()
    render(<ReviewSideTabs tabs={TABS} />)
    expect(screen.getByRole('tab', { name: 'Knowledge graph' }).getAttribute('aria-selected')).toBe('true')
  })

  it('falls back to the first tab when the stored id no longer exists', () => {
    // A renamed or removed tab must not leave the section showing nothing.
    window.localStorage.setItem('lakercm.reviewSideTab', 'a-tab-that-was-removed')
    render(<ReviewSideTabs tabs={TABS} />)
    expect(screen.getByRole('tab', { name: 'Notes' }).getAttribute('aria-selected')).toBe('true')
  })

  describe('resizing', () => {
    const handle = () => screen.getByRole('separator')

    it('is keyboard-operable, not pointer-only', () => {
      // A pane you cannot size without a pointer is a pane some reviewers cannot
      // size at all — the same reason the assistant's handle takes arrow keys.
      render(<ReviewSideTabs tabs={TABS} />)
      const h = handle()
      expect(Number(h.getAttribute('aria-valuenow'))).toBe(DEFAULT_HEIGHT)
      fireEvent.keyDown(h, { key: 'ArrowUp' })
      expect(Number(h.getAttribute('aria-valuenow'))).toBeGreaterThan(DEFAULT_HEIGHT)
      fireEvent.keyDown(h, { key: 'ArrowDown' })
      fireEvent.keyDown(h, { key: 'ArrowDown' })
      expect(Number(h.getAttribute('aria-valuenow'))).toBeLessThan(DEFAULT_HEIGHT)
    })

    it('clamps to its bounds', () => {
      render(<ReviewSideTabs tabs={TABS} />)
      const h = handle()
      fireEvent.keyDown(h, { key: 'Home' })
      expect(Number(h.getAttribute('aria-valuenow'))).toBe(MAX_HEIGHT)
      fireEvent.keyDown(h, { key: 'End' })
      expect(Number(h.getAttribute('aria-valuenow'))).toBe(MIN_HEIGHT)
      // Past the floor it stays at the floor rather than collapsing to nothing.
      for (let i = 0; i < 20; i += 1) fireEvent.keyDown(h, { key: 'ArrowDown' })
      expect(Number(h.getAttribute('aria-valuenow'))).toBe(MIN_HEIGHT)
    })

    it('resets to the default on Enter', () => {
      render(<ReviewSideTabs tabs={TABS} />)
      const h = handle()
      fireEvent.keyDown(h, { key: 'Home' })
      fireEvent.keyDown(h, { key: 'Enter' })
      expect(Number(h.getAttribute('aria-valuenow'))).toBe(DEFAULT_HEIGHT)
    })

    it('remembers the height', () => {
      const { unmount } = render(<ReviewSideTabs tabs={TABS} />)
      fireEvent.keyDown(handle(), { key: 'Home' })
      unmount()
      render(<ReviewSideTabs tabs={TABS} />)
      expect(Number(handle().getAttribute('aria-valuenow'))).toBe(MAX_HEIGHT)
    })

    it('grows when dragged UP', () => {
      // The handle is on the TOP edge, so the delta is inverted. Getting this
      // backwards makes the section shrink as you pull it open.
      render(<ReviewSideTabs tabs={TABS} />)
      const h = handle()
      fireEvent.mouseDown(h, { clientY: 500 })
      fireEvent.mouseMove(window, { clientY: 420 })
      fireEvent.mouseUp(window)
      expect(Number(h.getAttribute('aria-valuenow'))).toBe(DEFAULT_HEIGHT + 80)
    })
  })
})
