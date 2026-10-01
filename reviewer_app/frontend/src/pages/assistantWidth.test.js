import { describe, expect, it } from 'vitest'
import { clampAssistantWidth } from './DocumentDetailPage'

// jsdom has no layout engine, so the rendered width of the drawer cannot be
// asserted anywhere in this suite. The clamp is where every real bug in a
// resizable panel lives, so it is a pure function and tested directly: a stored
// width that survives onto a smaller screen, a drag past either bound, and a
// corrupted localStorage value are the three ways this breaks.

const WIDE = 1920
const LAPTOP = 1280

describe('clampAssistantWidth', () => {
  it('leaves a sensible width alone', () => {
    expect(clampAssistantWidth(500, WIDE)).toBe(500)
    expect(clampAssistantWidth(420, WIDE)).toBe(420)
  })

  it('holds the floor so the composer and mic stay usable', () => {
    expect(clampAssistantWidth(100, WIDE)).toBe(360)
    expect(clampAssistantWidth(0, WIDE)).toBe(360)
    expect(clampAssistantWidth(-500, WIDE)).toBe(360)
  })

  it('holds the ceiling at the width where the page stops reflowing', () => {
    expect(clampAssistantWidth(5000, WIDE)).toBe(720)
    expect(clampAssistantWidth(721, WIDE)).toBe(720)
  })

  it('lets 92vw win on a narrow viewport', () => {
    // 92% of 700 is 644, below the 720 ceiling — a width stored on a wide monitor
    // must not come back verbatim on a small window, or the drawer covers the
    // page and the drag handle is off-screen.
    expect(clampAssistantWidth(720, 700)).toBe(644)
    expect(clampAssistantWidth(700, LAPTOP)).toBe(700)
  })

  it('never returns less than the floor even when 92vw is smaller', () => {
    // Below ~640px CSS takes the drawer to 100vw, so the floor winning here is
    // inert rather than wrong — but it must not return something absurd.
    expect(clampAssistantWidth(400, 320)).toBe(360)
  })

  it('falls back to the default for a value that is not a number', () => {
    // localStorage is a string store; a hand-edited or half-written value must
    // not produce NaN and a drawer with no width at all.
    expect(clampAssistantWidth(NaN, WIDE)).toBe(420)
    expect(clampAssistantWidth(undefined, WIDE)).toBe(420)
    expect(clampAssistantWidth(null, WIDE)).toBe(420)
  })

  it('assumes the ceiling when the viewport is unknown', () => {
    expect(clampAssistantWidth(500, undefined)).toBe(500)
    expect(clampAssistantWidth(5000, undefined)).toBe(720)
  })
})
