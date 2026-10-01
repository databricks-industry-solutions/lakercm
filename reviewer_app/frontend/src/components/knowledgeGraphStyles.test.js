import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

// jsdom does not apply stylesheets (vitest runs with css: false), so the two
// stylesheet bugs this panel shipped with are pinned by reading the CSS itself.

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8')
const KG = read('./KnowledgeGraphPanel.css')
const DESIGN_SYSTEM = read('../design-system.css')

// Custom properties the panel sets inline from JSX rather than in a stylesheet.
const SET_INLINE = new Set(['--i', '--kg-c'])

const declared = (css) =>
  new Set([...css.matchAll(/(--[a-z0-9-]+)\s*:/gi)].map((m) => m[1]))

describe('KnowledgeGraphPanel.css', () => {
  it('resolves every var() to a token that exists', () => {
    // A var() whose token is defined nowhere renders its fallback hex and never
    // follows the palette. The first version of this file did that with every
    // colour (--gray-*, --blue-*, --danger-*): it drew in Tailwind blue and grey,
    // which the app does not use, and ignored dark mode.
    const known = new Set([...declared(DESIGN_SYSTEM), ...declared(KG), ...SET_INLINE])
    const used = new Set([...KG.matchAll(/var\(\s*(--[a-z0-9-]+)/gi)].map((m) => m[1]))
    expect([...used].filter((token) => !known.has(token))).toEqual([])
  })

  it('never lets the entrance animation outlive itself', () => {
    // fill-mode `both`/`forwards` keeps the END keyframe applied after the
    // animation, and an animation outranks normal rules. It pinned every node at
    // opacity 1, so engaging one dimmed only the lines and the highlight barely
    // showed.
    const animations = [...KG.matchAll(/animation\s*:\s*([^;]+);/g)].map((m) => m[1])
    expect(animations.length).toBeGreaterThan(0)
    for (const a of animations) expect(a).not.toMatch(/\b(both|forwards)\b/)
    expect(KG).not.toMatch(/animation-fill-mode\s*:\s*(both|forwards)/)
  })
})
