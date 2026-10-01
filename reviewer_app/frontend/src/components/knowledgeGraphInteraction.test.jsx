import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import KnowledgeGraphPanel from './KnowledgeGraphPanel'

vi.mock('../api/lakeRcmApi', () => ({
  getDocumentKgNeighbourhood: vi.fn(),
}))
const { getDocumentKgNeighbourhood } = await import('../api/lakeRcmApi')

const PATIENT_DOCS = Array.from({ length: 9 }, (_, i) => ({
  name: `synthetic-1-00${String(i + 31).padStart(2, '0')}-referral.pdf`,
  // One document the graph knows but the review queue does not yet.
  id: i === 1 ? null : `id-${i + 31}`,
  status: i === 0 ? 'pending' : 'auto_verified',
  label: 'referral_workqueue',
}))

const READY = {
  available: true,
  node: 'synthetic-1-0027-prior-authorization.pdf',
  node_type: 'prior_authorization',
  edges: [
    { edge: 'billedTo', label: 'payer', neighbour: 'oakhollow', direction: 'out' },
    { edge: 'hasDiagnosis', label: 'diagnosis', neighbour: 'M54.50', direction: 'out' },
    { edge: 'documentsPatient', label: 'patient', neighbour: '837274cc5d1a635f', direction: 'out' },
  ],
  siblings: PATIENT_DOCS.map((d) => d.name),
  details: {
    oakhollow: { cls: 'Payer', label: 'Oakhollow', props: {} },
    'M54.50': {
      cls: 'DiagnosisCode',
      label: 'M54.50',
      props: {
        description: 'Low back pain, unspecified',
        category: 'Musculoskeletal',
        is_billable: 'true',
      },
    },
    // Deliberately absent: 837274cc5d1a635f (the patient) has no stored detail,
    // which the panel must say rather than render an empty card.
  },
  linked: {
    'documentsPatient:837274cc5d1a635f': { total: 9, documents: PATIENT_DOCS },
    'hasDiagnosis:M54.50': {
      total: 40,
      documents: [{ name: 'synthetic-1-0040-denial.pdf', id: 'id-40', status: 'pending', label: 'denial_management' }],
    },
    // The payer shares nothing: the card must say so, not show an empty list.
  },
}

const nodeFor = (container, text) =>
  [...container.querySelectorAll('.kg-node')].find((n) =>
    n.getAttribute('aria-label')?.toLowerCase().includes(text.toLowerCase())
  )

const ready = async (props = {}) => {
  const utils = render(<KnowledgeGraphPanel documentId="d1" {...props} />)
  await waitFor(() => expect(utils.container.querySelectorAll('.kg-node')).toHaveLength(3))
  return utils
}

beforeEach(() => {
  vi.clearAllMocks()
  getDocumentKgNeighbourhood.mockResolvedValue(READY)
})

describe('KnowledgeGraphPanel', () => {
  it('draws one spoke per edge plus a hub', async () => {
    const { container } = await ready()
    expect(container.querySelectorAll('.kg-dot')).toHaveLength(3)
    expect(container.querySelectorAll('.kg-edge')).toHaveLength(3)
    expect(container.querySelectorAll('.kg-hub')).toHaveLength(1)
  })

  it('names the hub by what the document is, not as an anonymous dot', async () => {
    // The one node the diagram is about was the only one without a label.
    const { container } = await ready()
    const hub = container.querySelector('.kg-hub')
    expect(hub.textContent).toContain('prior authorization')
    expect(hub.getAttribute('title')).toBe(READY.node)
  })

  it('puts the patient spoke first, so it is the one the eye lands on', async () => {
    const { container } = await ready()
    const first = container.querySelector('.kg-node')
    expect(first.getAttribute('class')).toContain('kg-node--documentsPatient')
  })

  it('labels a node with what it means, not only its code', async () => {
    const { container } = await ready()
    const dx = nodeFor(container, 'M54.50')
    expect(dx.querySelector('.kg-label-main').textContent).toBe('M54.50')
    expect(dx.querySelector('.kg-label-sub').textContent).toMatch(/Low back pain/)
    // A patient has no description; it says how many documents it ties together.
    const patient = nodeFor(container, '837274cc5d1a635f')
    expect(patient.querySelector('.kg-label-sub').textContent).toBe('10 documents')
  })

  it('reads out the full value on hover, since ring labels are truncated', async () => {
    const { container } = await ready()
    // Untouched: a summary, not a value.
    expect(screen.getByText(/3 connections/i)).toBeTruthy()
    fireEvent.mouseEnter(nodeFor(container, 'oakhollow'))
    // Scoped to the card: the value also appears as the ring label, and that
    // ambiguity is the point — the label is truncated, the card is not.
    const card = container.querySelector('.kg-detail')
    expect(card.querySelector('.kg-detail-value').textContent).toBe('Oakhollow')
    expect(card.querySelector('.kg-detail-cls').textContent).toBe('Payer')
  })

  it('dims the rest while one spoke is engaged', async () => {
    // Without this a 12-node ring is unreadable when you are chasing one code.
    // (The CSS half -- that the dimming is not overridden -- is pinned in
    // knowledgeGraphStyles.test.js; jsdom does not apply stylesheets.)
    const { container } = await ready()
    expect(container.querySelector('.kg-stage').getAttribute('class')).not.toContain('is-focused')
    fireEvent.mouseEnter(container.querySelector('.kg-node'))
    expect(container.querySelector('.kg-stage').getAttribute('class')).toContain('is-focused')
  })

  it('is operable by keyboard, not pointer only', async () => {
    // Real buttons, so Tab reaches them and Enter/Space activate them natively.
    const { container } = await ready()
    const node = container.querySelector('.kg-node')
    expect(node.tagName).toBe('BUTTON')
    fireEvent.focus(node)
    expect(container.querySelector('.kg-stage').getAttribute('class')).toContain('is-focused')
    // Activating it pins the selection so it survives focus moving on.
    fireEvent.click(node)
    fireEvent.blur(node)
    expect(node.getAttribute('aria-pressed')).toBe('true')
    expect(container.querySelector('.kg-node.is-selected')).toBeTruthy()
  })

  it('unpins on Escape without letting Escape leave the page', async () => {
    // Escape on this page means "back to the documents".
    const { container } = await ready()
    const node = container.querySelector('.kg-node')
    fireEvent.click(node)
    const onWindowKey = vi.fn()
    window.addEventListener('keydown', onWindowKey)
    fireEvent.keyDown(node, { key: 'Escape' })
    window.removeEventListener('keydown', onWindowKey)
    expect(node.getAttribute('aria-pressed')).toBe('false')
    expect(onWindowKey).not.toHaveBeenCalled()
  })

  it('explains WHAT a node is, not just which one it is', async () => {
    // The ring shows "M54.50"; on its own that tells a reviewer which node, not
    // that it means low back pain.
    const { container } = await ready()
    fireEvent.mouseEnter(nodeFor(container, 'M54.50'))
    const card = container.querySelector('.kg-detail')
    expect(card.textContent).toMatch(/ICD-10-CM diagnosis code/i)
    expect(card.textContent).toContain('Low back pain, unspecified')
    expect(card.textContent).toContain('Musculoskeletal')
  })

  it('renders property names as prose and booleans as Yes/No', async () => {
    const { container } = await ready()
    fireEvent.mouseEnter(nodeFor(container, 'M54.50'))
    const props = container.querySelector('.kg-detail-props')
    expect(props.textContent).toContain('Billable')
    expect(props.textContent).toContain('Yes')
    expect(props.textContent).not.toContain('is_billable')
    expect(props.textContent).not.toContain('true')
  })

  it('says so when an entity has no stored detail', async () => {
    // "Nothing is stored about this" and "the graph is stale" are different
    // claims; an empty card would imply the second.
    const { container } = await ready()
    fireEvent.mouseEnter(nodeFor(container, '837274cc5d1a635f'))
    expect(container.querySelector('.kg-detail').textContent).toMatch(
      /No further detail is stored/i
    )
  })

  it('invites the click instead of leaving the card blank', async () => {
    const { container } = await ready()
    expect(container.querySelector('.kg-detail').textContent).toMatch(
      /select a node to see what it is and which other documents share it/i
    )
  })

  it('names how many other documents share a node, and lists them', async () => {
    // The question the user asked of the graph: click a node, see the documents
    // it is associated with.
    const { container } = await ready()
    fireEvent.mouseEnter(nodeFor(container, '837274cc5d1a635f'))
    const card = container.querySelector('.kg-detail')
    expect(card.querySelector('.kg-linked-title').textContent).toBe(
      '9 other documents for this patient'
    )
    expect(card.querySelectorAll('.kg-linked-doc')).toHaveLength(9)
    // Hover previews; the card says how to pin it, since the pointer cannot
    // reach the list without leaving the node.
    expect(card.textContent).toMatch(/Click the node to pin it/)
  })

  it('says how many it did not list when the cap applied', async () => {
    const { container } = await ready()
    fireEvent.mouseEnter(nodeFor(container, 'M54.50'))
    const card = container.querySelector('.kg-detail')
    expect(card.querySelector('.kg-linked-title').textContent).toBe(
      '40 other documents with this diagnosis'
    )
    expect(card.textContent).toContain('+39 more not listed')
  })

  it('says so when no other document shares a node', async () => {
    const { container } = await ready()
    fireEvent.mouseEnter(nodeFor(container, 'oakhollow'))
    expect(container.querySelector('.kg-linked-title').textContent).toBe(
      'No other documents billed to this payer.'
    )
    expect(container.querySelector('.kg-linked-doc')).toBeNull()
  })

  it('opens a linked document by its id, auto-verified ones included', async () => {
    // The id comes from the API. The old path matched a NAME against the cached
    // j/k page, which excludes auto-verified documents, so most links did nothing.
    const onOpenDocument = vi.fn()
    const { container } = await ready({ onOpenDocument })
    fireEvent.click(nodeFor(container, '837274cc5d1a635f'))
    const rows = container.querySelectorAll('.kg-linked-doc')
    fireEvent.click(rows[2])
    expect(onOpenDocument).toHaveBeenCalledWith({
      id: 'id-33',
      name: PATIENT_DOCS[2].name,
    })
    expect(rows[2].textContent).toContain('Auto-Verified')
  })

  it('does not offer to open a document the review queue does not have yet', async () => {
    const onOpenDocument = vi.fn()
    const { container } = await ready({ onOpenDocument })
    fireEvent.click(nodeFor(container, '837274cc5d1a635f'))
    const row = container.querySelectorAll('.kg-linked-doc')[1]
    expect(row.disabled).toBe(true)
    fireEvent.click(row)
    expect(onOpenDocument).not.toHaveBeenCalled()
  })

  it('draws a pinned node’s documents on the graph, numbered like the list', async () => {
    const onOpenDocument = vi.fn()
    const { container } = await ready({ onOpenDocument })
    // Hover alone does not redraw the graph: that would jump as the pointer
    // crosses the ring.
    fireEvent.mouseEnter(nodeFor(container, '837274cc5d1a635f'))
    expect(container.querySelectorAll('.kg-doc')).toHaveLength(0)

    fireEvent.click(nodeFor(container, '837274cc5d1a635f'))
    const tiles = [...container.querySelectorAll('.kg-doc')]
    // Nine documents, eight places: seven tiles and a "+2".
    expect(tiles).toHaveLength(8)
    expect(tiles.slice(0, 7).map((t) => t.textContent)).toEqual(['1', '2', '3', '4', '5', '6', '7'])
    expect(tiles[7].textContent).toBe('+2')
    expect(container.querySelectorAll('.kg-fan-edge')).toHaveLength(8)

    fireEvent.click(tiles[0])
    expect(onOpenDocument).toHaveBeenCalledWith({ id: 'id-31', name: PATIENT_DOCS[0].name })
  })

  it('numbers the tiles in reading order on every side of the hub', async () => {
    // Four spokes put one node at each compass point. Walking the arc the same
    // way for all of them read 1..3 above the hub and 3..1 below it.
    const docs = (p) => ({
      total: 3,
      documents: [1, 2, 3].map((n) => ({ name: `${p}-${n}.pdf`, id: `${p}${n}` })),
    })
    const edges = [
      ['documentsPatient', 'pk'],
      ['billedTo', 'payer'],
      ['deniedFor', '197'],
      ['hasDiagnosis', 'M54.50'],
    ]
    getDocumentKgNeighbourhood.mockResolvedValue({
      available: true,
      node: 'x.pdf',
      edges: edges.map(([edge, neighbour]) => ({ edge, label: edge, neighbour, direction: 'out' })),
      details: {},
      linked: Object.fromEntries(edges.map(([e, n]) => [`${e}:${n}`, docs(e)])),
    })
    const { container } = render(<KnowledgeGraphPanel documentId="d1" onOpenDocument={vi.fn()} />)
    await waitFor(() => expect(container.querySelectorAll('.kg-node')).toHaveLength(4))
    // Order 0..3 is top, right, bottom, left.
    const axis = ['left', 'top', 'left', 'top']
    for (let i = 0; i < 4; i++) {
      fireEvent.click(container.querySelectorAll('.kg-node')[i])
      const pos = [...container.querySelectorAll('.kg-doc')].map((t) =>
        parseFloat(t.style[axis[i]])
      )
      expect(pos).toHaveLength(3)
      expect([...pos].sort((a, b) => a - b)).toEqual(pos)
      fireEvent.click(container.querySelectorAll('.kg-node')[i])
    }
  })

  it('ties a tile and its list row together on hover', async () => {
    const { container } = await ready({ onOpenDocument: vi.fn() })
    fireEvent.click(nodeFor(container, '837274cc5d1a635f'))
    fireEvent.mouseEnter(container.querySelectorAll('.kg-linked-doc')[0])
    expect(container.querySelectorAll('.kg-doc')[0].getAttribute('class')).toContain('is-hovered')
  })

  it('asks the page to light the node up, and says what it lit', async () => {
    const onHighlightNode = vi.fn((node) => (node?.neighbour === 'M54.50' ? 'diagnosis code 1' : null))
    const { container } = await ready({ onHighlightNode })
    fireEvent.mouseEnter(nodeFor(container, 'M54.50'))
    expect(onHighlightNode).toHaveBeenLastCalledWith(
      { edge: 'hasDiagnosis', neighbour: 'M54.50', label: 'M54.50' },
      { scroll: false }
    )
    expect(container.querySelector('.kg-detail').textContent).toContain(
      'Highlighted on the page: diagnosis code 1'
    )

    // Pinning scrolls the document to it; hover never does.
    fireEvent.click(nodeFor(container, 'M54.50'))
    expect(onHighlightNode).toHaveBeenLastCalledWith(
      { edge: 'hasDiagnosis', neighbour: 'M54.50', label: 'M54.50' },
      { scroll: true }
    )

    // A node the page cannot find claims nothing.
    fireEvent.mouseEnter(nodeFor(container, 'oakhollow'))
    expect(container.querySelector('.kg-detail').textContent).not.toContain('Highlighted on the page')
  })

  it('clears the page highlight when nothing is engaged', async () => {
    const onHighlightNode = vi.fn(() => null)
    const { container } = await ready({ onHighlightNode })
    fireEvent.mouseEnter(nodeFor(container, 'M54.50'))
    fireEvent.mouseLeave(container.querySelector('.kg-stage'))
    expect(onHighlightNode).toHaveBeenLastCalledWith(null, { scroll: false })
  })

  it('states a fact, not a cause, when there are no connections', async () => {
    // An earlier version blamed a stale graph — which was false while the
    // URI-matching bug was live, and told the reviewer to wait rather than report.
    getDocumentKgNeighbourhood.mockResolvedValue({ available: true, node: 'x.pdf', edges: [], siblings: [] })
    render(<KnowledgeGraphPanel documentId="d1" />)
    expect(await screen.findByText(/No graph connections found/i)).toBeTruthy()
    expect(screen.queryByText(/rebuilt after a deploy/i)).toBeNull()
  })

  it('says so when the graph is not enabled, rather than looking empty', async () => {
    getDocumentKgNeighbourhood.mockResolvedValue({ available: false, node: null, edges: [], siblings: [] })
    render(<KnowledgeGraphPanel documentId="d1" />)
    expect(await screen.findByText(/not enabled in this workspace/i)).toBeTruthy()
  })
})
