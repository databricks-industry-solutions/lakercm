import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { getDocumentKgNeighbourhood } from '../api/lakeRcmApi'
import { getStatusLabel } from './DocumentCard'
import './KnowledgeGraphPanel.css'

// Hand-rolled, no graph library.
//
// A one-hop neighbourhood of a claim document is single-figure to low tens of
// nodes, which needs no layout engine, and package.json carries 8 dependencies.
// The repo already hand-rolls SVG twice -- DonutChart.jsx and ConnectionLine.jsx
// -- so a radial spoke diagram is the established idiom here, not a new one.
//
// Lines are SVG; everything a reviewer reads or clicks is HTML placed over them.
// The all-SVG version scaled its TEXT with the drawing, so at the default pane
// height a label rendered at about 7px and at full height at 18px. HTML keeps
// type at the app's sizes whatever the pane does, and makes every node a real
// <button> rather than a <g role="button">.
//
// Laid out in real pixels from the stage's measured size, NOT a fixed square
// scaled to fit. A scaled square grows with the pane while the labels hanging
// off its sides stay the same width, so in a wide-but-short or narrow pane the
// side labels ran off the column and into the verdict form. Measuring lets the
// ring take whatever room is left after reserving each side label's width.

// Room a side label needs beside its node -- half the node's hit area, the
// label's max-width (.kg-label in KnowledgeGraphPanel.css), and a margin -- and
// what a top/bottom label needs above or below it. The ring gets the rest.
const SIDE_ROOM = 14 + 112 + 6
const VERTICAL_ROOM = 36
// Size assumed before the first measurement (and in jsdom, which has no layout).
const DEFAULT_SIZE = { w: 440, h: 260 }

// How many of a pinned node's documents are drawn around it. The card lists up
// to the API's cap; the graph only needs enough to read as "these, and more".
const FAN_MAX = 8
// How far beyond its node a pinned node's documents sit, and how far apart.
const FAN_GAP = 46
const FAN_SPACING = 27
// Tiles stay this far inside the stage edge.
const EDGE_MARGIN = 13

// Draw order, so the patient spoke is always first and the long tail of codes
// last. Keys match EDGE_LABELS in reviewer_app/routes/kg.py.
const EDGE_ORDER = [
  'documentsPatient',
  'billedTo',
  'deniedFor',
  'hasDiagnosis',
  'hasProcedure',
]

// The colour each kind of node is drawn in: custom properties set in
// KnowledgeGraphPanel.css from the design-system palette, so the graph follows
// the theme -- dark mode included -- instead of carrying hexes of its own.
const KIND_VAR = {
  documentsPatient: '--kg-patient',
  billedTo: '--kg-payer',
  deniedFor: '--kg-denial',
  hasDiagnosis: '--kg-diagnosis',
  hasProcedure: '--kg-procedure',
}
const kindStyle = (edge) => ({ '--kg-c': `var(${KIND_VAR[edge] || '--kg-other'})` })

const KIND_NAMES = {
  documentsPatient: 'Patient',
  billedTo: 'Payer',
  deniedFor: 'Denial reason',
  hasDiagnosis: 'Diagnosis',
  hasProcedure: 'Procedure',
}

// What a node's other documents are relative to it: "7 other documents for
// this patient", "15 other documents with this diagnosis".
const LINK_PHRASE = {
  documentsPatient: 'for this patient',
  billedTo: 'billed to this payer',
  deniedFor: 'denied for this reason',
  hasDiagnosis: 'with this diagnosis',
  hasProcedure: 'with this procedure',
}

const shorten = (v, max = 20) =>
  !v ? '' : v.length <= max ? v : `${v.slice(0, max - 1)}…`

// Keep the END of a file name. Siblings differ only in their tail
// ("…0001-lab-results.pdf" against "…0002-prior-authorization.pdf"), which is
// exactly what an end-ellipsis throws away. The cut lands on a separator, so it
// never opens mid-token ("…0-0003-denial.pdf").
const tailName = (s, max = 26) => {
  if (!s || s.length <= max) return s || ''
  const tail = s.slice(-(max - 1))
  const cut = tail.search(/[-_ ]/)
  return `…${cut >= 0 && cut < tail.length - 6 ? tail.slice(cut + 1) : tail}`
}

// "denial_management" -> "denial management", capitalised in CSS the way the
// extraction panel's type pill is.
const prettyType = (t) => (t ? String(t).replace(/_/g, ' ') : '')

const at = (x, y) => ({ left: `${x}px`, top: `${y}px` })
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// Which side of its dot a label sits on: away from the hub, so text never lies
// across the spokes.
const sideOf = (x, y, cx, cy) =>
  Math.abs(x - cx) < 18 ? (y < cy ? 'top' : 'bottom') : x > cx ? 'right' : 'left'

// The stage's rendered size, kept current as the pane is resized. A callback ref
// rather than useRef, because the stage only mounts once the graph has loaded.
function useMeasured() {
  const [el, setEl] = useState(null)
  const [size, setSize] = useState(DEFAULT_SIZE)
  useEffect(() => {
    if (!el || typeof ResizeObserver === 'undefined') return undefined
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (width > 0 && height > 0) setSize({ w: width, h: height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [el])
  return [setEl, size]
}

// What each class IS, in one line. Mirrors the `comment` on each class in
// scripts/lakercm_domain_content.py -- the graph stores those on the ontology, not
// on the instances, so repeating them here avoids a second round trip for seven
// strings that change about once a year. If a class is added there and not here,
// the card falls back to showing the class name alone rather than inventing a
// description.
const CLASS_ABOUT = {
  Document: 'A claim document processed by LakeRCM.',
  Patient:
    'One patient, identified only by a pseudonymous key — the graph holds no name, date of birth or member id.',
  Payer: 'An insurance payer.',
  PayerPolicy: 'A payer coverage or coding policy, which governs specific codes.',
  DiagnosisCode: 'An ICD-10-CM diagnosis code.',
  ProcedureCode: 'A CPT or HCPCS procedure code.',
  DenialReason: 'A CARC denial-reason code, explaining why a claim was denied.',
}

// Human labels for the stored property names. Anything unmapped is de-snaked
// rather than hidden, so a new ontology property still reads sensibly.
const PROP_LABELS = {
  description: 'Description',
  category: 'Category',
  is_billable: 'Billable',
  code_type: 'Code type',
  denial_category: 'Denial category',
  policy_type: 'Policy type',
  title: 'Title',
  citation_label: 'Citation',
  effective_date: 'Effective',
  document_type: 'Document type',
  confidence_score: 'Confidence',
  payer_name: 'Payer',
  review_reasons: 'Held for',
}

const propLabel = (k) =>
  PROP_LABELS[k] || k.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())

// "true"/"false" arrive as strings from the triplestore; everything else is shown
// as stored.
const propValue = (v) =>
  v === 'true' ? 'Yes' : v === 'false' ? 'No' : String(v ?? '')

const mainLabel = (s) => s.detail?.label || s.neighbour

// The second line under a node: what it means, where the store says. A patient
// has no description -- it is a pseudonymous key -- so it says how many documents
// it ties together instead, which is the reason it is on the graph at all.
const subLabel = (s) => {
  if (s.edge === 'documentsPatient') {
    const n = (s.linked.total || 0) + 1
    return `${n} document${n === 1 ? '' : 's'}`
  }
  return s.detail?.props?.description || KIND_NAMES[s.edge] || s.label
}

// What a set of file names all share, cut back to a separator, so a list can
// show what DIFFERS. Siblings "synthetic-20260930-0004-referral.pdf" and
// "synthetic-20260930-0007-referral.pdf" read as "…0004-referral.pdf" and
// "…0007-referral.pdf". A fixed-length tail cannot do this: the type suffix
// varies in length, so "…authorization.pdf" lost the very number that told two
// prior authorizations apart.
const sharedPrefix = (names) => {
  if (names.length < 2) return ''
  let p = names[0]
  for (const n of names.slice(1)) {
    let i = 0
    while (i < p.length && i < n.length && p[i] === n[i]) i++
    p = p.slice(0, i)
  }
  const cut = Math.max(p.lastIndexOf('-'), p.lastIndexOf('_'), p.lastIndexOf(' '))
  const prefix = cut >= 0 ? p.slice(0, cut + 1) : ''
  // Not worth an ellipsis for a few characters.
  return prefix.length >= 6 ? prefix : ''
}

function LinkedDocuments({ spoke, pinned, hoveredDoc, onHoverDoc, onOpen, canOpen }) {
  const docs = spoke.linked.documents || []
  const total = Math.max(spoke.linked.total || 0, docs.length)
  const phrase = LINK_PHRASE[spoke.edge] || 'that share it'
  const prefix = sharedPrefix(docs.map((d) => d.name))
  const shown = (name) =>
    prefix && name.startsWith(prefix) ? `…${name.slice(prefix.length)}` : tailName(name, 30)
  return (
    <div className="kg-linked">
      <p className="kg-linked-title">
        {total === 0
          ? `No other documents ${phrase}.`
          : `${total} other document${total === 1 ? '' : 's'} ${phrase}`}
      </p>
      {/* Hover only previews: the pointer cannot reach this list without
          leaving the node. Saying how to pin it is what makes the list usable. */}
      {total > 0 && !pinned && (
        <p className="kg-linked-hint">Click the node to pin it and draw these on the graph.</p>
      )}
      {docs.length > 0 && (
        <ol className="kg-linked-list">
          {docs.map((d, i) => (
            <li key={d.name}>
              <button
                type="button"
                className={`kg-linked-doc${d.name === hoveredDoc ? ' is-hovered' : ''}`}
                disabled={!d.id || !canOpen}
                title={d.id ? d.name : `${d.name} is not in the review queue yet`}
                onClick={() => onOpen(d)}
                onMouseEnter={() => onHoverDoc(d.name)}
                onMouseLeave={() => onHoverDoc(null)}
                onFocus={() => onHoverDoc(d.name)}
                onBlur={() => onHoverDoc(null)}
              >
                <span className="kg-linked-n">{i + 1}</span>
                <span className="kg-linked-text">
                  <span className="kg-linked-name">{shown(d.name)}</span>
                  {d.label && <span className="kg-linked-type">{prettyType(d.label)}</span>}
                </span>
                {d.status && (
                  <span className={`kg-pill kg-pill--${d.status}`}>
                    {getStatusLabel(d.status)}
                  </span>
                )}
              </button>
            </li>
          ))}
        </ol>
      )}
      {total > docs.length && (
        <p className="kg-linked-more">+{total - docs.length} more not listed</p>
      )}
    </div>
  )
}

function KnowledgeGraphPanel({ documentId, onOpenDocument, onHighlightNode }) {
  const [data, setData] = useState(null)
  const [state, setState] = useState('idle') // idle|loading|ready|empty|off
  const [hovered, setHovered] = useState(null)
  const [selected, setSelected] = useState(null)
  const [hoveredDoc, setHoveredDoc] = useState(null)
  const [pageMatch, setPageMatch] = useState(null)
  const liveRef = useRef(true)
  const highlightRef = useRef(onHighlightNode)
  const [stageRef, size] = useMeasured()

  useEffect(() => {
    highlightRef.current = onHighlightNode
  }, [onHighlightNode])

  useEffect(() => {
    if (!documentId) return
    liveRef.current = true
    setState('loading')
    setHovered(null)
    setSelected(null)
    setHoveredDoc(null)
    getDocumentKgNeighbourhood(documentId)
      .then((r) => {
        if (!liveRef.current) return
        if (!r?.available) {
          setState('off')
          return
        }
        setData(r)
        setState((r.edges || []).length ? 'ready' : 'empty')
      })
      .catch(() => liveRef.current && setState('empty'))
    return () => {
      liveRef.current = false
    }
  }, [documentId])

  // The ring, in pixels. Its horizontal radius is what the width leaves after a
  // side label on each side; its vertical radius what the height leaves after a
  // top and bottom label, held near round so a tall pane does not stretch it
  // into a diamond.
  const edgeCount = data?.edges?.length || 0
  const ring = useMemo(() => {
    const cx = size.w / 2
    const cy = size.h / 2
    // How far out the widest SIDE node sits, as a fraction of rx: with six
    // spokes that is cos 30deg, not 1, so sizing rx as if a node sat on the
    // horizontal axis under-reserves the label room. Top and bottom nodes centre
    // their labels and do not count.
    const n = Math.max(1, edgeCount)
    let reach = 0
    for (let i = 0; i < n; i++) {
      const c = Math.abs(Math.cos((i / n) * 2 * Math.PI - Math.PI / 2))
      if (c > 0.2) reach = Math.max(reach, c)
    }
    const rx = clamp((cx - SIDE_ROOM) / (reach || 1), 56, 170)
    const ry = clamp(cy - VERTICAL_ROOM, 46, rx * 1.1)
    return { cx, cy, rx, ry }
  }, [size, edgeCount])

  const spokes = useMemo(() => {
    const edges = data?.edges || []
    const sorted = [...edges].sort(
      (a, b) => EDGE_ORDER.indexOf(a.edge) - EDGE_ORDER.indexOf(b.edge)
    )
    const n = sorted.length || 1
    const { cx, cy, rx, ry } = ring
    return sorted.map((e, i) => {
      // Start at the top: the patient sorts first, so it is the spoke the eye
      // lands on.
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2
      const x = cx + rx * Math.cos(angle)
      const y = cy + ry * Math.sin(angle)
      const key = `${e.edge}:${e.neighbour}`
      return {
        ...e,
        key,
        angle,
        x,
        y,
        side: sideOf(x, y, cx, cy),
        // Fetched with the neighbourhood, so opening a node costs no round trip.
        detail: data?.details?.[e.neighbour] || null,
        linked: data?.linked?.[key] || { total: 0, documents: [] },
      }
    })
  }, [data, ring])

  const active = hovered || selected
  const activeSpoke = spokes.find((s) => s.key === active) || null
  const selectedSpoke = spokes.find((s) => s.key === selected) || null
  const detail = activeSpoke?.detail || null

  // The pinned node's documents, fanned out on an arc beyond it. This is the
  // graph answering "what is this connected to" in the graph itself, not only in
  // a list beside it.
  const fan = useMemo(() => {
    if (!selectedSpoke) return []
    const docs = selectedSpoke.linked.documents || []
    if (!docs.length) return []
    const total = Math.max(selectedSpoke.linked.total || 0, docs.length)
    const overflow = total > FAN_MAX
    const shown = docs.slice(0, overflow ? FAN_MAX - 1 : FAN_MAX)
    const items = shown.map((d, i) => ({ ...d, n: i + 1 }))
    if (overflow) items.push({ more: total - shown.length })
    // An arc on a ring FAN_GAP outside the node ring, centred on the node, with
    // the angle per tile chosen so neighbouring tiles sit FAN_SPACING apart.
    // Clamped inside the stage: a node at the top or bottom edge flattens its arc
    // against the edge rather than pushing tiles out of view.
    const { cx, cy, rx, ry } = ring
    const fx = rx + FAN_GAP
    const fy = ry + FAN_GAP
    const step = FAN_SPACING / ((fx + fy) / 2)
    const spread = Math.min(Math.PI * 0.62, (items.length - 1) * step)
    // Number in READING order wherever the node sits: left to right on an arc
    // above or below the hub, top to bottom beside it. Walking the angle one way
    // for every node read 1..7 above the hub and 7..1 below it.
    const cos = Math.cos(selectedSpoke.angle)
    const sin = Math.sin(selectedSpoke.angle)
    const reverse = Math.abs(cos) >= Math.abs(sin) ? cos < 0 : sin > 0
    return items.map((it, i) => {
      const u = items.length === 1 ? 0 : i / (items.length - 1) - 0.5
      const t = reverse ? -u : u
      const a = selectedSpoke.angle + t * spread
      return {
        ...it,
        x: clamp(cx + fx * Math.cos(a), EDGE_MARGIN, size.w - EDGE_MARGIN),
        y: clamp(cy + fy * Math.sin(a), EDGE_MARGIN, size.h - EDGE_MARGIN),
      }
    })
  }, [selectedSpoke, ring, size])

  // The page highlight. Engaging a node lights up where its value appears on the
  // document, through the same field<->bbox highlight the extraction panel
  // drives; pinning one also scrolls the document to it. The page answers with
  // the field it matched (or null), so the card can say what it lit up instead
  // of leaving the reviewer to spot it.
  const activeKey = activeSpoke?.key || null
  const pinned = activeKey !== null && activeKey === selected
  useEffect(() => {
    const notify = highlightRef.current
    if (!notify) return
    const s = spokes.find((x) => x.key === activeKey)
    const field = notify(
      s ? { edge: s.edge, neighbour: s.neighbour, label: s.detail?.label || null } : null,
      { scroll: pinned }
    )
    setPageMatch(s && field ? field : null)
  }, [activeKey, pinned, spokes])

  // Leaving the document must not leave a box lit on the next one.
  useEffect(() => () => highlightRef.current?.(null, { scroll: false }), [])

  const toggle = useCallback(
    (key) => setSelected((cur) => (cur === key ? null : key)),
    []
  )

  // Escape unpins. It stops here: on this page Escape otherwise means "back to
  // the documents", which is not what someone dismissing a node intends.
  const onKeyNode = useCallback(
    (e) => {
      if (e.key === 'Escape' && (selected || hovered)) {
        e.stopPropagation()
        setSelected(null)
        setHovered(null)
      }
    },
    [selected, hovered]
  )

  const openDoc = useCallback(
    (d) => {
      if (d?.id && onOpenDocument) onOpenDocument({ id: d.id, name: d.name })
    },
    [onOpenDocument]
  )

  const kinds = EDGE_ORDER.filter((k) => spokes.some((s) => s.edge === k))
  const hubType = prettyType(data?.node_type)

  return (
    <div className="kg-panel">
      {state === 'loading' && (
        <p className="kg-panel-note" role="status" aria-live="polite">
          Loading graph…
        </p>
      )}
      {state === 'off' && (
        <p className="kg-panel-note">
          The knowledge graph is not enabled in this workspace.
        </p>
      )}
      {/* States a fact, not a cause. An earlier version asserted "the graph is
          rebuilt after a deploy, so a freshly ingested document can lag behind"
          -- and while the URI-matching bug was live that was false: the document
          had 15 triples and the query was wrong. A confident wrong explanation
          tells the reviewer to wait instead of reporting it. */}
      {state === 'empty' && (
        <p className="kg-panel-note">No graph connections found for this document.</p>
      )}

      {state === 'ready' && (
        <div className="kg-body">
          <div className="kg-graph">
            <div
              ref={stageRef}
              className={`kg-stage${active ? ' is-focused' : ''}`}
              role="group"
              aria-label={`Graph neighbourhood: ${spokes.length} connected entities`}
              onMouseLeave={() => setHovered(null)}
            >
              <svg
                className="kg-svg"
                width={size.w}
                height={size.h}
                viewBox={`0 0 ${size.w} ${size.h}`}
                aria-hidden="true"
                focusable="false"
              >
                {spokes.map((s) => (
                  <line
                    key={`l-${s.key}`}
                    className={`kg-edge kg-edge--${s.edge}${
                      active === s.key ? ' is-active' : ''
                    }${selected === s.key ? ' is-selected' : ''}`}
                    style={kindStyle(s.edge)}
                    x1={ring.cx}
                    y1={ring.cy}
                    x2={s.x}
                    y2={s.y}
                  />
                ))}
                {selectedSpoke &&
                  fan.map((t) => (
                    <line
                      key={`f-${t.name || 'more'}`}
                      className={`kg-fan-edge${
                        t.name && t.name === hoveredDoc ? ' is-hovered' : ''
                      }`}
                      style={kindStyle(selectedSpoke.edge)}
                      x1={selectedSpoke.x}
                      y1={selectedSpoke.y}
                      x2={t.x}
                      y2={t.y}
                    />
                  ))}
              </svg>

              {/* The hub says what the document IS. It used to be an anonymous
                  dot with its name in a tooltip, so the one node the whole
                  diagram is about was the only one without a label. */}
              <div className="kg-hub" style={at(ring.cx, ring.cy)} title={data?.node}>
                <span className="kg-hub-type">{hubType || 'This document'}</span>
                <span className="kg-hub-name">{tailName(data?.node, 20)}</span>
              </div>

              {spokes.map((s, i) => (
                <button
                  key={`n-${s.key}`}
                  type="button"
                  className={`kg-node kg-node--${s.edge}${
                    active === s.key ? ' is-active' : ''
                  }${selected === s.key ? ' is-selected' : ''}`}
                  style={{ ...at(s.x, s.y), ...kindStyle(s.edge), '--i': i }}
                  aria-label={`${s.label}: ${mainLabel(s)}`}
                  aria-pressed={selected === s.key}
                  onMouseEnter={() => setHovered(s.key)}
                  onFocus={() => setHovered(s.key)}
                  onBlur={() => setHovered(null)}
                  onClick={() => toggle(s.key)}
                  onKeyDown={onKeyNode}
                >
                  <span className="kg-dot" />
                  <span className={`kg-label kg-label--${s.side}`}>
                    <span className="kg-label-main">{shorten(mainLabel(s), 20)}</span>
                    <span className="kg-label-sub">
                      {shorten(subLabel(s), s.side === 'top' || s.side === 'bottom' ? 34 : 26)}
                    </span>
                  </span>
                </button>
              ))}

              {selectedSpoke &&
                fan.map((t, i) =>
                  t.more ? (
                    <span
                      key="more"
                      className="kg-doc kg-doc--more"
                      style={{ ...at(t.x, t.y), ...kindStyle(selectedSpoke.edge), '--i': i }}
                      title={`${t.more} more in the list`}
                    >
                      +{t.more}
                    </span>
                  ) : (
                    <button
                      key={`d-${t.name}`}
                      type="button"
                      className={`kg-doc${t.name === hoveredDoc ? ' is-hovered' : ''}`}
                      style={{ ...at(t.x, t.y), ...kindStyle(selectedSpoke.edge), '--i': i }}
                      disabled={!t.id || !onOpenDocument}
                      title={t.id ? t.name : `${t.name} is not in the review queue yet`}
                      aria-label={`Open ${t.name}`}
                      onClick={() => openDoc(t)}
                      onMouseEnter={() => setHoveredDoc(t.name)}
                      onMouseLeave={() => setHoveredDoc(null)}
                      onFocus={() => setHoveredDoc(t.name)}
                      onBlur={() => setHoveredDoc(null)}
                    >
                      {t.n}
                    </button>
                  )
                )}
            </div>

            {kinds.length > 0 && (
              <ul className="kg-legend" aria-label="Legend">
                {kinds.map((k) => (
                  <li key={k}>
                    <span className="kg-swatch" style={kindStyle(k)} />
                    {KIND_NAMES[k]}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* An EXPLAIN card, not just a label. The ring shows "M54.50"; on its own
              that tells a reviewer which node, not what it is. This names the class,
              says in one line what that class means, lists the entity's stored
              properties -- and then the other documents that share it, which is
              what a click on the graph is for. */}
          <div
            className="kg-detail"
            style={activeSpoke ? kindStyle(activeSpoke.edge) : undefined}
          >
            <div className="kg-detail-summary" role="status" aria-live="polite">
              {activeSpoke ? (
                <>
                  <div className="kg-detail-head">
                    <span className="kg-swatch" />
                    <span className="kg-detail-value">
                      {detail?.label || activeSpoke.neighbour}
                    </span>
                    {detail?.cls && <span className="kg-detail-cls">{detail.cls}</span>}
                  </div>

                  {detail?.cls && CLASS_ABOUT[detail.cls] && (
                    <p className="kg-detail-about">{CLASS_ABOUT[detail.cls]}</p>
                  )}

                  {detail && Object.keys(detail.props || {}).length > 0 && (
                    <dl className="kg-detail-props">
                      {Object.entries(detail.props).map(([k, v]) => (
                        <div className="kg-detail-prop" key={k}>
                          <dt>{propLabel(k)}</dt>
                          <dd>{propValue(v)}</dd>
                        </div>
                      ))}
                    </dl>
                  )}

                  {/* Said out loud rather than shown as an empty card: "nothing is
                      stored about this entity" and "the graph has not been rebuilt"
                      are different claims. */}
                  {!detail && (
                    <p className="kg-detail-about kg-detail-about--muted">
                      Connected as {activeSpoke.label}. No further detail is stored for
                      this entity.
                    </p>
                  )}

                  {pageMatch && (
                    <p className="kg-detail-onpage">
                      Highlighted on the page: <strong>{pageMatch}</strong>
                    </p>
                  )}
                </>
              ) : (
                <>
                  <div className="kg-detail-head">
                    <span className="kg-swatch kg-swatch--hub" />
                    <span className="kg-detail-value kg-detail-value--type">
                      {hubType || 'This document'}
                    </span>
                    <span className="kg-detail-cls">Document</span>
                  </div>
                  <p className="kg-detail-name" title={data?.node}>
                    {data?.node}
                  </p>
                  <span className="kg-readout-hint">
                    {spokes.length} connection{spokes.length === 1 ? '' : 's'} — select a
                    node to see what it is and which other documents share it
                  </span>
                </>
              )}
            </div>

            {activeSpoke && (
              <LinkedDocuments
                spoke={activeSpoke}
                pinned={pinned}
                hoveredDoc={hoveredDoc}
                onHoverDoc={setHoveredDoc}
                onOpen={openDoc}
                canOpen={Boolean(onOpenDocument)}
              />
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default KnowledgeGraphPanel
