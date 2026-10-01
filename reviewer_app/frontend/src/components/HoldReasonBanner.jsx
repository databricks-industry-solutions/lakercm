import { useEffect, useState } from 'react'
import { getDocumentHoldReasons } from '../api/lakeRcmApi'
import { formatConfidencePct } from '../lib/confidence'
import './HoldReasonBanner.css'

// What replaced a single fabricated sentence.
//
// The banner this supersedes said, on a document measuring 99.96%:
//
//   Held for human review — Extraction confidence was 100%, below the 92%
//   auto-verify threshold, so the pipeline routed this document to a reviewer.
//   Every code on it passed validation.
//
// Three separate faults produced that. The verdict came from React props the
// routed page never passed, so every document read as held. The percentage was
// Math.round of 0.9996. And the explanation was inferred from an EMPTY reason
// list — "no rule fired" was read as "the score was low".
//
// So this component renders only what the server states, and the server refuses
// to state a cause it cannot support. There is no branch here that composes an
// explanation; every sentence shown is a `detail` computed next to the data it
// describes.

const STATE_COPY = {
  held: {
    icon: '!',
    title: 'Held for human review',
    className: 'hrb--held',
  },
  auto_verified: {
    icon: '✓',
    title: 'Auto-verified by the extraction pipeline',
    className: 'hrb--auto',
  },
}

function ReasonRow({ item }) {
  return (
    <li className={`hrb-reason hrb-reason--${item.severity}`}>
      <div className="hrb-reason-head">
        <span className="hrb-reason-title">{item.title}</span>
        {item.observed_code && (
          <code className="hrb-reason-code">{item.observed_code}</code>
        )}
        {item.field_name && (
          <span className="hrb-reason-field">{item.field_name}</span>
        )}
      </div>
      <p className="hrb-reason-detail">{item.detail}</p>
      {item.guidance && <p className="hrb-reason-guidance">{item.guidance}</p>}
      {item.candidates?.length > 0 && (
        <p className="hrb-reason-candidates">
          Terminology accepts:{' '}
          {item.candidates.map((c) => (
            <code key={c.code} className="hrb-candidate">
              {c.code}
            </code>
          ))}
          {item.candidates_truncated && <span className="hrb-more"> and more</span>}
        </p>
      )}
    </li>
  )
}

export default function HoldReasonBanner({ documentId }) {
  const [data, setData] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!documentId) return
    let active = true
    setData(null)
    setFailed(false)
    getDocumentHoldReasons(documentId)
      .then((d) => active && setData(d))
      .catch(() => active && setFailed(true))
    return () => {
      active = false
    }
  }, [documentId])

  if (failed) {
    // Say which thing is unknown. The predecessor's failure mode was to render a
    // confident verdict regardless of whether it had the data for one.
    return (
      <div className="hrb hrb--unknown">
        <span className="hrb-icon">?</span>
        <div className="hrb-body">
          <strong>Review status could not be determined</strong>
          <span>
            The routing decision for this document could not be read. Review the
            extraction on its merits.
          </span>
        </div>
      </div>
    )
  }

  if (!data) return null

  // "unknown" is a real answer, not a reason to guess. It means is_automated
  // could not be read — no gold row yet, or the sync has not caught up — and the
  // honest rendering is no verdict at all.
  if (data.state === 'unknown') return null

  const copy = STATE_COPY[data.state]
  if (!copy) return null

  const pct = formatConfidencePct(data.confidence_score)
  const thresholdPct = formatConfidencePct(data.auto_verdict_threshold)
  const blocking = data.blocking || []
  const advisory = data.advisory || []

  return (
    <div className={`hrb ${copy.className}`}>
      <span className="hrb-icon">{copy.icon}</span>
      <div className="hrb-body">
        <strong>{copy.title}</strong>

        {data.state === 'auto_verified' && (
          <span className="hrb-summary">
            {pct
              ? `Extraction confidence was ${pct}, at or above the ${thresholdPct} threshold, and nothing else was flagged. Submit a verdict below to override.`
              : 'Nothing was flagged on this document. Submit a verdict below to override.'}
          </span>
        )}

        {data.state === 'held' && (
          <span className="hrb-summary">
            {blocking.length === 1
              ? 'One thing needs a person:'
              : `${blocking.length} things need a person:`}
          </span>
        )}

        {blocking.length > 0 && (
          <ul className="hrb-reasons">
            {blocking.map((item, i) => (
              <ReasonRow key={`${item.code}-${item.observed_code || i}`} item={item} />
            ))}
          </ul>
        )}

        {advisory.length > 0 && (
          <div className="hrb-advisory">
            {/* Kept visually separate and labelled: these did NOT hold the
                document, and presenting them as if they had would misrepresent
                why it is on this screen. */}
            <span className="hrb-advisory-label">
              Worth checking — these did not hold the document
            </span>
            <ul className="hrb-reasons">
              {advisory.map((item, i) => (
                <ReasonRow key={`${item.code}-${i}`} item={item} />
              ))}
            </ul>
          </div>
        )}

        {data.degraded?.includes('advisory_unavailable') && (
          // "Nothing to report" and "could not check" look identical otherwise,
          // and only one of them is reassuring.
          <span className="hrb-degraded">
            Additional checks could not be run for this document.
          </span>
        )}
      </div>
    </div>
  )
}
