import { useEffect, useState } from 'react'
import { getDocumentRemediation } from '../api/lakeRcmApi'
import './RemediationPanel.css'

// What the terminology permits for a held document, shown next to the reason it
// was held. The pipeline says WHY a document needs a reviewer; this says what
// the possible fixes are, so the reviewer is not left to work out which codes
// exist from memory.
//
// Read-only on purpose. Applying a candidate means writing it against the
// reviewer UI's `id:<n>` correction key, which is POSITIONAL into the
// identifiers array — mapping a field name back to an index here would put a
// value on the wrong field the first time extraction reindexed. The in-document
// assistant applies candidates instead: it reads the real keys from
// get_active_review_context and stages a card the reviewer approves.

const RESOLUTION_COPY = {
  deterministic: {
    label: 'One option',
    hint: 'The terminology leaves a single valid replacement. Confirm it against the document.',
  },
  needs_judgment: {
    label: 'Needs the document',
    hint: 'Several codes are valid. Only the document decides — site, laterality or severity.',
  },
  not_resolvable: {
    label: 'Cannot be resolved here',
    hint: 'There is no value to fill in. It has to come from somewhere else.',
  },
}

function RemediationItem({ item }) {
  const copy = RESOLUTION_COPY[item.resolution] || {
    label: item.resolution,
    hint: '',
  }
  const hasCandidates = (item.candidates || []).length > 0

  return (
    <li className={`rmd-item rmd-${item.resolution}`}>
      <div className="rmd-item-head">
        {item.observed_code && (
          <code className="rmd-observed">{item.observed_code}</code>
        )}
        {item.field_name && (
          <span className="rmd-field">on {item.field_name}</span>
        )}
        <span className="rmd-resolution">{copy.label}</span>
      </div>

      <p className="rmd-guidance">{item.guidance}</p>

      {hasCandidates && (
        <div className="rmd-candidates">
          <span className="rmd-candidates-label">
            {item.candidates.length === 1 ? 'Candidate' : 'Candidates'}
          </span>
          <ul className="rmd-candidate-list">
            {item.candidates.map((c) => (
              <li key={c.code} className="rmd-candidate">
                <code>{c.code}</code>
                <span className="rmd-candidate-desc">{c.description}</span>
              </li>
            ))}
          </ul>
          {item.candidates_truncated && (
            <p className="rmd-truncated">
              Only the first {item.candidates.length} are shown — the observed
              value is too ambiguous to shortlist reliably.
            </p>
          )}
        </div>
      )}
    </li>
  )
}

export default function RemediationPanel({ documentId }) {
  const [state, setState] = useState({ status: 'loading', data: null })

  useEffect(() => {
    if (!documentId) return
    let cancelled = false
    setState({ status: 'loading', data: null })
    getDocumentRemediation(documentId)
      .then((data) => {
        if (!cancelled) setState({ status: 'ready', data })
      })
      .catch((e) => {
        // Never block the review on this. A reviewer can still work the
        // document from the reason alone, which is what they had before.
        if (!cancelled) setState({ status: 'error', data: null, error: e.message })
      })
    return () => {
      cancelled = true
    }
  }, [documentId])

  if (state.status === 'loading') {
    // Was `return null`, which showed nothing at all while two warehouse reads
    // ran (flagged items, then the terminology). On a cold SQL warehouse that is
    // seconds of blank space directly under "this was held for a person",
    // reading as "there are no suggested fixes" -- the opposite of the truth.
    //
    // Deliberately NOT called "agent verification": this shortlist is computed
    // in code from the curated terminology (services/remediation.py), with no
    // model involved. The agent strip is a separate component, shown only when a
    // turn is actually streaming.
    return (
      <p className="rmd-loading" role="status" aria-live="polite">
        Checking the terminology for possible fixes…
      </p>
    )
  }
  if (state.status === 'error') {
    // Said out loud rather than rendered as an empty list: "no suggestions"
    // and "could not look" are different claims, and a reviewer would act on
    // them differently.
    return (
      <p className="rmd-unavailable">
        Suggested fixes are unavailable right now ({state.error}).
      </p>
    )
  }

  const items = state.data?.items || []
  if (!state.data?.is_held || items.length === 0) return null

  return (
    <div className="rmd-panel">
      <div className="rmd-head">
        <strong>Possible fixes</strong>
        <span className="rmd-sub">
          from the reference terminology — nothing is applied until you approve
          it
        </span>
      </div>
      <ul className="rmd-list">
        {items.map((item, i) => (
          <RemediationItem
            key={`${item.review_reason}-${item.observed_code || i}`}
            item={item}
          />
        ))}
      </ul>
    </div>
  )
}
