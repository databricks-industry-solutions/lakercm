import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ExtractedDataPanel } from './ExtractionComparison'

// ai_classify v2.1 adds two things to this header: the classifier's own
// confidence and a grounded rationale. Both are absent on a document classified
// before the upgrade, and absent for the whole window between the pipeline
// emitting the new gold columns and the Lakebase sync picking them up — so the
// header has to render identically without them. That degrade path is the only
// part of this change that can break in production, and it is invisible to every
// other test in the repo.
//
// The second thing pinned here is that the classify confidence is NOT a
// ConfidenceBar. That widget colours against the auto-verdict threshold (0.92),
// and ai_classify's confidence tops out around 0.78, so routing it through the
// bar would paint every document in the app as "below threshold" on a signal the
// threshold was never set for.

const RATIONALE =
  'The document states the claim was denied because prior authorization was not obtained.'

function panel(record) {
  return render(
    <ExtractedDataPanel data={{ items: record ? [record] : [] }} corrections={{}} />,
  )
}

const BASE = {
  document_path: '/Volumes/x/doc.pdf',
  document_name: 'doc.pdf',
  label: 'denial_management',
  identifiers: [{ name: 'claim_id', value: '8842', confidence: 0.99 }],
  confidence_score: 0.94,
}

describe('ExtractedDataPanel classification header', () => {
  it('renders the rationale under the label', () => {
    panel({ ...BASE, classify_confidence: 0.68, classify_rationale: RATIONALE })
    expect(screen.getByText(RATIONALE)).toBeTruthy()
  })

  it('shows the classifier confidence on its own scale, not as a confidence bar', () => {
    const { container } = panel({
      ...BASE,
      classify_confidence: 0.68,
      classify_rationale: RATIONALE,
    })
    // 0.68 -> "type 68 %" (thin space between number and sign).
    expect(screen.getByText(/type\s*68/)).toBeTruthy()
    // Scoped to the classification row: exactly one bar there, the blended
    // score's. (ConfidenceBar is also used per identifier field, so an unscoped
    // count would just track the fixture's field count.) If a future refactor
    // routes the classify number through ConfidenceBar too, this fails.
    const row = container.querySelector('.ls-classification-row')
    expect(row.querySelectorAll('.ls-confidence-bar')).toHaveLength(1)
    expect(row.querySelectorAll('.ls-classify-confidence')).toHaveLength(1)
  })

  it('renders the header unchanged when the synced columns are absent', () => {
    const { container } = panel(BASE)
    // The label pill and the blended bar still render...
    expect(screen.getByText('denial management')).toBeTruthy()
    const row = container.querySelector('.ls-classification-row')
    expect(row.querySelectorAll('.ls-confidence-bar')).toHaveLength(1)
    // ...and nothing classify-specific appears.
    expect(container.querySelectorAll('.ls-classify-confidence')).toHaveLength(0)
    expect(container.querySelectorAll('.ls-classify-rationale')).toHaveLength(0)
  })

  it('treats a blank rationale as absent', () => {
    // gold stores whatever the model returned; an empty or whitespace string is
    // not a rationale and must not render an empty paragraph.
    const { container } = panel({ ...BASE, classify_rationale: '   ' })
    expect(container.querySelectorAll('.ls-classify-rationale')).toHaveLength(0)
  })

  it('renders a zero classify confidence rather than hiding it', () => {
    // 0 is a real measurement and the worst case the blend can see — it must not
    // be swallowed by a falsy check.
    panel({ ...BASE, classify_confidence: 0 })
    expect(screen.getByText(/type\s*0/)).toBeTruthy()
  })
})
