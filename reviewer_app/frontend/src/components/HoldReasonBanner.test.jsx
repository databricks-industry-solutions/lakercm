import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import HoldReasonBanner from './HoldReasonBanner'

// The component that replaced a banner which stated a falsehood confidently.
//
// The first test is the one that matters: a document held with no recorded
// reason at 99.96% confidence must never be told it was below a 92% threshold.
// The rest pin the mechanisms that made that possible — a rounded percentage, a
// strict `=== true` on a stringified boolean, and an "unknown" routing decision
// being rendered as a verdict.
//
// Absorbs the coverage that lived in FidelityNotice.test.jsx: rewritten codes are
// now one advisory row in this banner rather than a separate notice, so a
// silently altered code appears exactly once on the screen.

vi.mock('../api/lakeRcmApi', () => ({
  getDocumentHoldReasons: vi.fn(),
}))

const { getDocumentHoldReasons } = await import('../api/lakeRcmApi')

const HELD_LOW_CONFIDENCE = {
  document_id: 'doc-1',
  state: 'held',
  is_automated: false,
  confidence_score: 0.841,
  auto_verdict_threshold: 0.92,
  reasons_recorded: true,
  unexplained: false,
  blocking: [
    {
      code: 'low_confidence',
      severity: 'blocking',
      source: 'derived_from_confidence',
      title: 'Low confidence',
      detail:
        'Blended extraction confidence was 84.1%, below the 92% auto-verify threshold.',
      guidance: 'Re-read every extracted value against the document.',
      candidates: [],
    },
  ],
  advisory: [],
  degraded: [],
}

beforeEach(() => {
  getDocumentHoldReasons.mockResolvedValue(HELD_LOW_CONFIDENCE)
})

describe('HoldReasonBanner', () => {
  it('never claims a 99.96% document was below the threshold', async () => {
    // THE regression. The old banner rendered "Extraction confidence was 100%,
    // below the 92% auto-verify threshold" for exactly this payload.
    getDocumentHoldReasons.mockResolvedValue({
      ...HELD_LOW_CONFIDENCE,
      confidence_score: 0.9996,
      reasons_recorded: false,
      unexplained: true,
      blocking: [
        {
          code: 'reason_not_recorded',
          severity: 'blocking',
          source: 'unexplained',
          title: 'Reason not recorded',
          detail:
            'The pipeline routed this document to a reviewer but recorded no ' +
            'reason, and its confidence of 99.96% is at or above the 92% ' +
            'threshold. Nothing here explains the hold.',
          candidates: [],
        },
      ],
    })
    render(<HoldReasonBanner documentId="doc-1" />)

    expect(await screen.findByText('Reason not recorded')).toBeInTheDocument()
    const text = document.querySelector('.hrb').textContent
    expect(text).not.toMatch(/below the/i)
    expect(text).not.toContain('100%')
    expect(text).toContain('99.96%')
  })

  it('names the measured numbers on a genuine low-confidence hold', async () => {
    render(<HoldReasonBanner documentId="doc-1" />)

    expect(await screen.findByText('Held for human review')).toBeInTheDocument()
    const text = document.querySelector('.hrb').textContent
    expect(text).toContain('84.1%')
    expect(text).toContain('92%')
    expect(text).toMatch(/below the/i)
  })

  it('itemises every blocking reason with its specific code and field', async () => {
    getDocumentHoldReasons.mockResolvedValue({
      ...HELD_LOW_CONFIDENCE,
      blocking: [
        {
          code: 'invalid_code',
          severity: 'blocking',
          source: 'pipeline',
          title: 'Invalid code',
          detail: 'M5450 (ICD-10-CM) in diagnosis_codes is not in the terminology.',
          guidance: 'Confirm the code against the document.',
          observed_code: 'M5450',
          field_name: 'diagnosis_codes',
          candidates: [{ code: 'M54.50', description: 'Low back pain', method: 'x' }],
        },
        {
          code: 'missing_member_id',
          severity: 'blocking',
          source: 'pipeline',
          title: 'Missing member ID',
          detail: 'No member or subscriber ID was found on this document.',
          candidates: [],
        },
      ],
    })
    render(<HoldReasonBanner documentId="doc-1" />)

    expect(await screen.findByText('Invalid code')).toBeInTheDocument()
    expect(screen.getByText('Missing member ID')).toBeInTheDocument()
    expect(screen.getByText('M5450')).toBeInTheDocument()
    expect(screen.getByText('diagnosis_codes')).toBeInTheDocument()
    // The candidate fix, so the reviewer is not left to look it up.
    expect(screen.getByText('M54.50')).toBeInTheDocument()
    expect(screen.getByText(/2 things need a person/)).toBeInTheDocument()
  })

  it('shows an auto-verified document its real score, not a rounded one', async () => {
    getDocumentHoldReasons.mockResolvedValue({
      ...HELD_LOW_CONFIDENCE,
      state: 'auto_verified',
      is_automated: true,
      confidence_score: 0.9996,
      blocking: [],
    })
    render(<HoldReasonBanner documentId="doc-1" />)

    expect(
      await screen.findByText('Auto-verified by the extraction pipeline')
    ).toBeInTheDocument()
    const text = document.querySelector('.hrb').textContent
    expect(text).toContain('99.96%')
    expect(text).not.toContain('100%')
    expect(text).not.toMatch(/held for human review/i)
  })

  it('renders no verdict at all when the routing decision is unknown', async () => {
    // Previously a NULL is_automated read as "held" and got an invented reason.
    getDocumentHoldReasons.mockResolvedValue({
      document_id: 'doc-1',
      state: 'unknown',
      is_automated: null,
      confidence_score: 0.9996,
      auto_verdict_threshold: 0.92,
      blocking: [],
      advisory: [],
      degraded: [],
    })
    render(<HoldReasonBanner documentId="doc-1" />)

    await waitFor(() => expect(getDocumentHoldReasons).toHaveBeenCalled())
    expect(document.querySelector('.hrb')).toBeNull()
  })

  it('separates advisory findings from the reasons that held it', async () => {
    getDocumentHoldReasons.mockResolvedValue({
      ...HELD_LOW_CONFIDENCE,
      advisory: [
        {
          code: 'code_rewritten',
          severity: 'advisory',
          source: 'analytics',
          title: 'A code was silently rewritten',
          detail: 'The parser stored I10 where the page shows I1O.',
          candidates: [],
        },
        {
          code: 'uncaptured_code',
          severity: 'advisory',
          source: 'pipeline_advisory',
          title: 'Code on the page was not captured',
          detail: 'E11.9 appears in the page text but is not on the claim.',
          candidates: [],
        },
      ],
    })
    render(<HoldReasonBanner documentId="doc-1" />)

    expect(
      await screen.findByText('A code was silently rewritten')
    ).toBeInTheDocument()
    expect(
      screen.getByText(/these did not hold the document/i)
    ).toBeInTheDocument()
    // Advisories must not be counted as reasons the document is held.
    expect(screen.getByText(/One thing needs a person/)).toBeInTheDocument()
    expect(document.querySelectorAll('.hrb-reason--advisory')).toHaveLength(2)
  })

  it('shows a rewritten code on an auto-verified document', async () => {
    // Migrated from FidelityNotice.test.jsx and still the case worth
    // interrupting a reviewer for: a clinical code was altered and the document
    // auto-verified anyway. Measured 97 of 99 rewrites on dev.
    getDocumentHoldReasons.mockResolvedValue({
      ...HELD_LOW_CONFIDENCE,
      state: 'auto_verified',
      is_automated: true,
      confidence_score: 0.9996,
      blocking: [],
      advisory: [
        {
          code: 'code_rewritten',
          severity: 'advisory',
          source: 'analytics',
          title: 'A code was silently rewritten',
          detail: 'The parser stored I10 where the page shows I1O.',
          candidates: [],
        },
      ],
    })
    render(<HoldReasonBanner documentId="doc-1" />)

    expect(
      await screen.findByText('A code was silently rewritten')
    ).toBeInTheDocument()
    expect(document.querySelector('.hrb').textContent).toContain('I1O')
  })

  it('says so when the extra checks could not run', async () => {
    getDocumentHoldReasons.mockResolvedValue({
      ...HELD_LOW_CONFIDENCE,
      degraded: ['advisory_unavailable'],
    })
    render(<HoldReasonBanner documentId="doc-1" />)

    expect(
      await screen.findByText(/Additional checks could not be run/i)
    ).toBeInTheDocument()
  })

  it('reports an unreadable status instead of asserting one', async () => {
    getDocumentHoldReasons.mockRejectedValue(new Error('503'))
    render(<HoldReasonBanner documentId="doc-1" />)

    expect(
      await screen.findByText(/could not be determined/i)
    ).toBeInTheDocument()
    expect(screen.queryByText(/held for human review/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/auto-verified/i)).not.toBeInTheDocument()
  })

  it('does not fetch without a document', () => {
    render(<HoldReasonBanner documentId={null} />)
    expect(getDocumentHoldReasons).not.toHaveBeenCalled()
  })
})
