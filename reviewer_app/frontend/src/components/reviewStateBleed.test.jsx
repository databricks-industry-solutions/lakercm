import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { ReviewPanel } from './ExtractionComparison'

// Cross-document state bleed in the review form.
//
// DocumentDetailPage is NOT remounted between documents: the route is
// `/review/documents/:documentId` with no `key` (App.jsx:83), `documentId` comes
// from useParams(), and j/k navigation calls navigate() to the SAME route
// (DocumentDetailPage.jsx:212). React therefore keeps the instance — and every
// piece of form state in it — while only the param changes.
//
// The load effect set verdict/reasoning only on the SUCCESS path, and its catch
// reset just `savedCorrections`. So moving from a reviewed document to an
// unreviewed one — whose `GET /review` 404s — left the previous document's
// verdict selected, its reasoning in the textarea, and reviewSaved true, which
// made the button read "Update Review" for a document that has no review. One
// click then submitted the wrong verdict against the wrong document.
//
// This is a correctness bug, not a cosmetic one, which is why it gets its own
// file rather than a case appended to ExtractionComparison.test.jsx (that suite
// deliberately renders only ExtractedDataPanel and mocks nothing).

vi.mock('../api/lakeRcmApi', () => ({
  getExtractionReview: vi.fn(),
  submitExtractionReview: vi.fn(),
  getDocumentHoldReasons: vi.fn(),
  getDocumentRemediation: vi.fn(),
  // ReviewPanel now autosaves an unsubmitted draft through useReviewDraft, so
  // these have to exist on the mock or the hook's load effect throws.
  getReviewDraft: vi.fn(),
  saveReviewDraft: vi.fn(),
  discardReviewDraft: vi.fn(),
  beaconReviewDraft: vi.fn(),
}))

const {
  getExtractionReview,
  getDocumentHoldReasons,
  getDocumentRemediation,
  getReviewDraft,
  saveReviewDraft,
  discardReviewDraft,
} = await import('../api/lakeRcmApi')

const NO_DRAFT = { exists: false }

const REVIEWED = {
  verdict: 'incorrect',
  reasoning: 'The member ID on this claim is wrong.',
  corrections: { 'id:0': 'CORRECTED-A' },
}

beforeEach(() => {
  vi.clearAllMocks()
  // Children of ReviewPanel fetch on mount; keep them inert and quiet.
  getDocumentHoldReasons.mockResolvedValue({ state: 'unknown', reasons: [] })
  getDocumentRemediation.mockResolvedValue({ findings: [] })
  // Default: no draft anywhere. Individual tests opt in.
  getReviewDraft.mockResolvedValue(NO_DRAFT)
  saveReviewDraft.mockResolvedValue({ exists: true })
  discardReviewDraft.mockResolvedValue({ deleted: true })
})

function panel(documentId) {
  return (
    <ReviewPanel
      documentId={documentId}
      hasRecord
      corrections={{}}
      onCorrectionsLoaded={() => {}}
      proposedReview={null}
    />
  )
}

describe('review form state does not bleed between documents', () => {
  it('clears a previous document verdict when the next one has no review', async () => {
    // doc-1 is reviewed; doc-2 has no review, so the endpoint rejects.
    getExtractionReview.mockImplementation((id) =>
      id === 'doc-1'
        ? Promise.resolve(REVIEWED)
        : Promise.reject(new Error('Request failed with status code 404')),
    )

    const { rerender } = render(panel('doc-1'))
    await waitFor(() =>
      expect(screen.getByDisplayValue(REVIEWED.reasoning)).toBeTruthy(),
    )

    rerender(panel('doc-2'))

    // The reasoning textarea must be empty, not carrying doc-1's text.
    await waitFor(() =>
      expect(screen.queryByDisplayValue(REVIEWED.reasoning)).toBeNull(),
    )
    // And no verdict radio may be left selected.
    const checked = screen
      .queryAllByRole('radio')
      .filter((r) => r.checked)
    expect(checked).toHaveLength(0)
  })

  it('does not claim an unreviewed document already has a review', async () => {
    // reviewSaved drove the button label. Left true from the previous document,
    // it told the reviewer they were UPDATING a review that does not exist.
    //
    // doc-1's review carries NO corrections here on purpose: the label is
    // `reviewSaved && unsavedCount === 0` (:1221-1223), so a corrections
    // mismatch against the `corrections` prop would read "Submit Review"
    // regardless and the test would prove nothing about reviewSaved.
    getExtractionReview.mockImplementation((id) =>
      id === 'doc-1'
        ? Promise.resolve({ verdict: 'correct', reasoning: 'Looks right.' })
        : Promise.reject(new Error('Request failed with status code 404')),
    )

    const { rerender } = render(panel('doc-1'))
    await waitFor(() => expect(screen.getByText(/Update Review/i)).toBeTruthy())

    rerender(panel('doc-2'))

    await waitFor(() => expect(screen.queryByText(/Update Review/i)).toBeNull())
  })

  it('still loads a real review for a document that has one', async () => {
    // Regression guard on the reset: clearing unconditionally must not stop the
    // success path from populating the form.
    getExtractionReview.mockResolvedValue(REVIEWED)
    render(panel('doc-1'))
    await waitFor(() =>
      expect(screen.getByDisplayValue(REVIEWED.reasoning)).toBeTruthy(),
    )
  })
})
