import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { ReviewPanel } from './ExtractionComparison'

// Draft autosave for the review form.
//
// Before this, nothing in the form persisted until Submit. A reviewer could pick
// a verdict, type reasoning, correct three fields, press `j` to peek at the next
// document, and lose all of it -- DocumentDetailPage is not remounted between
// documents (see reviewStateBleed.test.jsx), so the form is simply reset.
//
// The riskiest part is not the saving, it is the ORDER. ReviewPanel resets
// verdict/reasoning to '' at the top of its documentId effect, synchronously,
// before the draft GET resolves. An autosave wired naively to that state fires
// on the reset and writes an EMPTY draft over the good one -- destroying the work
// the feature exists to protect. `loadedFor` in useReviewDraft is what prevents
// it, and the third test here is the one that proves it.

vi.mock('../api/lakeRcmApi', () => ({
  getExtractionReview: vi.fn(),
  submitExtractionReview: vi.fn(),
  getDocumentHoldReasons: vi.fn(),
  getDocumentRemediation: vi.fn(),
  getReviewDraft: vi.fn(),
  saveReviewDraft: vi.fn(),
  discardReviewDraft: vi.fn(),
  beaconReviewDraft: vi.fn(),
}))

const {
  getExtractionReview,
  submitExtractionReview,
  getDocumentHoldReasons,
  getDocumentRemediation,
  getReviewDraft,
  saveReviewDraft,
  discardReviewDraft,
} = await import('../api/lakeRcmApi')

const NO_REVIEW = () =>
  Promise.reject(new Error('Request failed with status code 404'))

const DRAFT = {
  exists: true,
  verdict: 'partially_correct',
  reasoning: 'Dx code on page 2 looks wrong.',
  corrections: { 'id:3': 'E11.9' },
}

beforeEach(() => {
  vi.clearAllMocks()
  getDocumentHoldReasons.mockResolvedValue({ state: 'unknown', reasons: [] })
  getDocumentRemediation.mockResolvedValue({ findings: [] })
  getExtractionReview.mockImplementation(NO_REVIEW)
  getReviewDraft.mockResolvedValue({ exists: false })
  saveReviewDraft.mockResolvedValue({ exists: true })
  discardReviewDraft.mockResolvedValue({ deleted: true })
  submitExtractionReview.mockResolvedValue({})
})

function panel(documentId, { corrections = {}, onCorrectionsLoaded } = {}) {
  return (
    <ReviewPanel
      documentId={documentId}
      hasRecord
      corrections={corrections}
      onCorrectionsLoaded={onCorrectionsLoaded || (() => {})}
      proposedReview={null}
    />
  )
}

describe('review draft autosave', () => {
  it('restores a saved draft into the form', async () => {
    const onCorrectionsLoaded = vi.fn()
    getReviewDraft.mockResolvedValue(DRAFT)

    render(panel('doc-1', { onCorrectionsLoaded }))

    await waitFor(() =>
      expect(screen.getByDisplayValue(DRAFT.reasoning)).toBeTruthy(),
    )
    const checked = screen.queryAllByRole('radio').filter((r) => r.checked)
    expect(checked).toHaveLength(1)
    expect(checked[0].value).toBe('partially_correct')
    // Corrections live in the parent, so they go back up rather than into state.
    expect(onCorrectionsLoaded).toHaveBeenCalledWith(DRAFT.corrections)
  })

  it('prefers the draft over an already-submitted review', async () => {
    // Both exist and they are two independent fetches, so without an explicit
    // precedence rule the slower one wins at random. The draft is the newer,
    // unsubmitted edit -- and it only exists until a submit deletes it.
    getExtractionReview.mockResolvedValue({
      verdict: 'correct',
      reasoning: 'Submitted earlier: looked fine.',
    })
    getReviewDraft.mockResolvedValue(DRAFT)

    render(panel('doc-1'))

    await waitFor(() =>
      expect(screen.getByDisplayValue(DRAFT.reasoning)).toBeTruthy(),
    )
    expect(screen.queryByDisplayValue(/Submitted earlier/)).toBeNull()
  })

  it('never autosaves before the draft has loaded', async () => {
    // THE GUARD. Hold the GET open, let the form reset run, and assert nothing
    // is written. Without loadedFor this writes {verdict:null,reasoning:null}
    // over a real draft and the reviewer's work is gone.
    let release
    getReviewDraft.mockReturnValue(
      new Promise((resolve) => {
        release = () => resolve(DRAFT)
      }),
    )

    const { rerender } = render(panel('doc-1', { corrections: {} }))

    // A corrections change arriving while the GET is still open must not save.
    rerender(panel('doc-1', { corrections: { 'id:9': 'X' } }))
    await Promise.resolve()
    expect(saveReviewDraft).not.toHaveBeenCalled()

    release()
    await waitFor(() =>
      expect(screen.getByDisplayValue(DRAFT.reasoning)).toBeTruthy(),
    )
  })

  it('flushes a verdict immediately rather than debouncing it', async () => {
    render(panel('doc-1'))
    await waitFor(() => expect(getReviewDraft).toHaveBeenCalled())

    const incorrect = screen
      .getAllByRole('radio')
      .find((r) => r.value === 'incorrect')
    fireEvent.click(incorrect)

    // No timer advance: a radio selection is one discrete event, so waiting
    // only widens the window in which it can be lost.
    await waitFor(() => expect(saveReviewDraft).toHaveBeenCalled())
    const [, payload] = saveReviewDraft.mock.calls.at(-1)
    expect(payload.verdict).toBe('incorrect')
  })

  it('debounces reasoning instead of saving every keystroke', async () => {
    render(panel('doc-1'))
    await waitFor(() => expect(getReviewDraft).toHaveBeenCalled())

    fireEvent.click(
      screen.getAllByRole('radio').find((r) => r.value === 'incorrect'),
    )
    await waitFor(() => expect(saveReviewDraft).toHaveBeenCalled())
    const afterVerdict = saveReviewDraft.mock.calls.length

    const box = await screen.findByPlaceholderText(/Explain what was incorrect/i)
    fireEvent.change(box, { target: { value: 'a' } })
    fireEvent.change(box, { target: { value: 'ab' } })
    fireEvent.change(box, { target: { value: 'abc' } })

    // Still nothing extra: three keystrokes coalesce into one pending write.
    expect(saveReviewDraft.mock.calls.length).toBe(afterVerdict)

    await waitFor(
      () => expect(saveReviewDraft.mock.calls.length).toBeGreaterThan(afterVerdict),
      { timeout: 3000 },
    )
    expect(saveReviewDraft.mock.calls.at(-1)[1].reasoning).toBe('abc')
  })

  it('does not create a draft row for a form nobody touched', async () => {
    // Otherwise every document merely OPENED would show a Draft badge in the
    // queue, and the badge would stop meaning anything.
    render(panel('doc-1'))
    await waitFor(() => expect(getReviewDraft).toHaveBeenCalled())
    await new Promise((r) => setTimeout(r, 1500))
    expect(saveReviewDraft).not.toHaveBeenCalled()
  })

  it('discards the draft once the review is submitted', async () => {
    // The server deletes it too; this clears the client's copy so the form stops
    // showing submitted work as outstanding.
    getReviewDraft.mockResolvedValue(DRAFT)
    render(panel('doc-1', { corrections: DRAFT.corrections }))
    await waitFor(() =>
      expect(screen.getByDisplayValue(DRAFT.reasoning)).toBeTruthy(),
    )

    fireEvent.click(screen.getByRole('button', { name: /Review/i }))

    await waitFor(() => expect(submitExtractionReview).toHaveBeenCalled())
    await waitFor(() => expect(discardReviewDraft).toHaveBeenCalledWith('doc-1'))
  })
})
