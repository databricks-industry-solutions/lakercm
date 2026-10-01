import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ReviewNotes from './ReviewNotes'

// Autosave is the kind of feature that looks fine and loses work. The status text
// saying "Saved" proves nothing on its own — these assert the REQUEST went out,
// with the right body, and that the paths which can drop keystrokes do not.

vi.mock('../api/lakeRcmApi', () => ({
  getDocumentNotes: vi.fn(),
  saveDocumentNotes: vi.fn(),
}))

const { getDocumentNotes, saveDocumentNotes } = await import('../api/lakeRcmApi')

beforeEach(() => {
  getDocumentNotes.mockResolvedValue({ note_text: '' })
  saveDocumentNotes.mockResolvedValue({ note_text: '' })
})

const lastSaved = () =>
  saveDocumentNotes.mock.calls[saveDocumentNotes.mock.calls.length - 1]

describe('ReviewNotes', () => {
  it('loads the existing note for the document', async () => {
    getDocumentNotes.mockResolvedValue({ note_text: 'Queried M17.11' })
    render(<ReviewNotes documentId="doc-1" />)

    await waitFor(() =>
      expect(screen.getByRole('textbox')).toHaveValue(
        'Queried M17.11'
      )
    )
    expect(getDocumentNotes).toHaveBeenCalledWith('doc-1')
  })

  it('autosaves what was typed, with the text', async () => {
    const user = userEvent.setup()
    render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    await user.type(screen.getByRole('textbox'), 'Checked p2')

    await waitFor(() => expect(saveDocumentNotes).toHaveBeenCalled(), {
      timeout: 4000,
    })
    expect(lastSaved()).toEqual(['doc-1', 'Checked p2'])
    expect(await screen.findByText(/^Saved$/)).toBeInTheDocument()
  })

  it('debounces instead of saving on every keystroke', async () => {
    const user = userEvent.setup()
    render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    await user.type(screen.getByRole('textbox'), 'abcdefghij')

    await waitFor(() => expect(saveDocumentNotes).toHaveBeenCalled(), {
      timeout: 4000,
    })
    // Ten keystrokes must not be ten writes to Lakebase.
    expect(saveDocumentNotes.mock.calls.length).toBeLessThan(4)
  })

  it('flushes a pending edit when the document changes', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    await user.type(screen.getByRole('textbox'), 'half a thought')
    // Switch documents while the debounce is still pending: this is the
    // verdict-then-next-document sequence, and it is where an autosaver loses the
    // reviewer's last sentence.
    rerender(<ReviewNotes documentId="doc-2" />)

    await waitFor(() =>
      expect(saveDocumentNotes).toHaveBeenCalledWith('doc-1', 'half a thought')
    )
  })

  it('reports a failed save instead of showing Saved', async () => {
    const user = userEvent.setup()
    saveDocumentNotes.mockRejectedValue(new Error('503'))
    render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    await user.type(screen.getByRole('textbox'), 'x')

    expect(await screen.findByText(/not saved/i, {}, { timeout: 4000 })).toBeInTheDocument()
    expect(screen.queryByText(/^Saved$/)).not.toBeInTheDocument()
  })

  it('renders markdown in preview rather than literal syntax', async () => {
    const user = userEvent.setup()
    getDocumentNotes.mockResolvedValue({ note_text: 'Queried **M17.11**' })
    render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() =>
      expect(screen.getByRole('textbox')).toHaveValue(
        'Queried **M17.11**'
      )
    )

    await user.click(screen.getByRole('button', { name: 'Preview' }))

    // Bold as an element, not asterisks on screen — the whole point of markdown.
    const strong = document.querySelector('.rn-preview strong')
    expect(strong).toHaveTextContent('M17.11')
    expect(document.querySelector('.rn-preview').textContent).not.toContain('**')
  })

  it('disables preview while there is nothing to render', async () => {
    render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    expect(screen.getByRole('button', { name: 'Preview' })).toBeDisabled()
  })

  it('a reload does not clobber an unsaved edit', async () => {
    const user = userEvent.setup()
    render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    await user.type(screen.getByRole('textbox'), 'local edit')

    // The focus listener exists because this note record has two editors (this
    // pane and the assistant notepad). Refetching must never overwrite what the
    // reviewer is mid-way through typing, or the mitigation causes the data loss
    // it was added to prevent.
    getDocumentNotes.mockResolvedValue({ note_text: 'text from the assistant' })
    window.dispatchEvent(new Event('focus'))

    await new Promise((r) => setTimeout(r, 50))
    expect(screen.getByRole('textbox')).toHaveValue('local edit')
  })

  it('merges an agent append on top of unsaved local edits', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    await user.type(screen.getByRole('textbox'), 'checked p2')
    // The assistant appends WHILE there is an unsaved local edit. This is the
    // race the old design lost: the pane would GET the (stale) server text,
    // append, and POST — and the reviewer's own pending save would then overwrite
    // it, or vice versa. Applying it here, on the live buffer, has one writer.
    rerender(
      <ReviewNotes documentId="doc-1" appendRequest={{ id: 1, text: 'agent line' }} />
    )

    await waitFor(() =>
      expect(screen.getByRole('textbox')).toHaveValue('checked p2\nagent line')
    )
    await waitFor(() => expect(lastSaved()).toEqual(['doc-1', 'checked p2\nagent line']), {
      timeout: 4000,
    })
  })

  it('applies an append request only once', async () => {
    const { rerender } = render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    const req = { id: 7, text: 'one note' }
    rerender(<ReviewNotes documentId="doc-1" appendRequest={req} />)
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue('one note'))
    // A parent state change re-renders with the same request; without the id
    // guard the note would double every time anything else on the page moved.
    rerender(<ReviewNotes documentId="doc-1" appendRequest={req} />)
    await new Promise((r) => setTimeout(r, 30))
    expect(screen.getByRole('textbox')).toHaveValue('one note')
  })

  it('appends to an empty note without a leading blank line', async () => {
    const { rerender } = render(<ReviewNotes documentId="doc-1" />)
    await waitFor(() => expect(getDocumentNotes).toHaveBeenCalled())

    rerender(
      <ReviewNotes documentId="doc-1" appendRequest={{ id: 2, text: 'first note' }} />
    )
    await waitFor(() =>
      expect(screen.getByRole('textbox')).toHaveValue('first note')
    )
  })

  it('does not fetch without a document', () => {
    render(<ReviewNotes documentId={null} />)
    expect(getDocumentNotes).not.toHaveBeenCalled()
  })
})
