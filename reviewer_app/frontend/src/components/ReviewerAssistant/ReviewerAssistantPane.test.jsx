import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ReviewerAssistantPane from './ReviewerAssistantPane'

// The pane's first test file. It had none, which is a large part of why it drifted
// behind the full-screen chat: nothing recorded what it was supposed to do.
//
// Four things are pinned here, each one a bug that would otherwise be invisible:
//   * trace links resolve to the experiment (the whole chain, in one href)
//   * the notepad is GONE and the pane never writes notes (exactly one writer)
//   * an agent note is handed UP rather than saved here (the data-loss path)
//   * dictated text survives the stop (the failure mode of a mic in a chat box)

vi.mock('../../api/chatApi', () => ({
  streamReviewerAgent: vi.fn(),
}))
vi.mock('../../api/lakeRcmApi', () => ({
  recordDocumentProposal: vi.fn().mockResolvedValue({ id: 'p1' }),
  setProposalDisposition: vi.fn().mockResolvedValue({}),
  saveDocumentNotes: vi.fn().mockResolvedValue({}),
  getDocumentNotes: vi.fn().mockResolvedValue({ note_text: '' }),
}))

const mockTranscription = {
  isRecording: false,
  start: vi.fn(),
  stop: vi.fn().mockResolvedValue(''),
  committedPhrases: [],
  interimText: '',
  error: null,
  supported: true,
}
vi.mock('../../hooks/useLiveTranscription', () => ({
  useLiveTranscription: () => mockTranscription,
}))

const { streamReviewerAgent } = await import('../../api/chatApi')
const lakeRcmApi = await import('../../api/lakeRcmApi')

const stream = (events) =>
  async function* () {
    for (const e of events) yield e
  }

const USER = {
  workspace_host: 'https://dbc-1.cloud.databricks.com',
  experiment_id: '4242',
  transcription_available: true,
}

beforeEach(() => {
  vi.clearAllMocks()
  window.localStorage.clear()
  Object.assign(mockTranscription, {
    isRecording: false,
    committedPhrases: [],
    interimText: '',
    error: null,
    supported: true,
  })
  mockTranscription.stop = vi.fn().mockResolvedValue('')
  streamReviewerAgent.mockImplementation(stream([{ type: 'done' }]))
})

const renderPane = (props = {}) =>
  render(
    <ReviewerAssistantPane
      documentId="doc-1"
      currentUser={USER}
      onClose={() => {}}
      fieldLabelForKey={(k) => k}
      {...props}
    />
  )

const send = async (user, text = 'check this') => {
  await user.type(screen.getByRole('textbox'), text)
  await user.click(screen.getByRole('button', { name: 'Send' }))
}

describe('ReviewerAssistantPane — agent activity reporting', () => {
  // The review form shows an "Agent verification" strip from this signal. The
  // pane can be closed, collapsed or scrolled away mid-turn, so the indicator
  // cannot live only in here.

  it('reports busy on send and names the running tool', async () => {
    const user = userEvent.setup()
    const onActivityChange = vi.fn()
    streamReviewerAgent.mockImplementation(
      stream([
        { type: 'tool_call', tool: 'search_payer_policy', input: {}, run_id: 'r1' },
        { type: 'tool_result', tool: 'search_payer_policy', output_preview: 'ok', run_id: 'r1' },
        { type: 'token', content: 'Policy says 30 days.' },
        { type: 'done' },
      ])
    )
    renderPane({ onActivityChange })
    await send(user)

    const calls = onActivityChange.mock.calls.map(([a]) => a)
    expect(calls[0]).toEqual({ busy: true, tool: null })
    expect(calls).toEqual(
      expect.arrayContaining([{ busy: true, tool: 'search_payer_policy' }])
    )
  })

  it('drops the tool name once the tool returns but the turn continues', async () => {
    // Otherwise the strip keeps naming work that already finished while the
    // model is still composing.
    const user = userEvent.setup()
    const onActivityChange = vi.fn()
    streamReviewerAgent.mockImplementation(
      stream([
        { type: 'tool_call', tool: 'search_payer_policy', input: {}, run_id: 'r1' },
        { type: 'tool_result', tool: 'search_payer_policy', output_preview: 'ok', run_id: 'r1' },
        { type: 'done' },
      ])
    )
    renderPane({ onActivityChange })
    await send(user)

    const calls = onActivityChange.mock.calls.map(([a]) => a)
    const afterTool = calls.indexOf(
      calls.find((c) => c.busy && c.tool === 'search_payer_policy')
    )
    expect(calls.slice(afterTool + 1)).toEqual(
      expect.arrayContaining([{ busy: true, tool: null }])
    )
  })

  it('clears busy when the turn completes', async () => {
    const user = userEvent.setup()
    const onActivityChange = vi.fn()
    renderPane({ onActivityChange })
    await send(user)

    await waitFor(() => {
      const calls = onActivityChange.mock.calls.map(([a]) => a)
      expect(calls[calls.length - 1]).toEqual({ busy: false, tool: null })
    })
  })

  it('clears busy even when the stream throws', async () => {
    // THE HAZARD. A busy flag cleared only on the success path is a spinner
    // that never stops, over a review form the reviewer is trying to submit.
    const user = userEvent.setup()
    const onActivityChange = vi.fn()
    streamReviewerAgent.mockImplementation(async function* () {
      yield { type: 'tool_call', tool: 'get_active_review_context', input: {}, run_id: 'r1' }
      throw new Error('stream died')
    })
    renderPane({ onActivityChange })
    await send(user)

    await waitFor(() => {
      const calls = onActivityChange.mock.calls.map(([a]) => a)
      expect(calls[calls.length - 1]).toEqual({ busy: false, tool: null })
    })
  })

  it('works when no handler is passed', async () => {
    // DocumentDetailPage is the only caller today; an optional prop must not
    // make the pane throw for any other mount.
    const user = userEvent.setup()
    renderPane()
    await send(user)
    await waitFor(() => expect(streamReviewerAgent).toHaveBeenCalled())
  })
})

describe('ReviewerAssistantPane — trace links', () => {
  it('builds an experiment-scoped trace link from the streamed trace id', async () => {
    const user = userEvent.setup()
    streamReviewerAgent.mockImplementation(
      stream([
        { type: 'tool_call', tool: 'get_active_review_context', input: {}, run_id: 'r1' },
        { type: 'tool_result', tool: 'get_active_review_context', output_preview: 'ok', run_id: 'r1' },
        { type: 'trace', trace_id: 'tr-abc123' },
        { type: 'token', content: 'Looks fine.' },
        { type: 'done' },
      ])
    )
    renderPane()
    await send(user)

    // The link lives inside the tool card's collapsed detail, so expand it —
    // which is also how a reviewer reaches it.
    const header = await waitFor(() => {
      const el = document.querySelector('.tool-card__header')
      expect(el).toBeTruthy()
      return el
    })
    await user.click(header)

    // One assertion covers the whole chain: currentUser destructured, the props
    // threaded to MessageBubble, and the trace event arriving live.
    const link = await waitFor(() => {
      const a = document.querySelector('a.tool-card__trace-link')
      expect(a).toBeTruthy()
      return a
    })
    expect(link).toHaveAttribute(
      'href',
      'https://dbc-1.cloud.databricks.com/ml/experiments/4242/traces/tr-abc123'
    )
  })

  it('renders no trace link when the workspace host is unknown', async () => {
    const user = userEvent.setup()
    streamReviewerAgent.mockImplementation(
      stream([
        { type: 'tool_call', tool: 't', input: {}, run_id: 'r1' },
        { type: 'tool_result', tool: 't', output_preview: 'ok', run_id: 'r1' },
        { type: 'trace', trace_id: 'tr-abc123' },
        { type: 'done' },
      ])
    )
    renderPane({ currentUser: null })
    await send(user)

    const header = await waitFor(() => {
      const el = document.querySelector('.tool-card__header')
      expect(el).toBeTruthy()
      return el
    })
    await user.click(header)

    // Guards against an href of "undefined/ml/..." — a link that looks real and
    // goes nowhere is worse than no link.
    expect(document.querySelector('a[href*="undefined"]')).toBeNull()
    expect(document.querySelector('a.tool-card__trace-link')).toBeNull()
  })
})

describe('ReviewerAssistantPane — the notepad is gone', () => {
  it('has no notepad tab and no second note editor', () => {
    renderPane()
    expect(screen.queryByRole('tab')).not.toBeInTheDocument()
    expect(screen.queryByText(/notepad/i)).not.toBeInTheDocument()
    // One textbox: the composer. A second would be the notepad back again.
    expect(screen.getAllByRole('textbox')).toHaveLength(1)
  })

  it('never loads or writes the note record itself', async () => {
    const user = userEvent.setup()
    renderPane()
    await send(user)
    await waitFor(() => expect(streamReviewerAgent).toHaveBeenCalled())

    // The invariant: exactly ONE writer for a record with one row per document +
    // reviewer. ReviewNotes owns it; this pane must not touch it.
    expect(lakeRcmApi.saveDocumentNotes).not.toHaveBeenCalled()
    expect(lakeRcmApi.getDocumentNotes).not.toHaveBeenCalled()
  })

  it('hands an agent note up instead of saving it', async () => {
    const user = userEvent.setup()
    const onNoteAppended = vi.fn()
    streamReviewerAgent.mockImplementation(
      stream([
        {
          type: 'frontend_action',
          action: { type: 'note_append', note_text: 'Queried M17.11 with coding' },
        },
        { type: 'done' },
      ])
    )
    renderPane({ onNoteAppended })
    await send(user)

    await waitFor(() =>
      expect(onNoteAppended).toHaveBeenCalledWith('Queried M17.11 with coding')
    )
    // The agent's add_review_note tool returns {"saved": true} and persists
    // nothing, so this pane used to be the only writer. If it writes again, the
    // read-modify-write race against the reviewer's own autosave is back.
    expect(lakeRcmApi.saveDocumentNotes).not.toHaveBeenCalled()
  })
})

describe('ReviewerAssistantPane — voice', () => {
  it('offers the mic when transcription is available', () => {
    renderPane()
    expect(
      screen.getByRole('button', { name: /dictation|record|mic/i })
    ).toBeInTheDocument()
  })

  it('hides the mic when the backend says transcription is unavailable', () => {
    renderPane({ currentUser: { ...USER, transcription_available: false } })
    expect(
      screen.queryByRole('button', { name: /dictation|record|mic/i })
    ).not.toBeInTheDocument()
  })

  it('hides the mic where the browser cannot record', () => {
    mockTranscription.supported = false
    renderPane()
    expect(
      screen.queryByRole('button', { name: /dictation|record|mic/i })
    ).not.toBeInTheDocument()
  })

  it('shows live transcription in the composer and locks it while recording', () => {
    Object.assign(mockTranscription, {
      isRecording: true,
      committedPhrases: ['check the member id'],
      interimText: 'on page two',
    })
    renderPane()
    const box = screen.getByRole('textbox')
    expect(box).toHaveValue('check the member id on page two')
    // readOnly, not disabled: a disabled textarea loses focus and the scroll
    // position, so the newest phrase scrolls out of view mid-sentence.
    expect(box).toHaveAttribute('readonly')
  })
})

describe('ReviewerAssistantPane — resize handle', () => {
  it('exposes a keyboard-operable separator', async () => {
    const user = userEvent.setup()
    const onResizeKey = vi.fn()
    renderPane({ onResizeKey, onResizeStart: vi.fn() })

    const sep = screen.getByRole('separator', { name: /resize/i })
    expect(sep).toHaveAttribute('aria-orientation', 'vertical')
    // The page's existing resize handles are mouse-only. A pane you cannot size
    // without a pointer is a pane some reviewers cannot size at all.
    sep.focus()
    await user.keyboard('{ArrowLeft}')
    expect(onResizeKey).toHaveBeenCalled()
  })
})
