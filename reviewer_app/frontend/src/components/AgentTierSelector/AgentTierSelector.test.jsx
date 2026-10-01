import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import AgentTierSelector, {
  TIER_STORAGE_KEY,
  readStoredTier,
  storeTier,
  tierToWire,
} from './AgentTierSelector'

// The control is simple; the thing worth testing is the ESCALATION notice.
// Safety escalation outranks an explicit selection by deliberate decision, which
// only works as a product if the UI admits it. A selector that silently disagrees
// with what actually ran teaches reviewers something false about cost and depth.

beforeEach(() => {
  window.localStorage.clear()
})

describe('tier storage', () => {
  it('defaults to auto with nothing stored', () => {
    expect(readStoredTier()).toBe('auto')
  })

  it('round-trips a stored choice', () => {
    storeTier('high')
    expect(window.localStorage.getItem(TIER_STORAGE_KEY)).toBe('high')
    expect(readStoredTier()).toBe('high')
  })

  it('ignores a value it does not recognise', () => {
    window.localStorage.setItem(TIER_STORAGE_KEY, 'turbo')
    expect(readStoredTier()).toBe('auto')
  })
})

describe('tierToWire', () => {
  it('maps the UI label onto the tier the pipeline names', () => {
    // The UI says "Medium" because "Med" reads as an abbreviation nobody asked
    // for; the tiers are low|med|high.
    expect(tierToWire('medium')).toBe('med')
    expect(tierToWire('low')).toBe('low')
    expect(tierToWire('high')).toBe('high')
    expect(tierToWire('auto')).toBe('auto')
    expect(tierToWire(undefined)).toBe('auto')
  })
})

describe('AgentTierSelector', () => {
  it('offers all four tiers and marks the current one', () => {
    render(<AgentTierSelector value="medium" onChange={() => {}} />)
    const options = screen.getAllByRole('radio')
    expect(options.map((o) => o.textContent)).toEqual([
      'Auto',
      'Low',
      'Medium',
      'High',
    ])
    expect(screen.getByRole('radio', { name: 'Medium' })).toHaveAttribute(
      'aria-checked',
      'true'
    )
  })

  it('reports the chosen tier', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<AgentTierSelector value="auto" onChange={onChange} />)

    await user.click(screen.getByRole('radio', { name: 'High' }))
    expect(onChange).toHaveBeenCalledWith('high')
  })

  it('says so when a safety cue overrode the selection', async () => {
    render(
      <AgentTierSelector
        value="low"
        onChange={() => {}}
        effective={{ tier: 'high', source: 'user_override_escalated' }}
      />
    )
    expect(await screen.findByRole('status')).toHaveTextContent(
      /Escalated to High/i
    )
  })

  it('stays quiet when the tier that ran is the one that was asked for', () => {
    render(
      <AgentTierSelector
        value="high"
        onChange={() => {}}
        effective={{ tier: 'high', source: 'complexity_cue' }}
      />
    )
    // Nothing was overridden — the reviewer and the cue agree. A warning here
    // would flag a screen with nothing wrong on it.
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('reports the classifier choice on auto, without calling it an override', () => {
    // CHANGED DELIBERATELY. This used to assert the selector stayed silent on
    // auto. Silence was the wrong call: auto is the DEFAULT, so the common case
    // -- most reviewers, most turns -- gave no indication of which reasoning tier
    // actually ran, which is the one thing the control is about. It is reported
    // now, but as a delegated decision rather than an override: nothing
    // contradicted the reviewer, so nothing should read as a warning.
    render(
      <AgentTierSelector
        value="auto"
        onChange={() => {}}
        effective={{ tier: 'high', source: 'complexity_cue' }}
      />
    )
    expect(screen.getByText(/Classifier chose High/i)).toBeTruthy()
    expect(screen.queryByText(/Escalated/i)).toBeNull()
  })

  it('says nothing on auto until a turn has actually reported a tier', () => {
    // No effective routing yet means no claim to make.
    render(<AgentTierSelector value="auto" onChange={() => {}} effective={null} />)
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('prefers the escalation message over the classifier line', () => {
    // Both could apply to a turn; an escalation contradicts an explicit choice
    // and must not be softened into "the classifier chose".
    render(
      <AgentTierSelector
        value="low"
        onChange={() => {}}
        effective={{ tier: 'high', source: 'user_override_escalated' }}
      />
    )
    expect(screen.getByText(/Escalated to High/i)).toBeTruthy()
    expect(screen.queryByText(/Classifier chose/i)).toBeNull()
  })

  it('disables every option while a turn is in flight', () => {
    render(<AgentTierSelector value="auto" onChange={() => {}} disabled />)
    screen.getAllByRole('radio').forEach((o) => expect(o).toBeDisabled())
  })
})
