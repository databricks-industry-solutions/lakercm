import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import AgentVerification, { TOOL_PHRASES } from './AgentVerification'

// The "Agent verification" strip beside the review form.
//
// It exists because the assistant pane can be closed, collapsed or scrolled away
// while a turn is still streaming, and the decision the agent informs is made
// over in the review form. A reviewer looking at the verdict radios otherwise has
// no indication that a verdict proposal is seconds away.

describe('AgentVerification', () => {
  it('renders nothing when the agent is idle', () => {
    const { container } = render(<AgentVerification busy={false} tool={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('announces itself politely, not assertively', () => {
    // This appears and disappears on every turn; assertive would interrupt a
    // screen-reader user mid-sentence each time. WCAG 2.2 SC 4.1.3.
    render(<AgentVerification busy tool={null} />)
    const status = screen.getByRole('status')
    expect(status.getAttribute('aria-live')).toBe('polite')
    expect(status.getAttribute('aria-atomic')).toBe('true')
  })

  it('names the work in the reviewer language, not the tool id', () => {
    render(<AgentVerification busy tool="search_payer_policy" />)
    expect(screen.getByText(/Looking up the payer policy/i)).toBeTruthy()
    expect(screen.queryByText(/search_payer_policy/)).toBeNull()
  })

  it('falls back to a generic phrase for an unmapped tool', () => {
    // A new tool in agent/tools.py must not leak its identifier into the UI.
    render(<AgentVerification busy tool="some_future_tool" />)
    expect(screen.getByText(/Reviewing this document/i)).toBeTruthy()
    expect(screen.queryByText(/some_future_tool/)).toBeNull()
  })

  it('shows the generic phrase while the model composes between tools', () => {
    // tool=null with busy=true is the post-tool_result state.
    render(<AgentVerification busy tool={null} />)
    expect(screen.getByText(/Reviewing this document/i)).toBeTruthy()
  })

  it('maps every reviewer-facing tool to a phrase without an identifier', () => {
    for (const phrase of Object.values(TOOL_PHRASES)) {
      expect(phrase).not.toContain('_')
      expect(phrase[0]).toBe(phrase[0].toUpperCase())
    }
  })
})
