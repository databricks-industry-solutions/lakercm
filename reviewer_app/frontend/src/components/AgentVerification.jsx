import { wireToLabel } from './AgentTierSelector/AgentTierSelector'
import './AgentVerification.css'

// What the agent is actually doing, in the reviewer's language.
//
// Tool names are the contract with agent_app/agent/tools.py. An unmapped tool
// falls back to the generic phrase rather than leaking an identifier like
// `search_document_chunks` into the UI.
const TOOL_PHRASES = {
  get_active_review_context: 'Reading the open document',
  get_review_remediation: 'Checking why this was held',
  search_document_chunks: 'Searching the document text',
  semantic_search_documents: 'Searching similar documents',
  get_document_details: 'Reading the document record',
  get_extraction_results: 'Reading the extracted fields',
  search_payer_policy: 'Looking up the payer policy',
  traverse_claims_graph: 'Following the claims graph',
  query_lakehouse: 'Querying the lakehouse',
  propose_extraction_edit: 'Preparing a field correction',
  propose_review_verdict: 'Preparing a verdict',
  add_review_note: 'Writing a note',
}

/**
 * "Agent verification" strip, shown beside the review form while the assistant
 * is reasoning over this document.
 *
 * WHY IT LIVES HERE AND NOT ONLY IN THE ASSISTANT PANE. The agent's work is
 * about the held document's error, and the decision it informs is made in the
 * review form — but the pane can be closed, collapsed, or scrolled away while a
 * turn is still streaming. Without this, a reviewer looking at the verdict
 * radios has no indication that a verdict proposal is seconds away, so they
 * either submit early or assume nothing is happening.
 *
 * Deliberately NOT shown for RemediationPanel's load: that shortlist is
 * computed deterministically in code from the curated terminology
 * (services/remediation.py), so calling it "agent verification" would be a
 * plain misstatement of what ran.
 *
 * @param {boolean} busy       whether a turn is in flight
 * @param {string|null} tool   the tool currently running, if any
 * @param {string|null} tier   reasoning tier actually serving this turn (low|med|high)
 * @param {string|null} tierSource why that tier -- 'user_override_escalated' when
 *        high-risk wording overrode the reviewer's own choice
 */
export default function AgentVerification({ busy, tool, tier, tierSource }) {
  if (!busy) return null
  const phrase = (tool && TOOL_PHRASES[tool]) || 'Reviewing this document'
  const escalated = tierSource === 'user_override_escalated'
  return (
    <div
      className="agent-verify"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <span className="agent-verify-spinner" aria-hidden="true" />
      <span className="agent-verify-text">
        <strong>Agent verification</strong>
        <span className="agent-verify-detail">{phrase}…</span>
      </span>
      {/* The reasoning strength actually serving this turn. Shown here because
          the selector lives in the assistant pane's toolbar, which is invisible
          while the pane is closed -- and cost/depth is exactly what a reviewer
          wants to know before trusting a proposed verdict. */}
      {tier && (
        <span
          className={`agent-verify-tier${escalated ? ' agent-verify-tier--escalated' : ''}`}
          title={
            escalated
              ? 'Escalated automatically: high-risk wording overrides the selected tier'
              : 'Reasoning tier serving this turn'
          }
        >
          {wireToLabel(tier)}
          {escalated ? ' \u2191' : ''}
        </span>
      )}
    </div>
  )
}

export { TOOL_PHRASES }
