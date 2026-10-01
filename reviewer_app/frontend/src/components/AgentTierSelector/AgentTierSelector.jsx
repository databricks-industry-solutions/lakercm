import './AgentTierSelector.css'

// One selector, used by BOTH the in-document pane and the full-screen chat.
//
// Deliberately shared rather than implemented twice. The reviewer pane and
// ChatInterface are parallel chat UIs, and that is exactly why the pane spent
// this long without voice input or trace links: a feature added to one surface
// does not arrive at the other. A second copy of this control would drift the
// same way.
//
// "Auto" is the default and means what it did before a selector existed: the
// complexity classifier picks, and a thread pin keeps the choice stable.

export const TIER_OPTIONS = [
  { value: 'auto', label: 'Auto', hint: 'Let the classifier choose per question' },
  { value: 'low', label: 'Low', hint: 'Cheapest and fastest; simple lookups' },
  { value: 'medium', label: 'Medium', hint: 'Default balance of cost and depth' },
  { value: 'high', label: 'High', hint: 'Most capable; multi-step reasoning' },
]

export const TIER_STORAGE_KEY = 'lakercm.agentTier'

// The wire values are low|med|high; the UI says "Medium" because "Med" reads as
// an abbreviation nobody asked for. normalize_requested_tier on the agent side
// accepts both spellings, so this mapping is display-only.
const WIRE_BY_VALUE = { auto: 'auto', low: 'low', medium: 'med', high: 'high' }
const LABEL_BY_WIRE = { low: 'Low', med: 'Medium', high: 'High' }

/** The tier to send on the wire for a stored UI value. */
export const tierToWire = (value) => WIRE_BY_VALUE[value] || 'auto'

/** Human label for a tier that came back from the server. */
export const wireToLabel = (wire) => LABEL_BY_WIRE[wire] || wire || ''

export function readStoredTier() {
  if (typeof window === 'undefined') return 'auto'
  const raw = window.localStorage.getItem(TIER_STORAGE_KEY)
  return TIER_OPTIONS.some((o) => o.value === raw) ? raw : 'auto'
}

export function storeTier(value) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(TIER_STORAGE_KEY, value)
  } catch {
    /* a private-mode storage failure must not break the chat */
  }
}

/**
 * @param {string} value            current selection ('auto'|'low'|'medium'|'high')
 * @param {(v: string) => void} onChange
 * @param {{tier: string, source: string}|null} effective
 *        What the server reported actually serving the last turn. Used ONLY to
 *        explain an override — see below.
 * @param {boolean} compact         tighter styling for the 360px-wide pane
 * @param {boolean} disabled
 */
export default function AgentTierSelector({
  value,
  onChange,
  effective = null,
  compact = false,
  disabled = false,
}) {
  // Safety escalation outranks the selector by design: a turn containing
  // high-risk wording (fraud, reconcile, discrepancy) goes to the high tier even
  // if the reviewer asked for something cheaper. That is a deliberate product
  // decision, and it obliges the UI to SAY so — a control that is silently
  // ignored is worse than no control, because the reviewer draws conclusions
  // about cost and depth that are not true.
  const overridden =
    effective?.source === 'user_override_escalated' &&
    value !== 'auto' &&
    tierToWire(value) !== effective?.tier

  // THE AUTO BLIND SPOT. "Auto" is the default, and on Auto the classifier's
  // choice was never shown anywhere: the override message below only fires for an
  // EXPLICIT selection that got escalated. So the common case -- most reviewers,
  // most turns -- gave no indication of which reasoning tier actually ran, which
  // is the one thing the control is about. Reporting it also makes the selector
  // legible as a real decision rather than a preference that may or may not
  // matter.
  const autoResolved = value === 'auto' && Boolean(effective?.tier)

  return (
    <div className={`ats${compact ? ' ats--compact' : ''}`}>
      <div
        className="ats-group"
        role="radiogroup"
        aria-label="Assistant reasoning tier"
      >
        {TIER_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={value === opt.value}
            className={`ats-option${value === opt.value ? ' is-selected' : ''}`}
            title={opt.hint}
            disabled={disabled}
            onClick={() => onChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>
      {overridden && (
        <span className="ats-override" role="status">
          Escalated to {wireToLabel(effective.tier)} — high-risk wording
        </span>
      )}
      {autoResolved && !overridden && (
        <span className="ats-resolved" role="status">
          Classifier chose {wireToLabel(effective.tier)}
          {effective.source === 'user_override_escalated'
            ? ' — high-risk wording'
            : ''}
        </span>
      )}
    </div>
  )
}
