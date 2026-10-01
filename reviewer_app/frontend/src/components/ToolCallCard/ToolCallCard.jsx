import { useEffect, useState } from 'react'
import './ToolCallCard.css'

const TOOL_LABELS = {
  search_documents: 'Searching documents',
  get_document_details: 'Fetching document details',
  get_extraction_results: 'Retrieving extraction results',
  get_review_statistics: 'Computing review statistics',
  get_recent_reviews: 'Loading recent reviews',
  search_extractions_by_label: 'Searching by label',
}

// Industry-standard duration formatting:
//   < 100ms  -> "47 ms"        (sub-second tools, common for warm Lakebase)
//   < 1000ms -> "0.4s"         (under one second, one-decimal tenths)
//   < 10s    -> "1.2s"         (single-digit seconds, one decimal)
//   >= 10s   -> "34s"          (longer ops, integer seconds)
function formatMs(ms) {
  if (ms == null) return null
  const n = Math.max(0, ms)
  if (n < 100) return `${Math.round(n)} ms`
  if (n < 1000) return `0.${Math.round(n / 100)}s`
  if (n < 10000) return `${(n / 1000).toFixed(1)}s`
  return `${Math.round(n / 1000)}s`
}

function ToolCallCard({ toolCall, traceId, workspaceHost, experimentId }) {
  const [expanded, setExpanded] = useState(false)
  const [now, setNow] = useState(() => Date.now())

  const isRunning = toolCall.status === 'running'

  // Live ticker — only runs while the tool is in flight. Updates every
  // 100ms so the user gets a visible "this is working" affordance.
  useEffect(() => {
    if (!isRunning || !toolCall.startedAt) return undefined
    const id = setInterval(() => setNow(Date.now()), 100)
    return () => clearInterval(id)
  }, [isRunning, toolCall.startedAt])

  const label = TOOL_LABELS[toolCall.tool] || toolCall.tool

  // Prefer the server-measured duration_ms (accurate to the actual tool
  // execution time). Fall back to the SSE-arrival diff if the agent was
  // running an older build that didn't include duration_ms.
  let durationMs = null
  if (isRunning && toolCall.startedAt) {
    durationMs = now - toolCall.startedAt
  } else if (toolCall.duration_ms != null) {
    durationMs = toolCall.duration_ms
  } else if (toolCall.duration != null) {
    durationMs = toolCall.duration
  }
  const durationLabel = formatMs(durationMs)

  // Per-tool MLflow trace deep-link. Verified live 2026-09-29: the agent emits
  // the MLflow 3 id as `tr-<32-hex>` (no slashes), and
  // /ml/experiments/<id>/traces/tr-<hex> redirects to the canonical
  // ?selectedEvaluationId= route and opens that trace. UC-backed trace storage
  // reports the longer `trace:/catalog.schema.table/<32-hex>` form, so keep
  // taking the last `/`-segment — it is a no-op on the `tr-` form and strips the
  // table prefix off the other.
  const traceUuid = traceId
    ? (() => {
        const parts = String(traceId).split('/')
        return parts[parts.length - 1] || traceId
      })()
    : null
  const traceUrl =
    traceUuid && workspaceHost
      ? experimentId
        ? `${workspaceHost}/ml/experiments/${experimentId}/traces/${traceUuid}`
        : `${workspaceHost}/ml/traces/${traceUuid}`
      : null

  return (
    <div className={`tool-card ${isRunning ? 'tool-card--running' : 'tool-card--done'}`}>
      <button className="tool-card__header" onClick={() => setExpanded(!expanded)}>
        <span className="tool-card__indicator">
          {isRunning ? (
            <span className="tool-card__spinner" />
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          )}
        </span>
        <span className="tool-card__label">{label}</span>
        {durationLabel && <span className="tool-card__duration">{durationLabel}</span>}
        <span className={`tool-card__chevron ${expanded ? 'tool-card__chevron--open' : ''}`}>
          &#9662;
        </span>
      </button>
      {expanded && (
        <div className="tool-card__details">
          {toolCall.input && Object.keys(toolCall.input).length > 0 && (
            <div className="tool-card__section">
              <span className="tool-card__section-label">Input</span>
              <pre className="tool-card__json">{JSON.stringify(toolCall.input, null, 2)}</pre>
            </div>
          )}
          {toolCall.output_preview && (
            <div className="tool-card__section">
              <span className="tool-card__section-label">Output</span>
              <pre className="tool-card__json">{toolCall.output_preview.slice(0, 300)}</pre>
            </div>
          )}
          {traceUrl && (
            <div className="tool-card__section">
              <a
                className="tool-card__trace-link"
                href={traceUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                View trace <span aria-hidden="true">↗</span>
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ToolCallCard
