import React, { useState, useEffect, useMemo, useRef, createContext, useContext } from 'react';
import {
  getExtractionComparisons,
  submitExtractionReview,
  getExtractionReview,
} from '../api/lakeRcmApi';
import RemediationPanel from './RemediationPanel';
import HoldReasonBanner from './HoldReasonBanner';
import { useReviewDraft } from '../hooks/useReviewDraft';
import { formatConfidencePct } from '../lib/confidence';
import './ExtractionComparison.css';

const VERDICT_OPTIONS = [
  { value: 'correct', label: 'Correct', icon: '✓' },
  { value: 'partially_correct', label: 'Partially Correct', icon: '~' },
  { value: 'incorrect', label: 'Incorrect', icon: '✗' },
];

// Fallback only. The authoritative value is the `auto_verdict_threshold` bundle
// variable (variables.yml), surfaced to the frontend via /api/me and supplied
// through AutoVerdictThresholdContext. This constant is used solely when that
// backend value is unavailable (e.g. before /api/me resolves).
export const AUTO_VERDICT_THRESHOLD = 0.92;

// Holds the live threshold sourced from the backend. App.jsx wraps the tree with
// the provider, passing currentUser.auto_verdict_threshold; consumers read it via
// useAutoVerdictThreshold() so the displayed threshold never drifts from the
// pipeline-side source of truth.
const AutoVerdictThresholdContext = createContext(AUTO_VERDICT_THRESHOLD);

export function AutoVerdictThresholdProvider({ value, children }) {
  const threshold =
    typeof value === 'number' && value > 0 ? value : AUTO_VERDICT_THRESHOLD;
  return (
    <AutoVerdictThresholdContext.Provider value={threshold}>
      {children}
    </AutoVerdictThresholdContext.Provider>
  );
}

export function useAutoVerdictThreshold() {
  return useContext(AutoVerdictThresholdContext);
}

function cleanFieldName(name, sectionValues) {
  const nameLower = name.toLowerCase();
  for (const prefix of sectionValues) {
    const prefixLower = prefix.toLowerCase();
    if (nameLower.startsWith(prefixLower + ' ') && name.length > prefix.length + 1) {
      const stripped = name.substring(prefix.length).trim();
      if (stripped.length > 0) return stripped;
    }
  }
  const words = name.split(' ');
  const half = Math.floor(words.length / 2);
  if (half > 0) {
    const first = words.slice(0, half).join(' ').toLowerCase();
    const second = words.slice(half).join(' ').toLowerCase();
    if (first === second) return words.slice(half).join(' ');
  }
  return name;
}

/**
 * Some upstream extractor variants emit names/values as JSON-encoded
 * `{"value": "..."}` strings instead of raw strings. Detect that shape and
 * unwrap it once at parse time so the row/section/table-header regexes below
 * can match the literal name (otherwise everything ends up in `topLevel`).
 */
function unwrapJsonString(s) {
  if (typeof s !== 'string') return s;
  if (s.length === 0 || s[0] !== '{') return s;
  try {
    const parsed = JSON.parse(s);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      const inner = parsed.value ?? parsed.VALUE;
      if (inner != null) return String(inner);
    }
  } catch {
    // Not valid JSON — return as-is.
  }
  return s;
}

/**
 * Parse flat identifiers + stable field keys so we can wire attribution hover
 * state. Each field gets a unique `key` derived from its (section, index) and
 * carries through the per-field confidence + citations from ai_extract v2.1.
 */
export function parseIdentifiers(identifiers) {
  const topLevel = [];
  const sectionHeaders = [];
  const tableHeaders = [];
  const rows = {};

  (identifiers || []).forEach((id, originalIdx) => {
    const raw = unwrapJsonString(id.name || '');
    const value = unwrapJsonString(id.value);
    const confidence =
      typeof id.confidence === 'number' ? id.confidence : null;
    const citations = Array.isArray(id.citations) ? id.citations : null;
    const norm = raw.toLowerCase().replace(/_/g, ' ');
    const rowMatch = norm.match(/^row\s+(\d+)\s+(.+)$/);
    const sectionMatch = norm.match(/^section\s+header\s+(.+)$/);
    const tableMatch = norm.match(/^table\s+header\s+(.+)$/);

    const base = {
      value,
      confidence,
      citations,
      key: `id:${originalIdx}`,
    };

    if (rowMatch) {
      const num = rowMatch[1];
      if (!rows[num]) rows[num] = [];
      rows[num].push({ ...base, name: rowMatch[2] });
    } else if (sectionMatch) {
      sectionHeaders.push({ ...base, name: sectionMatch[1] });
    } else if (tableMatch) {
      tableHeaders.push({ ...base, name: tableMatch[1] });
    } else {
      topLevel.push({ ...base, name: raw.replace(/_/g, ' ') });
    }
  });

  const sectionValues = sectionHeaders.map((h) => h.value).filter(Boolean);
  topLevel.forEach((field) => {
    field.name = cleanFieldName(field.name, sectionValues);
  });

  const sortedRows = Object.keys(rows)
    .sort((a, b) => parseInt(a) - parseInt(b))
    .map((num) => ({ rowNum: num, fields: rows[num] }));

  return { topLevel, sectionHeaders, tableHeaders, rows: sortedRows };
}

/**
 * Test whether two `[x0, y0, x1, y1]` rectangles overlap, with a small
 * tolerance to absorb rounding differences between ai_extract citations
 * and ai_parse_document element bboxes.
 */
function rectsOverlap(a, b, tolerance = 2) {
  return !(
    a[2] + tolerance < b[0] ||
    b[2] + tolerance < a[0] ||
    a[3] + tolerance < b[1] ||
    b[3] + tolerance < a[1]
  );
}

/**
 * Resolve an ai_extract citation `{coord, page_id}` to the indices of every
 * element whose bbox overlaps it on the same page. The model's citation may
 * span multiple elements (e.g. a multi-line value), so we return all matches
 * rather than just the closest one.
 */
function findElementsByCitation(citation, elements) {
  const matches = [];
  if (
    !citation ||
    !Array.isArray(citation.coord) ||
    citation.coord.length < 4
  ) {
    return matches;
  }
  const cRect = citation.coord;
  const cPage = citation.page_id;
  for (let ei = 0; ei < elements.length; ei++) {
    const el = elements[ei];
    const bb = el?.bounding_box;
    if (!bb) continue;
    const ePage = (el.page_number ?? 1) - 1;
    if (ePage !== cPage) continue;
    const eRect = [bb.x, bb.y, bb.x + bb.width, bb.y + bb.height];
    if (rectsOverlap(eRect, cRect)) matches.push(ei);
  }
  return matches;
}

/**
 * Substring-match fallback for legacy rows (or future field types) that
 * don't carry per-field citations. Kept narrow — capped at 3 matches and
 * skipping values shorter than 3 chars to limit "Yes" / "1" false positives.
 */
function attributeBySubstring(value, normElements) {
  const v = (value ?? '').toString().toLowerCase().replace(/\s+/g, ' ').trim();
  if (!v || v === '—') return [];
  return normElements
    .filter(({ text }) => {
      if (!text) return false;
      if (text === v) return true;
      if (v.length >= 3 && text.includes(v)) return true;
      return false;
    })
    .sort((a, b) => a.text.length - b.text.length)
    .slice(0, 3)
    .map(({ ei }) => ei);
}

/**
 * Build authoritative bidirectional attribution between extracted identifiers
 * and parsed document elements. Prefers ai_extract v2.1 per-field citations
 * (every highlighted bbox is the model's own source — no heuristic guesswork),
 * and falls back to substring matching only when an identifier arrived
 * without citations (legacy rows). The fallback is what makes this safe to
 * ship before the silver full refresh has reprocessed every document.
 */
export function buildAttribution(identifiers, elements) {
  const forward = new Map();
  const inverse = new Map();
  if (!identifiers || !elements || elements.length === 0) {
    return { forward, inverse };
  }

  const normElements = elements.map((el, ei) => ({
    ei,
    text: (el?.text_content ?? '')
      .toString()
      .toLowerCase()
      .replace(/\s+/g, ' ')
      .trim(),
  }));

  identifiers.forEach((f, fi) => {
    const key = `id:${fi}`;
    let elementIdxs = [];

    if (Array.isArray(f.citations) && f.citations.length > 0) {
      const seen = new Set();
      f.citations.forEach((c) => {
        findElementsByCitation(c, elements).forEach((ei) => {
          if (!seen.has(ei)) {
            seen.add(ei);
            elementIdxs.push(ei);
          }
        });
      });
    }

    if (elementIdxs.length === 0) {
      elementIdxs = attributeBySubstring(f.value, normElements);
    }

    elementIdxs.forEach((ei) => {
      if (!forward.has(key)) forward.set(key, new Set());
      forward.get(key).add(ei);
      if (!inverse.has(ei)) inverse.set(ei, new Set());
      inverse.get(ei).add(key);
    });
  });

  return { forward, inverse };
}

// ── Sub-components ──────────────────────────────────────────────────────────

// 4-segment quartile bar replacing the inline confidence percentage pill.
// Hits the auto-verdict threshold → fills all 4 in primary-600;
// below that, fills proportional segments in neutral-300.
// `variant="compact"` shrinks the bar to fit inline in a field row (no pill
// chrome, smaller cells); the default variant is sized for the panel header.
function ConfidenceBar({ score, variant = 'default' }) {
  const threshold = useAutoVerdictThreshold();
  if (typeof score !== 'number') return null;
  // Floored, not rounded: this badge sits next to a threshold comparison, and
  // Math.round turned 0.9996 into "100%" -- a score reported as perfect on the
  // screen that was telling the reviewer it had fallen short.
  const pct = formatConfidencePct(score);
  const filled = Math.max(1, Math.min(4, Math.ceil(score * 4)));
  const meets = score >= threshold;
  const compact = variant === 'compact';
  // States the measurement, not the routing outcome. "(auto-verdict threshold
  // met)" read as "this was auto-verified" on documents sitting in the review
  // queue: a held document with one non-billable code measured 99.99%
  // confidence, met the threshold, and was still correctly held — is_automated
  // is `confidence >= threshold AND no review reasons`, and this badge only
  // knows the confidence half. The banner states the routing decision; the badge
  // must not contradict it.
  const confidenceTitle =
    `Extraction confidence: ${pct} ` +
    `(${meets ? 'at or above' : 'below'} the ` +
    `${formatConfidencePct(threshold)} auto-verify threshold)`;
  return (
    <span
      className={[
        'ls-confidence-bar',
        meets ? 'meets' : 'below',
        compact ? 'is-compact' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      title={confidenceTitle}
      role="img"
      aria-label={`Confidence ${pct}`}
    >
      {[0, 1, 2, 3].map((i) => (
        <span
          key={i}
          className={`ls-conf-cell ${i < filled ? 'is-filled' : ''}`}
        />
      ))}
      <span className="ls-conf-value">{pct}</span>
    </span>
  );
}

// Confidence below this threshold flags an identifier row with an amber rail
// so reviewers can scan the panel for "what needs attention" without reading
// every percentage. Set above the noise floor of typical OCR extractions.
const LOW_CONFIDENCE_THRESHOLD = 0.7;

// Filled pin icon used as the "selected" affordance inside the inline detail.
function FilledPin() {
  return (
    <svg
      viewBox="0 0 16 16"
      width="13"
      height="13"
      aria-hidden="true"
      focusable="false"
    >
      <path
        fill="currentColor"
        d="M9.5 1.5a1 1 0 0 0-.7 1.71l.4.4-3.7 3.7-1.7-.4a1 1 0 0 0-1 1.7l3.3 3.3-2.6 2.6a.75.75 0 1 0 1.06 1.06l2.6-2.6 3.3 3.3a1 1 0 0 0 1.7-1l-.4-1.7 3.7-3.7.4.4a1 1 0 0 0 1.71-.7v-3a1 1 0 0 0-1-1h-3z"
      />
    </svg>
  );
}

const FieldRow = React.forwardRef(function FieldRow(
  {
    fieldKey,
    name,
    value,
    confidence,
    citations,
    hoveredField,
    hoveredBboxIdx,
    selectedField,
    selectedBboxIdx,
    attribution,
    onFieldHover,
    onFieldClick,
    onUnlock,
    onScrollToSource,
    corrections,
    onFieldEdit,
  },
  ref
) {
  const [copied, setCopied] = useState(false);
  const isSourceForHoveredBbox =
    hoveredBboxIdx != null &&
    attribution?.inverse?.get(hoveredBboxIdx)?.has(fieldKey);
  const isSourceForSelectedBbox =
    selectedBboxIdx != null &&
    attribution?.inverse?.get(selectedBboxIdx)?.has(fieldKey);
  const isSelf = hoveredField === fieldKey;
  const isLocked = selectedField === fieldKey;
  const hasAttribution = attribution?.forward?.has(fieldKey);
  const isLowConfidence =
    typeof confidence === 'number' && confidence < LOW_CONFIDENCE_THRESHOLD;

  const correctedValue =
    fieldKey != null && corrections ? corrections[fieldKey] : undefined;
  const isCorrected = correctedValue !== undefined && correctedValue !== value;
  const displayValue = isCorrected ? correctedValue : value;
  const editable = Boolean(onFieldEdit && fieldKey != null);
  const hasValue =
    displayValue != null && String(displayValue).trim().length > 0;

  const className = [
    'ls-field-row',
    isSelf ? 'is-hovered' : '',
    isSourceForHoveredBbox ? 'is-source-match' : '',
    isSourceForSelectedBbox && !isLocked ? 'is-source-locked' : '',
    isLocked ? 'is-locked' : '',
    isLowConfidence ? 'is-low-confidence' : '',
    hasAttribution ? 'has-attribution' : '',
    isCorrected ? 'is-corrected' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const handleBlur = (e) => {
    const next = e.currentTarget.innerText.trim();
    if (next === (value ?? '').toString().trim()) {
      // value reverted to original — clear any prior correction
      if (correctedValue !== undefined) onFieldEdit?.(fieldKey, null);
    } else if (next !== (correctedValue ?? '').trim()) {
      onFieldEdit?.(fieldKey, next);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      e.currentTarget.blur();
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      e.currentTarget.innerText = displayValue ?? '';
      e.currentTarget.blur();
    }
  };

  const handleCopy = async (e) => {
    e.stopPropagation();
    if (!hasValue) return;
    try {
      await navigator.clipboard.writeText(String(displayValue));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1100);
    } catch {
      // Clipboard blocked — fall back to range selection so the user can copy.
      const sel = window.getSelection();
      const range = window.document.createRange();
      const valueEl = e.currentTarget
        .closest('.ls-field-row')
        ?.querySelector('.ls-field-value');
      if (valueEl) {
        range.selectNodeContents(valueEl);
        sel?.removeAllRanges();
        sel?.addRange(range);
      }
    }
  };

  // Click anywhere outside the editable value or copy button locks/unlocks
  // the field. This drives selection state in the parent: a locked field
  // syncs scroll, anchors the InspectorCard, and keeps the bbox highlighted
  // until the user clicks elsewhere or hits Esc.
  const handleRowClick = (e) => {
    if (!onFieldClick) return;
    const target = e.target;
    if (target?.closest?.('.ls-field-value.is-editable')) return;
    if (target?.closest?.('.ls-copy-btn')) return;
    onFieldClick(fieldKey);
  };

  const ariaLabel = isLocked
    ? `${name} (selected — press Escape to deselect)`
    : name;

  return (
    <div
      ref={ref}
      className={className}
      onMouseEnter={() => onFieldHover?.(fieldKey)}
      onMouseLeave={() => onFieldHover?.(null)}
      onClick={handleRowClick}
      role={onFieldClick ? 'button' : undefined}
      tabIndex={onFieldClick ? 0 : undefined}
      aria-pressed={onFieldClick ? isLocked : undefined}
      aria-label={onFieldClick ? ariaLabel : undefined}
      onKeyDown={(e) => {
        if (!onFieldClick) return;
        if (e.target !== e.currentTarget) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onFieldClick(fieldKey);
        }
      }}
    >
      <div className="ls-field-label" title={name}>
        {name}
      </div>
      <div
        className={`ls-field-value ${editable ? 'is-editable' : ''}`}
        contentEditable={editable}
        suppressContentEditableWarning
        spellCheck={false}
        onBlur={editable ? handleBlur : undefined}
        onKeyDown={editable ? handleKeyDown : undefined}
        title={editable ? 'Click to correct this value' : undefined}
      >
        {hasValue ? (
          displayValue
        ) : editable ? (
          ''
        ) : (
          <span className="ls-empty-value">—</span>
        )}
      </div>
      <div className="ls-field-actions">
        {typeof confidence === 'number' && (
          <ConfidenceBar score={confidence} variant="compact" />
        )}
        {isCorrected && (
          <span
            className="ls-correction-badge"
            title={`Original: ${value || '—'}`}
          >
            edited
          </span>
        )}
        {hasValue && (
          <button
            type="button"
            className={`ls-copy-btn ${copied ? 'is-copied' : ''}`}
            onClick={handleCopy}
            title={copied ? 'Copied' : 'Copy value'}
            aria-label={copied ? 'Copied' : 'Copy value'}
            tabIndex={-1}
          >
            {copied ? '✓' : '⧉'}
          </button>
        )}
      </div>
      {isLocked && (
        <InlineFieldDetail
          name={name}
          value={displayValue}
          hasValue={hasValue}
          confidence={confidence}
          citations={citations}
          onUnlock={onUnlock}
          onScrollToSource={onScrollToSource}
        />
      )}
    </div>
  );
});

function InlineFieldDetail({
  name,
  value,
  hasValue,
  confidence,
  citations,
  onUnlock,
  onScrollToSource,
}) {
  const [copied, setCopied] = useState(false);
  const citationCount = Array.isArray(citations) ? citations.length : 0;
  const pageRefs = Array.isArray(citations)
    ? Array.from(new Set(citations.map((c) => c?.page_id).filter((p) => p != null))).sort(
        (a, b) => a - b
      )
    : [];

  const handleCopy = async (e) => {
    e.stopPropagation();
    if (!hasValue) return;
    try {
      await navigator.clipboard.writeText(String(value));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1100);
    } catch {
      // Clipboard blocked — silent failure.
    }
  };

  return (
    <div
      className="ls-inline-detail"
      onClick={(e) => e.stopPropagation()}
      role="region"
      aria-label={`Selected field detail: ${name}`}
    >
      <header className="ls-inline-detail-head">
        <span className="ls-inline-detail-eyebrow">Selected field</span>
        {onUnlock && (
          <button
            type="button"
            className="ls-inline-detail-unpin"
            onClick={(e) => {
              e.stopPropagation();
              onUnlock();
            }}
            title="Unlock selection (Esc)"
            aria-label="Unlock selection"
          >
            <FilledPin />
          </button>
        )}
      </header>

      <div className="ls-inline-detail-value">
        <p className="ls-inline-detail-value-text">
          {hasValue ? value : <span className="ls-inline-detail-empty">No value extracted</span>}
        </p>
        {hasValue && (
          <button
            type="button"
            className={`ls-inline-detail-copy ${copied ? 'is-copied' : ''}`}
            onClick={handleCopy}
            aria-label={copied ? 'Copied' : 'Copy value'}
            title={copied ? 'Copied' : 'Copy value to clipboard'}
          >
            {copied ? '✓ Copied' : 'Copy'}
          </button>
        )}
      </div>

      <div className="ls-inline-detail-meta">
        <ConfidenceBar score={confidence} />
        {citationCount > 0 ? (
          <span
            className="ls-inline-detail-chip"
            title={`${citationCount} model-asserted source region${citationCount === 1 ? '' : 's'}`}
          >
            {citationCount} source{citationCount === 1 ? '' : 's'}
          </span>
        ) : (
          <span
            className="ls-inline-detail-chip is-warn"
            title="No model citations — attribution falls back to substring matching"
          >
            no citations
          </span>
        )}
        {pageRefs.length > 0 && (
          <span
            className="ls-inline-detail-chip"
            title={`Source on page ${pageRefs.map((p) => p + 1).join(', ')}`}
          >
            page {pageRefs.map((p) => p + 1).join(', ')}
          </span>
        )}
      </div>

      {onScrollToSource && citationCount > 0 && (
        <button
          type="button"
          className="ls-inline-detail-action"
          onClick={(e) => {
            e.stopPropagation();
            onScrollToSource();
          }}
        >
          Scroll to source →
        </button>
      )}
    </div>
  );
}

function SectionGroup({
  title,
  icon,
  fields,
  defaultOpen = true,
  hoveredField,
  hoveredBboxIdx,
  selectedField,
  selectedBboxIdx,
  attribution,
  onFieldHover,
  onFieldClick,
  onUnlock,
  onScrollToSource,
  corrections,
  onFieldEdit,
  fieldRefs,
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="ls-section-group">
      <button className="ls-section-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="ls-section-chevron">{open ? '▾' : '▸'}</span>
        {icon ? <span className="ls-section-icon">{icon}</span> : null}
        <span className="ls-section-label">{title}</span>
        <span className="ls-section-count">{fields.length}</span>
      </button>
      {open && (
        <div className="ls-section-body">
          {fields.map((f, i) => (
            <FieldRow
              key={f.key ?? `${f.name}-${i}`}
              ref={(el) => {
                if (!fieldRefs || f.key == null) return;
                if (el) fieldRefs.current.set(f.key, el);
                else fieldRefs.current.delete(f.key);
              }}
              fieldKey={f.key}
              name={f.name}
              value={f.value}
              confidence={f.confidence}
              citations={f.citations}
              hoveredField={hoveredField}
              hoveredBboxIdx={hoveredBboxIdx}
              selectedField={selectedField}
              selectedBboxIdx={selectedBboxIdx}
              attribution={attribution}
              onFieldHover={onFieldHover}
              onFieldClick={onFieldClick}
              onUnlock={onUnlock}
              onScrollToSource={onScrollToSource}
              corrections={corrections}
              onFieldEdit={onFieldEdit}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function RowCard({
  rowNum,
  fields,
  hoveredField,
  hoveredBboxIdx,
  selectedField,
  selectedBboxIdx,
  attribution,
  onFieldHover,
  onFieldClick,
  onUnlock,
  onScrollToSource,
  corrections,
  onFieldEdit,
  fieldRefs,
}) {
  const [open, setOpen] = useState(true);
  const nameLike = fields.find((f) =>
    /name|patient|member|provider/.test(f.name.toLowerCase())
  );
  const summary = nameLike?.value || `Row ${rowNum}`;

  return (
    <div className="ls-row-card">
      <button className="ls-row-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="ls-section-chevron">{open ? '▾' : '▸'}</span>
        <span className="ls-row-num">#{rowNum}</span>
        <span className="ls-row-summary">{summary}</span>
        <span className="ls-section-count">{fields.length} fields</span>
      </button>
      {open && (
        <div className="ls-row-body">
          {fields.map((f, i) => (
            <FieldRow
              key={f.key ?? `${f.name}-${i}`}
              ref={(el) => {
                if (!fieldRefs || f.key == null) return;
                if (el) fieldRefs.current.set(f.key, el);
                else fieldRefs.current.delete(f.key);
              }}
              fieldKey={f.key}
              name={f.name}
              value={f.value}
              confidence={f.confidence}
              citations={f.citations}
              hoveredField={hoveredField}
              hoveredBboxIdx={hoveredBboxIdx}
              selectedField={selectedField}
              selectedBboxIdx={selectedBboxIdx}
              attribution={attribution}
              onFieldHover={onFieldHover}
              onFieldClick={onFieldClick}
              onUnlock={onUnlock}
              onScrollToSource={onScrollToSource}
              corrections={corrections}
              onFieldEdit={onFieldEdit}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function PanelHeader({ title, subtitle }) {
  return (
    <div className="ls-panel-header">
      <div className="ls-header-top">
        <h2 className="ls-panel-title">{title}</h2>
      </div>
      {subtitle && <p className="comparison-subtitle">{subtitle}</p>}
    </div>
  );
}

// ── Extracted Data panel (new, split) ───────────────────────────────────────

/**
 * Pure render of the extracted data view. Data fetching is lifted to the
 * parent so attribution state can be shared with the bbox overlay.
 */
export function ExtractedDataPanel({
  data,
  loading,
  error,
  onRetry,
  attribution,
  hoveredField,
  hoveredBboxIdx,
  selectedField,
  selectedBboxIdx,
  onFieldHover,
  onFieldClick,
  onUnlock,
  onScrollToSource,
  corrections,
  onFieldEdit,
  fieldRefs,
}) {
  if (loading) {
    return (
      <div className="comparison-panel">
        <PanelHeader title="Extracted Data" subtitle="AI extraction results" />
        <div className="comparison-loading">
          <div className="loading-spinner" />
          <p>Loading extraction data…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="comparison-panel">
        <PanelHeader title="Extracted Data" subtitle="AI extraction results" />
        <div className="comparison-empty-state">
          <div className="empty-icon">—</div>
          <p className="empty-title">Unable to load extractions</p>
          <p className="empty-subtitle">{error}</p>
          {onRetry && (
            <button className="btn-retry" onClick={onRetry}>
              Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  const record = data?.items?.[0] || null;
  const label = unwrapJsonString(record?.label);
  const identifiers = record?.identifiers || [];
  const { topLevel, sectionHeaders, tableHeaders, rows } =
    parseIdentifiers(identifiers);
  const totalFields = identifiers.length;
  const confidenceScore =
    typeof record?.confidence_score === 'number' ? record.confidence_score : null;
  // ai_classify v2.1. Both are absent on a document classified before the
  // upgrade, and during the window where the Lakebase synced table has not picked
  // the columns up yet — so both render conditionally and the header otherwise
  // falls back to exactly what it showed before.
  const classifyConfidence =
    typeof record?.classify_confidence === 'number'
      ? record.classify_confidence
      : null;
  const classifyRationale =
    typeof record?.classify_rationale === 'string' &&
    record.classify_rationale.trim()
      ? record.classify_rationale.trim()
      : null;

  return (
    <div className="comparison-panel">
      <div className="ls-panel-header">
        <div className="ls-header-top">
          <h2 className="ls-panel-title">Extracted Data</h2>
          {totalFields > 0 && (
            <span className="ls-field-count-badge">{totalFields} fields</span>
          )}
        </div>
        {label && (
          <div className="ls-classification-row">
            <span className="ls-classification-pill">
              {label.replace(/_/g, ' ')}
            </span>
            {/* The classifier's OWN confidence, on its own scale. Deliberately
                NOT a ConfidenceBar: that widget colours against the auto-verdict
                threshold, and ai_classify never returns much above 0.78, so every
                document would read "below threshold" on a signal the threshold was
                never set for. A quiet pill with a tooltip says what it is. */}
            {classifyConfidence !== null && (
              <span
                className="ls-classify-confidence"
                title={
                  'Classifier confidence in this document type: ' +
                  `${Math.round(classifyConfidence * 100)} percent. That is ` +
                  "ai_classify's own scale, not the blended extraction " +
                  'confidence beside it.'
                }
              >
                type {Math.round(classifyConfidence * 100)}
                {' '}%
              </span>
            )}
            <ConfidenceBar score={confidenceScore} />
          </div>
        )}
        {classifyRationale && (
          <p className="ls-classify-rationale">{classifyRationale}</p>
        )}
      </div>

      <div className="ls-annotations-body">
        {!record ? (
          <div className="comparison-empty-state">
            <div className="empty-icon">⏳</div>
            <p className="empty-title">No extraction results yet</p>
            <p className="empty-subtitle">
              The pipeline may still be processing this document — check back
              shortly
            </p>
          </div>
        ) : (
          <>
            {topLevel.length > 0 && (
              <SectionGroup
                title="Document Identifiers"
                fields={topLevel}
                defaultOpen={true}
                hoveredField={hoveredField}
                hoveredBboxIdx={hoveredBboxIdx}
                selectedField={selectedField}
                selectedBboxIdx={selectedBboxIdx}
                attribution={attribution}
                onFieldHover={onFieldHover}
                onFieldClick={onFieldClick}
                onUnlock={onUnlock}
                onScrollToSource={onScrollToSource}
                corrections={corrections}
                onFieldEdit={onFieldEdit}
                fieldRefs={fieldRefs}
              />
            )}
            {sectionHeaders.length > 0 && (
              <SectionGroup
                title="Section Info"
                fields={sectionHeaders}
                defaultOpen={true}
                hoveredField={hoveredField}
                hoveredBboxIdx={hoveredBboxIdx}
                selectedField={selectedField}
                selectedBboxIdx={selectedBboxIdx}
                attribution={attribution}
                onFieldHover={onFieldHover}
                onFieldClick={onFieldClick}
                onUnlock={onUnlock}
                onScrollToSource={onScrollToSource}
                corrections={corrections}
                onFieldEdit={onFieldEdit}
                fieldRefs={fieldRefs}
              />
            )}
            {tableHeaders.length > 0 && (
              <SectionGroup
                title="Table Headers"
                fields={tableHeaders}
                defaultOpen={true}
                hoveredField={hoveredField}
                hoveredBboxIdx={hoveredBboxIdx}
                selectedField={selectedField}
                selectedBboxIdx={selectedBboxIdx}
                attribution={attribution}
                onFieldHover={onFieldHover}
                onFieldClick={onFieldClick}
                onUnlock={onUnlock}
                onScrollToSource={onScrollToSource}
                corrections={corrections}
                onFieldEdit={onFieldEdit}
                fieldRefs={fieldRefs}
              />
            )}
            {rows.length > 0 && (
              <div className="ls-rows-section">
                <div className="ls-rows-header">
                  <span className="ls-rows-title">Extracted Rows</span>
                  <span className="ls-section-count">{rows.length}</span>
                </div>
                <div className="ls-rows-list">
                  {rows.map((row) => (
                    <RowCard
                      key={row.rowNum}
                      rowNum={row.rowNum}
                      fields={row.fields}
                      hoveredField={hoveredField}
                      hoveredBboxIdx={hoveredBboxIdx}
                      selectedField={selectedField}
                      selectedBboxIdx={selectedBboxIdx}
                      attribution={attribution}
                      onFieldHover={onFieldHover}
                      onFieldClick={onFieldClick}
                      onUnlock={onUnlock}
                      onScrollToSource={onScrollToSource}
                      corrections={corrections}
                      onFieldEdit={onFieldEdit}
                      fieldRefs={fieldRefs}
                    />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Data Extraction Review panel (new, split) ──────────────────────────────

/**
 * Review form (verdict radios + reasoning). Owns its own review state +
 * server calls. Rendered in a pinned-bottom slot by DocumentDetailPage.
 */
export function ReviewPanel({
  documentId,
  // NOTE: `reviewReasons` and `pipelineAutoVerified` USED to be props here, and
  // removing them is the point. DocumentDetailPage — the only routed document
  // page — never passed either, so they sat at their defaults ([] and false) and
  // every document rendered "Held for human review ... below the 92% threshold",
  // auto-verified ones included. A verdict assembled from props a caller can
  // omit will eventually be the wrong verdict, stated confidently. HoldReasonBanner
  // fetches the decision from the server instead, so there is nothing to forget.
  hasRecord,
  corrections,
  onCorrectionsLoaded,
  // Set by the in-document assistant when the reviewer approves a proposed
  // verdict card. A new object identity (fresh `ts`) fills the form; the
  // reviewer still presses Submit. Never auto-submits.
  proposedReview,
}) {
  const [verdict, setVerdict] = useState('');
  const [reasoning, setReasoning] = useState('');
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewSaved, setReviewSaved] = useState(false);
  const [reviewError, setReviewError] = useState(null);
  // Snapshot of corrections as last persisted on the server. Diff against
  // the live `corrections` prop powers the "N unsaved edits" pill.
  const [savedCorrections, setSavedCorrections] = useState({});

  // Autosave of the UNSUBMITTED form. Writing a draft submits nothing; the
  // server deletes it when a real review lands.
  const {
    status: draftStatus,
    draft,
    save: saveDraft,
    suspend: suspendDraft,
    discard: discardDraft,
  } = useReviewDraft(documentId);

  // Whether a draft has already populated the form for this document. The draft
  // and the submitted review are two independent fetches, so without this the
  // slower one wins at random. A draft is the NEWER, unsubmitted edit, so it
  // takes precedence -- and it only exists until a submit deletes it.
  const draftAppliedFor = useRef(null);

  useEffect(() => {
    if (!documentId) return;
    // Reset the saved-snapshot on doc change so a stale prior-doc snapshot
    // doesn't make the new doc's loaded corrections look "unsaved".
    setSavedCorrections({});
    // Reset the FORM too, and unconditionally. Only the success path below set
    // these, and the catch reset just savedCorrections -- so moving from a
    // reviewed document to an unreviewed one (whose GET /review 404s) left the
    // previous document's verdict selected, its reasoning in the textarea, and
    // reviewSaved true, which made the button read "Update Review" for a
    // document that has no review. One click then submitted the wrong verdict.
    setVerdict('');
    setReasoning('');
    setReviewSaved(false);
    setReviewError(null);
    (async () => {
      try {
        const review = await getExtractionReview(documentId);
        // A draft already filled the form: it is the newer edit, so keep it and
        // only take the review's corrections snapshot (which drives the
        // "N unsaved edits" diff).
        if (draftAppliedFor.current !== documentId) {
          setVerdict(review.verdict);
          setReasoning(review.reasoning || '');
          setReviewSaved(true);
        }
        if (review.corrections) {
          setSavedCorrections(review.corrections);
          if (onCorrectionsLoaded) onCorrectionsLoaded(review.corrections);
        } else {
          setSavedCorrections({});
        }
      } catch {
        setSavedCorrections({});
      }
    })();
    // onCorrectionsLoaded is intentionally not in the dep array — we only
    // want this effect to run when documentId changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId]);

  // Apply the draft the hook loaded. Runs after the reset at the top of the
  // documentId effect, so it restores rather than races it.
  useEffect(() => {
    if (!draft) return;
    draftAppliedFor.current = documentId;
    if (draft.verdict) setVerdict(draft.verdict);
    setReasoning(draft.reasoning || '');
    if (draft.corrections && Object.keys(draft.corrections).length > 0) {
      if (onCorrectionsLoaded) onCorrectionsLoaded(draft.corrections);
    }
    // There IS unsubmitted work, whether or not a review already exists, so the
    // button must not read "Update Review" as though everything were persisted.
    setReviewSaved(false);
    // onCorrectionsLoaded is deliberately out of the deps, as above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, documentId]);

  // Autosave the inline field corrections. They live in DocumentDetailPage and
  // arrive as a prop, so there is no onChange to hang this off -- unlike the
  // verdict and reasoning below. The hook drops writes that are identical to
  // what the server already has, so the draft load pushing corrections back
  // down here does not echo straight back out as a save.
  useEffect(() => {
    saveDraft({ verdict, reasoning, corrections: corrections || {} });
  }, [corrections, verdict, reasoning, saveDraft]);

  // Fill the form from an assistant-approved proposal. Keyed on the object
  // identity (fresh each approval) so re-proposing the same verdict re-applies.
  // Marks the review unsaved so the reviewer knows they still need to Submit.
  useEffect(() => {
    if (!proposedReview) return;
    if (proposedReview.verdict) setVerdict(proposedReview.verdict);
    setReasoning(proposedReview.reasoning || '');
    setReviewSaved(false);
    setReviewError(null);
  }, [proposedReview]);

  // Count of corrections whose value differs from the last-saved snapshot.
  // Counts new edits, value changes, and reverts/clears equally.
  const unsavedCount = useMemo(() => {
    const cur = corrections || {};
    const saved = savedCorrections || {};
    const keys = new Set([...Object.keys(cur), ...Object.keys(saved)]);
    let n = 0;
    for (const k of keys) {
      const a = cur[k] ?? null;
      const b = saved[k] ?? null;
      if (a !== b) n += 1;
    }
    return n;
  }, [corrections, savedCorrections]);

  // The routing verdict and its explanation now come from
  // GET /api/documents/{id}/hold-reasons, rendered by HoldReasonBanner. Nothing
  // about held-vs-auto-verified is decided in this component any more.

  const handleSubmitReview = async () => {
    if (!verdict) return;
    if (verdict !== 'correct' && !reasoning.trim()) {
      setReviewError(
        'Reasoning is required when the extraction is not fully correct.'
      );
      return;
    }
    try {
      setReviewLoading(true);
      setReviewError(null);
      // Stop autosaving before the submit goes out. A debounced write landing
      // after the server deleted the draft would recreate it, and on the next
      // load that resurrected draft would take precedence over the review just
      // submitted.
      suspendDraft();
      const cleanCorrections =
        corrections && Object.keys(corrections).length > 0
          ? Object.fromEntries(
              Object.entries(corrections).filter(([, v]) => v != null && v !== '')
            )
          : null;
      await submitExtractionReview(documentId, {
        verdict,
        reasoning: reasoning.trim() || null,
        corrections:
          cleanCorrections && Object.keys(cleanCorrections).length > 0
            ? cleanCorrections
            : null,
      });
      setSavedCorrections(cleanCorrections || {});
      setReviewSaved(true);
      // The verdict is on the shared review row now. The server drops the draft
      // as part of the submit; this clears the client's copy so the form does not
      // keep showing it as outstanding work.
      draftAppliedFor.current = null;
      discardDraft();
    } catch (err) {
      setReviewError(
        err.response?.data?.detail || err.message || 'Failed to submit review'
      );
    } finally {
      setReviewLoading(false);
    }
  };

  if (!hasRecord) return null;

  return (
    <div className="review-section review-section-standalone">
      <div className="review-header">
        <h3 className="review-title">Data Extraction Review</h3>
        <p className="review-subtitle">
          Rate the accuracy of the extraction above
        </p>
      </div>

      {/* The routing verdict AND every reason behind it, from one server
          payload. This replaced two hand-composed banners whose "why" was
          inferred in the browser: one asserted "below the 92% auto-verify
          threshold" whenever the reason list was empty, which on a 99.96%
          document was simply false. Nothing is composed here now.

          HoldReasonBanner also absorbs what FidelityNotice used to render, so a
          silently rewritten code appears exactly once, as an advisory, alongside
          everything else that is wrong with the document rather than in a
          separate notice with its own fetch. */}
      <HoldReasonBanner documentId={documentId} />

      {/* What the terminology would accept for each flagged code. Below the
          banner because it is the detail that follows from it, and still fetched
          separately: it renders nothing when there is no flagged code, so an
          auto-verified or confidence-only hold pays nothing for it. */}
      <RemediationPanel documentId={documentId} />

      <div className="review-verdict-group">
        {VERDICT_OPTIONS.map((opt) => (
          <label
            key={opt.value}
            className={`review-radio-option ${
              verdict === opt.value ? 'selected' : ''
            }`}
            data-verdict={opt.value}
          >
            <input
              type="radio"
              name="verdict"
              value={opt.value}
              checked={verdict === opt.value}
              onChange={(e) => {
                setVerdict(e.target.value);
                setReviewSaved(false);
                setReviewError(null);
                // Immediate, not debounced: a radio selection is one discrete
                // event with nothing to coalesce, so waiting only widens the
                // window in which it can be lost.
                saveDraft(
                  {
                    verdict: e.target.value,
                    reasoning,
                    corrections: corrections || {},
                  },
                  { immediate: true }
                );
              }}
            />
            <span className="verdict-icon">{opt.icon}</span>
            <span className="radio-label">{opt.label}</span>
          </label>
        ))}
      </div>

      {verdict && verdict !== 'correct' && (
        <textarea
          className="review-reasoning"
          placeholder="Explain what was incorrect or missing… (⏎ to submit, Shift+⏎ for newline)"
          value={reasoning}
          onChange={(e) => {
            setReasoning(e.target.value);
            setReviewSaved(false);
            setReviewError(null);
            // Debounced: this one IS a keystroke stream.
            saveDraft({
              verdict,
              reasoning: e.target.value,
              corrections: corrections || {},
            });
          }}
          onKeyDown={(e) => {
            // Plain Enter submits the review (chat-app pattern). Shift+Enter
            // still inserts a newline if the reviewer needs multi-line.
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (verdict && (verdict === 'correct' || reasoning.trim())) {
                handleSubmitReview();
              }
            }
          }}
          rows={3}
        />
      )}

      {reviewError && <p className="review-error">{reviewError}</p>}

      <div className="review-actions">
        <button
          className="review-submit-btn"
          onClick={handleSubmitReview}
          disabled={!verdict || reviewLoading}
        >
          {reviewLoading
            ? 'Saving…'
            : reviewSaved && unsavedCount === 0
              ? 'Update Review'
              : 'Submit Review'}
        </button>
        {unsavedCount > 0 && !reviewLoading && (
          <span
            className="review-unsaved-pill"
            title="These field corrections are stored locally — submit the review to persist them."
            role="status"
          >
            <span className="review-unsaved-dot" aria-hidden="true" />
            {unsavedCount} unsaved {unsavedCount === 1 ? 'edit' : 'edits'}
          </span>
        )}
        {reviewSaved && unsavedCount === 0 && !reviewLoading && (
          <span className="review-saved-badge">✓ Saved</span>
        )}
        {/* Draft autosave state. aria-live="polite" (never "assertive") so a
            screen reader is not interrupted on every keystroke pause, and
            aria-atomic so the phrase is announced whole -- WCAG 2.2 SC 4.1.3
            (Status Messages, AA). Same markup as the notepad's rn-status. */}
        <span
          className={`review-draft-status review-draft-status--${draftStatus}`}
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          {draftStatus === 'saving'
            ? 'Saving draft…'
            : draftStatus === 'saved'
              ? 'Draft saved'
              : draftStatus === 'error'
                ? 'Draft not saved'
                : ''}
        </span>
      </div>
    </div>
  );
}

// ── Legacy combined component (kept for backward compatibility) ────────────

const ExtractionComparison = ({ documentId }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!documentId) return;
    (async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await getExtractionComparisons(documentId);
        setData(result);
      } catch (err) {
        setError(err.message || 'Failed to load comparison data');
      } finally {
        setLoading(false);
      }
    })();
  }, [documentId]);

  const record = data?.items?.[0] || null;

  return (
    <>
      <ExtractedDataPanel
        data={data}
        loading={loading}
        error={error}
        onRetry={() => {
          setError(null);
        }}
      />
      <ReviewPanel
        documentId={documentId}
        hasRecord={Boolean(record)}
      />
    </>
  );
};

export default ExtractionComparison;
