import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router';
import './ReviewsTable.css';

const VERDICT_CONFIG = {
  correct: { label: 'Correct', className: 'verdict-correct' },
  partially_correct: { label: 'Partial', className: 'verdict-partial' },
  incorrect: { label: 'Incorrect', className: 'verdict-incorrect' },
};

const VERDICT_OPTIONS = [
  { value: '', label: 'All Verdicts' },
  { value: 'correct', label: 'Correct' },
  { value: 'partially_correct', label: 'Partially Correct' },
  { value: 'incorrect', label: 'Incorrect' },
];

function timeAgo(dateStr) {
  const now = new Date();
  const date = new Date(dateStr);
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHrs = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHrs / 24);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHrs < 24) return `${diffHrs}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

const ReviewsTable = ({
  reviews = [],
  totalCount = 0,
  offset = 0,
  onFilterChange,
  pageSize = 20,
  includeAutomated,
  onIncludeAutomatedChange,
  reviewers,
  selectedReviewer,
  onReviewerChange,
}) => {
  const navigate = useNavigate();
  const [verdict, setVerdict] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [search, setSearch] = useState('');
  const searchTimer = useRef(null);
  const showAutoToggle = typeof onIncludeAutomatedChange === 'function';
  const showReviewerFilter =
    typeof onReviewerChange === 'function' && Array.isArray(reviewers) && reviewers.length > 0;

  const emitFilters = (overrides = {}) => {
    if (!onFilterChange) return;
    const filters = {
      verdict: overrides.verdict !== undefined ? overrides.verdict : verdict,
      date_from: overrides.date_from !== undefined ? overrides.date_from : dateFrom,
      search: overrides.search !== undefined ? overrides.search : search,
      offset: overrides.offset !== undefined ? overrides.offset : 0,
    };
    onFilterChange(filters);
  };

  const handleVerdictChange = (val) => {
    setVerdict(val);
    emitFilters({ verdict: val, offset: 0 });
  };

  const handleDateFromChange = (val) => {
    setDateFrom(val);
    emitFilters({ date_from: val, offset: 0 });
  };

  const handleSearchChange = (val) => {
    setSearch(val);
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => {
      emitFilters({ search: val, offset: 0 });
    }, 300);
  };

  useEffect(() => {
    return () => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
  }, []);

  const handleClearFilters = () => {
    setVerdict('');
    setDateFrom('');
    setSearch('');
    if (showReviewerFilter) onReviewerChange('');
    if (onFilterChange) onFilterChange({ verdict: '', date_from: '', search: '', offset: 0 });
  };

  const hasFilters = verdict || dateFrom || search || (showReviewerFilter && selectedReviewer);

  const currentPage = Math.floor(offset / pageSize) + 1;
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));
  const showingFrom = totalCount > 0 ? offset + 1 : 0;
  const showingTo = Math.min(offset + pageSize, totalCount);

  const handlePrev = () => {
    if (offset > 0) emitFilters({ offset: Math.max(0, offset - pageSize) });
  };
  const handleNext = () => {
    if (offset + pageSize < totalCount) emitFilters({ offset: offset + pageSize });
  };

  return (
    <div>
      {/* Filter Bar */}
      <div className="reviews-filter-bar">
        <div className="reviews-filter-group">
          <select
            className="reviews-filter-select"
            value={verdict}
            onChange={(e) => handleVerdictChange(e.target.value)}
          >
            {VERDICT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <input
            type="date"
            className="reviews-filter-date"
            value={dateFrom}
            onChange={(e) => handleDateFromChange(e.target.value)}
            title="Reviews from this date onwards"
          />
          {showReviewerFilter && (
            <select
              className="reviews-filter-select"
              value={selectedReviewer || ''}
              onChange={(e) => onReviewerChange(e.target.value)}
              title="Filter by reviewer"
            >
              <option value="">All Reviewers</option>
              {reviewers.map((r) => (
                <option key={r.value} value={r.value}>{r.label}</option>
              ))}
            </select>
          )}
          <input
            type="text"
            className="reviews-filter-search"
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder="Search documents..."
          />
          {showAutoToggle && (
            <label className="reviews-auto-verified-toggle">
              <input
                type="checkbox"
                checked={!!includeAutomated}
                onChange={(e) => onIncludeAutomatedChange(e.target.checked)}
              />
              Show auto-verified
            </label>
          )}
        </div>
        {hasFilters && (
          <button className="reviews-filter-clear" onClick={handleClearFilters}>
            Clear filters
          </button>
        )}
      </div>

      {/* Table */}
      {reviews.length === 0 ? (
        <div className="reviews-empty">
          <div className="reviews-empty-icon">&#128203;</div>
          <p className="reviews-empty-title">{hasFilters ? 'No matching reviews' : 'No reviews yet'}</p>
          <p className="reviews-empty-subtitle">
            {hasFilters
              ? 'Try adjusting your filters'
              : 'Reviews will appear here after documents are reviewed'}
          </p>
        </div>
      ) : (
        <>
          <div className="reviews-table-wrapper">
            <table className="reviews-table">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Reviewer</th>
                  <th>Verdict</th>
                  <th>Reasoning</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {reviews.map((review) => {
                  const config = VERDICT_CONFIG[review.verdict] || VERDICT_CONFIG.correct;
                  const reasoning = review.reasoning
                    ? review.reasoning.length > 80
                      ? review.reasoning.substring(0, 80) + '...'
                      : review.reasoning
                    : '';
                  const reviewerLabel = review.reviewer_display_name || review.reviewer_email;
                  const isAuto = !!review.is_automated;

                  return (
                    <tr
                      key={review.id}
                      className={`reviews-row${isAuto ? ' reviews-row--auto' : ''}`}
                      onClick={() => navigate(`/review/documents/${review.document_id}`)}
                    >
                      <td className="reviews-cell-doc">
                        {review.document_name}
                      </td>
                      <td className="reviews-cell-reviewer">
                        {isAuto ? (
                          <span
                            className="reviewer-chip--auto"
                            title="Auto-verified above confidence threshold"
                          >
                            Auto-verified
                          </span>
                        ) : (
                          <span className="reviewer-chip" title={review.reviewer_email}>
                            {reviewerLabel}
                          </span>
                        )}
                      </td>
                      <td>
                        <span className={`verdict-badge ${config.className}`}>
                          {config.label}
                        </span>
                      </td>
                      <td className="reviews-cell-reasoning">
                        {reasoning ? (
                          reasoning
                        ) : (
                          <span
                            className="reviews-no-reasoning"
                            title={isAuto ? 'Auto-verified above confidence threshold' : undefined}
                          >
                            —
                          </span>
                        )}
                      </td>
                      <td className="reviews-cell-time">
                        {timeAgo(review.created_at)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="reviews-pagination">
            <span className="reviews-pagination-info">
              Showing {showingFrom}–{showingTo} of {totalCount}
            </span>
            <div className="reviews-pagination-controls">
              <button
                className="reviews-pagination-btn"
                onClick={handlePrev}
                disabled={offset === 0}
              >
                Prev
              </button>
              <span className="reviews-pagination-page">
                Page {currentPage} of {totalPages}
              </span>
              <button
                className="reviews-pagination-btn"
                onClick={handleNext}
                disabled={offset + pageSize >= totalCount}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ReviewsTable;
