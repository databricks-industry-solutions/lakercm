import { useState, useEffect, useRef, useCallback } from 'react';
import { getAnalyticsSummary, getProcessingMetrics, syncDocumentStatus, getRecentReviews, getReviewerList, getDocumentStatusCounts } from '../api/lakeRcmApi';
import DonutChart from '../components/DonutChart';
import ProgressRing from '../components/ProgressRing';
import ReviewsTable from '../components/ReviewsTable';
import './PhysicianDashboard.css';

const LS_OVERVIEW_SHOW_AUTO = 'overview.reviews.showAutoVerified';
const OVERVIEW_REVIEWS_PAGE_SIZE = 10;

const formatDuration = (seconds) => {
  if (seconds == null) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)} sec`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)} hrs`;
  return `${(seconds / 86400).toFixed(1)} days`;
};

// Count-up hook for the hero number — ease-out over ~600ms
const useCountUp = (target, duration = 600) => {
  const [value, setValue] = useState(0);
  const rafRef = useRef(null);

  useEffect(() => {
    if (target == null) return;
    const start = performance.now();
    const initial = 0;

    const tick = (now) => {
      const elapsed = now - start;
      const t = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(initial + (target - initial) * eased));
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, duration]);

  return value;
};

const PhysicianDashboard = () => {
  // Reviews state (Lakebase — fast)
  const [reviews, setReviews] = useState([]);
  const [reviewsTotalCount, setReviewsTotalCount] = useState(0);
  const [reviewsOffset, setReviewsOffset] = useState(0);
  const [reviewsLoading, setReviewsLoading] = useState(true);
  const [reviewsError, setReviewsError] = useState(null);

  // Accuracy state (DBSQL — slower)
  const [summary, setSummary] = useState(null);
  const [accuracyLoading, setAccuracyLoading] = useState(true);
  const [accuracyError, setAccuracyError] = useState(null);
  const [accuracyVisible, setAccuracyVisible] = useState(false);

  // Processing metrics state (Lakebase — fast)
  const [processingMetrics, setProcessingMetrics] = useState(null);

  // Document status counts (Lakebase — fast); powers the Documents Processed tile
  const [statusCounts, setStatusCounts] = useState(null);

  // Shared state
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [reviewers, setReviewers] = useState([]);
  const [selectedReviewer, setSelectedReviewer] = useState('');
  const [reviewFilters, setReviewFilters] = useState({});
  const [showAutoVerified, setShowAutoVerified] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem(LS_OVERVIEW_SHOW_AUTO) === 'true';
  });
  const intervalRef = useRef(null);

  // Fetch reviews from Lakebase (fast path)
  const fetchReviews = useCallback(async (filters = {}, isRefresh = false) => {
    try {
      if (!isRefresh) setReviewsLoading(true);
      setReviewsError(null);
      const params = {
        reviewer: selectedReviewer || undefined,
        limit: OVERVIEW_REVIEWS_PAGE_SIZE,
        include_automated: showAutoVerified,
        ...filters,
      };
      Object.keys(params).forEach((k) => {
        if (params[k] === '' || params[k] === undefined) delete params[k];
      });
      const data = await getRecentReviews(params);
      setReviews(data.reviews || []);
      setReviewsTotalCount(data.total_count || 0);
      setReviewsOffset(data.offset || 0);
    } catch (err) {
      console.error('Failed to load reviews:', err);
      if (!isRefresh) setReviewsError(err.message || 'Failed to load recent reviews');
    } finally {
      setReviewsLoading(false);
    }
  }, [selectedReviewer, showAutoVerified]);

  // Fetch accuracy from DBSQL (slow path). Dashboard-level metrics are
  // always all-reviewers — reviewer scoping only applies to the Recent
  // Reviews table below, not the summary accuracy / processing tiles.
  const fetchAccuracy = useCallback(async (isRefresh = false) => {
    try {
      if (!isRefresh) {
        setAccuracyLoading(true);
        setAccuracyVisible(false);
      }
      setAccuracyError(null);
      // Sync pipeline status from SDP gold table before fetching metrics
      await syncDocumentStatus().catch(() => {});
      const [summaryData, metricsData, countsData] = await Promise.all([
        getAnalyticsSummary(undefined),
        getProcessingMetrics().catch(() => null),
        getDocumentStatusCounts(false).catch(() => null),
      ]);
      setSummary(summaryData);
      if (metricsData) setProcessingMetrics(metricsData);
      if (countsData) setStatusCounts(countsData);
      // Trigger fade-in animation
      setTimeout(() => setAccuracyVisible(true), 50);
    } catch (err) {
      console.error('Failed to load accuracy:', err);
      if (!isRefresh) setAccuracyError(err.message || 'Failed to load extraction accuracy data');
    } finally {
      setAccuracyLoading(false);
      if (isRefresh) setRefreshing(false);
    }
  }, []);

  // Accuracy loads once on mount and on explicit refresh. Reviewer
  // changes only re-fetch the Recent Reviews table below.
  useEffect(() => {
    fetchAccuracy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchReviews(reviewFilters);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedReviewer]);

  useEffect(() => {
    getReviewerList()
      .then((data) => setReviewers(data.reviewers || []))
      .catch((err) => console.error('Failed to load reviewers:', err));
  }, []);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        fetchReviews(reviewFilters, true);
        fetchAccuracy(true);
      }, 60000);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, fetchReviews, fetchAccuracy, reviewFilters]);

  const handleReviewFilterChange = (filters) => {
    setReviewFilters(filters);
    fetchReviews(filters);
  };

  const handleIncludeAutomatedChange = (checked) => {
    setShowAutoVerified(checked);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LS_OVERVIEW_SHOW_AUTO, String(checked));
    }
    const nextFilters = { ...reviewFilters, offset: 0 };
    setReviewFilters(nextFilters);
    // Use a direct call with the new value to avoid one render of stale state
    const params = {
      reviewer: selectedReviewer || undefined,
      limit: OVERVIEW_REVIEWS_PAGE_SIZE,
      include_automated: checked,
      ...nextFilters,
    };
    Object.keys(params).forEach((k) => {
      if (params[k] === '' || params[k] === undefined) delete params[k];
    });
    setReviewsLoading(true);
    setReviewsError(null);
    getRecentReviews(params)
      .then((data) => {
        setReviews(data.reviews || []);
        setReviewsTotalCount(data.total_count || 0);
        setReviewsOffset(data.offset || 0);
      })
      .catch((err) => {
        console.error('Failed to load reviews:', err);
        setReviewsError(err.message || 'Failed to load recent reviews');
      })
      .finally(() => setReviewsLoading(false));
  };

  const handleRefresh = () => {
    setRefreshing(true);
    fetchReviews(reviewFilters, true);
    fetchAccuracy(true);
  };

  // Derived values from summary (computed once)
  const humanDocs = summary?.human_documents_reviewed ?? 0;
  const humanAccuracy = summary?.human_accuracy_pct ?? 0;
  const humanCorrect = summary?.human_correct_count ?? 0;
  const humanPartial = summary?.human_partially_correct_count ?? 0;
  const humanIncorrect = summary?.human_incorrect_count ?? 0;

  // Documents Processed tile: full lifecycle breakdown from /status-counts.
  // auto_verified here is the live soft-delete-aware count from Lakebase
  // (medical_documents), so stale gold_extraction_labels rows left behind
  // by deleted uploads don't inflate the auto-verification rate.
  const processingCount = statusCounts?.processing ?? 0;
  const pendingCount = statusCounts?.pending ?? 0;
  const reviewedCount = statusCounts?.reviewed ?? 0;
  const autoVerifiedCount = statusCounts?.auto_verified ?? 0;
  const lifecycleTotal = statusCounts?.total ?? (processingCount + pendingCount + reviewedCount + autoVerifiedCount);
  const autoDocs = autoVerifiedCount;
  const autoRate = lifecycleTotal > 0 ? (autoDocs / lifecycleTotal) * 100 : 0;
  const processingPct = lifecycleTotal > 0 ? (processingCount / lifecycleTotal) * 100 : 0;
  const pendingPct = lifecycleTotal > 0 ? (pendingCount / lifecycleTotal) * 100 : 0;
  const reviewedPct = lifecycleTotal > 0 ? (reviewedCount / lifecycleTotal) * 100 : 0;
  const autoVerifiedPct = lifecycleTotal > 0 ? (autoVerifiedCount / lifecycleTotal) * 100 : 0;

  const animatedTotal = useCountUp(accuracyVisible ? lifecycleTotal : 0);

  return (
    <div className="dashboard-page">
      <div className="container">
        {/* Page Header */}
        <div className="dashboard-header animate-slideUp">
          <div className="dashboard-header-left">
            <h1 className="dashboard-title">Review Dashboard</h1>
            <p className="dashboard-subtitle">
              Extraction accuracy, processing performance, and recent review activity
            </p>
          </div>
          <div className="dashboard-header-right">
            <label className="auto-refresh-toggle">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              <span className="toggle-track">
                <span className="toggle-thumb" />
              </span>
              <span className="toggle-label">Auto-refresh</span>
            </label>
            <button
              className={`dashboard-btn-refresh ${refreshing ? 'refreshing' : ''}`}
              onClick={handleRefresh}
              disabled={refreshing}
            >
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* ─── Extraction Accuracy (loads with skeleton) ─── */}
        <div className="dashboard-section">
          {accuracyLoading ? (
            <div className="accuracy-loading-container">
              <div className="accuracy-loading-bar" />
              <div className="accuracy-loading-content">
                <div className="accuracy-loading-label">
                  <span className="accuracy-loading-text">Querying data warehouse</span>
                  <span className="accuracy-loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                  </span>
                </div>
                <div className="accuracy-skeleton-kpis accuracy-skeleton-kpis--3">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="accuracy-skeleton-kpi">
                      <div className="skeleton-line" style={{ width: 36, height: 36, borderRadius: '0.75rem' }} />
                      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
                        <div className="skeleton-line" style={{ width: '60%', height: 20 }} />
                        <div className="skeleton-line" style={{ width: '80%', height: 12 }} />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="accuracy-skeleton-charts">
                  <div className="skeleton-panel" />
                  <div className="skeleton-panel" />
                </div>
              </div>
            </div>
          ) : accuracyError ? (
            <div className="dashboard-panel">
              <div className="section-error">
                <h2 className="panel-title">Extraction Accuracy</h2>
                <p>{accuracyError}</p>
                <button className="dashboard-btn-retry" onClick={() => fetchAccuracy()}>Retry</button>
              </div>
            </div>
          ) : (
            <div className={`accuracy-content ${accuracyVisible ? 'accuracy-visible' : ''}`}>
              <h2 className="section-heading">Extraction Accuracy</h2>
              <p className="section-subheading">
                Aggregate accuracy metrics from human review of AI document extractions
              </p>

              {/* Hero KPI strip — 3 balanced cards */}
              {summary && (
                <div className="stat-row">
                  <article className="stat-card stat-card--volume">
                    <h3 className="stat-card__label">Documents Processed</h3>
                    <div className="stat-card__viz">
                      <div className="hero-number">
                        {animatedTotal.toLocaleString()}
                      </div>
                      <div
                        className="stacked-bar"
                        role="img"
                        aria-label={`${processingCount} processing, ${pendingCount} pending, ${reviewedCount} reviewed, ${autoVerifiedCount} auto-verified`}
                      >
                        {processingPct > 0 && (
                          <div
                            className="stacked-bar__seg stacked-bar__seg--amber"
                            style={{ width: `${processingPct}%` }}
                          />
                        )}
                        {pendingPct > 0 && (
                          <div
                            className="stacked-bar__seg stacked-bar__seg--blue"
                            style={{ width: `${pendingPct}%` }}
                          />
                        )}
                        {reviewedPct > 0 && (
                          <div
                            className="stacked-bar__seg stacked-bar__seg--emerald"
                            style={{ width: `${reviewedPct}%` }}
                          />
                        )}
                        {autoVerifiedPct > 0 && (
                          <div
                            className="stacked-bar__seg stacked-bar__seg--violet"
                            style={{ width: `${autoVerifiedPct}%` }}
                          />
                        )}
                      </div>
                    </div>
                    <div className="stat-card__caption stat-card__caption--lifecycle">
                      <span className="legend-item">
                        <span className="legend-item__dot legend-item__dot--amber" />
                        <span className="legend-item__value">{processingCount}</span>
                        <span className="legend-item__label">processing</span>
                      </span>
                      <span className="legend-item">
                        <span className="legend-item__dot legend-item__dot--blue" />
                        <span className="legend-item__value">{pendingCount}</span>
                        <span className="legend-item__label">pending</span>
                      </span>
                      <span className="legend-item">
                        <span className="legend-item__dot legend-item__dot--emerald" />
                        <span className="legend-item__value">{reviewedCount}</span>
                        <span className="legend-item__label">reviewed</span>
                      </span>
                      <span className="legend-item">
                        <span className="legend-item__dot legend-item__dot--violet" />
                        <span className="legend-item__value">{autoVerifiedCount}</span>
                        <span className="legend-item__label">auto-verified</span>
                      </span>
                    </div>
                  </article>

                  <article className="stat-card stat-card--leverage">
                    <h3 className="stat-card__label">Auto-Verification Rate</h3>
                    <div className="stat-card__viz stat-card__viz--ring">
                      <ProgressRing
                        percent={autoRate}
                        color="oklch(0.55 0.20 290)"
                        trackColor="oklch(0.95 0.005 80)"
                        size={148}
                        strokeWidth={12}
                      />
                    </div>
                    <div className="stat-card__caption">
                      <strong>{autoDocs}</strong> of {lifecycleTotal} docs bypassed human review
                    </div>
                  </article>

                  <article className="stat-card stat-card--quality">
                    <h3 className="stat-card__label">Human Review Accuracy</h3>
                    <div className="stat-card__viz stat-card__viz--ring">
                      <ProgressRing
                        percent={humanAccuracy}
                        color="oklch(0.55 0.20 155)"
                        trackColor="oklch(0.95 0.005 80)"
                        size={148}
                        strokeWidth={12}
                      />
                    </div>
                    <div className="stat-card__caption">
                      <strong>{humanCorrect}</strong> of {humanDocs} reviewed correct
                    </div>
                  </article>
                </div>
              )}

              {/* Quality Spotlight — Verdict Distribution + Insights */}
              {summary && (
                <div className="spotlight-section" style={{ animationDelay: '80ms' }}>
                  <div className="spotlight-grid">
                    <div className="spotlight-panel spotlight-panel--donut">
                      <div className="spotlight-panel__header">
                        <h3 className="spotlight-panel__title">Verdict Distribution</h3>
                        <p className="spotlight-panel__subtitle">
                          Breakdown of all {humanDocs} human-reviewed extractions
                        </p>
                      </div>
                      <div className="spotlight-donut">
                        <DonutChart
                          correct={humanCorrect}
                          partiallyCorrect={humanPartial}
                          incorrect={humanIncorrect}
                        />
                      </div>
                      <div className="spotlight-legend">
                        <span className="spotlight-pill spotlight-pill--success">
                          <span className="spotlight-pill__dot" />
                          <span className="spotlight-pill__value">{humanCorrect}</span>
                          <span className="spotlight-pill__label">Correct</span>
                        </span>
                        <span className="spotlight-pill spotlight-pill--warning">
                          <span className="spotlight-pill__dot" />
                          <span className="spotlight-pill__value">{humanPartial}</span>
                          <span className="spotlight-pill__label">Partial</span>
                        </span>
                        <span className="spotlight-pill spotlight-pill--error">
                          <span className="spotlight-pill__dot" />
                          <span className="spotlight-pill__value">{humanIncorrect}</span>
                          <span className="spotlight-pill__label">Incorrect</span>
                        </span>
                      </div>
                    </div>

                    <div className="spotlight-panel spotlight-panel--insights">
                      <div className="spotlight-panel__header">
                        <h3 className="spotlight-panel__title">Quality Insights</h3>
                        <p className="spotlight-panel__subtitle">
                          Key signals from this reviewer cohort
                        </p>
                      </div>
                      <ul className="insight-list">
                        <li className="insight-card insight-card--success">
                          <div className="insight-card__accent" />
                          <div className="insight-card__body">
                            <div className="insight-card__metric">
                              <span className="insight-card__value">{humanCorrect}</span>
                              <span className="insight-card__unit">of {humanDocs}</span>
                            </div>
                            <p className="insight-card__copy">
                              Documents fully correct on first review — no reviewer edits needed.
                            </p>
                          </div>
                        </li>
                        <li className="insight-card insight-card--warning">
                          <div className="insight-card__accent" />
                          <div className="insight-card__body">
                            <div className="insight-card__metric">
                              <span className="insight-card__value">{humanPartial}</span>
                              <span className="insight-card__unit">flagged</span>
                            </div>
                            <p className="insight-card__copy">
                              Required partial corrections — candidates for prompt tuning or schema refinement.
                            </p>
                          </div>
                        </li>
                        <li className="insight-card insight-card--error">
                          <div className="insight-card__accent" />
                          <div className="insight-card__body">
                            <div className="insight-card__metric">
                              <span className="insight-card__value">{humanIncorrect}</span>
                              <span className="insight-card__unit">flagged</span>
                            </div>
                            <p className="insight-card__copy">
                              Marked incorrect by reviewers — review for document-type or upstream quality issues.
                            </p>
                          </div>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {/* Processing Performance — pipeline + review side by side.
                  Both tiles always render when the section is visible; each
                  renders em-dashes with an explanatory count when its own
                  dataset is empty. */}
              {processingMetrics && (processingMetrics.pipeline.total > 0 || processingMetrics.review.total > 0) && (
                <div className="performance-section" style={{ animationDelay: '160ms' }}>
                  <h2 className="section-heading">Processing Performance</h2>
                  <p className="section-subheading">
                    End-to-end timing from document upload through extraction and human review
                  </p>
                  <div className="performance-grid">
                    {[
                      {
                        title: 'Extraction Pipeline',
                        data: processingMetrics.pipeline,
                        emptyLabel: 'No batches yet',
                      },
                      {
                        title: 'Human Review Turnaround',
                        data: processingMetrics.review,
                        emptyLabel: 'No reviews yet',
                      },
                    ].map(({ title, data, emptyLabel }) => {
                      const hasData = data.total > 0;
                      return (
                        <div className="performance-col" key={title}>
                          <h3 className="performance-col__title">{title}</h3>
                          <div className="performance-col__stats">
                            <div className="perf-stat">
                              <span className="perf-stat__value perf-stat__value--accent">{hasData ? formatDuration(data.avg_seconds) : '—'}</span>
                              <span className="perf-stat__label">Average</span>
                            </div>
                            <div className="perf-stat">
                              <span className="perf-stat__value">{hasData ? formatDuration(data.median_seconds) : '—'}</span>
                              <span className="perf-stat__label">Median</span>
                            </div>
                            <div className="perf-stat">
                              <span className="perf-stat__value perf-stat__value--success">{hasData ? formatDuration(data.min_seconds) : '—'}</span>
                              <span className="perf-stat__label">Fastest</span>
                            </div>
                            <div className="perf-stat">
                              <span className="perf-stat__value perf-stat__value--warning">{hasData ? formatDuration(data.max_seconds) : '—'}</span>
                              <span className="perf-stat__label">Slowest</span>
                            </div>
                            <span className="performance-col__count">
                              {hasData ? `${data.total} documents` : emptyLabel}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ─── Recent Reviews (Lakebase — loads fast) ─── */}
        <div className="dashboard-section animate-slideUp" style={{ animationDelay: '240ms' }}>
          <div className="dashboard-panel dashboard-reviews-panel">
            <h2 className="panel-title">Recent Reviews</h2>
            {reviewsLoading ? (
              <div className="reviews-skeleton">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="reviews-skeleton-row">
                    <div className="skeleton-line" style={{ width: '30%', height: 16 }} />
                    <div className="skeleton-line" style={{ width: '20%', height: 16 }} />
                    <div className="skeleton-line" style={{ width: '15%', height: 16 }} />
                    <div className="skeleton-line" style={{ width: '25%', height: 16 }} />
                  </div>
                ))}
              </div>
            ) : reviewsError ? (
              <div className="section-error">
                <p>{reviewsError}</p>
                <button className="dashboard-btn-retry" onClick={() => fetchReviews(reviewFilters)}>Retry</button>
              </div>
            ) : (
              <ReviewsTable
                reviews={reviews}
                totalCount={reviewsTotalCount}
                offset={reviewsOffset}
                onFilterChange={handleReviewFilterChange}
                pageSize={OVERVIEW_REVIEWS_PAGE_SIZE}
                includeAutomated={showAutoVerified}
                onIncludeAutomatedChange={handleIncludeAutomatedChange}
                reviewers={reviewers}
                selectedReviewer={selectedReviewer}
                onReviewerChange={setSelectedReviewer}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PhysicianDashboard;
