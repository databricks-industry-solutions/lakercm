import { useEffect, useRef, useState, useCallback } from 'react'
import { Link } from 'react-router'
import {
  getAdminLakebaseInstance,
  getAdminLakebaseMetrics,
  getAdminLakebaseEvents,
  getAdminLakebaseSessions,
  getAdminLakebaseSlowQueries,
  getAdminLakebaseTopTables,
  getAdminLakebaseReplication,
  getAdminLakebaseIndexes,
  getAdminRuntime,
  getAdminHealth,
  getCurrentUser,
} from '../api/lakeRcmApi'
import './AdminPage.css'

const SPARK_HISTORY = 60

const STATE_PILL_CLASS = {
  ACTIVE: 'pill-ok',
  IDLE: 'pill-idle',
  INIT: 'pill-warn',
}

const INTERVAL_OPTIONS = [
  { ms: 5000, label: '5s' },
  { ms: 10000, label: '10s' },
  { ms: 30000, label: '30s' },
  { ms: 60000, label: '1m' },
]

const EVENTS_RANGE_OPTIONS = [
  { hours: 1, label: '1h' },
  { hours: 24, label: '24h' },
  { hours: 168, label: '7d' },
]

const SESSION_THRESHOLD_OPTIONS = [
  { seconds: 30, label: '30s' },
  { seconds: 60, label: '60s' },
  { seconds: 120, label: '2m' },
  { seconds: 300, label: '5m' },
]

const TOP_TABLES_FILTER_OPTIONS = [
  { key: 'app', label: 'App tables' },
  { key: 'all', label: 'All tables' },
  { key: 'managed', label: 'Databricks-managed' },
]

// Returns true if a session is "running something" — used to gate the
// duration-based row coloring. Plain `idle` pool connections never count
// (their query_start is the time of the last query they ran, which gets
// misread as "running for X minutes").
function isSessionRunning(s) {
  if (!s || !s.state) return false
  if (s.state === 'active') return true
  if (s.state.startsWith('idle in transaction')) return true
  if (s.wait_event_type === 'Lock') return true
  return false
}

// Top-tables filter — segregates app tables from Databricks-managed
// internals (Sync driver tables, CDC tracking, LangGraph checkpoints).
const MANAGED_SCHEMAS = new Set(['__db_system', 'wal2delta', '__databricks_internal'])
const MANAGED_TABLE_PREFIXES = ['checkpoint_', 'store_']

function isManagedTable(t) {
  if (!t) return false
  if (MANAGED_SCHEMAS.has(t.schema)) return true
  if (MANAGED_TABLE_PREFIXES.some((p) => (t.table || '').startsWith(p))) return true
  return false
}

// ---------------------------------------------------------------------------
// formatters
// ---------------------------------------------------------------------------

function formatBytes(n) {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${units[i]}`
}

function formatDuration(seconds) {
  if (seconds == null || isNaN(seconds)) return '—'
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`
  return `${(seconds / 3600).toFixed(1)}h`
}

function formatTs(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function formatNumber(n) {
  if (n == null) return '—'
  return Number(n).toLocaleString()
}

function endpointTypeLabel(value) {
  if (!value) return '—'
  if (value === 'ENDPOINT_TYPE_READ_WRITE') return 'Read-write'
  if (value === 'ENDPOINT_TYPE_READ_ONLY') return 'Read-only'
  return value
}

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

function Sparkline({ values, width = 110, height = 28 }) {
  if (!values || values.length < 2) {
    return <div className="sparkline-empty">collecting…</div>
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const stepX = width / (values.length - 1)
  const points = values
    .map((v, i) => `${(i * stepX).toFixed(1)},${(height - ((v - min) / range) * height).toFixed(1)}`)
    .join(' ')
  return (
    <svg
      className="sparkline"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
    >
      <polyline points={points} fill="none" strokeWidth="1.5" />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Metric tile (rate-aware)
// ---------------------------------------------------------------------------

function RateTile({ title, value, unit, history, hint }) {
  return (
    <div className="metric-tile">
      <div className="metric-tile-head">
        <div className="metric-tile-title">{title}</div>
        <Sparkline values={history} />
      </div>
      <div className="metric-tile-value">
        {value}
        {unit && <span className="metric-tile-unit"> {unit}</span>}
      </div>
      {hint && <div className="metric-tile-hint">{hint}</div>}
    </div>
  )
}

function pushRing(ring, key, value) {
  const arr = (ring[key] || []).concat(value)
  if (arr.length > SPARK_HISTORY) arr.shift()
  return { ...ring, [key]: arr }
}

// ---------------------------------------------------------------------------
// Section error wrapper
// ---------------------------------------------------------------------------

function SectionError({ message, onRetry }) {
  return (
    <div className="admin-section-error">
      <span>Couldn&apos;t load: {message || 'unknown error'}</span>
      {onRetry && (
        <button type="button" className="admin-retry" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}

function UnavailableHint({ reason }) {
  return (
    <div className="admin-unavailable">
      <span className="admin-unavailable-icon" aria-hidden="true">ⓘ</span>
      <span>Not available — {reason || 'feature disabled'}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// AdminPage
// ---------------------------------------------------------------------------

function AdminPage() {
  const [authLoading, setAuthLoading] = useState(true)
  const [authorized, setAuthorized] = useState(false)

  // Data buckets keyed by section.
  const [data, setData] = useState({})
  const [errors, setErrors] = useState({})
  const [lastRefresh, setLastRefresh] = useState(null)

  // Auto-refresh control.
  const [intervalMs, setIntervalMs] = useState(() => {
    const stored = parseInt(localStorage.getItem('lakercm.admin.interval') || '', 10)
    return [5000, 10000, 30000, 60000].includes(stored) ? stored : 10000
  })
  const [paused, setPaused] = useState(false)

  // Events range.
  const [eventsHours, setEventsHours] = useState(24)

  // Sessions: filter toggle (idle pool hidden by default) + threshold for
  // long-running highlight.
  const [showIdlePool, setShowIdlePool] = useState(
    () => localStorage.getItem('lakercm.admin.showIdlePool') === '1'
  )
  const [sessionThresholdSec, setSessionThresholdSec] = useState(() => {
    const stored = parseInt(
      localStorage.getItem('lakercm.admin.sessionThresholdSec') || '',
      10
    )
    return [30, 60, 120, 300].includes(stored) ? stored : 60
  })

  // Top tables: app/all/managed filter (persisted).
  const [topTablesFilter, setTopTablesFilter] = useState(
    () => localStorage.getItem('lakercm.admin.topTablesFilter') || 'app'
  )

  // Ring buffer for rate sparklines and previous metric snapshot for delta.
  const ringRef = useRef({})
  const prevMetricsRef = useRef(null)
  const [, setTick] = useState(0)

  // -------------------------------------------------------------------------
  // Server-verified admin gate.
  // -------------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const me = await getCurrentUser()
        if (cancelled) return
        setAuthorized(Boolean(me?.is_admin))
      } catch {
        if (!cancelled) setAuthorized(false)
      } finally {
        if (!cancelled) setAuthLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  // -------------------------------------------------------------------------
  // Per-section refresher.
  // -------------------------------------------------------------------------
  const refreshSection = useCallback(async (key, fetcher) => {
    try {
      const value = await fetcher()
      setData((d) => ({ ...d, [key]: value }))
      setErrors((e) => {
        if (!(key in e)) return e
        const next = { ...e }
        delete next[key]
        return next
      })
      return value
    } catch (err) {
      setErrors((e) => ({ ...e, [key]: err?.message || 'failed' }))
      return null
    }
  }, [])

  const refreshAll = useCallback(async () => {
    if (!authorized) return
    const tasks = [
      ['instance', getAdminLakebaseInstance],
      ['metrics', getAdminLakebaseMetrics],
      ['sessions', () => getAdminLakebaseSessions(!showIdlePool)],
      ['slowQueries', getAdminLakebaseSlowQueries],
      ['topTables', getAdminLakebaseTopTables],
      ['replication', getAdminLakebaseReplication],
      ['indexes', getAdminLakebaseIndexes],
      ['runtime', getAdminRuntime],
      ['health', getAdminHealth],
      ['events', () => getAdminLakebaseEvents(eventsHours)],
    ]
    const results = await Promise.all(
      tasks.map(([k, fn]) => refreshSection(k, fn))
    )

    // Update ring buffer for rate sparklines using metrics result.
    const metrics = results[1]
    if (metrics) {
      const prev = prevMetricsRef.current
      const ring = ringRef.current

      const sampledAt = metrics.sampled_at
        ? new Date(metrics.sampled_at).getTime()
        : Date.now()
      const dtSec = prev ? (sampledAt - prev.sampledAt) / 1000 : 0

      const rate = (cur, prevVal) =>
        prev && dtSec > 0 ? Math.max(0, (cur - prevVal) / dtSec) : null

      const commitsPerSec = rate(
        metrics.transactions?.commit || 0,
        prev?.transactionsCommit || 0
      )
      const rollbacksPerSec = rate(
        metrics.transactions?.rollback || 0,
        prev?.transactionsRollback || 0
      )
      const rowsWritten =
        (metrics.rows?.inserted || 0) +
        (metrics.rows?.updated || 0) +
        (metrics.rows?.deleted || 0)
      const rowsWrittenPerSec = rate(rowsWritten, prev?.rowsWritten || 0)
      const rowsFetchedPerSec = rate(
        metrics.rows?.fetched || 0,
        prev?.rowsFetched || 0
      )
      const deadlocksPerMin =
        prev && dtSec > 0
          ? Math.max(
              0,
              ((metrics.deadlocks || 0) - (prev.deadlocks || 0)) / (dtSec / 60)
            )
          : null
      const walBytesPerSec = rate(
        metrics.wal?.wal_bytes || 0,
        prev?.walBytes || 0
      )

      let r = ring
      if (commitsPerSec != null) r = pushRing(r, 'commitsPerSec', commitsPerSec)
      if (rollbacksPerSec != null)
        r = pushRing(r, 'rollbacksPerSec', rollbacksPerSec)
      if (rowsWrittenPerSec != null)
        r = pushRing(r, 'rowsWrittenPerSec', rowsWrittenPerSec)
      if (rowsFetchedPerSec != null)
        r = pushRing(r, 'rowsFetchedPerSec', rowsFetchedPerSec)
      if (metrics.cache?.hit_ratio_pct != null)
        r = pushRing(r, 'cacheHitRatio', metrics.cache.hit_ratio_pct)
      if (deadlocksPerMin != null)
        r = pushRing(r, 'deadlocksPerMin', deadlocksPerMin)
      if (walBytesPerSec != null)
        r = pushRing(r, 'walBytesPerSec', walBytesPerSec)
      r = pushRing(r, 'activeConnections', metrics.connections?.active || 0)
      ringRef.current = r

      prevMetricsRef.current = {
        sampledAt,
        transactionsCommit: metrics.transactions?.commit || 0,
        transactionsRollback: metrics.transactions?.rollback || 0,
        rowsWritten,
        rowsFetched: metrics.rows?.fetched || 0,
        deadlocks: metrics.deadlocks || 0,
        walBytes: metrics.wal?.wal_bytes || 0,
        commitsPerSec,
        rollbacksPerSec,
        rowsWrittenPerSec,
        rowsFetchedPerSec,
        deadlocksPerMin,
        walBytesPerSec,
      }
      setTick((t) => t + 1)
    }

    setLastRefresh(new Date())
  }, [authorized, eventsHours, refreshSection, showIdlePool])

  // Initial fetch.
  useEffect(() => {
    if (authorized) refreshAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authorized])

  // Polling.
  useEffect(() => {
    if (!authorized || paused || intervalMs === 0) return
    const id = setInterval(refreshAll, intervalMs)
    return () => clearInterval(id)
  }, [authorized, paused, intervalMs, refreshAll])

  // Refresh when events range changes.
  useEffect(() => {
    if (authorized) refreshSection('events', () => getAdminLakebaseEvents(eventsHours))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eventsHours, authorized])

  const onIntervalChange = (ms) => {
    setIntervalMs(ms)
    localStorage.setItem('lakercm.admin.interval', String(ms))
  }

  const onShowIdlePoolChange = (next) => {
    setShowIdlePool(next)
    localStorage.setItem('lakercm.admin.showIdlePool', next ? '1' : '0')
    refreshSection('sessions', () => getAdminLakebaseSessions(!next))
  }

  const onSessionThresholdChange = (sec) => {
    setSessionThresholdSec(sec)
    localStorage.setItem('lakercm.admin.sessionThresholdSec', String(sec))
  }

  const onTopTablesFilterChange = (key) => {
    setTopTablesFilter(key)
    localStorage.setItem('lakercm.admin.topTablesFilter', key)
  }

  // -------------------------------------------------------------------------
  // Render gates
  // -------------------------------------------------------------------------
  if (authLoading) {
    return (
      <div className="admin-page">
        <div className="admin-container">
          <div className="admin-loading">Checking permissions…</div>
        </div>
      </div>
    )
  }

  if (!authorized) {
    return (
      <div className="admin-page">
        <div className="admin-container">
          <div className="admin-denied">
            <h1>Not authorized</h1>
            <p>This page requires workspace admin access.</p>
            <Link to="/overview" className="admin-link">Back to overview</Link>
          </div>
        </div>
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Sections
  // -------------------------------------------------------------------------
  return (
    <div className="admin-page">
      <div className="admin-container">
        <header className="admin-header">
          <div>
            <h1 className="admin-title">Admin Diagnostics</h1>
            <p className="admin-subtitle">
              Lakebase endpoint, live Postgres metrics, sessions, and runtime telemetry
            </p>
          </div>
          <div className="admin-controls">
            <label className="admin-control-toggle">
              <input
                type="checkbox"
                checked={!paused}
                onChange={(e) => setPaused(!e.target.checked)}
              />
              Auto-refresh
            </label>
            <select
              className="admin-control-interval"
              value={intervalMs}
              onChange={(e) => onIntervalChange(Number(e.target.value))}
              disabled={paused}
              aria-label="Refresh interval"
            >
              {INTERVAL_OPTIONS.map((opt) => (
                <option key={opt.ms} value={opt.ms}>{opt.label}</option>
              ))}
            </select>
            <button
              type="button"
              className="admin-control-btn"
              onClick={refreshAll}
            >
              Refresh
            </button>
            <span className="admin-refresh">
              {lastRefresh && <>Last refresh {lastRefresh.toLocaleTimeString()}</>}
            </span>
          </div>
        </header>

        <InstanceSection
          instance={data.instance}
          error={errors.instance}
          onRetry={() => refreshSection('instance', getAdminLakebaseInstance)}
        />

        <ThroughputSection
          metrics={data.metrics}
          ring={ringRef.current}
          error={errors.metrics}
          onRetry={() => refreshSection('metrics', getAdminLakebaseMetrics)}
        />

        <ConnectionsSection
          metrics={data.metrics}
          ring={ringRef.current}
        />

        <SessionsSection
          sessions={data.sessions}
          error={errors.sessions}
          showIdlePool={showIdlePool}
          onShowIdlePoolChange={onShowIdlePoolChange}
          thresholdSec={sessionThresholdSec}
          onThresholdChange={onSessionThresholdChange}
          onRetry={() =>
            refreshSection('sessions', () =>
              getAdminLakebaseSessions(!showIdlePool)
            )
          }
        />

        <SlowQueriesSection
          slowQueries={data.slowQueries}
          error={errors.slowQueries}
          onRetry={() => refreshSection('slowQueries', getAdminLakebaseSlowQueries)}
        />

        <TopTablesSection
          topTables={data.topTables}
          error={errors.topTables}
          filter={topTablesFilter}
          onFilterChange={onTopTablesFilterChange}
          onRetry={() => refreshSection('topTables', getAdminLakebaseTopTables)}
        />

        <IndexesSection
          indexes={data.indexes}
          error={errors.indexes}
          onRetry={() => refreshSection('indexes', getAdminLakebaseIndexes)}
        />

        <ReplicationSection
          replication={data.replication}
          error={errors.replication}
          onRetry={() => refreshSection('replication', getAdminLakebaseReplication)}
        />

        <WalSection metrics={data.metrics} />

        <EventsSection
          events={data.events}
          eventsHours={eventsHours}
          setEventsHours={setEventsHours}
          error={errors.events}
          onRetry={() => refreshSection('events', () => getAdminLakebaseEvents(eventsHours))}
        />

        <HealthSection
          health={data.health}
          runtime={data.runtime}
          error={errors.health}
          onRetry={() => refreshSection('health', getAdminHealth)}
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section components
// ---------------------------------------------------------------------------

function InstanceSection({ instance, error, onRetry }) {
  const stateClass = instance?.current_state
    ? STATE_PILL_CLASS[instance.current_state] || 'pill-warn'
    : 'pill-warn'

  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Instance</h2>
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !instance ? (
        <div className="admin-empty">loading…</div>
      ) : (
        <div className="instance-grid">
          <div className="instance-state">
            <span className={`pill ${stateClass}`}>
              {instance.current_state || '—'}
            </span>
            {instance.pending_state && (
              <span className="pill pill-warn">
                pending → {instance.pending_state}
              </span>
            )}
            {instance.disabled === true && (
              <span className="pill pill-warn">disabled</span>
            )}
          </div>
          <dl className="instance-fields">
            <div className="wide">
              <dt>Endpoint</dt>
              <dd className="mono small">{instance.name}</dd>
            </div>
            <div className="wide">
              <dt>Host</dt>
              <dd className="mono small">{instance.host || '—'}</dd>
            </div>
            <div>
              <dt>Region</dt>
              <dd>{instance.region || '—'}</dd>
            </div>
            <div>
              <dt>Type</dt>
              <dd>{endpointTypeLabel(instance.endpoint_type)}</dd>
            </div>
            <div>
              <dt>Postgres version</dt>
              <dd>{instance.pg_version || '—'}</dd>
            </div>
            <div>
              <dt>Capacity (CU)</dt>
              <dd>
                {instance.min_cu ?? '—'} – {instance.max_cu ?? '—'}
              </dd>
            </div>
            <div>
              <dt>Suspend timeout</dt>
              <dd
                title={
                  instance.suspend_timeout_seconds === 0
                    ? 'Endpoint inherits the project-level default suspend policy.'
                    : undefined
                }
              >
                {instance.suspend_timeout_seconds == null
                  ? '—'
                  : instance.suspend_timeout_seconds > 0
                    ? formatDuration(instance.suspend_timeout_seconds)
                    : 'Default (project setting)'}
              </dd>
            </div>
            <div>
              <dt>Created</dt>
              <dd>{formatTs(instance.create_time)}</dd>
            </div>
            <div>
              <dt>Last update</dt>
              <dd>{formatTs(instance.update_time)}</dd>
            </div>
          </dl>
        </div>
      )}
    </section>
  )
}

function ThroughputSection({ metrics, ring, error, onRetry }) {
  const fmt = (n, digits = 2) => (n == null ? '—' : Number(n).toFixed(digits))
  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Throughput</h2>
        <div className="admin-card-meta">
          {metrics?.sampled_at && <>Sampled {formatTs(metrics.sampled_at)}</>}
        </div>
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !metrics ? (
        <div className="admin-empty">loading…</div>
      ) : (
        <div className="metric-grid">
          <RateTile
            title="Commits / sec"
            value={fmt(ring.commitsPerSec?.[ring.commitsPerSec.length - 1])}
            history={ring.commitsPerSec}
            hint={`cum: ${formatNumber(metrics.transactions?.commit)}`}
          />
          <RateTile
            title="Rollbacks / sec"
            value={fmt(ring.rollbacksPerSec?.[ring.rollbacksPerSec.length - 1])}
            history={ring.rollbacksPerSec}
            hint={`cum: ${formatNumber(metrics.transactions?.rollback)}`}
          />
          <RateTile
            title="Rows fetched / sec"
            value={fmt(ring.rowsFetchedPerSec?.[ring.rowsFetchedPerSec.length - 1], 1)}
            history={ring.rowsFetchedPerSec}
            hint={`cum: ${formatNumber(metrics.rows?.fetched)}`}
          />
          <RateTile
            title="Rows written / sec"
            value={fmt(ring.rowsWrittenPerSec?.[ring.rowsWrittenPerSec.length - 1], 1)}
            history={ring.rowsWrittenPerSec}
            hint={`I:${formatNumber(metrics.rows?.inserted)} U:${formatNumber(metrics.rows?.updated)} D:${formatNumber(metrics.rows?.deleted)}`}
          />
          <RateTile
            title="Cache hit ratio"
            value={metrics.cache?.hit_ratio_pct != null ? metrics.cache.hit_ratio_pct.toFixed(2) : '—'}
            unit="%"
            history={ring.cacheHitRatio}
            hint={`hit:${formatNumber(metrics.cache?.blks_hit)} miss:${formatNumber(metrics.cache?.blks_read)}`}
          />
          <RateTile
            title="Deadlocks / min"
            value={fmt(ring.deadlocksPerMin?.[ring.deadlocksPerMin.length - 1], 2)}
            history={ring.deadlocksPerMin}
            hint={`cum: ${formatNumber(metrics.deadlocks)}`}
          />
          <RateTile
            title="WAL bytes / sec"
            value={
              ring.walBytesPerSec?.length
                ? formatBytes(
                    ring.walBytesPerSec[ring.walBytesPerSec.length - 1]
                  )
                : '—'
            }
            history={ring.walBytesPerSec}
            hint={
              metrics.wal
                ? `cum: ${formatBytes(metrics.wal.wal_bytes)} · ${formatNumber(metrics.wal.wal_records)} records`
                : 'wal stats unavailable'
            }
          />
          <RateTile
            title="DB size"
            value={formatBytes(metrics.db_size_bytes)}
          />
          <RateTile
            title="Temp files / bytes"
            value={`${formatNumber(metrics.temp_files)} / ${formatBytes(metrics.temp_bytes)}`}
          />
        </div>
      )}
    </section>
  )
}

function ConnectionsSection({ metrics, ring }) {
  if (!metrics?.connections) {
    return (
      <section className="admin-card">
        <header className="admin-card-header">
          <h2>Connections</h2>
        </header>
        <div className="admin-empty">loading…</div>
      </section>
    )
  }
  const c = metrics.connections
  const max = c.max || 0
  const seg = (n) => (max > 0 ? (n / max) * 100 : 0)
  const utilPct = max > 0 ? (c.total / max) * 100 : 0
  const onLock = c.on_lock ?? 0
  const longestIdleInTx = c.longest_idle_in_tx_seconds
  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Connections</h2>
        <div className="admin-card-meta">
          {c.total} of {max} ({utilPct.toFixed(1)}% utilization)
        </div>
      </header>
      <div className="conn-bar-wrap">
        <div className="conn-bar" role="img" aria-label={`${c.total} connections of ${max}`}>
          <span className="conn-seg conn-active" style={{ width: `${seg(c.active)}%` }} />
          <span className="conn-seg conn-idle" style={{ width: `${seg(c.idle)}%` }} />
          <span className="conn-seg conn-idle-tx" style={{ width: `${seg(c.idle_in_transaction)}%` }} />
        </div>
        <div className="conn-legend">
          <span><i className="conn-dot conn-active" /> Active {c.active}</span>
          <span><i className="conn-dot conn-idle" /> Idle {c.idle}</span>
          <span><i className="conn-dot conn-idle-tx" /> Idle in tx {c.idle_in_transaction}</span>
          <span className={onLock > 0 ? 'conn-util-high' : ''}>
            On lock {onLock}
          </span>
        </div>
        <Sparkline values={ring.activeConnections} width={220} height={36} />
        {(longestIdleInTx != null || c.idle_in_transaction > 0) && (
          <div className="conn-detail">
            <span className="conn-detail-label">Longest idle-in-tx:</span>
            <span className="conn-detail-value mono">
              {longestIdleInTx != null ? formatDuration(longestIdleInTx) : '—'}
            </span>
            {longestIdleInTx != null && longestIdleInTx > 60 && (
              <span className="pill pill-warn">attention</span>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

function SessionsSection({
  sessions,
  error,
  showIdlePool,
  onShowIdlePoolChange,
  thresholdSec,
  onThresholdChange,
  onRetry,
}) {
  const dangerSec = Math.max(thresholdSec * 5, thresholdSec + 60)

  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Active sessions</h2>
        <div className="admin-card-meta admin-controls-inline">
          <label className="admin-control-toggle">
            <input
              type="checkbox"
              checked={showIdlePool}
              onChange={(e) => onShowIdlePoolChange(e.target.checked)}
            />
            Show idle pool
          </label>
          <label className="admin-inline-label">
            Highlight if &gt;
            <select
              value={thresholdSec}
              onChange={(e) => onThresholdChange(Number(e.target.value))}
              className="admin-control-interval"
              aria-label="Long-running session threshold"
            >
              {SESSION_THRESHOLD_OPTIONS.map((o) => (
                <option key={o.seconds} value={o.seconds}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !sessions ? (
        <div className="admin-empty">loading…</div>
      ) : !sessions.available ? (
        <UnavailableHint reason={sessions.reason} />
      ) : sessions.sessions.length === 0 ? (
        <div className="admin-empty">
          {showIdlePool
            ? 'No sessions in this database.'
            : 'No active or idle-in-tx sessions. Toggle "Show idle pool" to see pooled connections.'}
        </div>
      ) : (
        <table className="admin-table sessions-table">
          <thead>
            <tr>
              <th>PID</th>
              <th>User</th>
              <th>App</th>
              <th>State</th>
              <th>Wait</th>
              <th>Running for</th>
              <th>Query</th>
            </tr>
          </thead>
          <tbody>
            {sessions.sessions.map((s) => {
              // Only color rows when the session is actually doing
              // something. Plain `idle` connections never get red — their
              // `query_start` is the time of the last query they ran, not
              // a "running for X minutes" measurement.
              const running = isSessionRunning(s)
              const dur = s.duration_seconds
              const cls =
                !running || dur == null
                  ? ''
                  : dur > dangerSec
                    ? 'row-danger'
                    : dur > thresholdSec
                      ? 'row-warn'
                      : ''
              const dimmed = !running ? 'row-dimmed' : ''
              return (
                <tr key={s.pid} className={`${cls} ${dimmed}`.trim()}>
                  <td className="mono">{s.pid}</td>
                  <td className="mono small">{s.user || '—'}</td>
                  <td className="small">{s.app || '—'}</td>
                  <td>
                    <span className="state-mini">{s.state || '—'}</span>
                  </td>
                  <td className="small">{s.wait || '—'}</td>
                  <td className="mono small">
                    {running
                      ? formatDuration(dur)
                      : s.state_age_seconds != null
                        ? `idle ${formatDuration(s.state_age_seconds)}`
                        : '—'}
                  </td>
                  <td className="mono query-cell" title={s.query || ''}>
                    {s.query || '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </section>
  )
}

function SlowQueriesSection({ slowQueries, error, onRetry }) {
  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Slow queries</h2>
        {slowQueries && (
          <div className="admin-card-meta">
            {slowQueries.available
              ? 'pg_stat_statements (top 10 by total time)'
              : 'pg_stat_statements unavailable'}
          </div>
        )}
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !slowQueries ? (
        <div className="admin-empty">loading…</div>
      ) : !slowQueries.available ? (
        <div className="admin-unavailable admin-unavailable-block">
          <div>
            <span className="admin-unavailable-icon" aria-hidden="true">ⓘ</span>
            <span>Not available — {slowQueries.reason || 'feature disabled'}</span>
          </div>
          <div className="admin-unavailable-howto">
            To enable: as a Lakebase superuser, run{' '}
            <code>CREATE EXTENSION pg_stat_statements;</code> then add{' '}
            <code>pg_stat_statements</code> to{' '}
            <code>shared_preload_libraries</code> in the project settings and
            restart the endpoint. Then{' '}
            <code>GRANT SELECT ON pg_stat_statements TO PUBLIC;</code>.
          </div>
        </div>
      ) : slowQueries.queries.length === 0 ? (
        <div className="admin-empty">No queries recorded.</div>
      ) : (
        <table className="admin-table slow-queries-table">
          <thead>
            <tr>
              <th>Calls</th>
              <th>Total time</th>
              <th>Mean time</th>
              <th>Rows</th>
              <th>Query</th>
            </tr>
          </thead>
          <tbody>
            {slowQueries.queries.map((q) => (
              <tr key={q.queryid}>
                <td className="mono">{formatNumber(q.calls)}</td>
                <td className="mono small">{q.total_exec_ms.toFixed(0)} ms</td>
                <td className="mono small">{q.mean_exec_ms.toFixed(2)} ms</td>
                <td className="mono">{formatNumber(q.rows)}</td>
                <td className="mono query-cell" title={q.query || ''}>
                  {q.query || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

function TopTablesSection({
  topTables,
  error,
  filter,
  onFilterChange,
  onRetry,
}) {
  // Apply the app/all/managed filter to the row set client-side. The server
  // returns the top 10 by raw size; the filter cuts noisy Databricks-managed
  // tables (sync internals, CDC tracking, agent checkpoints) so the actual
  // app tables are easier to find.
  const allRows = topTables?.tables || []
  const visibleRows =
    filter === 'all'
      ? allRows
      : filter === 'managed'
        ? allRows.filter(isManagedTable)
        : allRows.filter((t) => !isManagedTable(t))

  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Top tables by size</h2>
        <div className="admin-card-meta">
          {visibleRows.length} of {allRows.length} · rows + dead tuple ratio · last vacuum
          <span className="admin-range-picker">
            {TOP_TABLES_FILTER_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                type="button"
                className={filter === opt.key ? 'active' : ''}
                onClick={() => onFilterChange(opt.key)}
              >
                {opt.label}
              </button>
            ))}
          </span>
        </div>
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !topTables ? (
        <div className="admin-empty">loading…</div>
      ) : !topTables.available ? (
        <UnavailableHint reason={topTables.reason} />
      ) : visibleRows.length === 0 ? (
        <div className="admin-empty">
          {filter === 'app'
            ? 'No app tables in the top 10. Try "All tables" to see Databricks-managed tables.'
            : 'No tables match this filter.'}
        </div>
      ) : (
        <table className="admin-table top-tables-table">
          <thead>
            <tr>
              <th>Table</th>
              <th>Live rows</th>
              <th>Dead %</th>
              <th>Size</th>
              <th>Last vacuum</th>
              <th>Last autovacuum</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((t) => {
              const dead = t.dead_ratio_pct || 0
              const managed = isManagedTable(t)
              const cls = [
                dead > 20 ? 'row-warn' : '',
                filter === 'all' && managed ? 'row-dimmed' : '',
              ]
                .filter(Boolean)
                .join(' ')
              return (
                <tr key={`${t.schema}.${t.table}`} className={cls}>
                  <td className="mono small">
                    {t.schema}.{t.table}
                    {managed && filter === 'all' && (
                      <span className="row-tag">managed</span>
                    )}
                  </td>
                  <td className="mono">{formatNumber(t.live_rows)}</td>
                  <td className="mono">{dead.toFixed(1)}%</td>
                  <td className="mono">{formatBytes(t.total_bytes)}</td>
                  <td className="small">{formatTs(t.last_vacuum)}</td>
                  <td className="small">{formatTs(t.last_autovacuum)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </section>
  )
}

function IndexesSection({ indexes, error, onRetry }) {
  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Indexes</h2>
        <div className="admin-card-meta">
          top 15 by size · scans + tuples read · unused = dead weight
        </div>
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !indexes ? (
        <div className="admin-empty">loading…</div>
      ) : !indexes.available ? (
        <UnavailableHint reason={indexes.reason} />
      ) : indexes.indexes.length === 0 ? (
        <div className="admin-empty">No user indexes visible.</div>
      ) : (
        <table className="admin-table indexes-table">
          <thead>
            <tr>
              <th>Index</th>
              <th>Table</th>
              <th>Scans</th>
              <th>Tuples read</th>
              <th>Size</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {indexes.indexes.map((ix) => {
              const unused = ix.idx_scan === 0 && !ix.is_primary
              return (
                <tr
                  key={`${ix.schema}.${ix.table}.${ix.index}`}
                  className={unused ? 'row-warn' : ''}
                >
                  <td className="mono small">{ix.index}</td>
                  <td className="mono small">
                    {ix.schema}.{ix.table}
                  </td>
                  <td className="mono">{formatNumber(ix.idx_scan)}</td>
                  <td className="mono">{formatNumber(ix.idx_tup_read)}</td>
                  <td className="mono">{formatBytes(ix.bytes)}</td>
                  <td className="small">
                    {ix.is_primary && <span className="row-tag">primary</span>}
                    {ix.is_unique && !ix.is_primary && (
                      <span className="row-tag">unique</span>
                    )}
                    {unused && <span className="row-tag warn">unused</span>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </section>
  )
}

function ReplicationSection({ replication, error, onRetry }) {
  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Replication</h2>
        <div className="admin-card-meta">
          {replication?.available === false ? 'unavailable' : 'pg_stat_replication'}
        </div>
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !replication ? (
        <div className="admin-empty">loading…</div>
      ) : !replication.available ? (
        <UnavailableHint reason={replication.reason} />
      ) : replication.replicas.length === 0 ? (
        <div className="admin-empty">No replicas configured.</div>
      ) : (
        <table className="admin-table replication-table">
          <thead>
            <tr>
              <th>App</th>
              <th>State</th>
              <th>Sync</th>
              <th>Write lag</th>
              <th>Replay lag</th>
              <th>Lag bytes</th>
            </tr>
          </thead>
          <tbody>
            {replication.replicas.map((r) => (
              <tr key={r.pid}>
                <td>{r.app || '—'}</td>
                <td><span className="state-mini">{r.state || '—'}</span></td>
                <td className="small">{r.sync_state || '—'}</td>
                <td className="mono">{formatDuration(r.write_lag_seconds)}</td>
                <td className="mono">{formatDuration(r.replay_lag_seconds)}</td>
                <td className="mono">{formatBytes(r.lag_bytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

function WalSection({ metrics }) {
  if (!metrics) return null
  const bg = metrics.bgwriter
  const locks = metrics.locks
  // WAL bytes/sec moved to the Throughput section. This panel keeps just
  // checkpoints + locks (both useful but secondary signals).
  if (!bg && !locks) return null
  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Checkpoints · locks</h2>
      </header>
      <div className="metric-grid">
        {bg && (
          <RateTile
            title="Checkpoints (timed / requested)"
            value={`${formatNumber(bg.checkpoints_timed)} / ${formatNumber(bg.checkpoints_req)}`}
            hint={`buffers written ${formatNumber(bg.buffers_checkpoint)} · source ${bg.source || 'pg_stat_bgwriter'}`}
          />
        )}
        {bg && (
          <RateTile
            title="Buffer writes"
            value={formatNumber(bg.buffers_backend)}
            hint={`bgwriter ${formatNumber(bg.buffers_clean)} · alloc ${formatNumber(bg.buffers_alloc)}`}
          />
        )}
        {locks && (
          <RateTile
            title="Waiting locks"
            value={formatNumber(locks.waiting_total)}
            hint={
              locks.by_mode.length > 0
                ? locks.by_mode.map((l) => `${l.mode}:${l.waiting}`).join(' · ')
                : 'no contention'
            }
          />
        )}
      </div>
    </section>
  )
}

function EventsSection({ events, eventsHours, setEventsHours, error, onRetry }) {
  const sortedEvents = (events?.events || []).slice(0, 50)
  const slowest = events?.summary?.cold_start_max_seconds
  const avg = events?.summary?.cold_start_avg_seconds
  const wakeUps = (events?.events || []).filter(
    (e) => e.prev_state === 'IDLE' && e.new_state !== 'IDLE'
  ).length

  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Endpoint state transitions</h2>
        <div className="admin-card-meta">
          {events?.summary != null && (
            <>
              {wakeUps} wake-up{wakeUps === 1 ? '' : 's'}
              {avg != null && <> · avg cold-start {avg}s</>}
              {slowest != null && <> · max {slowest}s</>}
              {events.summary.cold_start_count > 0 && (
                <span className="admin-precision">≤30s precision</span>
              )}
            </>
          )}
          <span className="admin-range-picker">
            {EVENTS_RANGE_OPTIONS.map((opt) => (
              <button
                key={opt.hours}
                type="button"
                className={eventsHours === opt.hours ? 'active' : ''}
                onClick={() => setEventsHours(opt.hours)}
              >
                {opt.label}
              </button>
            ))}
          </span>
        </div>
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !events ? (
        <div className="admin-empty">loading…</div>
      ) : sortedEvents.length === 0 ? (
        <div className="admin-empty">
          No transitions in the selected range.
        </div>
      ) : (
        <table className="admin-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Transition</th>
              <th>Cold start</th>
            </tr>
          </thead>
          <tbody>
            {sortedEvents.map((e) => (
              <tr key={e.id}>
                <td>{formatTs(e.ts)}</td>
                <td>
                  <span className="state-mini">{e.prev_state || '∅'}</span>
                  <span className="state-arrow">→</span>
                  <span className="state-mini">{e.new_state}</span>
                </td>
                <td className="mono">
                  {e.cold_start_seconds != null
                    ? `${e.cold_start_seconds.toFixed(2)}s`
                    : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

function HealthSection({ health, runtime, error, onRetry }) {
  return (
    <section className="admin-card">
      <header className="admin-card-header">
        <h2>Component health</h2>
        <div className="admin-card-meta">
          {health?.checked_at && <>Checked {formatTs(health.checked_at)}</>}
        </div>
      </header>
      {error ? (
        <SectionError message={error} onRetry={onRetry} />
      ) : !health?.components ? (
        <div className="admin-empty">loading…</div>
      ) : (
        <ul className="health-list">
          {health.components.map((c) => (
            <li key={c.name} className="health-row">
              <span
                className={`health-dot ${c.ok ? 'ok' : 'fail'}`}
                aria-hidden="true"
              />
              <span className="health-name">{c.name}</span>
              <span className="health-latency">{c.latency_ms} ms</span>
              <span className="health-detail">{c.detail || ''}</span>
            </li>
          ))}
        </ul>
      )}
      {runtime && (
        <dl className="runtime-fields">
          <div>
            <dt>App uptime</dt>
            <dd className="mono">{formatDuration(runtime.uptime_seconds)}</dd>
          </div>
          <div>
            <dt>Memory (RSS)</dt>
            <dd className="mono">{runtime.rss_mb != null ? `${runtime.rss_mb} MB` : '—'}</dd>
          </div>
          <div>
            <dt>Threads</dt>
            <dd className="mono">{runtime.threads ?? '—'}</dd>
          </div>
          <div>
            <dt>PID</dt>
            <dd className="mono">{runtime.pid}</dd>
          </div>
          <div>
            <dt>Alembic head</dt>
            <dd className="mono small">
              {runtime.alembic_head?.revision || runtime.alembic_head?.error || '—'}
            </dd>
          </div>
          <div>
            <dt>Monitor last state</dt>
            <dd className="mono small">{runtime.monitor?.last_state || '—'}</dd>
          </div>
        </dl>
      )}
    </section>
  )
}

export default AdminPage
