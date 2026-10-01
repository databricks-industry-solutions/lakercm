/**
 * LakeRCM API Client
 */

import axios from 'axios'

const API_BASE = ''

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'An error occurred'
    console.error('[LakeRCM API]', message)
    return Promise.reject(new Error(message))
  }
)

// Health Check
export const checkHealth = async () => {
  const response = await api.get('/health')
  return response.data
}

// Documents
export const uploadDocument = async (formData) => {
  const response = await api.post('/api/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

export const listDocuments = async (params = {}) => {
  // Strip undefined/null/'all' so the backend default kicks in.
  const clean = {}
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null || v === '' || v === 'all') return
    clean[k] = v
  })
  const response = await api.get('/api/documents/', { params: clean })
  return response.data
}

export const getDocumentStatusCounts = async (includeAutoVerified = false) => {
  const response = await api.get('/api/documents/status-counts', {
    params: { include_auto_verified: includeAutoVerified },
  })
  return response.data
}

export const getDocument = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}`)
  return response.data
}

export const deleteDocument = async (documentId) => {
  const response = await api.delete(`/api/documents/${documentId}`)
  return response.data
}

export const getExtractionComparisons = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}/comparisons`)
  return response.data
}

export const submitExtractionReview = async (documentId, data) => {
  const response = await api.post(`/api/documents/${documentId}/review`, data)
  return response.data
}

export const getExtractionReview = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}/review`)
  return response.data
}

// Reviewer notepad — per (document, reviewer) private scratch notes, distinct
// from the shared review verdict. GET always 200s ({note_text: ''} when empty).
export const getDocumentNotes = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}/notes`)
  return response.data
}

export const saveDocumentNotes = async (documentId, noteText) => {
  const response = await api.post(`/api/documents/${documentId}/notes`, {
    note_text: noteText ?? '',
  })
  return response.data
}

// Review drafts — the UNSUBMITTED review autosaved as the reviewer works:
// verdict, reasoning and inline field corrections, per (document, reviewer).
// Writing one never submits anything. GET always 200s, with exists=false when
// there is no draft, so callers never treat "nothing saved yet" as an error.
export const getReviewDraft = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}/draft`)
  return response.data
}

export const saveReviewDraft = async (documentId, draft) => {
  const response = await api.post(`/api/documents/${documentId}/draft`, {
    verdict: draft?.verdict || null,
    reasoning: draft?.reasoning || null,
    corrections: draft?.corrections || {},
  })
  return response.data
}

// Which documents this reviewer has an unsubmitted draft for. Advisory only --
// it powers a badge in the queue, so callers should swallow errors rather than
// let a missing badge break the list.
export const getReviewDraftIds = async () => {
  const response = await api.get('/api/documents/drafts')
  return response.data?.document_ids || []
}

export const discardReviewDraft = async (documentId) => {
  const response = await api.delete(`/api/documents/${documentId}/draft`)
  return response.data
}

// Last-gasp flush when the page is being hidden (tab switch, navigation away).
//
// sendBeacon, not axios: the browser guarantees a queued beacon survives the
// page going away, whereas an in-flight XHR is cancelled. It is POST-only,
// which is why the draft endpoint is a POST.
//
// Returns false when the beacon could not be queued (unsupported, or over the
// ~64KB cap) so the caller can fall back to a normal request.
//
// The body MUST be a Blob typed application/json, not a bare string. A string
// beacon is sent as text/plain, and FastAPI only JSON-parses a body whose
// content-type maintype is `application` with a json subtype -- anything else
// arrives as raw bytes and fails pydantic validation with a 422. The flush would
// look like it worked and silently drop the reviewer's last edit, which is the
// one thing this function exists to prevent.
//
// Typing the blob is safe here: API_BASE is '' so these are same-origin
// requests, which are never preflighted.
export const beaconReviewDraft = (documentId, draft) => {
  if (typeof navigator === 'undefined' || !navigator.sendBeacon) return false
  try {
    const body = JSON.stringify({
      verdict: draft?.verdict || null,
      reasoning: draft?.reasoning || null,
      corrections: draft?.corrections || {},
    })
    const blob = new Blob([body], { type: 'application/json' })
    return navigator.sendBeacon(`/api/documents/${documentId}/draft`, blob)
  } catch {
    return false
  }
}

// Knowledge graph. Advisory: the panel renders nothing when the graph is off or
// mid-rebuild, and a failure here must never break the review page.
export const getKgStatus = async () => {
  const response = await api.get('/api/kg/status')
  return response.data
}

export const getDocumentKgNeighbourhood = async (documentId) => {
  const response = await api.get(`/api/kg/document/${documentId}/neighbourhood`)
  return response.data
}

// Agent-assisted review of held documents
export const getDocumentRemediation = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}/remediation`)
  return response.data
}

// Has a stored clinical code stopped matching what the page printed?
// The endpoint is designed never to fail — it answers has_finding=false for a
// faithful extraction, a real document with no planted ground truth, and an
// analytics pipeline that has not materialised gold_extraction_fidelity yet.
// Callers should still swallow errors: this is advisory, and a badge must never
// be able to break the review page.
export const getDocumentFidelity = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}/fidelity`)
  return response.data
}

// Why this document is where it is: the pipeline's routing decision plus every
// reason behind it, blocking and advisory, assembled server-side.
//
// The verdict is NOT derived in the browser any more. It used to be computed from
// two React props that the only routed page never passed, so every document
// rendered as "held" with a fabricated explanation. One payload, one source.
export const getDocumentHoldReasons = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}/hold-reasons`)
  return response.data
}

export const getDocumentProposals = async (documentId) => {
  const response = await api.get(`/api/documents/${documentId}/proposals`)
  return response.data
}

export const recordDocumentProposal = async (documentId, proposal) => {
  const response = await api.post(
    `/api/documents/${documentId}/proposals`,
    proposal
  )
  return response.data
}

// A proposal can only be dispositioned once. A repeat (a double-click, a
// replayed request) comes back 409 rather than overwriting the first decision,
// so callers treat that as already-recorded, not as an error to surface.
export const setProposalDisposition = async (
  proposalId,
  disposition,
  humanValue = null
) => {
  const response = await api.patch(`/api/proposals/${proposalId}`, {
    disposition,
    human_value: humanValue,
  })
  return response.data
}

// User
export const getCurrentUser = async () => {
  const response = await api.get('/api/me')
  return response.data
}

// Admin diagnostics (require workspace `admins` group)
export const getAdminLakebaseInstance = async () => {
  const response = await api.get('/api/admin/lakebase/instance')
  return response.data
}

export const getAdminLakebaseMetrics = async () => {
  const response = await api.get('/api/admin/lakebase/metrics')
  return response.data
}

export const getAdminLakebaseEvents = async (hours = 24) => {
  const response = await api.get('/api/admin/lakebase/events', {
    params: { hours },
  })
  return response.data
}

export const getAdminLakebaseSessions = async (interesting = true) => {
  const response = await api.get('/api/admin/lakebase/sessions', {
    params: { interesting },
  })
  return response.data
}

export const getAdminLakebaseIndexes = async () => {
  const response = await api.get('/api/admin/lakebase/indexes')
  return response.data
}

export const getAdminLakebaseSlowQueries = async () => {
  const response = await api.get('/api/admin/lakebase/slow_queries')
  return response.data
}

export const getAdminLakebaseTopTables = async () => {
  const response = await api.get('/api/admin/lakebase/top_tables')
  return response.data
}

export const getAdminLakebaseReplication = async () => {
  const response = await api.get('/api/admin/lakebase/replication')
  return response.data
}

export const getAdminRuntime = async () => {
  const response = await api.get('/api/admin/runtime')
  return response.data
}

export const getAdminHealth = async () => {
  const response = await api.get('/api/admin/health')
  return response.data
}

// Analytics (Physician View)
export const getAnalyticsSummary = async (reviewer) => {
  const params = reviewer ? { reviewer } : {}
  const response = await api.get('/api/analytics/summary', { params })
  return response.data
}

export const getAnalyticsTrend = async (reviewer) => {
  const params = reviewer ? { reviewer } : {}
  const response = await api.get('/api/analytics/trend', { params })
  return response.data
}

export const getProcessingMetrics = async () => {
  const response = await api.get('/api/analytics/processing-metrics')
  return response.data
}

export const syncDocumentStatus = async () => {
  const response = await api.post('/api/documents/sync-status')
  return response.data
}

export const getRecentReviews = async (filters = {}) => {
  const params = {}
  if (filters.reviewer) params.reviewer = filters.reviewer
  if (filters.verdict) params.verdict = filters.verdict
  if (filters.date_from) params.date_from = filters.date_from
  if (filters.date_to) params.date_to = filters.date_to
  if (filters.search) params.search = filters.search
  if (filters.limit) params.limit = filters.limit
  if (filters.offset) params.offset = filters.offset
  if (filters.include_automated !== undefined) params.include_automated = filters.include_automated
  const response = await api.get('/api/analytics/recent-reviews', { params })
  return response.data
}

export const getReviewerList = async () => {
  const response = await api.get('/api/analytics/reviewers')
  return response.data
}

export default api
