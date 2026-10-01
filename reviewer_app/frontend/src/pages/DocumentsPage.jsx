import { useState, useEffect, useRef, useCallback } from 'react';
import DocumentCard from '../components/DocumentCard';
import DocumentUpload from '../components/DocumentUpload';
import {
  uploadDocument,
  listDocuments as listDocumentsApi,
  deleteDocument as deleteDocumentApi,
  syncDocumentStatus,
  getReviewDraftIds,
} from '../api/lakeRcmApi';
import './DocumentsPage.css';

const LIST_POLL_INTERVAL_MS = 15000;
const LS_SHOW_AUTO_VERIFIED = 'documents.showAutoVerified';
const PAGE_SIZE = 9;
const SEARCH_DEBOUNCE_MS = 300;

// User-facing filter pills. 'all' = no server filter.
const STATUS_FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'processing', label: 'Processing' },
  { value: 'pending', label: 'Pending' },
  { value: 'reviewed', label: 'Reviewed' },
];

const DocumentsPage = () => {
  const [activeTab, setActiveTab] = useState('documents');
  const [documents, setDocuments] = useState([]);
  // Document ids with an unsubmitted draft, for the queue badge. A Set so the
  // per-card lookup stays O(1) across a full page of cards.
  const [draftIds, setDraftIds] = useState(() => new Set());
  const [totalCount, setTotalCount] = useState(0);
  const [filteredDocuments, setFilteredDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortBy, setSortBy] = useState('name');
  const [page, setPage] = useState(0);
  const [showAutoVerified, setShowAutoVerified] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem(LS_SHOW_AUTO_VERIFIED) === 'true';
  });
  const pollRef = useRef(null);
  const searchTimerRef = useRef(null);

  const documentTypes = [
    'All Types',
    'Medical Record',
    'Lab Result',
    'Prescription',
    'Imaging Report',
    'Discharge Summary',
    'Insurance Document',
    'Other'
  ];

  const fetchDocuments = useCallback(async ({ silent = false } = {}) => {
    try {
      if (!silent) setLoading(true);
      setError(null);
      const data = await listDocumentsApi({
        include_auto_verified: showAutoVerified,
        status: statusFilter,
        search: debouncedSearch || undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      });
      setDocuments(data.documents || []);
      setTotalCount(data.total_count || 0);
      // Advisory, and deliberately not awaited alongside the list in a
      // Promise.all: a draft-lookup failure must never blank the queue, so it
      // gets its own catch and simply leaves the badges off.
      getReviewDraftIds()
        .then((ids) => setDraftIds(new Set(ids)))
        .catch(() => {});
    } catch (err) {
      console.error('Error fetching documents:', err);
      if (!silent) setError(err.message || 'Failed to load documents');
      if (!silent) setDocuments([]);
    } finally {
      if (!silent) setLoading(false);
    }
  }, [showAutoVerified, statusFilter, debouncedSearch, page]);

  // Best-effort sync: promotes processing -> pending / auto_verified based
  // on the gold MV. Also promotes already-'pending' docs to auto_verified
  // once the MV catches up (closes the MV-refresh race).
  const syncThenFetch = useCallback(
    async ({ silent = false } = {}) => {
      try {
        await syncDocumentStatus();
      } catch (err) {
        console.warn('sync-status failed (non-fatal):', err);
      }
      await fetchDocuments({ silent });
    },
    [fetchDocuments]
  );

  useEffect(() => {
    syncThenFetch();
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [syncThenFetch]);

  // Poll only while there's extraction in flight. 'processing' = being
  // processed (pre-extraction); that's the state we're waiting on.
  useEffect(() => {
    const hasProcessing = documents.some(
      (d) => d.processing_status === 'processing'
    );
    if (hasProcessing && !pollRef.current) {
      pollRef.current = setInterval(
        () => syncThenFetch({ silent: true }),
        LIST_POLL_INTERVAL_MS
      );
    } else if (!hasProcessing && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, [documents, syncThenFetch]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(
        LS_SHOW_AUTO_VERIFIED,
        String(showAutoVerified)
      );
    }
  }, [showAutoVerified]);

  // Debounce the search box → debouncedSearch, which feeds the server query.
  useEffect(() => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(() => {
      setDebouncedSearch(searchQuery.trim());
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, [searchQuery]);

  // Type + sort layer on top of the server-returned page. Search is
  // server-side (see debouncedSearch → fetchDocuments).
  useEffect(() => {
    let filtered = [...documents];
    if (filterType !== 'all' && filterType !== 'All Types') {
      filtered = filtered.filter(doc => doc.document_type === filterType);
    }
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'recent':
          return new Date(b.upload_timestamp || b.uploaded_at) - new Date(a.upload_timestamp || a.uploaded_at);
        case 'oldest':
          return new Date(a.upload_timestamp || a.uploaded_at) - new Date(b.upload_timestamp || b.uploaded_at);
        case 'name':
          return (a.document_name || a.file_name || '').localeCompare(b.document_name || b.file_name || '');
        case 'type':
          return (a.document_type || '').localeCompare(b.document_type || '');
        default:
          return 0;
      }
    });
    setFilteredDocuments(filtered);
  }, [documents, filterType, sortBy]);

  // Reset to page 0 whenever any server-side filter changes (status,
  // auto-verified toggle, or search term).
  useEffect(() => {
    setPage(0);
  }, [statusFilter, showAutoVerified, debouncedSearch]);

  const handleUpload = async (formData) => {
    try {
      setUploading(true);
      setError(null);
      const uploadResult = await uploadDocument(formData);
      setActiveTab('documents');
      await fetchDocuments();
      return uploadResult;
    } catch (err) {
      console.error('Error uploading document:', err);
      throw new Error(err.message || 'Failed to upload document');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (documentId) => {
    if (!window.confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
      return;
    }
    try {
      setError(null);
      await deleteDocumentApi(documentId);
      await fetchDocuments({ silent: true });
    } catch (err) {
      console.error('Error deleting document:', err);
      setError(err.message || 'Failed to delete document');
    }
  };

  const totalPages = Math.max(1, Math.ceil(totalCount / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages - 1);
  const pageStart = totalCount === 0 ? 0 : currentPage * PAGE_SIZE + 1;
  const pageEnd = Math.min(totalCount, (currentPage + 1) * PAGE_SIZE);

  return (
    <div className="documents-page">
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">Document Review</h1>
          <p className="page-subtitle">
            Upload, view, and review the documents queue with AI-powered extraction
          </p>
        </div>

        {error && (
          <div className="page-error">
            <span className="error-icon">!</span>
            <span>{error}</span>
            <button className="btn-dismiss-error" onClick={() => setError(null)}>
              ×
            </button>
          </div>
        )}

        <div className="page-content">
        <div className="tabs-container">
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'documents' ? 'active' : ''}`}
              onClick={() => setActiveTab('documents')}
            >
              Documents
              <span className="tab-badge">{totalCount}</span>
            </button>
            <button
              className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              Upload New
            </button>
          </div>
        </div>

        {activeTab === 'documents' && (
          <div className="tab-content">
            {/* Status filter pills */}
            <div className="status-filter-pills" role="tablist" aria-label="Filter by status">
              {STATUS_FILTERS.map((f) => (
                <button
                  key={f.value}
                  role="tab"
                  aria-selected={statusFilter === f.value}
                  className={`status-pill ${statusFilter === f.value ? 'status-pill--active' : ''} status-pill--${f.value}`}
                  onClick={() => setStatusFilter(f.value)}
                >
                  {f.label}
                </button>
              ))}
            </div>

            <div className="controls-bar">
              <div className="search-box">
                <input
                  type="text"
                  className="search-input"
                  placeholder="Search documents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>

              <div className="filter-controls">
                <select
                  className="filter-select"
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                >
                  <option value="all">All Types</option>
                  {documentTypes.slice(1).map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>

                <select
                  className="filter-select"
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                >
                  <option value="recent">Most Recent</option>
                  <option value="oldest">Oldest First</option>
                  <option value="name">Name (A-Z)</option>
                  <option value="type">Document Type</option>
                </select>

                <label className="auto-verified-toggle">
                  <input
                    type="checkbox"
                    checked={showAutoVerified}
                    onChange={(e) => setShowAutoVerified(e.target.checked)}
                  />
                  <span>Show auto-verified</span>
                </label>
              </div>
            </div>

            {loading ? (
              <div className="documents-grid">
                {[...Array(PAGE_SIZE)].map((_, i) => (
                  <div key={i} className="document-card-skeleton-card">
                    <div className="document-card-skeleton-thumbnail">
                      <div className="document-card-skeleton-shimmer" />
                    </div>
                    <div className="document-card-skeleton-body">
                      <div className="skeleton-line" style={{ width: '80%' }} />
                      <div className="skeleton-line" style={{ width: '55%' }} />
                      <div className="skeleton-line" style={{ width: '65%' }} />
                    </div>
                  </div>
                ))}
              </div>
            ) : filteredDocuments.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">—</div>
                <h3 className="empty-title">
                  {searchQuery || filterType !== 'all' || statusFilter !== 'all'
                    ? 'No documents found'
                    : 'No documents yet'}
                </h3>
                <p className="empty-subtitle">
                  {searchQuery || filterType !== 'all' || statusFilter !== 'all'
                    ? 'Try adjusting your filters'
                    : 'Upload your first medical document to get started'}
                </p>
                {!searchQuery && filterType === 'all' && statusFilter === 'all' && (
                  <button
                    className="btn-empty-action"
                    onClick={() => setActiveTab('upload')}
                  >
                    Upload Document
                  </button>
                )}
              </div>
            ) : (
              <>
                <div className="documents-grid">
                  {filteredDocuments.map(document => (
                    <DocumentCard
                      key={document.id}
                      document={document}
                      onDelete={handleDelete}
                      hasDraft={draftIds.has(document.id)}
                    />
                  ))}
                </div>

                {totalCount > PAGE_SIZE && (
                  <nav className="documents-pagination" aria-label="Documents pagination">
                    <span className="documents-pagination__range">
                      Showing <strong>{pageStart}–{pageEnd}</strong> of <strong>{totalCount}</strong>
                    </span>
                    <div className="documents-pagination__controls">
                      <button
                        className="pagination-btn"
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                        disabled={currentPage === 0}
                        aria-label="Previous page"
                      >
                        ← Prev
                      </button>
                      <span className="pagination-indicator">
                        Page <strong>{currentPage + 1}</strong> of {totalPages}
                      </span>
                      <button
                        className="pagination-btn"
                        onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                        disabled={currentPage >= totalPages - 1}
                        aria-label="Next page"
                      >
                        Next →
                      </button>
                    </div>
                  </nav>
                )}
              </>
            )}
          </div>
        )}

        {activeTab === 'upload' && (
          <div className="tab-content">
            <DocumentUpload onUpload={handleUpload} isUploading={uploading} />
          </div>
        )}
      </div>
      </div>

    </div>
  );
};

export default DocumentsPage;
