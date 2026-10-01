import { useState } from 'react';
import { useNavigate } from 'react-router';
import './DocumentCard.css';

// Raw backend value -> human label. Backend values are the source of truth;
// label is display-only.
const STATUS_LABELS = {
  processing: 'Processing',
  pending: 'Pending',
  auto_verified: 'Auto-Verified',
  reviewed: 'Reviewed',
  failed: 'Failed',
};

// Status -> badge color. Tracks the doc's lifecycle:
//   processing   -> warning (in flight)
//   pending      -> info    (awaiting reviewer action)
//   reviewed     -> success (done)
//   auto_verified-> auto    (done, no human review)
//   failed       -> error
const STATUS_BADGE_CLASSES = {
  processing: 'status-badge-warning',
  pending: 'status-badge-info',
  reviewed: 'status-badge-success',
  auto_verified: 'status-badge-auto-verified',
  failed: 'status-badge-error',
};

export const getStatusLabel = (status) =>
  STATUS_LABELS[status?.toLowerCase()] || 'Processing';

export const getStatusBadgeClass = (status) =>
  STATUS_BADGE_CLASSES[status?.toLowerCase()] || 'status-badge-warning';

// `hasDraft` is advisory and defaults false, so a caller that does not know
// about drafts (or whose lookup failed) renders exactly as before.
const DocumentCard = ({ document, onDelete, hasDraft = false }) => {
  const navigate = useNavigate();
  const [imgError, setImgError] = useState(false);
  const [imgLoaded, setImgLoaded] = useState(false);

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getDocumentIcon = (fileName) => {
    const extension = fileName?.split('.').pop()?.toLowerCase();
    return extension === 'pdf' ? 'PDF' : 'IMG';
  };

  const isReviewed = Boolean(document.review_verdict);
  const badgeClass = isReviewed
    ? STATUS_BADGE_CLASSES.reviewed
    : getStatusBadgeClass(document.processing_status);
  // "Pending" told a reviewer nothing about which of five different problems a
  // document has. The server derives the primary reason through the same
  // function the detail page uses, so the card and the document it opens can
  // never name different reasons. Falls back to "Pending" when the gold sync has
  // not caught up with review_reasons yet, so an older backend still renders.
  const holdLabel = !isReviewed && document.hold_primary_label;
  const badgeLabel = isReviewed
    ? STATUS_LABELS.reviewed
    : holdLabel || getStatusLabel(document.processing_status);
  // Held with nothing accounting for it is a defect, not a category of problem.
  // It gets its own styling so it cannot be mistaken for a normal hold reason.
  const badgeExtra = document.hold_unexplained
    ? ' document-card-status-badge--unexplained'
    : '';
  const reviewerName = document.review_reviewer_name || document.review_reviewer_email;

  return (
    <div className="document-card">
      <div className="document-card-thumbnail">
        {!imgError ? (
          <>
            {!imgLoaded && (
              <div className="document-card-skeleton" aria-hidden="true">
                <div className="document-card-skeleton-shimmer" />
              </div>
            )}
            <img
              className={`document-card-img ${imgLoaded ? 'document-card-img--loaded' : ''}`}
              src={`/api/documents/${document.id}/image`}
              alt={document.document_name}
              onLoad={() => setImgLoaded(true)}
              onError={() => setImgError(true)}
            />
          </>
        ) : (
          <div className="document-card-icon">
            {getDocumentIcon(document.document_name)}
          </div>
        )}
        <div
          className={`document-card-status-badge ${badgeClass}${badgeExtra}`}
          title={
            holdLabel
              ? `Held for review: ${holdLabel}`
              : `Status: ${badgeLabel}`
          }
        >
          {badgeLabel}
        </div>
        {/* An unsubmitted draft is otherwise invisible until the document is
            reopened, so a reviewer who stepped away has no way to find the work
            they started. */}
        {hasDraft && (
          <div
            className="document-card-draft-badge"
            title="You have an unsubmitted draft review on this document"
          >
            Draft
          </div>
        )}
      </div>

      <div className="document-card-content">
        <h3 className="document-card-title" title={document.document_name}>
          {document.document_name}
        </h3>

        <div className="document-card-metadata">
          <div className="metadata-item">
            <span className="metadata-label">Type:</span>
            <span className="metadata-value">{document.document_type || 'Unspecified'}</span>
          </div>
          <div className="metadata-item">
            <span className="metadata-label">Uploaded:</span>
            <span className="metadata-value">{formatDate(document.upload_timestamp)}</span>
          </div>
          {document.num_pages && (
            <div className="metadata-item">
              <span className="metadata-label">Pages:</span>
              <span className="metadata-value">{document.num_pages}</span>
            </div>
          )}
          {isReviewed && reviewerName && (
            <div className="metadata-item">
              <span className="metadata-label">Reviewed by:</span>
              <span className="metadata-value" title={document.review_reviewer_email}>
                {reviewerName}
              </span>
            </div>
          )}
        </div>

        {document.notes && (
          <p className="document-card-notes">{document.notes}</p>
        )}
      </div>

      <div className="document-card-actions">
        <button
          className="btn-action btn-view"
          onClick={() => navigate(`/review/documents/${document.id}`)}
          title="View document"
        >
          View
        </button>
        <button
          className="btn-action btn-delete"
          onClick={() => onDelete(document.id)}
          title="Delete document"
        >
          Delete
        </button>
      </div>
    </div>
  );
};

export default DocumentCard;
