import { useState, useRef, useEffect } from 'react';
import ExtractionPanel from './ExtractionPanel';
import ExtractionComparison from './ExtractionComparison';
import BoundingBoxLayer from './BoundingBoxLayer';
import './DocumentViewer.css';

const DocumentViewer = ({ document, onClose }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [showExtractions, setShowExtractions] = useState(true);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [panelView, setPanelView] = useState('extractions'); // 'extractions' | 'comparison'
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const imageRef = useRef(null);
  const viewerRef = useRef(null);

  useEffect(() => {
    // Prevent body scroll when viewer is open
    // NOTE: Use window.document to avoid shadowing the `document` prop
    window.document.body.style.overflow = 'hidden';
    return () => {
      window.document.body.style.overflow = 'unset';
    };
  }, []);

  useEffect(() => {
    // Update image dimensions when loaded
    if (imageRef.current) {
      const img = imageRef.current;
      setImageDimensions({
        width: img.naturalWidth,
        height: img.naturalHeight
      });
    }
  }, [currentPage, document]);

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.25, 3));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.25, 0.5));
  };

  const handleResetZoom = () => {
    setZoom(1);
  };

  const handlePreviousPage = () => {
    setCurrentPage(prev => Math.max(prev - 1, 1));
  };

  const handleNextPage = () => {
    setCurrentPage(prev => Math.min(prev + 1, document.num_pages || 1));
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Escape') {
      onClose();
    } else if (e.key === 'ArrowLeft') {
      handlePreviousPage();
    } else if (e.key === 'ArrowRight') {
      handleNextPage();
    } else if (e.key === '+' || e.key === '=') {
      handleZoomIn();
    } else if (e.key === '-' || e.key === '_') {
      handleZoomOut();
    } else if (e.key === '0') {
      handleResetZoom();
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
    // handleKeyPress is re-created every render; listing it here would
    // re-subscribe the listener every render. It only closes over currentPage
    // and zoom, so re-subscribing when those change is what is needed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPage, zoom]);

  const getPageExtractions = () => {
    if (!document.extractions || !Array.isArray(document.extractions)) {
      return [];
    }
    return document.extractions.filter(e => e.page === currentPage);
  };

  const getBoundingBoxes = () => {
    const pageExtractions = getPageExtractions();
    return pageExtractions
      .filter(e => e.bounding_box)
      .map(e => ({
        ...e.bounding_box,
        type: e.type,
        text: e.text,
        confidence: e.confidence
      }));
  };

  const getDocumentImageUrl = () => {
    // Serve the document image from the backend endpoint that
    // reads the file from the Unity Catalog volume
    return `/api/documents/${document.id}/image`;
  };

  return (
    <div className="document-viewer-overlay" onClick={onClose}>
      <div
        className={`document-viewer ${showExtractions ? 'with-panel' : 'full-width'}`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="viewer-header">
          <div className="header-left">
            <h2 className="viewer-title">{document.document_name}</h2>
            {document.document_type && (
              <span className="viewer-document-type">{document.document_type}</span>
            )}
          </div>

          <div className="header-right">
            <button
              className="btn-header-action"
              onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
              title={showBoundingBoxes ? 'Hide annotations' : 'Show annotations'}
            >
              Annotations
            </button>

            <button
              className="btn-header-action"
              onClick={() => setShowExtractions(!showExtractions)}
              title={showExtractions ? 'Hide panel' : 'Show panel'}
            >
              Extractions
            </button>

            {showExtractions && (
              <button
                className={`btn-header-action ${panelView === 'comparison' ? 'active' : ''}`}
                onClick={() =>
                  setPanelView(panelView === 'comparison' ? 'extractions' : 'comparison')
                }
                title="Toggle extraction comparison view"
              >
                Comparison
              </button>
            )}

            <button
              className="btn-close"
              onClick={onClose}
              title="Close (Esc)"
            >
              ×
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="viewer-content">
          {/* Document Display */}
          <div className="viewer-document-container" ref={viewerRef}>
            <div
              className="viewer-document-wrapper"
              style={{
                transform: `scale(${zoom})`,
                transformOrigin: 'center top'
              }}
            >
              <div className="document-image-container">
                <img
                  ref={imageRef}
                  src={getDocumentImageUrl()}
                  alt={`${document.document_name} - Page ${currentPage}`}
                  className="document-image"
                  onLoad={(e) => {
                    setImageDimensions({
                      width: e.target.naturalWidth,
                      height: e.target.naturalHeight
                    });
                  }}
                />

                {showBoundingBoxes && (
                  <BoundingBoxLayer
                    boxes={getBoundingBoxes()}
                    imageWidth={imageDimensions.width}
                    imageHeight={imageDimensions.height}
                  />
                )}
              </div>
            </div>
          </div>

          {/* Extraction Panel / Comparison Panel */}
          {showExtractions && (
            <div className="viewer-extraction-panel">
              {panelView === 'comparison' ? (
                <ExtractionComparison
                  documentId={document.id}
                  isProcessed={
                    ['pending', 'auto_verified', 'reviewed'].includes(document.processing_status)
                    || Boolean(document.review_verdict)
                  }
                />
              ) : (
                <ExtractionPanel extractions={getPageExtractions()} />
              )}
            </div>
          )}
        </div>

        {/* Footer Controls */}
        <div className="viewer-footer">
          {/* Page Navigation */}
          <div className="footer-section">
            <button
              className="btn-footer-control"
              onClick={handlePreviousPage}
              disabled={currentPage <= 1}
              title="Previous page (←)"
            >
              ◀
            </button>

            <span className="page-indicator">
              Page {currentPage} of {document.num_pages || 1}
            </span>

            <button
              className="btn-footer-control"
              onClick={handleNextPage}
              disabled={currentPage >= (document.num_pages || 1)}
              title="Next page (→)"
            >
              ▶
            </button>
          </div>

          {/* Zoom Controls */}
          <div className="footer-section">
            <button
              className="btn-footer-control"
              onClick={handleZoomOut}
              disabled={zoom <= 0.5}
              title="Zoom out (-)"
            >
              −
            </button>

            <span className="zoom-indicator" onClick={handleResetZoom} title="Reset zoom (0)">
              {Math.round(zoom * 100)}%
            </span>

            <button
              className="btn-footer-control"
              onClick={handleZoomIn}
              disabled={zoom >= 3}
              title="Zoom in (+)"
            >
              +
            </button>
          </div>

          {/* Keyboard Shortcuts Hint */}
          <div className="footer-section keyboard-hints">
            <span className="hint">← → Navigate</span>
            <span className="hint">+ − Zoom</span>
            <span className="hint">Esc Close</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentViewer;
