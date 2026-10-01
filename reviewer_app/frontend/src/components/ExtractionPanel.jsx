import { useState } from 'react';
import './ExtractionPanel.css';

const ExtractionPanel = ({ extractions }) => {
  const [expandedSections, setExpandedSections] = useState({
    dates: true,
    medical: true,
    patient: true,
    metadata: false
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getEntityTypeColor = (type) => {
    const colors = {
      'DATE': '#3b82f6',
      'TIME': '#3b82f6',
      'MEDICAL_CONDITION': '#10b981',
      'MEDICATION': '#10b981',
      'PROCEDURE': '#10b981',
      'TEST': '#10b981',
      'PERSON': '#f59e0b',
      'AGE': '#f59e0b',
      'CONTACT': '#f59e0b',
      'ID': '#f59e0b'
    };
    return colors[type] || '#6b7280';
  };

  const groupExtractionsByType = () => {
    if (!extractions || !Array.isArray(extractions)) {
      return { dates: [], medical: [], patient: [] };
    }

    const dates = extractions.filter(e =>
      ['DATE', 'TIME'].includes(e.type)
    );

    const medical = extractions.filter(e =>
      ['MEDICAL_CONDITION', 'MEDICATION', 'PROCEDURE', 'TEST'].includes(e.type)
    );

    const patient = extractions.filter(e =>
      ['PERSON', 'AGE', 'CONTACT', 'ID'].includes(e.type)
    );

    return { dates, medical, patient };
  };

  const { dates, medical, patient } = groupExtractionsByType();

  const renderExtractionItem = (extraction) => (
    <div key={extraction.id} className="extraction-item">
      <div
        className="extraction-color-bar"
        style={{ backgroundColor: getEntityTypeColor(extraction.type) }}
      />
      <div className="extraction-content">
        <div className="extraction-text">{extraction.text}</div>
        <div className="extraction-meta">
          <span className="extraction-type">{extraction.type.replace(/_/g, ' ')}</span>
          {extraction.confidence && (
            <span className="extraction-confidence">
              {Math.round(extraction.confidence * 100)}% confident
            </span>
          )}
        </div>
      </div>
    </div>
  );

  const renderSection = (title, items, sectionKey, icon) => (
    <div className={`extraction-section ${expandedSections[sectionKey] ? 'expanded' : 'collapsed'}`}>
      <button
        className="section-header"
        onClick={() => toggleSection(sectionKey)}
      >
        <div className="section-title-wrapper">
          <span className="section-icon">{icon}</span>
          <span className="section-title">{title}</span>
          <span className="section-count">{items.length}</span>
        </div>
        <span className="section-toggle">{expandedSections[sectionKey] ? '▼' : '▶'}</span>
      </button>

      {expandedSections[sectionKey] && (
        <div className="section-content">
          {items.length > 0 ? (
            items.map(renderExtractionItem)
          ) : (
            <div className="section-empty">No {title.toLowerCase()} extracted</div>
          )}
        </div>
      )}
    </div>
  );

  const renderMetadataSection = () => (
    <div className={`extraction-section ${expandedSections.metadata ? 'expanded' : 'collapsed'}`}>
      <button
        className="section-header"
        onClick={() => toggleSection('metadata')}
      >
        <div className="section-title-wrapper">
          <span className="section-icon"></span>
          <span className="section-title">Document Metadata</span>
        </div>
        <span className="section-toggle">{expandedSections.metadata ? '▼' : '▶'}</span>
      </button>

      {expandedSections.metadata && (
        <div className="section-content">
          <div className="metadata-grid">
            <div className="metadata-item">
              <span className="metadata-label">Pages:</span>
              <span className="metadata-value">{extractions?.metadata?.page_count || 'N/A'}</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Processed:</span>
              <span className="metadata-value">
                {formatDate(extractions?.metadata?.processed_at)}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Total Entities:</span>
              <span className="metadata-value">{extractions?.length || 0}</span>
            </div>
            {extractions?.metadata?.processing_time && (
              <div className="metadata-item">
                <span className="metadata-label">Processing Time:</span>
                <span className="metadata-value">
                  {extractions.metadata.processing_time}s
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="extraction-panel">
      <div className="panel-header">
        <h2 className="panel-title">Extracted Information</h2>
        <p className="panel-subtitle">AI-powered document analysis</p>
      </div>

      <div className="panel-content">
        {renderSection('Dates & Times', dates, 'dates', '')}
        {renderSection('Medical Entities', medical, 'medical', '')}
        {renderSection('Patient Information', patient, 'patient', '')}
        {renderMetadataSection()}
      </div>

      {(!extractions || extractions.length === 0) && (
        <div className="panel-empty-state">
          <div className="empty-icon">—</div>
          <p className="empty-title">No extractions available</p>
          <p className="empty-subtitle">
            Document processing may still be in progress
          </p>
        </div>
      )}
    </div>
  );
};

export default ExtractionPanel;
