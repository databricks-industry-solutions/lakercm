import { useState, useRef } from 'react';
import './DocumentUpload.css';

const DocumentUpload = ({ onUpload, isUploading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [documentType, setDocumentType] = useState('');
  const [notes, setNotes] = useState('');
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ['application/pdf', 'image/png', 'image/jpeg'];
  const ALLOWED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg'];

  const documentTypes = [
    'Medical Record',
    'Lab Result',
    'Prescription',
    'Imaging Report',
    'Discharge Summary',
    'Insurance Document',
    'Other'
  ];

  const validateFile = (file) => {
    if (!file) {
      return 'No file selected';
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
      return 'Invalid file type. Only PDF, PNG, and JPEG files are allowed.';
    }

    if (file.size > MAX_FILE_SIZE) {
      return 'File size exceeds 10MB limit.';
    }

    return null;
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError('');

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelection = (file) => {
    const validationError = validateFile(file);

    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setError('');
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedFile) {
      setError('Please select a file to upload');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    if (documentType) formData.append('document_type', documentType);
    if (notes) formData.append('notes', notes);

    try {
      await onUpload(formData);
      // Reset form on success
      setSelectedFile(null);
      setDocumentType('');
      setNotes('');
      setError('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      setError(err.message || 'Upload failed. Please try again.');
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="document-upload">
      <form onSubmit={handleSubmit} className="upload-form">
        <div
          className={`upload-dropzone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="upload-input-hidden"
            onChange={handleFileInputChange}
            accept={ALLOWED_EXTENSIONS.join(',')}
            disabled={isUploading}
          />

          {!selectedFile ? (
            <div className="upload-prompt">
              <div className="upload-icon">+</div>
              <h3 className="upload-title">Drag and drop your document here</h3>
              <p className="upload-subtitle">or</p>
              <button
                type="button"
                className="btn-browse"
                onClick={handleBrowseClick}
                disabled={isUploading}
              >
                Browse Files
              </button>
              <p className="upload-hint">
                Supported formats: PDF, PNG, JPEG (max 10MB)
              </p>
            </div>
          ) : (
            <div className="upload-file-selected">
              <div className="file-icon">
                {selectedFile.type === 'application/pdf' ? 'PDF' : 'IMG'}
              </div>
              <div className="file-info">
                <h4 className="file-name">{selectedFile.name}</h4>
                <p className="file-size">{formatFileSize(selectedFile.size)}</p>
              </div>
              <button
                type="button"
                className="btn-remove-file"
                onClick={handleRemoveFile}
                disabled={isUploading}
              >
                ×
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="upload-error">
            <span className="error-icon">!</span>
            {error}
          </div>
        )}

        <div className="upload-form-fields">
          <div className="form-group">
            <label htmlFor="documentType" className="form-label">
              Document Type <span className="optional">(optional)</span>
            </label>
            <select
              id="documentType"
              className="form-select"
              value={documentType}
              onChange={(e) => setDocumentType(e.target.value)}
              disabled={isUploading}
            >
              <option value="">Select document type...</option>
              {documentTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="notes" className="form-label">
              Notes <span className="optional">(optional)</span>
            </label>
            <textarea
              id="notes"
              className="form-textarea"
              rows="3"
              placeholder="Add any relevant notes about this document..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              disabled={isUploading}
            />
          </div>
        </div>

        <button
          type="submit"
          className="btn-upload"
          disabled={!selectedFile || isUploading}
        >
          {isUploading ? (
            <>
              <span className="spinner"></span>
              Uploading...
            </>
          ) : (
            <>
                            Upload Document
            </>
          )}
        </button>
      </form>
    </div>
  );
};

export default DocumentUpload;
