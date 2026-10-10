import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles, RotateCcw, AlertTriangle, FileCheck, CheckCircle } from 'lucide-react';

export default function DiagnosticStudio({ onAnalyze, onReset, isAnalyzing, error, hasResult }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [samples, setSamples] = useState([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Load local sample presets
    fetch('/samples/samples.json')
      .then(res => res.json())
      .then(data => setSamples(data))
      .catch(() => {});
  }, []);

  const handleFileChange = (file) => {
    if (!file) return;
    
    // Clear previous prediction immediately as requested
    onReset();

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const loadSample = async (sample) => {
    try {
      const response = await fetch(sample.src);
      const blob = await response.blob();
      const file = new File([blob], sample.name, { type: 'image/jpeg' });
      handleFileChange(file);
    } catch (err) {
      console.error("Failed to load sample:", err);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    onReset();
  };

  const handleAnalyzeClick = () => {
    if (!selectedFile) return;
    onAnalyze(selectedFile);
  };

  return (
    <section id="scanner" style={{ marginTop: '54px' }}>
      <div className="section-header">
        <div>
          <h2 className="section-title">Potato Diagnostic Studio</h2>
          <p className="section-desc">Upload a high-resolution potato leaf photograph or test using verified pathology presets.</p>
        </div>
      </div>

      <div className="studio-grid">
        {/* Upload Panel */}
        <div className="glass-panel upload-panel">
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => handleFileChange(e.target.files?.[0])}
            accept="image/jpeg,image/png,image/webp,image/bmp"
            style={{ display: 'none' }}
          />

          <div
            className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
            onDragLeave={() => setIsDragActive(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon">
              <UploadCloud size={28} />
            </div>
            <h3 className="dropzone-title">Drag & Drop Leaf Photograph</h3>
            <p className="dropzone-sub">Supports JPEG, PNG, WEBP &bull; Max size 15MB</p>
            <button type="button" className="browse-btn" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
              <ImageIcon size={16} /> Browse From Device
            </button>
          </div>

          {/* Sample Presets */}
          {samples.length > 0 && (
            <div className="presets-section">
              <div className="presets-label">Quick Test Specimens (Held-Out Test Set)</div>
              <div className="presets-list">
                {samples.map((s, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className={`preset-chip ${s.category === 'negative' ? 'preset-negative' : (s.category === 'uncertain' ? 'preset-uncertain' : '')}`}
                    onClick={() => loadSample(s)}
                    title={`Load ${s.label}`}
                  >
                    <span>{s.category === 'negative' ? '🚫' : (s.category === 'uncertain' ? '⚠️' : '🍃')}</span> {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div style={{
              marginTop: '20px',
              padding: '14px 18px',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(239, 68, 68, 0.12)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: '#fca5a5',
              fontSize: '0.88rem',
              display: 'flex',
              alignItems: 'center',
              gap: '12px'
            }}>
              <AlertTriangle size={18} style={{ flexShrink: 0 }} />
              <div>{error}</div>
            </div>
          )}
        </div>

        {/* Preview & Scanner Panel */}
        <div className="glass-panel preview-panel">
          <div className="preview-wrapper">
            {previewUrl ? (
              <>
                <img src={previewUrl} alt="Leaf Specimen" className="preview-img" />
                {isAnalyzing && <div className="scanner-laser"></div>}
              </>
            ) : (
              <div className="preview-empty">
                <ImageIcon size={48} strokeWidth={1.5} />
                <span>No specimen selected</span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Select or drop a photo to preview</span>
              </div>
            )}
          </div>

          {selectedFile && (
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.82rem',
              color: 'var(--text-dim)',
              marginBottom: '16px',
              padding: '0 4px'
            }}>
              <span><strong>File:</strong> {selectedFile.name}</span>
              <span><strong>Size:</strong> {(selectedFile.size / 1024).toFixed(1)} KB</span>
            </div>
          )}

          <div className="action-buttons">
            <button
              type="button"
              className="btn-primary"
              disabled={!selectedFile || isAnalyzing}
              onClick={handleAnalyzeClick}
            >
              {isAnalyzing ? (
                <>
                  <Sparkles size={18} className="animate-spin" />
                  <span>Scanning Leaf Pathology...</span>
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  <span>Run Pathology Diagnosis</span>
                </>
              )}
            </button>

            {selectedFile && (
              <button
                type="button"
                className="btn-secondary"
                onClick={handleReset}
                disabled={isAnalyzing}
                title="Reset Specimen"
              >
                <RotateCcw size={16} />
                <span>Reset</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
