import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import CropSelector from './components/CropSelector';
import DiagnosticStudio from './components/DiagnosticStudio';
import DiagnosisResult from './components/DiagnosisResult';
import ValidationRejectionCard from './components/ValidationRejectionCard';
import ModelSpecs from './components/ModelSpecs';
import Footer from './components/Footer';
import { Clock, X } from 'lucide-react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function App() {
  const [activeCrop, setActiveCrop] = useState('potato');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState(null);
  const [error, setError] = useState(null);
  const [comingSoonCrop, setComingSoonCrop] = useState(null);

  const handleAnalyze = async (file) => {
    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Diagnostic failed with code ${response.status}`);
      }

      setDiagnosisResult(data);
    } catch (err) {
      console.error("Diagnosis error:", err);
      setError(err.message || "Failed to communicate with the Potato Doctor AI backend. Please verify that the API server is running.");
      setDiagnosisResult(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setDiagnosisResult(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <Navbar apiBaseUrl={API_BASE_URL} />

      <main className="main-content">
        <Hero />
        
        <CropSelector
          activeCrop={activeCrop}
          onSelectCrop={(cropId) => setActiveCrop(cropId)}
          onComingSoonClick={(cropName) => setComingSoonCrop(cropName)}
        />

        <DiagnosticStudio
          onAnalyze={handleAnalyze}
          onReset={handleReset}
          isAnalyzing={isAnalyzing}
          error={error}
          hasResult={!!diagnosisResult}
        />

        {diagnosisResult && (
          diagnosisResult.success ? (
            <DiagnosisResult
              result={diagnosisResult}
              apiBaseUrl={API_BASE_URL}
            />
          ) : (
            <ValidationRejectionCard
              result={diagnosisResult}
              onReset={handleReset}
            />
          )
        )}

        <ModelSpecs />
      </main>

      <Footer />

      {/* Coming Soon Modal */}
      {comingSoonCrop && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 999,
          padding: '20px'
        }}>
          <div className="glass-panel" style={{
            maxWidth: '480px',
            width: '100%',
            padding: '32px',
            position: 'relative',
            background: 'rgba(12, 24, 18, 0.95)',
            border: '1px solid rgba(16, 185, 129, 0.3)'
          }}>
            <button
              onClick={() => setComingSoonCrop(null)}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                background: 'transparent',
                color: 'var(--text-dim)',
                cursor: 'pointer'
              }}
            >
              <X size={20} />
            </button>

            <div style={{
              width: '54px',
              height: '54px',
              borderRadius: '50%',
              background: 'rgba(16, 185, 129, 0.12)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-light)',
              marginBottom: '20px'
            }}>
              <Clock size={28} />
            </div>

            <h3 style={{ fontSize: '1.5rem', marginBottom: '8px' }}>
              {comingSoonCrop} Disease Detection
            </h3>

            <div style={{
              display: 'inline-block',
              padding: '4px 10px',
              borderRadius: '999px',
              background: 'rgba(245, 158, 11, 0.15)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              color: '#fef08a',
              fontSize: '0.78rem',
              fontWeight: '700',
              textTransform: 'uppercase',
              marginBottom: '16px'
            }}>
              Coming Soon &bull; In Training
            </div>

            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.6', marginBottom: '24px' }}>
              The <strong>{comingSoonCrop}</strong> leaf pathology classification pipeline is currently undergoing dataset validation and transfer learning. Diagnostics for {comingSoonCrop} are disabled in this release to maintain scientific accuracy.
            </p>

            <button
              type="button"
              className="btn-primary"
              style={{ width: '100%' }}
              onClick={() => setComingSoonCrop(null)}
            >
              Return to Potato Diagnostics
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
