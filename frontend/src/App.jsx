import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import CropSelector from './components/CropSelector';
import DiagnosticStudio from './components/DiagnosticStudio';
import DiagnosisResult from './components/DiagnosisResult';
import ValidationRejectionCard from './components/ValidationRejectionCard';
import ModelSpecs from './components/ModelSpecs';
import ApiSettingsModal from './components/ApiSettingsModal';
import Footer from './components/Footer';
import { Clock, X, AlertCircle, Settings } from 'lucide-react';
import './App.css';

const getInitialApiUrl = () => {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem('potato_doctor_api_url');
    if (saved && saved.trim()) return saved.trim().replace(/\/+$/, '');
  }
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && envUrl.trim()) return envUrl.trim().replace(/\/+$/, '');
  
  if (typeof window !== 'undefined') {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://127.0.0.1:8000';
    }
  }
  return '';
};

export default function App() {
  const [apiUrl, setApiUrl] = useState(getInitialApiUrl);
  const [apiOnline, setApiOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const [activeCrop, setActiveCrop] = useState('potato');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState(null);
  const [error, setError] = useState(null);
  const [comingSoonCrop, setComingSoonCrop] = useState(null);

  // Periodic health check against the active API endpoint
  useEffect(() => {
    if (!apiUrl) {
      setApiOnline(false);
      setModelLoaded(false);
      return;
    }

    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${apiUrl}/health`);
        if (!isMounted) return;
        if (res.ok) {
          const data = await res.json();
          setApiOnline(true);
          const isReady = data.model_loaded === true || (data.disease_classifier?.loaded && data.potato_validator?.loaded);
          setModelLoaded(isReady);
        } else {
          setApiOnline(false);
          setModelLoaded(false);
        }
      } catch (err) {
        if (!isMounted) return;
        setApiOnline(false);
        setModelLoaded(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [apiUrl]);

  const handleSaveApiUrl = (newUrl) => {
    setApiUrl(newUrl);
    if (typeof window !== 'undefined') {
      if (newUrl) {
        localStorage.setItem('potato_doctor_api_url', newUrl);
      } else {
        localStorage.removeItem('potato_doctor_api_url');
      }
    }
  };

  const handleAnalyze = async (file) => {
    if (!apiUrl) {
      setError("Backend API is not connected. Click the status badge in the top navigation to connect your live server.");
      setIsSettingsOpen(true);
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
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
      setError(err.message || "Failed to communicate with the Potato Doctor AI backend. Please verify your server connection.");
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
      <Navbar 
        apiBaseUrl={apiUrl}
        apiOnline={apiOnline}
        modelLoaded={modelLoaded}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Production Offline Notice if API is not yet connected on cloud */}
      {!apiOnline && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.12)',
          borderBottom: '1px solid rgba(239, 68, 68, 0.25)',
          padding: '10px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          fontSize: '0.85rem',
          color: '#fca5a5'
        }}>
          <AlertCircle size={16} />
          <span>Backend API is currently offline. Deep learning model inference requires a running backend server.</span>
          <button
            onClick={() => setIsSettingsOpen(true)}
            style={{
              background: 'rgba(255, 255, 255, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.4)',
              color: '#fff',
              padding: '4px 12px',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Settings size={13} /> Configure API URL
          </button>
        </div>
      )}

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
              apiBaseUrl={apiUrl}
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

      {/* API Settings Modal */}
      <ApiSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        currentApiUrl={apiUrl}
        onSaveApiUrl={handleSaveApiUrl}
        apiOnline={apiOnline}
        modelLoaded={modelLoaded}
      />

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
            <p style={{ color: 'var(--text-muted)', lineHeight: 1.6, fontSize: '0.92rem' }}>
              We are actively curating and training pathology models for {comingSoonCrop}. This module will be released in an upcoming update to maintain scientific diagnostic precision.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
