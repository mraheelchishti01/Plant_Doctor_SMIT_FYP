import React, { useState } from 'react';
import { Server, CheckCircle2, AlertTriangle, X, RefreshCw, ExternalLink, ShieldCheck } from 'lucide-react';

export default function ApiSettingsModal({ isOpen, onClose, currentApiUrl, onSaveApiUrl, apiOnline, modelLoaded }) {
  const [testUrl, setTestUrl] = useState(currentApiUrl || '');
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  if (!isOpen) return null;

  const handleTest = async () => {
    const url = testUrl.trim().replace(/\/+$/, '');
    if (!url) {
      setTestResult({ success: false, message: 'Please enter a valid backend URL.' });
      return;
    }

    setTesting(true);
    setTestResult(null);

    const startTime = Date.now();
    try {
      const res = await fetch(`${url}/health`, { method: 'GET' });
      const latency = Date.now() - startTime;

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      const isReady = data.model_loaded === true || (data.disease_classifier?.loaded && data.potato_validator?.loaded);

      setTestResult({
        success: true,
        latency,
        data,
        isReady,
        message: isReady
          ? `Connected successfully (${latency}ms)! Both Botanical Validator and Disease Classifier are loaded.`
          : `Connected (${latency}ms), but AI models are still initializing.`
      });
    } catch (err) {
      setTestResult({
        success: false,
        message: `Connection failed: ${err.message || 'Cannot reach endpoint'}. Ensure the backend is running and CORS is enabled.`
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = () => {
    const cleaned = testUrl.trim().replace(/\/+$/, '');
    onSaveApiUrl(cleaned);
    onClose();
  };

  const handleReset = () => {
    const defaultUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
      ? 'http://127.0.0.1:8000'
      : '';
    setTestUrl(defaultUrl);
    onSaveApiUrl(defaultUrl);
    setTestResult(null);
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.8)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div className="glass-panel" style={{
        maxWidth: '560px',
        width: '100%',
        padding: '32px',
        position: 'relative',
        background: 'rgba(10, 20, 14, 0.98)',
        border: '1px solid rgba(16, 185, 129, 0.35)',
        borderRadius: 'var(--radius-lg)',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)'
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '18px',
            right: '18px',
            background: 'transparent',
            border: 'none',
            color: 'var(--text-dim)',
            cursor: 'pointer',
            padding: '4px'
          }}
        >
          <X size={20} />
        </button>

        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '20px' }}>
          <div style={{
            width: '46px',
            height: '46px',
            borderRadius: '12px',
            background: 'rgba(16, 185, 129, 0.15)',
            border: '1px solid rgba(16, 185, 129, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--accent-light)'
          }}>
            <Server size={22} />
          </div>
          <div>
            <h3 style={{ fontSize: '1.35rem', fontWeight: 700, margin: 0, color: '#fff' }}>
              Backend API Configuration
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
              Connect your Vercel frontend to the Potato Doctor AI inference server.
            </p>
          </div>
        </div>

        {/* Current Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 16px',
          borderRadius: 'var(--radius-sm)',
          background: apiOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
          border: apiOnline ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(245, 158, 11, 0.3)',
          marginBottom: '20px'
        }}>
          <span style={{ fontSize: '0.88rem', color: apiOnline ? '#a7f3d0' : '#fde68a', display: 'flex', alignItems: 'center', gap: '8px' }}>
            {apiOnline ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
            Status: <strong>{apiOnline ? (modelLoaded ? 'AI Model Online & Verified' : 'API Connected') : 'API Offline / Unreachable'}</strong>
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            {currentApiUrl ? new URL(currentApiUrl, window.location.href).origin : 'No URL Set'}
          </span>
        </div>

        {/* Input */}
        <div style={{ marginBottom: '18px' }}>
          <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '8px' }}>
            Backend API Base URL
          </label>
          <input
            type="url"
            value={testUrl}
            onChange={(e) => setTestUrl(e.target.value)}
            placeholder="e.g. https://plant-doctor-backend.onrender.com or http://127.0.0.1:8000"
            style={{
              width: '100%',
              padding: '12px 14px',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(0, 0, 0, 0.4)',
              border: '1px solid var(--border-subtle)',
              color: '#fff',
              fontSize: '0.92rem',
              outline: 'none',
              boxSizing: 'border-box'
            }}
          />
        </div>

        {/* Test Result Message */}
        {testResult && (
          <div style={{
            padding: '12px 16px',
            borderRadius: 'var(--radius-sm)',
            background: testResult.success ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
            border: testResult.success ? '1px solid rgba(16, 185, 129, 0.35)' : '1px solid rgba(239, 68, 68, 0.35)',
            color: testResult.success ? '#6ee7b7' : '#fca5a5',
            fontSize: '0.85rem',
            lineHeight: 1.5,
            marginBottom: '18px'
          }}>
            {testResult.message}
          </div>
        )}

        {/* Actions */}
        <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '24px' }}>
          <button
            type="button"
            className="btn-secondary"
            onClick={handleTest}
            disabled={testing}
            style={{ padding: '8px 16px', fontSize: '0.88rem' }}
          >
            {testing ? <RefreshCw size={15} className="animate-spin" /> : <RefreshCw size={15} />}
            <span>Test Connection</span>
          </button>

          <button
            type="button"
            className="btn-primary"
            onClick={handleSave}
            style={{ padding: '8px 20px', fontSize: '0.88rem' }}
          >
            <CheckCircle2 size={16} />
            <span>Save & Apply</span>
          </button>
        </div>

        {/* Production Deployment Guidance */}
        <div style={{
          marginTop: '22px',
          paddingTop: '16px',
          borderTop: '1px solid var(--border-subtle)',
          fontSize: '0.8rem',
          color: 'var(--text-dim)',
          lineHeight: 1.5
        }}>
          💡 <strong>Production Note:</strong> To set this permanently for all Vercel visitors, add <code style={{ color: 'var(--accent-light)' }}>VITE_API_BASE_URL</code> in your Vercel Project Settings &gt; Environment Variables.
        </div>
      </div>
    </div>
  );
}
