import React from 'react';
import { Settings, ShieldCheck, Server } from 'lucide-react';

export default function Navbar({ apiBaseUrl, apiOnline, modelLoaded, onOpenSettings }) {
  const getStatusText = () => {
    if (!apiOnline) return 'API Offline';
    if (modelLoaded) return 'AI Model Online';
    return 'API Connected';
  };

  const getStatusColor = () => {
    if (!apiOnline) return '#ef4444'; // Red for offline
    if (modelLoaded) return '#10b981'; // Green for online & verified
    return '#f59e0b'; // Amber for connected without model
  };

  return (
    <nav className="navbar">
      <div className="nav-wrapper">
        <a href="#" className="nav-brand">
          <div className="nav-brand-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
              <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
            </svg>
          </div>
          <div className="nav-brand-text">
            Potato Doctor <span>AI</span>
          </div>
        </a>

        <div className="nav-links">
          <a href="#crops" className="nav-link">Crops</a>
          <a href="#scanner" className="nav-link">Diagnostic Studio</a>
          <a href="#model-specs" className="nav-link">Model Specs</a>
          
          <button
            type="button"
            className="status-pill"
            onClick={onOpenSettings}
            title={apiOnline ? "Backend Connected & Verified (Click to configure)" : "API Offline: Click to connect live server"}
            style={{
              cursor: 'pointer',
              background: 'rgba(0, 0, 0, 0.4)',
              border: `1px solid ${apiOnline ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)'}`,
              transition: 'all 0.2s ease'
            }}
          >
            <span 
              className={`status-dot ${apiOnline ? '' : 'offline'}`} 
              style={{
                background: getStatusColor(),
                boxShadow: `0 0 10px ${getStatusColor()}`
              }} 
            />
            <span style={{ color: apiOnline ? '#fff' : '#fca5a5' }}>
              {getStatusText()}
            </span>
            <Settings size={13} style={{ marginLeft: '4px', opacity: 0.6 }} />
          </button>
        </div>
      </div>
    </nav>
  );
}
