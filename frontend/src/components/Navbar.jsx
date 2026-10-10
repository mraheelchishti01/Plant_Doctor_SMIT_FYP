import React, { useEffect, useState } from 'react';
import { Activity, ShieldCheck, Sparkles } from 'lucide-react';

export default function Navbar({ apiBaseUrl }) {
  const [apiOnline, setApiOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/health`);
        if (res.ok) {
          const data = await res.json();
          setApiOnline(true);
          setModelLoaded(data.model_loaded);
        } else {
          setApiOnline(false);
        }
      } catch (err) {
        setApiOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, [apiBaseUrl]);

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
          
          <div className="status-pill" title={apiOnline ? "Backend Connected" : "Local API Connecting..."}>
            <span className={`status-dot ${apiOnline ? '' : 'offline'}`} style={!apiOnline ? { background: '#f59e0b', boxShadow: '0 0 10px #f59e0b' } : {}} />
            <span>{apiOnline ? (modelLoaded ? 'AI Model Online' : 'API Connected') : 'Offline / Standby'}</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
