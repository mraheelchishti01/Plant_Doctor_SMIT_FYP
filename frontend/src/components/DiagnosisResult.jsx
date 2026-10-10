import React, { useState } from 'react';
import { 
  AlertOctagon, CheckCircle2, AlertTriangle, ShieldCheck, 
  HelpCircle, Sparkles, BookOpen, Clock, Send, Info
} from 'lucide-react';

export default function DiagnosisResult({ result, apiBaseUrl }) {
  const [activeTab, setActiveTab] = useState('immediate');
  const [userQuestion, setUserQuestion] = useState('');
  const [aiAdvice, setAiAdvice] = useState(null);
  const [isLoadingAi, setIsLoadingAi] = useState(false);

  if (!result) return null;

  const {
    disease_name,
    predicted_class,
    confidence,
    severity,
    scientific_name,
    probabilities = {},
    guidance = {},
    image_metadata = {}
  } = result;

  const isHealthy = predicted_class === 'Potato___healthy';
  const isLateBlight = predicted_class === 'Potato___Late_blight';
  
  const severityStyle = isHealthy ? 'healthy' : (isLateBlight ? 'critical' : 'warning');

  const handleAskAgronomist = async (e) => {
    e.preventDefault();
    if (!userQuestion.trim()) return;

    setIsLoadingAi(true);
    try {
      const res = await fetch(`${apiBaseUrl}/advice`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          predicted_class,
          question: userQuestion
        })
      });

      if (res.ok) {
        const data = await res.json();
        setAiAdvice(data);
      }
    } catch (err) {
      console.error("Agronomist call failed:", err);
    } finally {
      setIsLoadingAi(false);
    }
  };

  return (
    <section className="report-section">
      <div className={`glass-panel diagnosis-hero-card ${severityStyle}`}>
        <div className="diagnosis-top">
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <span className={`status-pill ${severityStyle}`} style={{ padding: '4px 12px' }}>
                <span className="status-dot"></span>
                Confirmed Pathology Diagnosis
              </span>
            </div>
            <h2 className="disease-title">{disease_name}</h2>
            <div className="pathogen-name">Pathogen / Etiology: {scientific_name}</div>
          </div>

          <div className={`severity-pill ${severityStyle}`}>
            {isHealthy ? <CheckCircle2 size={16} /> : (isLateBlight ? <AlertOctagon size={16} /> : <AlertTriangle size={16} />)}
            <span>Severity: {severity}</span>
          </div>
        </div>

        {/* Metrics Row */}
        <div className="metrics-row">
          <div className="confidence-circle">
            <div className="confidence-val" style={{
              color: isHealthy ? '#34d399' : (isLateBlight ? '#f87171' : '#fbbf24')
            }}>
              {confidence}%
            </div>
            <div className="confidence-label">Model Confidence</div>
          </div>

          <div className="prob-bars">
            {Object.entries(probabilities).map(([clsName, prob]) => (
              <div key={clsName}>
                <div className="prob-item-header">
                  <span style={{ fontWeight: clsName === disease_name ? '600' : '400', color: clsName === disease_name ? '#fff' : 'var(--text-muted)' }}>
                    {clsName}
                  </span>
                  <span style={{ fontWeight: '600', color: clsName === disease_name ? 'var(--accent-light)' : 'var(--text-dim)' }}>
                    {prob}%
                  </span>
                </div>
                <div className="prob-bar-bg">
                  <div
                    className="prob-bar-fill"
                    style={{
                      width: `${prob}%`,
                      background: clsName.includes('Late') 
                        ? 'linear-gradient(90deg, #dc2626, #ef4444)' 
                        : (clsName.includes('Early') 
                          ? 'linear-gradient(90deg, #d97706, #f59e0b)' 
                          : 'linear-gradient(90deg, #059669, #10b981)')
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {image_metadata.relevance_warning && (
          <div style={{
            marginTop: '20px',
            padding: '12px 16px',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.25)',
            fontSize: '0.84rem',
            color: '#fef08a',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <AlertTriangle size={16} style={{ flexShrink: 0 }} />
            <span>{image_metadata.relevance_warning}</span>
          </div>
        )}
      </div>

      {/* Guidance Panel */}
      <div className="glass-panel guidance-panel">
        <div className="tab-headers">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'immediate' ? 'active' : ''}`}
            onClick={() => setActiveTab('immediate')}
          >
            <Clock size={16} /> Immediate Actions (24-48h)
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'prevention' ? 'active' : ''}`}
            onClick={() => setActiveTab('prevention')}
          >
            <ShieldCheck size={16} /> Cultural Prevention
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'treatment' ? 'active' : ''}`}
            onClick={() => setActiveTab('treatment')}
          >
            <BookOpen size={16} /> Evidence-Based Treatments
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'agronomist' ? 'active' : ''}`}
            onClick={() => setActiveTab('agronomist')}
          >
            <Sparkles size={16} /> AI Agronomist Consultation
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'immediate' && (
            <div className="guidance-list">
              {guidance.immediate_actions?.map((action, i) => (
                <div key={i} className="guidance-item">
                  <div className="guidance-icon"><CheckCircle2 size={18} /></div>
                  <div className="guidance-text">{action}</div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'prevention' && (
            <div className="guidance-list">
              {guidance.prevention?.map((item, i) => (
                <div key={i} className="guidance-item">
                  <div className="guidance-icon"><ShieldCheck size={18} /></div>
                  <div className="guidance-text">{item}</div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'treatment' && (
            <div className="guidance-list">
              <div style={{ marginBottom: '14px', fontSize: '0.92rem', color: 'var(--text-muted)' }}>
                <strong>Evidence-Based Fungicides & Biocontrol Guidelines:</strong>
              </div>
              {guidance.treatments?.fungicides?.map((t, i) => (
                <div key={i} className="guidance-item">
                  <div className="guidance-icon">💊</div>
                  <div className="guidance-text">{t}</div>
                </div>
              ))}
              {guidance.treatments?.precautions && (
                <div style={{
                  padding: '14px 18px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(239, 68, 68, 0.08)',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  fontSize: '0.88rem',
                  color: '#fca5a5',
                  marginTop: '10px'
                }}>
                  <strong>Important Precaution:</strong> {guidance.treatments.precautions}
                </div>
              )}
            </div>
          )}

          {activeTab === 'agronomist' && (
            <div>
              <p style={{ fontSize: '0.92rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
                Ask specific agronomy questions regarding recovery plans, irrigation schedules, or regional soil conditions:
              </p>

              <form onSubmit={handleAskAgronomist} style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
                <input
                  type="text"
                  value={userQuestion}
                  onChange={(e) => setUserQuestion(e.target.value)}
                  placeholder="e.g., How often should I spray copper fungicide during humid weather?"
                  style={{
                    flex: 1,
                    padding: '12px 18px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid var(--border-subtle)',
                    color: '#fff',
                    outline: 'none',
                    fontSize: '0.92rem'
                  }}
                />
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={isLoadingAi || !userQuestion.trim()}
                  style={{ padding: '12px 20px', fontSize: '0.9rem' }}
                >
                  {isLoadingAi ? <Sparkles size={16} className="animate-spin" /> : <Send size={16} />}
                  <span>{isLoadingAi ? 'Consulting...' : 'Ask AI'}</span>
                </button>
              </form>

              {aiAdvice && (
                <div style={{
                  padding: '20px',
                  borderRadius: 'var(--radius-md)',
                  background: 'rgba(16, 185, 129, 0.08)',
                  border: '1px solid rgba(16, 185, 129, 0.25)',
                  fontSize: '0.92rem',
                  lineHeight: '1.7',
                  color: '#e2e8f0'
                }}>
                  <div style={{
                    fontSize: '0.8rem',
                    fontWeight: '700',
                    color: 'var(--accent-light)',
                    textTransform: 'uppercase',
                    marginBottom: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    <Sparkles size={14} /> Source: {aiAdvice.source}
                  </div>
                  <div style={{ whiteSpace: 'pre-wrap' }}>
                    {aiAdvice.ai_advice || aiAdvice.verified_knowledge?.expert_consultation}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Disclaimer */}
        <div className="disclaimer-box">
          <Info size={18} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--accent-light)' }} />
          <div>
            <strong>Diagnostic Disclaimer:</strong> {guidance.disclaimer || "Diagnostic guidance is based on machine vision and agricultural extension literature. Always comply with official pesticide regulations."}
            <div style={{ marginTop: '4px', fontSize: '0.78rem', color: 'var(--text-dim)' }}>
              Authoritative Sources: {guidance.sources?.join(', ') || 'Cornell University, UC Davis IPM, USDA-ARS, International Potato Center'}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
