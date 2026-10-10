import React from 'react';
import { 
  AlertOctagon, HelpCircle, AlertTriangle, ShieldAlert, 
  RotateCcw, Camera, Check, Info, ShieldCheck
} from 'lucide-react';

export default function ValidationRejectionCard({ result, onReset }) {
  if (!result) return null;

  const { status, title, message, validation = {} } = result;
  const isRejected = status === 'REJECTED_NOT_POTATO';
  const isUncertain = status === 'UNCERTAIN_LEAF';

  const confidencePct = validation.confidence_pct ?? (validation.confidence_raw ? Math.round(validation.confidence_raw * 100) : 0);

  return (
    <section className="report-section" style={{ marginTop: '40px' }}>
      <div 
        className={`glass-panel rejection-card ${isRejected ? 'rejection-critical' : 'rejection-uncertain'}`}
        style={{
          padding: '36px',
          borderRadius: 'var(--radius-lg)',
          background: isRejected 
            ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(185, 28, 28, 0.03) 100%)' 
            : 'linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(217, 119, 6, 0.03) 100%)',
          border: isRejected 
            ? '1px solid rgba(239, 68, 68, 0.35)' 
            : '1px solid rgba(245, 158, 11, 0.35)',
          boxShadow: isRejected
            ? '0 20px 40px -15px rgba(239, 68, 68, 0.2)'
            : '0 20px 40px -15px rgba(245, 158, 11, 0.2)',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        {/* Glow Accent */}
        <div style={{
          position: 'absolute',
          top: '-80px',
          right: '-80px',
          width: '240px',
          height: '240px',
          borderRadius: '50%',
          background: isRejected ? 'radial-gradient(circle, rgba(239,68,68,0.2) 0%, transparent 70%)' : 'radial-gradient(circle, rgba(245,158,11,0.2) 0%, transparent 70%)',
          pointerEvents: 'none'
        }} />

        {/* Header Badge */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 14px',
              borderRadius: '999px',
              fontSize: '0.82rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              background: isRejected ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.15)',
              border: isRejected ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(245, 158, 11, 0.4)',
              color: isRejected ? '#fca5a5' : '#fcd34d'
            }}>
              {isRejected ? <ShieldAlert size={14} /> : <HelpCircle size={14} />}
              {isRejected ? 'Species Validation: Filtered' : 'Validation: Inconclusive'}
            </span>

            <span style={{
              fontSize: '0.82rem',
              color: 'var(--text-dim)',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <ShieldCheck size={14} style={{ color: 'var(--accent-light)' }} />
              Two-Stage Safety Filter Enforced
            </span>
          </div>

          <div style={{
            fontSize: '0.85rem',
            padding: '4px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-muted)'
          }}>
            Potato Probability: <strong style={{ color: isRejected ? '#fca5a5' : '#fcd34d' }}>{confidencePct}%</strong>
          </div>
        </div>

        {/* Main Title & Prompt Requirement Text */}
        <div style={{ marginBottom: '28px' }}>
          <h2 style={{
            fontSize: '1.85rem',
            fontWeight: 800,
            letterSpacing: '-0.02em',
            marginBottom: '14px',
            color: isRejected ? '#fecaca' : '#fef3c7',
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}>
            {isRejected ? (
              <AlertOctagon size={28} style={{ color: '#ef4444', flexShrink: 0 }} />
            ) : (
              <AlertTriangle size={28} style={{ color: '#f59e0b', flexShrink: 0 }} />
            )}
            {title}
          </h2>

          <p style={{
            fontSize: '1.08rem',
            lineHeight: 1.6,
            color: '#ffffff',
            background: 'rgba(0, 0, 0, 0.25)',
            padding: '16px 20px',
            borderRadius: 'var(--radius-sm)',
            borderLeft: isRejected ? '4px solid #ef4444' : '4px solid #f59e0b',
            fontWeight: 500
          }}>
            "{message}"
          </p>
        </div>

        {/* Diagnostic Safety Explanation */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '20px',
          marginBottom: '28px'
        }}>
          <div style={{
            background: 'rgba(0, 0, 0, 0.25)',
            padding: '20px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)'
          }}>
            <h4 style={{ fontSize: '0.92rem', color: 'var(--accent-light)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Info size={16} /> Why Was Prediction Stopped?
            </h4>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              {isRejected 
                ? 'To ensure scientific diagnostic accuracy, Potato Doctor AI deploys an independent botanical validator before disease models. Non-potato foliage (such as tomato, corn, or pepper) and unrelated images are immediately rejected to prevent false diagnoses.'
                : 'The visual morphology of the leaf could not be verified with high confidence. Operating without leaf certainty could yield misleading disease diagnoses, so analysis was safely halted.'}
            </p>
          </div>

          <div style={{
            background: 'rgba(0, 0, 0, 0.25)',
            padding: '20px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)'
          }}>
            <h4 style={{ fontSize: '0.92rem', color: '#6ee7b7', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Camera size={16} /> Image Capture Guidelines
            </h4>
            <ul style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6, paddingLeft: '16px', margin: 0 }}>
              <li>Ensure the image depicts an authentic <strong>Solanum tuberosum</strong> (potato) leaf.</li>
              <li>Frame a single leaf or leaflet sharply in the center under clear natural light.</li>
              <li>Avoid blurred camera focus, dark shadows, or screenshots of text.</li>
            </ul>
          </div>
        </div>

        {/* Action Button */}
        {onReset && (
          <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
            <button 
              type="button" 
              className="btn-secondary"
              onClick={onReset}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 20px',
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid var(--border-subtle)',
                color: '#fff',
                borderRadius: 'var(--radius-sm)',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              <RotateCcw size={16} />
              <span>Upload New Specimen</span>
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
