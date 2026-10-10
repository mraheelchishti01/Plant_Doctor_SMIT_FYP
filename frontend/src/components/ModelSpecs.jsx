import React from 'react';
import { Cpu, Layers, Database, BarChart3, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ModelSpecs() {
  return (
    <section id="model-specs" style={{ marginTop: '64px' }}>
      <div className="section-header">
        <div>
          <h2 className="section-title">Model Architecture & Verification Telemetry</h2>
          <p className="section-desc">Audited machine learning specifications, dataset distribution, and held-out test evaluation.</p>
        </div>
      </div>

      <div className="specs-grid">
        <div className="glass-panel spec-card">
          <div className="spec-title">Neural Backbone</div>
          <div className="spec-value">MobileNetV2</div>
          <div className="spec-desc">Pre-trained ImageNet inverted residual blocks + custom dense classification head (2.43M total parameters, 11.6 MB model size).</div>
        </div>

        <div className="glass-panel spec-card">
          <div className="spec-title">Test Accuracy (Held-Out)</div>
          <div className="spec-value" style={{ color: '#34d399' }}>97.22%</div>
          <div className="spec-desc">Evaluated on 324 completely unseen test images (150 Early Blight, 150 Late Blight, 24 Healthy). Macro F1-score: 0.9566.</div>
        </div>

        <div className="glass-panel spec-card">
          <div className="spec-title">Workspace Dataset</div>
          <div className="spec-value">2,152 Leaves</div>
          <div className="spec-desc">Native dataset: 1,000 Early Blight, 1,000 Late Blight, 152 Healthy. Stratified 70% train (1,506), 15% val (322), 15% test (324).</div>
        </div>

        <div className="glass-panel spec-card">
          <div className="spec-title">Deployment Target</div>
          <div className="spec-value">Vercel + Python API</div>
          <div className="spec-desc">FastAPI microservice with exact MobileNetV2 preprocessing and edge-compatible Vite frontend.</div>
        </div>
      </div>

      {/* Metrics & Confusion Matrix Showcase */}
      <div style={{
        marginTop: '28px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
        gap: '24px'
      }}>
        {/* Table of Test Metrics */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={18} color="#34d399" />
            <span>Held-Out Test Set Metrics</span>
          </h3>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', textAlign: 'left', color: 'var(--text-dim)' }}>
                  <th style={{ padding: '10px 8px' }}>Class</th>
                  <th style={{ padding: '10px 8px' }}>Precision</th>
                  <th style={{ padding: '10px 8px' }}>Recall</th>
                  <th style={{ padding: '10px 8px' }}>F1-Score</th>
                  <th style={{ padding: '10px 8px' }}>Support</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid var(--border-glass)' }}>
                  <td style={{ padding: '12px 8px', fontWeight: '600' }}>Potato Early Blight</td>
                  <td style={{ padding: '12px 8px', color: '#34d399' }}>97.39%</td>
                  <td style={{ padding: '12px 8px' }}>99.33%</td>
                  <td style={{ padding: '12px 8px', fontWeight: '600' }}>0.9835</td>
                  <td style={{ padding: '12px 8px', color: 'var(--text-muted)' }}>150</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-glass)' }}>
                  <td style={{ padding: '12px 8px', fontWeight: '600' }}>Potato Late Blight</td>
                  <td style={{ padding: '12px 8px', color: '#34d399' }}>97.96%</td>
                  <td style={{ padding: '12px 8px' }}>96.00%</td>
                  <td style={{ padding: '12px 8px', fontWeight: '600' }}>0.9697</td>
                  <td style={{ padding: '12px 8px', color: 'var(--text-muted)' }}>150</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-glass)' }}>
                  <td style={{ padding: '12px 8px', fontWeight: '600' }}>Potato Healthy</td>
                  <td style={{ padding: '12px 8px', color: '#34d399' }}>91.67%</td>
                  <td style={{ padding: '12px 8px' }}>91.67%</td>
                  <td style={{ padding: '12px 8px', fontWeight: '600' }}>0.9167</td>
                  <td style={{ padding: '12px 8px', color: 'var(--text-muted)' }}>24</td>
                </tr>
                <tr style={{ fontWeight: '700', color: '#fff' }}>
                  <td style={{ padding: '12px 8px' }}>Macro Average</td>
                  <td style={{ padding: '12px 8px' }}>95.67%</td>
                  <td style={{ padding: '12px 8px' }}>95.67%</td>
                  <td style={{ padding: '12px 8px', color: '#34d399' }}>0.9566</td>
                  <td style={{ padding: '12px 8px', color: 'var(--text-muted)' }}>324</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Confusion Matrix Chart Preview */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <CheckCircle2 size={18} color="#34d399" />
            <span>Test Confusion Matrix</span>
          </h3>

          <div style={{
            flex: 1,
            borderRadius: 'var(--radius-md)',
            overflow: 'hidden',
            border: '1px solid var(--border-glass)',
            background: '#040906',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            maxHeight: '340px'
          }}>
            <img
              src="/confusion_matrix.png"
              alt="Test Confusion Matrix"
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
          </div>
        </div>
      </div>

      <div style={{
        marginTop: '24px',
        padding: '20px 24px',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(255, 255, 255, 0.03)',
        border: '1px solid var(--border-glass)',
        fontSize: '0.88rem',
        color: 'var(--text-muted)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '14px'
      }}>
        <AlertCircle size={20} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong style={{ color: '#fff' }}>Plant Pathology & Field Limitations:</strong> This model was trained on the standardized PlantVillage potato dataset. In real-world agricultural field settings, factors such as extreme shadows, soil splashing, insect chewing marks, and multiple simultaneous fungal infections can introduce visual ambiguity. Always verify AI diagnoses with on-site agronomists before large-scale pesticide applications.
        </div>
      </div>
    </section>
  );
}
