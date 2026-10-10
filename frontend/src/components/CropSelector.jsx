import React from 'react';
import { CheckCircle2, Lock, Sparkles, Clock } from 'lucide-react';

export default function CropSelector({ activeCrop, onSelectCrop, onComingSoonClick }) {
  const crops = [
    {
      id: 'potato',
      name: 'Potato',
      latin: 'Solanum tuberosum',
      status: 'AVAILABLE',
      isActive: true,
      avatar: '🥔',
      desc: 'Active diagnostic pipeline with 3 calibrated pathology classes: Early Blight, Late Blight, & Healthy.',
      classesCount: 3,
      datasetCount: '2,152 leaves'
    },
    {
      id: 'tomato',
      name: 'Tomato',
      latin: 'Solanum lycopersicum',
      status: 'COMING SOON',
      isActive: false,
      avatar: '🍅',
      desc: 'Multi-pathogen model (Bacterial spot, Yellow Leaf Curl, Septoria) currently under training.',
      classesCount: null,
      datasetCount: 'In Validation'
    },
    {
      id: 'corn',
      name: 'Corn (Maize)',
      latin: 'Zea mays',
      status: 'COMING SOON',
      isActive: false,
      avatar: '🌽',
      desc: 'Cercospora leaf spot, Common Rust, and Northern Leaf Blight pipeline scheduled for release.',
      classesCount: null,
      datasetCount: 'In Validation'
    },
    {
      id: 'pepper',
      name: 'Pepper (Bell)',
      latin: 'Capsicum annuum',
      status: 'COMING SOON',
      isActive: false,
      avatar: '🫑',
      desc: 'Bacterial Spot and Phytophthora blight diagnostic weights currently undergoing laboratory evaluation.',
      classesCount: null,
      datasetCount: 'In Validation'
    }
  ];

  return (
    <section id="crops">
      <div className="section-header">
        <div>
          <h2 className="section-title">Crop Coverage Portfolio</h2>
          <p className="section-desc">Select a target crop to activate the specialized pathology diagnostic engine.</p>
        </div>
      </div>

      <div className="crops-grid">
        {crops.map((crop) => (
          <div
            key={crop.id}
            className={`glass-panel crop-card ${crop.isActive ? 'active' : 'disabled'}`}
            onClick={() => {
              if (crop.isActive) {
                onSelectCrop(crop.id);
              } else {
                onComingSoonClick(crop.name);
              }
            }}
            style={{ cursor: crop.isActive ? 'default' : 'pointer' }}
          >
            <div className="crop-card-top">
              <div className="crop-avatar">{crop.avatar}</div>
              <span className={`crop-tag ${crop.isActive ? 'active' : 'soon'}`}>
                {crop.status}
              </span>
            </div>

            <h3 className="crop-name">{crop.name}</h3>
            <p className="crop-latin">{crop.latin}</p>
            <p className="crop-info">{crop.desc}</p>

            <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border-glass)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem', color: 'var(--text-dim)' }}>
              <span>{crop.datasetCount}</span>
              {crop.isActive ? (
                <span style={{ color: '#34d399', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: '600' }}>
                  <CheckCircle2 size={14} /> Diagnostic Engine Ready
                </span>
              ) : (
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Clock size={14} /> Coming Soon
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
