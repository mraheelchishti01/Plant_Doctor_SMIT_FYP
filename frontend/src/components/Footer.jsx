import React from 'react';

export default function Footer() {
  return (
    <footer className="footer">
      <div style={{ maxWidth: '1240px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div>
          Potato Doctor AI &bull; Leaf Pathology & Crop Disease Diagnostic System
        </div>
        <div>
          Developed by <strong>Muhammad Raheel Chishti</strong> &bull; SMIT FYP
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '4px' }}>
          Trained on Native Workspace Dataset &bull; Deployable to Vercel
        </div>
      </div>
    </footer>
  );
}
