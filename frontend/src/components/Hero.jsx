import React from 'react';
import { ShieldCheck, Cpu, Leaf, FileCheck2 } from 'lucide-react';

export default function Hero() {
  return (
    <section className="hero-section">
      <div className="hero-pill">
        <div className="status-pill">
          <span className="status-dot"></span>
          <span>Solanaceous Crop Pathology Intelligence &bull; Vercel Ready</span>
        </div>
      </div>

      <h1 className="hero-title">
        Precision Potato Leaf Pathology<br />& Agronomic Defense
      </h1>

      <p className="hero-subtitle">
        Antigravity-native deep learning vision model for instant diagnosis of Early Blight, Late Blight, 
        and Healthy potato foliage, backed by verified agricultural extension treatment protocols.
      </p>

      <div className="hero-badges">
        <div className="hero-badge-item">
          <Leaf size={16} color="#34d399" />
          <span>2,152 Native Workspace Images</span>
        </div>
        <div className="hero-badge-item">
          <Cpu size={16} color="#34d399" />
          <span>MobileNetV2 Transfer Learning</span>
        </div>
        <div className="hero-badge-item">
          <ShieldCheck size={16} color="#34d399" />
          <span>FRAC-Compliant Management</span>
        </div>
        <div className="hero-badge-item">
          <FileCheck2 size={16} color="#34d399" />
          <span>Held-Out Test Set Validated</span>
        </div>
      </div>
    </section>
  );
}
