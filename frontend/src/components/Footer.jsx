import React from 'react';
import { Sprout, Cpu, ShieldCheck } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="brand-icon" style={{ width: '30px', height: '30px' }}>
            <Sprout size={18} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>SMART CROP ADVISOR</div>
            <div className="footer-text">Precision Agriculture + Machine Learning Pipeline</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
          <span className="badge badge-success">
            <ShieldCheck size={14} /> GA-Optimized RF (Acc: 97.27%)
          </span>
          <span className="badge badge-info">
            <Cpu size={14} /> FastAPI + React Engine
          </span>
        </div>

        <div className="footer-text">
          © 2026 Smart Crop Advisor. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
