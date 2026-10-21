import React from 'react';
import { Info, Sprout, Cpu, ShieldCheck, Layers, GitMerge, CheckCircle2 } from 'lucide-react';

const About = () => {
  const modules = [
    { num: 'Module 1', title: 'Dataset Management', desc: 'Raw CSV structure, schema validation, and missing value checks without data alteration.' },
    { num: 'Module 2', title: 'Data Preprocessing', desc: 'Duplicate detection, invalid value handling, and clean dataset generation (2200 rows x 6 features).' },
    { num: 'Module 3', title: 'Feature Analysis', desc: 'Exploratory statistical distributions, feature correlations, and mutual information non-linear dependency scores.' },
    { num: 'Module 4', title: 'Train / Validation Split', desc: 'Stratified 80/20 train/validation split (1760 train samples, 440 validation samples, random_state=42).' },
    { num: 'Module 5', title: 'Baseline Random Forest', desc: 'Reference default Random Forest model (100 estimators) achieving 96.82% validation accuracy.' },
    { num: 'Module 6', title: 'Genetic Algorithm Optimization', desc: 'Population of 20 chromosomes evolved over 10 generations with 80% crossover and 20% mutation rates.' },
    { num: 'Module 7', title: 'GA-Optimized Random Forest', desc: 'Trained champion Random Forest (265 estimators, max_depth=29) achieving 97.27% validation accuracy.' },
    { num: 'Module 8', title: 'Performance Evaluation', desc: 'Side-by-side evaluation, per-class F1, confusion matrices, error analysis, and 100% reproducibility check.' },
    { num: 'Module 9', title: 'Explainable AI (LIME + SHAP)', desc: 'SHAP TreeExplainer global/local explanations and LIME tabular instance contribution breakdowns.' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      {/* Header */}
      <div className="section-header">
        <h1 className="section-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
          <Info color="var(--primary)" size={32} />
          <span>About Smart Crop Advisor</span>
        </h1>
        <p className="section-subtitle">
          An end-to-end Machine Learning and Precision Agriculture web application combining Genetic Algorithm hyperparameter optimization with Random Forest classification and SHAP explainability.
        </p>
      </div>

      {/* Domain & Methodology Banner */}
      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <Sprout className="title-icon" size={22} />
              <span>Project Overview</span>
            </h2>
            <span className="badge badge-success">Precision Agriculture</span>
          </div>
          <p style={{ color: 'var(--text-sub)', fontSize: '0.95rem', lineHeight: '1.6', marginBottom: '1rem' }}>
            <strong>Smart Crop Advisor</strong> aims to assist farmers, agricultural extension officers, and agronomists in making data-driven crop selection decisions. By analyzing key soil nutrient levels and microclimate conditions, the system predicts the most suitable crop out of 22 supported agricultural species.
          </p>
          <div style={{ background: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', fontSize: '0.88rem' }}>
            <div><strong>Domain:</strong> Machine Learning + Precision Agriculture</div>
            <div><strong>Core Algorithm:</strong> Genetic Algorithm (GA) + Random Forest (RF)</div>
            <div><strong>Frontend Stack:</strong> React + Vite + Vanilla CSS (JavaScript)</div>
            <div><strong>Backend Stack:</strong> Python + FastAPI + Uvicorn + scikit-learn + SHAP</div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <Cpu className="title-icon" color="var(--accent-cyan)" size={22} />
              <span>6 Canonical Input Parameters</span>
            </h2>
            <span className="badge badge-info">Unscaled Metric Units</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.9rem' }}>
            <div style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <strong>N (Nitrogen):</strong> Soil nitrogen content in mg/kg (0 to 140 mg/kg)
            </div>
            <div style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <strong>P (Phosphorus):</strong> Soil phosphorus content in mg/kg (5 to 145 mg/kg)
            </div>
            <div style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <strong>K (Potassium):</strong> Soil potassium content in mg/kg (5 to 205 mg/kg)
            </div>
            <div style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <strong>Temperature:</strong> Air temperature in degrees Celsius (0 to 50 °C)
            </div>
            <div style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <strong>Humidity:</strong> Relative atmospheric humidity percentage (10 to 100 %)
            </div>
            <div style={{ padding: '0.4rem 0' }}>
              <strong>Soil pH:</strong> Soil acidity/alkalinity scale value (3.5 to 9.5)
            </div>
          </div>
        </div>
      </div>

      {/* 9-Module Pipeline Architecture */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <Layers className="title-icon" color="var(--accent-gold)" size={22} />
            <span>9-Module Research & Pipeline Architecture</span>
          </h2>
          <span className="badge badge-warning">All Modules Executed (PASS)</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
          {modules.map((mod, idx) => (
            <div key={idx} style={{ background: 'var(--bg-input)', padding: '1rem 1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <span style={{ fontWeight: 800, color: 'var(--primary)', fontSize: '0.9rem', width: '85px' }}>{mod.num}</span>
                <div>
                  <h4 style={{ fontSize: '1.05rem', color: 'var(--text-main)' }}>{mod.title}</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>{mod.desc}</p>
                </div>
              </div>

              <span className="badge badge-success" style={{ fontSize: '0.75rem' }}>
                <CheckCircle2 size={14} /> PASS
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default About;
