import React, { useState } from 'react';
import { Award, HelpCircle, CheckCircle2, AlertCircle, ChevronRight, BarChart2, ShieldCheck } from 'lucide-react';
import { explainPrediction } from '../services/api';

const CROP_EMOJIS = {
  rice: '🌾',
  maize: '🌽',
  chickpea: '🧆',
  kidneybeans: '🫘',
  pigeonpeas: '🫘',
  mothbeans: '🫘',
  mungbean: '🫘',
  blackgram: '🫘',
  lentil: '🥣',
  pomegranate: '🍎',
  banana: '🍌',
  mango: '🥭',
  grapes: '🍇',
  watermelon: '🍉',
  muskmelon: '🍈',
  apple: '🍎',
  orange: '🍊',
  papaya: '🍈',
  coconut: '🥥',
  cotton: '☁️',
  jute: '🌾',
  coffee: '☕',
};

const ResultCard = ({ predictionResult, inputData }) => {
  const [explanation, setExplanation] = useState(null);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [explainError, setExplainError] = useState(null);

  if (!predictionResult) return null;

  const { predicted_crop, confidence, model, top_predictions, input_summary } = predictionResult;
  const cropLower = predicted_crop.toLowerCase();
  const emoji = CROP_EMOJIS[cropLower] || '🌱';

  const handleFetchExplanation = async () => {
    setLoadingExplain(true);
    setExplainError(null);
    try {
      const data = await explainPrediction(inputData);
      setExplanation(data);
    } catch (err) {
      setExplainError(err.message || 'Failed to load SHAP explanation.');
    } finally {
      setLoadingExplain(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Recommended Crop Hero Card */}
      <div className="card" style={{ background: 'linear-gradient(135deg, #1e293b, #0f1d2e)', borderColor: 'var(--primary)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Recommended Crop
            </div>
            <h2 style={{ fontSize: '2.4rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.2rem' }}>
              <span>{emoji}</span>
              <span style={{ textTransform: 'uppercase', letterSpacing: '0.02em' }}>{predicted_crop}</span>
            </h2>
          </div>
          <span className="badge badge-success" style={{ fontSize: '0.85rem', padding: '0.4rem 0.8rem' }}>
            <Award size={16} /> Optimal Selection
          </span>
        </div>

        {/* Confidence Gauge */}
        <div style={{ marginTop: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-sub)', fontWeight: 600 }}>Prediction Confidence</span>
            <span style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--primary)' }}>{confidence}%</span>
          </div>
          <div className="progress-container" style={{ height: '12px' }}>
            <div className="progress-bar" style={{ width: `${Math.min(confidence, 100)}%` }}></div>
          </div>
        </div>

        <div style={{ marginTop: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          <ShieldCheck size={16} color="var(--primary)" />
          <span>Model Engine: <strong>{model}</strong></span>
        </div>
      </div>

      {/* Top 3 Predictions Breakdown */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <BarChart2 className="title-icon" size={20} />
            <span>Top 3 Candidate Crops</span>
          </h3>
          <span className="badge badge-info">Probability Breakdown</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {top_predictions.map((item, idx) => (
            <div key={idx} style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.8rem 1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '1.2rem' }}>{CROP_EMOJIS[item.crop.toLowerCase()] || '🌱'}</span>
                  <span style={{ fontWeight: 700, textTransform: 'capitalize', fontSize: '1rem' }}>{item.crop}</span>
                  {idx === 0 && <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>Top Pick</span>}
                </div>
                <span style={{ fontWeight: 700, color: idx === 0 ? 'var(--primary)' : 'var(--text-sub)' }}>{item.probability}%</span>
              </div>
              <div className="progress-container">
                <div 
                  className="progress-bar" 
                  style={{ 
                    width: `${Math.min(item.probability, 100)}%`,
                    background: idx === 0 ? 'linear-gradient(90deg, #10b981, #06b6d4)' : 'linear-gradient(90deg, #3b82f6, #6366f1)' 
                  }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Input Summary Verification Table */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Input Parameters Summary</h3>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Verified 6 Canonical Features</span>
        </div>

        <div className="grid-3" style={{ gap: '0.75rem' }}>
          <div style={{ background: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Nitrogen (N)</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>{input_summary.N} <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>mg/kg</span></div>
          </div>
          <div style={{ background: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Phosphorus (P)</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>{input_summary.P} <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>mg/kg</span></div>
          </div>
          <div style={{ background: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Potassium (K)</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>{input_summary.K} <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>mg/kg</span></div>
          </div>
          <div style={{ background: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Temperature</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>{input_summary.temperature} <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>°C</span></div>
          </div>
          <div style={{ background: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Humidity</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>{input_summary.humidity} <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>%</span></div>
          </div>
          <div style={{ background: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Soil pH</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>{input_summary.ph}</div>
          </div>
        </div>
      </div>

      {/* SHAP Local Explainability Trigger */}
      <div className="card" style={{ borderColor: explanation ? 'var(--accent-cyan)' : 'var(--border-color)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h3 className="card-title" style={{ fontSize: '1.1rem' }}>
              <HelpCircle className="title-icon" color="var(--accent-cyan)" size={20} />
              <span>Why this crop? (SHAP Local Explanation)</span>
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Analyze feature contributions that pushed the Random Forest model toward recommending <strong>{predicted_crop}</strong>.
            </p>
          </div>

          <button className="btn btn-secondary" onClick={handleFetchExplanation} disabled={loadingExplain}>
            {loadingExplain ? (
              <>
                <div className="spinner"></div>
                <span>Computing SHAP...</span>
              </>
            ) : (
              <>
                <span>{explanation ? 'Refresh Explanation' : 'Explain Prediction'}</span>
                <ChevronRight size={18} />
              </>
            )}
          </button>
        </div>

        {explainError && (
          <div style={{ marginTop: '1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--accent-red)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', color: 'var(--accent-red)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={16} />
            <span>{explainError}</span>
          </div>
        )}

        {/* Displayed Local SHAP Feature Contributions */}
        {explanation && (
          <div style={{ marginTop: '1.5rem', paddingTop: '1.2rem', borderTop: '1px solid var(--border-color)' }}>
            <div style={{ marginBottom: '1rem', fontSize: '0.9rem', color: 'var(--text-sub)' }}>
              SHAP feature contribution breakdown for <strong>{explanation.predicted_crop}</strong> (Base Model Value: {explanation.base_value}):
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              {explanation.feature_contributions.map((feat, idx) => (
                <div key={idx} style={{ background: 'var(--bg-input)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>{feat.feature}</span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>(Val: {feat.feature_value})</span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className={`badge ${feat.direction === 'supports' ? 'badge-success' : 'badge-warning'}`}>
                        {feat.direction === 'supports' ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                        {feat.direction === 'supports' ? 'Supports' : 'Contradicts'}
                      </span>
                      <span style={{ fontWeight: 700, fontSize: '0.9rem', color: feat.direction === 'supports' ? 'var(--primary)' : 'var(--accent-gold)' }}>
                        {feat.shap_value > 0 ? `+${feat.shap_value}` : feat.shap_value}
                      </span>
                    </div>
                  </div>

                  {/* Horizontal Bar Visual */}
                  <div className="progress-container" style={{ height: '8px' }}>
                    <div
                      className="progress-bar"
                      style={{
                        width: `${Math.min(feat.absolute_impact * 200, 100)}%`,
                        background: feat.direction === 'supports' ? 'var(--primary)' : 'var(--accent-gold)',
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultCard;
