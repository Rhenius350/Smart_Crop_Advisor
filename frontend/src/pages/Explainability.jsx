import React, { useEffect, useState } from 'react';
import { HelpCircle, BarChart2, CheckCircle2, AlertCircle, RefreshCw, Zap } from 'lucide-react';
import { getFeatureImportance, explainPrediction } from '../services/api';

const Explainability = () => {
  const [globalImportances, setGlobalImportances] = useState([]);
  const [loadingGlobal, setLoadingGlobal] = useState(true);
  const [globalError, setGlobalError] = useState(null);

  // Sandbox state for local explanation testing
  const [sandboxData, setSandboxData] = useState({
    N: '90',
    P: '42',
    K: '43',
    temperature: '20.87',
    humidity: '82.00',
    ph: '6.50',
  });

  const [localExplanation, setLocalExplanation] = useState(null);
  const [loadingLocal, setLoadingLocal] = useState(false);
  const [localError, setLocalError] = useState(null);

  useEffect(() => {
    const fetchGlobalImportance = async () => {
      setLoadingGlobal(true);
      try {
        const data = await getFeatureImportance();
        setGlobalImportances(data);
      } catch (err) {
        setGlobalError(err.message || 'Failed to fetch global SHAP feature importances.');
      } finally {
        setLoadingGlobal(false);
      }
    };

    fetchGlobalImportance();
    handleRunLocalExplain(); // Auto run initial sandbox sample
  }, []);

  const handleSandboxChange = (e) => {
    const { name, value } = e.target;
    setSandboxData((prev) => ({ ...prev, [name]: value }));
  };

  const handleRunLocalExplain = async (e) => {
    if (e) e.preventDefault();

    const payload = {
      N: parseFloat(sandboxData.N) || 0,
      P: parseFloat(sandboxData.P) || 0,
      K: parseFloat(sandboxData.K) || 0,
      temperature: parseFloat(sandboxData.temperature) || 0,
      humidity: parseFloat(sandboxData.humidity) || 0,
      ph: parseFloat(sandboxData.ph) || 0,
    };

    setLoadingLocal(true);
    setLocalError(null);
    try {
      const data = await explainPrediction(payload);
      setLocalExplanation(data);
    } catch (err) {
      setLocalError(err.message || 'Failed to calculate local SHAP explanation.');
    } finally {
      setLoadingLocal(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      {/* Header */}
      <div className="section-header">
        <h1 className="section-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
          <HelpCircle color="var(--accent-cyan)" size={32} />
          <span>Explainable AI (SHAP Framework)</span>
        </h1>
        <p className="section-subtitle">
          SHAP (SHapley Additive exPlanations) attributes feature importance values to quantify the contribution of each agricultural measurement to the Random Forest's final crop prediction.
        </p>
      </div>

      {/* Global SHAP Feature Importance Card */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <BarChart2 className="title-icon" color="var(--accent-cyan)" size={22} />
            <span>Global Feature Importance (Module 9 SHAP Analysis)</span>
          </h2>
          <span className="badge badge-info">Mean Absolute SHAP Value</span>
        </div>

        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
          Ranks overall feature influence across the validation dataset. High values indicate features that drive model predictions most strongly across all crop classes.
        </p>

        {loadingGlobal ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <div className="spinner" style={{ margin: '0 auto 0.5rem auto' }}></div>
            <span style={{ color: 'var(--text-muted)' }}>Loading SHAP importances...</span>
          </div>
        ) : globalError ? (
          <div style={{ color: 'var(--accent-red)', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius-sm)' }}>
            {globalError}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {globalImportances.map((item, idx) => {
              const maxVal = globalImportances[0]?.shap_importance || 0.04;
              const percent = Math.min(((item.shap_importance || item.rf_importance) / maxVal) * 100, 100);

              return (
                <div key={idx} style={{ background: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span style={{ width: '24px', height: '24px', background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.8rem' }}>
                        {item.rank}
                      </span>
                      <span style={{ fontWeight: 700, fontSize: '1rem' }}>{item.feature}</span>
                    </div>

                    <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.85rem' }}>
                      <span>SHAP Mean |Value|: <strong style={{ color: 'var(--accent-cyan)' }}>{item.shap_importance ?? 'N/A'}</strong></span>
                      <span>RF Impurity: <strong style={{ color: 'var(--text-sub)' }}>{item.rf_importance}</strong></span>
                    </div>
                  </div>

                  <div className="progress-container" style={{ height: '10px' }}>
                    <div className="progress-bar" style={{ width: `${percent}%`, background: 'linear-gradient(90deg, #06b6d4, #10b981)' }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Interactive Local Explanation Sandbox */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <Zap className="title-icon" color="var(--accent-gold)" size={22} />
            <span>Interactive Local Prediction Explanation Sandbox</span>
          </h2>
          <span className="badge badge-warning">Instance SHAP TreeExplainer</span>
        </div>

        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
          Adjust the parameters below to compute real-time SHAP feature contribution breakdowns for any specific agricultural condition.
        </p>

        <form onSubmit={handleRunLocalExplain} style={{ marginBottom: '2rem' }}>
          <div className="grid-3" style={{ marginBottom: '1.2rem' }}>
            <div className="form-group">
              <label className="form-label">Nitrogen (N)</label>
              <input type="number" step="any" name="N" className="form-input" value={sandboxData.N} onChange={handleSandboxChange} required />
            </div>
            <div className="form-group">
              <label className="form-label">Phosphorus (P)</label>
              <input type="number" step="any" name="P" className="form-input" value={sandboxData.P} onChange={handleSandboxChange} required />
            </div>
            <div className="form-group">
              <label className="form-label">Potassium (K)</label>
              <input type="number" step="any" name="K" className="form-input" value={sandboxData.K} onChange={handleSandboxChange} required />
            </div>
            <div className="form-group">
              <label className="form-label">Temperature (°C)</label>
              <input type="number" step="any" name="temperature" className="form-input" value={sandboxData.temperature} onChange={handleSandboxChange} required />
            </div>
            <div className="form-group">
              <label className="form-label">Humidity (%)</label>
              <input type="number" step="any" name="humidity" className="form-input" value={sandboxData.humidity} onChange={handleSandboxChange} required />
            </div>
            <div className="form-group">
              <label className="form-label">Soil pH</label>
              <input type="number" step="any" name="ph" className="form-input" value={sandboxData.ph} onChange={handleSandboxChange} required />
            </div>
          </div>

          <button type="submit" className="btn btn-primary" disabled={loadingLocal}>
            {loadingLocal ? (
              <>
                <div className="spinner"></div>
                <span>Calculating SHAP Values...</span>
              </>
            ) : (
              <>
                <HelpCircle size={18} />
                <span>Calculate Local SHAP Explanation</span>
              </>
            )}
          </button>
        </form>

        {localError && (
          <div style={{ color: 'var(--accent-red)', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius-sm)', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={18} />
            <span>{localError}</span>
          </div>
        )}

        {/* Render Sandbox Local Explanation Result */}
        {localExplanation && (
          <div style={{ paddingTop: '1.5rem', borderTop: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Predicted Crop Candidate:</span>
                <h3 style={{ fontSize: '1.8rem', color: 'var(--primary)', textTransform: 'uppercase' }}>{localExplanation.predicted_crop}</h3>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span className="badge badge-success" style={{ fontSize: '0.9rem', padding: '0.4rem 0.8rem' }}>
                  Confidence: {localExplanation.confidence}%
                </span>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                  Expected Base Model Value: {localExplanation.base_value}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              {localExplanation.feature_contributions.map((feat, idx) => (
                <div key={idx} style={{ background: 'var(--bg-input)', padding: '0.8rem 1.2rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span style={{ fontWeight: 700 }}>{feat.feature}</span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>(Value: {feat.feature_value})</span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span className={`badge ${feat.direction === 'supports' ? 'badge-success' : 'badge-warning'}`}>
                        {feat.direction === 'supports' ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                        {feat.direction === 'supports' ? 'Supports Prediction' : 'Contradicts Prediction'}
                      </span>
                      <span style={{ fontWeight: 700, color: feat.direction === 'supports' ? 'var(--primary)' : 'var(--accent-gold)' }}>
                        {feat.shap_value > 0 ? `+${feat.shap_value}` : feat.shap_value}
                      </span>
                    </div>
                  </div>

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

export default Explainability;
