import React, { useState } from 'react';
import { TestTube, Droplets, Thermometer, Compass, AlertCircle, RefreshCw, Sparkles, CheckCircle2 } from 'lucide-react';
import { predictCrop } from '../services/api';
import ResultCard from '../components/ResultCard';

const PRESETS = [
  {
    name: '🍊 Orange Sample',
    data: { N: 13, P: 23, K: 6, temperature: 23.96, humidity: 90.26, ph: 7.37 },
  },
  {
    name: '🌾 Rice Sample',
    data: { N: 90, P: 42, K: 43, temperature: 20.87, humidity: 82.00, ph: 6.50 },
  },
  {
    name: '🍎 Apple Sample',
    data: { N: 20, P: 125, K: 200, temperature: 22.50, humidity: 92.00, ph: 5.90 },
  },
  {
    name: '☕ Coffee Sample',
    data: { N: 100, P: 28, K: 30, temperature: 25.40, humidity: 62.00, ph: 6.80 },
  },
];

const Prediction = () => {
  const [formData, setFormData] = useState({
    N: '13',
    P: '23',
    K: '6',
    temperature: '23.96',
    humidity: '90.26',
    ph: '7.37',
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleLoadPreset = (presetData) => {
    setFormData({
      N: String(presetData.N),
      P: String(presetData.P),
      K: String(presetData.K),
      temperature: String(presetData.temperature),
      humidity: String(presetData.humidity),
      ph: String(presetData.ph),
    });
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    // Form input validation
    const numN = parseFloat(formData.N);
    const numP = parseFloat(formData.P);
    const numK = parseFloat(formData.K);
    const numTemp = parseFloat(formData.temperature);
    const numHum = parseFloat(formData.humidity);
    const numPh = parseFloat(formData.ph);

    if ([numN, numP, numK, numTemp, numHum, numPh].some((v) => isNaN(v))) {
      setError('Please fill out all six parameter fields with valid numerical values.');
      return;
    }

    if (numN < 0 || numP < 0 || numK < 0) {
      setError('Soil nutrients (N, P, K) cannot be negative.');
      return;
    }

    if (numPh <= 0 || numPh > 14) {
      setError('Soil pH must be between 0.1 and 14.0.');
      return;
    }

    if (numHum < 0 || numHum > 100) {
      setError('Humidity percentage must be between 0% and 100%.');
      return;
    }

    const payload = {
      N: numN,
      P: numP,
      K: numK,
      temperature: numTemp,
      humidity: numHum,
      ph: numPh,
    };

    setLoading(true);
    try {
      const data = await predictCrop(payload);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to generate crop prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div className="section-header">
        <h1 className="section-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
          <TestTube color="var(--primary)" size={32} />
          <span>Crop Recommendation Engine</span>
        </h1>
        <p className="section-subtitle">
          Input your farm's soil nutrient analysis and current microclimate measurements to predict the most suitable crop using our GA-Optimized Random Forest.
        </p>
      </div>

      {/* Preset Quick Loader */}
      <div className="card" style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '1rem 1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--text-sub)', fontWeight: 600 }}>
            <Sparkles size={16} color="var(--accent-gold)" />
            <span>Load Quick Test Samples:</span>
          </div>

          <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
            {PRESETS.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => handleLoadPreset(preset.data)}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Grid: Left Form, Right Result */}
      <div className="grid-2" style={{ alignItems: 'flex-start' }}>
        {/* Prediction Form */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title" style={{ fontSize: '1.25rem' }}>
              <Compass className="title-icon" size={22} />
              <span>Input Parameters</span>
            </h2>
            <span className="badge badge-info">6 Canonical Features</span>
          </div>

          <form onSubmit={handleSubmit}>
            {/* Section 1: Soil Parameters */}
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--primary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Droplets size={16} /> Soil Nutrients Analysis
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="N">
                  <span>Nitrogen (N)</span>
                  <span className="unit-badge">mg/kg</span>
                </label>
                <input
                  type="number"
                  step="any"
                  id="N"
                  name="N"
                  className="form-input"
                  value={formData.N}
                  onChange={handleChange}
                  placeholder="e.g. 90"
                  required
                />
                <span className="form-hint">Ratio of nitrogen content in soil (0 - 140 mg/kg)</span>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="P">
                  <span>Phosphorus (P)</span>
                  <span className="unit-badge">mg/kg</span>
                </label>
                <input
                  type="number"
                  step="any"
                  id="P"
                  name="P"
                  className="form-input"
                  value={formData.P}
                  onChange={handleChange}
                  placeholder="e.g. 42"
                  required
                />
                <span className="form-hint">Ratio of phosphorus content in soil (5 - 145 mg/kg)</span>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="K">
                  <span>Potassium (K)</span>
                  <span className="unit-badge">mg/kg</span>
                </label>
                <input
                  type="number"
                  step="any"
                  id="K"
                  name="K"
                  className="form-input"
                  value={formData.K}
                  onChange={handleChange}
                  placeholder="e.g. 43"
                  required
                />
                <span className="form-hint">Ratio of potassium content in soil (5 - 205 mg/kg)</span>
              </div>
            </div>

            {/* Section 2: Environmental Parameters */}
            <div style={{ marginBottom: '1.5rem', paddingTop: '1.2rem', borderTop: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Thermometer size={16} /> Microclimate Parameters
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="temperature">
                  <span>Air Temperature</span>
                  <span className="unit-badge">°C</span>
                </label>
                <input
                  type="number"
                  step="any"
                  id="temperature"
                  name="temperature"
                  className="form-input"
                  value={formData.temperature}
                  onChange={handleChange}
                  placeholder="e.g. 20.87"
                  required
                />
                <span className="form-hint">Average air temperature in degrees Celsius (0 - 50 °C)</span>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="humidity">
                  <span>Relative Humidity</span>
                  <span className="unit-badge">%</span>
                </label>
                <input
                  type="number"
                  step="any"
                  id="humidity"
                  name="humidity"
                  className="form-input"
                  value={formData.humidity}
                  onChange={handleChange}
                  placeholder="e.g. 82.0"
                  required
                />
                <span className="form-hint">Relative humidity percentage (10 - 100 %)</span>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="ph">
                  <span>Soil pH Level</span>
                  <span className="unit-badge">pH Scale</span>
                </label>
                <input
                  type="number"
                  step="any"
                  id="ph"
                  name="ph"
                  className="form-input"
                  value={formData.ph}
                  onChange={handleChange}
                  placeholder="e.g. 6.5"
                  required
                />
                <span className="form-hint">pH value of the soil (3.5 - 9.5)</span>
              </div>
            </div>

            {error && (
              <div style={{ marginBottom: '1.25rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--accent-red)', padding: '0.8rem 1rem', borderRadius: 'var(--radius-md)', color: 'var(--accent-red)', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}

            <button type="submit" className="btn btn-primary" style={{ width: '100%', padding: '1rem' }} disabled={loading}>
              {loading ? (
                <>
                  <div className="spinner"></div>
                  <span>Analyzing Soil & Climate Conditions...</span>
                </>
              ) : (
                <>
                  <TestTube size={20} />
                  <span>Predict Suitable Crop</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Prediction Output Column */}
        <div>
          {result ? (
            <ResultCard
              predictionResult={result}
              inputData={{
                N: parseFloat(formData.N),
                P: parseFloat(formData.P),
                K: parseFloat(formData.K),
                temperature: parseFloat(formData.temperature),
                humidity: parseFloat(formData.humidity),
                ph: parseFloat(formData.ph),
              }}
            />
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem', color: 'var(--text-muted)' }}>
              <div className="brand-icon" style={{ width: '64px', height: '64px', margin: '0 auto 1.25rem auto', background: 'rgba(16, 185, 129, 0.1)', color: 'var(--primary)' }}>
                <TestTube size={32} />
              </div>
              <h3 style={{ color: 'var(--text-main)', marginBottom: '0.5rem' }}>Awaiting Prediction Input</h3>
              <p style={{ fontSize: '0.95rem', maxWidth: '380px', margin: '0 auto' }}>
                Enter parameters on the left and click "Predict Suitable Crop" to generate your recommendation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Prediction;
