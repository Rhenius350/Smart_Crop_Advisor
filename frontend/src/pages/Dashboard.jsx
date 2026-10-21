import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sprout, TestTube, Award, ShieldCheck, Zap, ArrowRight, Layers, HelpCircle, BarChart3 } from 'lucide-react';
import { getHealth, getModelInfo, getPerformance } from '../services/api';

const Dashboard = () => {
  const navigate = useNavigate();
  const [modelInfo, setModelInfo] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [serverOnline, setServerOnline] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        await getHealth();
        setServerOnline(true);

        const info = await getModelInfo();
        setModelInfo(info);

        const perf = await getPerformance();
        setPerformance(perf);
      } catch (err) {
        console.warn('Backend server status check:', err.message);
        setServerOnline(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      {/* Hero Section */}
      <div className="card" style={{ background: 'linear-gradient(135deg, #0f1d2e 0%, #1e293b 100%)', borderColor: 'var(--primary)', padding: '3rem 2.5rem', position: 'relative', overflow: 'hidden' }}>
        <div style={{ maxWidth: '720px', zIndex: 2, position: 'relative' }}>
          <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center', marginBottom: '1rem' }}>
            <span className="badge badge-success">
              <ShieldCheck size={14} /> Production Model Active
            </span>
            <span className="badge badge-info">
              <Zap size={14} /> GA-Optimized Random Forest
            </span>
          </div>

          <h1 style={{ fontSize: '2.8rem', lineHeight: '1.2', marginBottom: '1.2rem', background: 'linear-gradient(90deg, #ffffff, #cbd5e1)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Find the Right Crop for Your Soil & Climate
          </h1>

          <p style={{ fontSize: '1.1rem', color: 'var(--text-sub)', marginBottom: '2rem', lineHeight: '1.6' }}>
            Enter your soil nutrients (N, P, K) and environmental conditions (temperature, humidity, pH) to receive precise, instant crop recommendations powered by Genetic Algorithm hyperparameter optimization and SHAP explainable AI.
          </p>

          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <button className="btn btn-primary" onClick={() => navigate('/predict')}>
              <TestTube size={20} />
              <span>Start Crop Prediction</span>
              <ArrowRight size={18} />
            </button>

            <button className="btn btn-secondary" onClick={() => navigate('/performance')}>
              <BarChart3 size={20} />
              <span>View Model Metrics</span>
            </button>
          </div>
        </div>

        {/* Hero Background Plant Icon Visual */}
        <div style={{ position: 'absolute', right: '-20px', bottom: '-40px', opacity: 0.06, pointerEvents: 'none' }}>
          <Sprout size={360} />
        </div>
      </div>

      {/* Quick Stats Banner */}
      <div className="grid-4">
        <div className="card card-hover" style={{ textAlign: 'center', padding: '1.5rem 1rem' }}>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--primary)', fontFamily: 'var(--font-heading)' }}>
            97.27%
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-sub)', fontWeight: 600 }}>Validation Accuracy</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>+0.45% over Baseline</div>
        </div>

        <div className="card card-hover" style={{ textAlign: 'center', padding: '1.5rem 1rem' }}>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--accent-cyan)', fontFamily: 'var(--font-heading)' }}>
            22
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-sub)', fontWeight: 600 }}>Supported Crop Classes</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Cereals, Pulses, Fruits, Cash Crops</div>
        </div>

        <div className="card card-hover" style={{ textAlign: 'center', padding: '1.5rem 1rem' }}>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--accent-gold)', fontFamily: 'var(--font-heading)' }}>
            6
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-sub)', fontWeight: 600 }}>Canonical Parameters</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>N, P, K, Temp, Humidity, pH</div>
        </div>

        <div className="card card-hover" style={{ textAlign: 'center', padding: '1.5rem 1rem' }}>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--primary)', fontFamily: 'var(--font-heading)' }}>
            265
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-sub)', fontWeight: 600 }}>Optimized Decision Trees</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Max Depth = 29</div>
        </div>
      </div>

      {/* Feature Cards Grid */}
      <div>
        <div style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.6rem' }}>Core System Capabilities</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>Full-stack machine learning solution for precision agriculture advice.</p>
        </div>

        <div className="grid-3">
          <div className="card card-hover" onClick={() => navigate('/predict')} style={{ cursor: 'pointer' }}>
            <div className="card-header">
              <div className="brand-icon" style={{ width: '42px', height: '42px' }}>
                <TestTube size={22} />
              </div>
              <ArrowRight size={20} color="var(--text-muted)" />
            </div>
            <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>AI Crop Prediction</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
              Input soil N-P-K nutrient values and weather metrics to generate instant top-3 crop recommendations with confidence probabilities.
            </p>
          </div>

          <div className="card card-hover" onClick={() => navigate('/explainability')} style={{ cursor: 'pointer' }}>
            <div className="card-header">
              <div className="brand-icon" style={{ width: '42px', height: '42px', background: 'linear-gradient(135deg, #06b6d4, #0284c7)' }}>
                <HelpCircle size={22} />
              </div>
              <ArrowRight size={20} color="var(--text-muted)" />
            </div>
            <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>SHAP Explainable AI</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
              Inspect feature-level contributions (SHAP TreeExplainer) to understand exactly why the Random Forest chose a specific crop.
            </p>
          </div>

          <div className="card card-hover" onClick={() => navigate('/performance')} style={{ cursor: 'pointer' }}>
            <div className="card-header">
              <div className="brand-icon" style={{ width: '42px', height: '42px', background: 'linear-gradient(135deg, #f59e0b, #d97706)' }}>
                <BarChart3 size={22} />
              </div>
              <ArrowRight size={20} color="var(--text-muted)" />
            </div>
            <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>GA vs Baseline Evaluation</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
              Explore formal validation benchmarks comparing the unoptimized Baseline RF (96.82%) against the GA-Optimized RF (97.27%).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
