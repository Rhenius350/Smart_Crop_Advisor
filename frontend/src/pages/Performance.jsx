import React, { useEffect, useState } from 'react';
import { BarChart3, Award, ShieldCheck, Zap, Layers, RefreshCw, AlertCircle, ArrowUpRight } from 'lucide-react';
import { getPerformance, getModelInfo } from '../services/api';

const Performance = () => {
  const [perfData, setPerfData] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPerformance = async () => {
      setLoading(true);
      try {
        const perf = await getPerformance();
        setPerfData(perf);
        const info = await getModelInfo();
        setModelInfo(info);
      } catch (err) {
        setError(err.message || 'Failed to load performance metrics.');
      } finally {
        setLoading(false);
      }
    };

    fetchPerformance();
  }, []);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '4rem 2rem' }}>
        <div className="spinner" style={{ margin: '0 auto 1rem auto', width: '32px', height: '32px' }}></div>
        <p style={{ color: 'var(--text-sub)' }}>Loading Module 8 Model Evaluation Metrics...</p>
      </div>
    );
  }

  const baseline = perfData?.baseline || {
    accuracy: 96.82,
    weighted_precision: 96.91,
    weighted_recall: 96.82,
    weighted_f1: 96.78,
    macro_f1: 96.78,
    train_val_gap: 3.18,
  };

  const ga = perfData?.ga_optimized || {
    accuracy: 97.27,
    weighted_precision: 97.34,
    weighted_recall: 97.27,
    weighted_f1: 97.26,
    macro_f1: 97.26,
    train_val_gap: 1.93,
  };

  const improvements = perfData?.improvements || {
    accuracy_diff: '+0.45',
    weighted_f1_diff: '+0.47',
    weighted_precision_diff: '+0.43',
    macro_f1_diff: '+0.48',
    gap_reduction: '-1.25',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      {/* Header */}
      <div className="section-header">
        <h1 className="section-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
          <BarChart3 color="var(--primary)" size={32} />
          <span>Model Performance & Evaluation</span>
        </h1>
        <p className="section-subtitle">
          Formal evaluation results from Module 8 comparing the unoptimized Baseline Random Forest against the Genetic Algorithm (GA) Optimized Random Forest using identical validation data.
        </p>
      </div>

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--accent-red)', padding: '1rem', borderRadius: 'var(--radius-md)', color: 'var(--accent-red)', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Top 3 Metric Summary Cards */}
      <div className="grid-3">
        {/* Baseline RF Card */}
        <div className="card" style={{ borderTop: '4px solid var(--text-muted)' }}>
          <div className="card-header">
            <div>
              <span className="badge badge-info">Module 5 Model</span>
              <h3 style={{ fontSize: '1.25rem', marginTop: '0.4rem' }}>Baseline Random Forest</h3>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Validation Accuracy</div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-sub)' }}>{baseline.accuracy}%</div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.8rem', paddingTop: '0.8rem', borderTop: '1px solid var(--border-color)' }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Weighted F1</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>{baseline.weighted_f1}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Macro F1</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>{baseline.macro_f1}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Weighted Precision</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>{baseline.weighted_precision}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Train-Val Gap</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-gold)' }}>{baseline.train_val_gap}%</div>
              </div>
            </div>
          </div>
        </div>

        {/* GA-Optimized RF Card */}
        <div className="card" style={{ borderTop: '4px solid var(--primary)', background: 'linear-gradient(135deg, #1e293b, #0f1d2e)' }}>
          <div className="card-header">
            <div>
              <span className="badge badge-success">Module 7 Champion Model</span>
              <h3 style={{ fontSize: '1.25rem', marginTop: '0.4rem', color: 'var(--primary)' }}>GA-Optimized Random Forest</h3>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Validation Accuracy</div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--primary)' }}>{ga.accuracy}%</div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.8rem', paddingTop: '0.8rem', borderTop: '1px solid var(--border-color)' }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Weighted F1</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary)' }}>{ga.weighted_f1}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Macro F1</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary)' }}>{ga.macro_f1}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Weighted Precision</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary)' }}>{ga.weighted_precision}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Train-Val Gap</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary)' }}>{ga.train_val_gap}%</div>
              </div>
            </div>
          </div>
        </div>

        {/* Measured Improvement Card */}
        <div className="card" style={{ borderTop: '4px solid var(--accent-cyan)' }}>
          <div className="card-header">
            <div>
              <span className="badge badge-info">Optimization Delta</span>
              <h3 style={{ fontSize: '1.25rem', marginTop: '0.4rem', color: 'var(--accent-cyan)' }}>Measured Improvement</h3>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Accuracy Increase</div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <span>{improvements.accuracy_diff}</span>
                <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>percentage points</span>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.8rem', paddingTop: '0.8rem', borderTop: '1px solid var(--border-color)' }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Weighted F1 Delta</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>{improvements.weighted_f1_diff} pp</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Macro F1 Delta</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>{improvements.macro_f1_diff} pp</div>
              </div>
              <div style={{ gridColumn: 'span 2' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Generalization Gap Reduction</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary)' }}>{improvements.gap_reduction} percentage points</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Visual Metric Bar Chart Comparison */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <BarChart3 className="title-icon" size={20} />
            <span>Metric-by-Metric Visual Comparison</span>
          </h3>
          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.85rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)' }}>
              <span style={{ width: '12px', height: '12px', background: 'var(--text-muted)', borderRadius: '3px' }}></span> Baseline RF
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--primary)' }}>
              <span style={{ width: '12px', height: '12px', background: 'var(--primary)', borderRadius: '3px' }}></span> GA-Optimized RF
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1rem' }}>
          {[
            { label: 'Accuracy', b: baseline.accuracy, g: ga.accuracy },
            { label: 'Weighted Precision', b: baseline.weighted_precision, g: ga.weighted_precision },
            { label: 'Weighted Recall', b: baseline.weighted_recall, g: ga.weighted_recall },
            { label: 'Weighted F1 Score', b: baseline.weighted_f1, g: ga.weighted_f1 },
            { label: 'Macro F1 Score', b: baseline.macro_f1, g: ga.macro_f1 },
          ].map((item, idx) => (
            <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', fontWeight: 600 }}>
                <span>{item.label}</span>
                <span>Baseline: {item.b}% | GA-RF: <strong style={{ color: 'var(--primary)' }}>{item.g}%</strong></span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div className="progress-container" style={{ height: '8px' }}>
                  <div className="progress-bar" style={{ width: `${item.b}%`, background: 'var(--text-muted)' }}></div>
                </div>
                <div className="progress-container" style={{ height: '8px' }}>
                  <div className="progress-bar" style={{ width: `${item.g}%`, background: 'var(--primary)' }}></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* GA Hyperparameter & Search Space Details */}
      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <Zap className="title-icon" color="var(--accent-gold)" size={20} />
              <span>Genetic Algorithm Parameters</span>
            </h3>
            <span className="badge badge-warning">Module 6 Hyperparameters</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Population Size</span>
              <strong style={{ color: 'var(--text-main)' }}>20 Chromosomes</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Generations</span>
              <strong style={{ color: 'var(--text-main)' }}>10 Evolution Cycles</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Crossover Probability ($p_c$)</span>
              <strong style={{ color: 'var(--primary)' }}>80.0%</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Mutation Probability ($p_m$)</span>
              <strong style={{ color: 'var(--accent-gold)' }}>20.0%</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Tournament Size</span>
              <strong style={{ color: 'var(--text-main)' }}>3 Candidates</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0' }}>
              <span style={{ color: 'var(--text-muted)' }}>Elite Preservation Count</span>
              <strong style={{ color: 'var(--text-main)' }}>1 Top Chromosome</strong>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <Award className="title-icon" color="var(--primary)" size={20} />
              <span>Discovered Best Parameters</span>
            </h3>
            <span className="badge badge-success">Optimized RF Architecture</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Number of Estimators (n_estimators)</span>
              <strong style={{ color: 'var(--primary)' }}>{modelInfo?.n_estimators || 265}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Maximum Tree Depth (max_depth)</span>
              <strong style={{ color: 'var(--primary)' }}>{modelInfo?.max_depth || 29}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Min Samples Split (min_samples_split)</span>
              <strong style={{ color: 'var(--primary)' }}>{modelInfo?.min_samples_split || 3}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Min Samples Leaf (min_samples_leaf)</span>
              <strong style={{ color: 'var(--primary)' }}>{modelInfo?.min_samples_leaf || 4}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Max Feature Split Strategy</span>
              <strong style={{ color: 'var(--primary)' }}>{modelInfo?.max_features || 'sqrt'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0' }}>
              <span style={{ color: 'var(--text-muted)' }}>Splitting Criterion</span>
              <strong style={{ color: 'var(--text-main)' }}>Gini Impurity</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Performance;
