import { useState, useEffect, useCallback } from 'react';
import { loginAdmin, getAnalyticsDashboard } from '../services/api';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, RadialLinearScale, Filler, Tooltip, Legend, Title,
} from 'chart.js';
import { Bar, Line, Doughnut, Pie, Radar, Scatter } from 'react-chartjs-2';
import '../styles/AnalyticsDashboard.css';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement, PointElement,
  ArcElement, RadialLinearScale, Filler, Tooltip, Legend, Title,
);

const PALETTE = ['#0284C7','#8B5CF6','#10B981','#F59E0B','#EF4444','#EC4899','#06B6D4','#6366F1','#14B8A6','#F97316','#A855F7','#E11D48','#84CC16','#0EA5E9'];
const CONFIDENCE_COLORS = (bins) => bins.map(b => {
  const lo = parseFloat(b.bin);
  if (lo >= 0.8) return '#10B981';
  if (lo >= 0.5) return '#F59E0B';
  return '#EF4444';
});

const CHART_BASE = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { font: { family: 'Inter', size: 11 }, color: '#64748B', padding: 12 } },
    tooltip: { backgroundColor: '#0F172A', titleFont: { family: 'Inter' }, bodyFont: { family: 'Inter' }, cornerRadius: 6, padding: 10 },
  },
  scales: {
    x: { ticks: { font: { family: 'Inter', size: 10 }, color: '#94A3B8' }, grid: { color: 'rgba(0,0,0,0.04)' } },
    y: { ticks: { font: { family: 'Inter', size: 10 }, color: '#94A3B8' }, grid: { color: 'rgba(0,0,0,0.04)' } },
  },
};
const DOUGHNUT_BASE = { responsive: true, maintainAspectRatio: false, plugins: { ...CHART_BASE.plugins } };

const fmt = (v, d = 2) => v == null ? '—' : typeof v === 'number' ? (v >= 1000 ? v.toLocaleString('en-IN', { maximumFractionDigits: d }) : Number(v.toFixed(d)).toString()) : v;

const isAuthError = (err) => {
  const m = String(err?.message || '').toLowerCase();
  return m.includes('401') || m.includes('token') || m.includes('log in');
};

export default function AnalyticsDashboard() {
  const [token, setToken] = useState(localStorage.getItem('bbmp_token') || '');
  const [loggedIn, setLoggedIn] = useState(!!localStorage.getItem('bbmp_token'));
  const [loginError, setLoginError] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [langFilter, setLangFilter] = useState('');
  const [verifyOpen, setVerifyOpen] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    const form = e.target;
    setLoginError(''); setLoginLoading(true);
    try {
      const res = await loginAdmin(form.username.value, form.password.value);
      setToken(res.access_token);
      localStorage.setItem('bbmp_token', res.access_token);
      setLoggedIn(true);
    } catch (err) { setLoginError(err.message); }
    setLoginLoading(false);
  };

  const handleLogout = useCallback(() => {
    setToken(''); localStorage.removeItem('bbmp_token');
    setLoggedIn(false); setData(null);
  }, []);

  const fetchData = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const res = await getAnalyticsDashboard(token, { startDate, endDate, language: langFilter });
      setData(res);
    } catch (err) {
      if (isAuthError(err)) handleLogout();
      else setError(err.message);
    }
    setLoading(false);
  }, [token, startDate, endDate, langFilter, handleLogout]);

  useEffect(() => { if (loggedIn) fetchData(); }, [loggedIn, fetchData]);

  /* ── Login Screen ──────────────────────────────────────────── */
  if (!loggedIn) {
    return (
      <div className="login-wrapper gravless-container">
        <div className="login-card">
          <div className="login-header">
            <div className="login-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0284C7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"/>
                <path d="M3 5v14a2 2 0 0 0 2 2h16v-5"/>
                <path d="M18 12a2 2 0 1 0 0 4 2 2 0 0 0 0-4Z"/>
              </svg>
            </div>
            <h2>NLP Analytics</h2>
            <p className="login-subtext">Sign in to view system analytics</p>
          </div>
          <form onSubmit={handleLogin} className="login-form">
            <div className="form-group">
              <label htmlFor="a-username" className="form-label">Username</label>
              <input id="a-username" name="username" className="input" placeholder="Enter username" required autoComplete="username" />
            </div>
            <div className="form-group">
              <label htmlFor="a-password" className="form-label">Password</label>
              <input id="a-password" name="password" type="password" className="input" placeholder="Enter password" required autoComplete="current-password" />
            </div>
            {loginError && <div className="login-error">{loginError}</div>}
            <button type="submit" className="btn btn-primary btn-lg login-btn" disabled={loginLoading}>
              {loginLoading ? <><span className="spinner" /> Signing in…</> : 'Sign In'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  /* ── Loading ────────────────────────────────────────────────── */
  if (loading && !data) {
    return (
      <div className="analytics-loading gravless-container">
        <div className="analytics-spinner" />
        <p>Loading analytics…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="analytics-empty gravless-container">
        <div className="analytics-empty-icon">⚠️</div>
        <h3>Error loading analytics</h3>
        <p>{error}</p>
        <button className="btn btn-primary" onClick={fetchData}>Retry</button>
      </div>
    );
  }

  if (!data) return null;

  const cs = data.complaint_stats || {};
  const ns = data.nlp_stats || {};
  const es = data.energy_stats || {};
  const er = data.error_stats || {};
  const ch = data.charts || {};
  const ds = data.data_sources || {};

  /* ── Charts Config ─────────────────────────────────────────── */
  const energyByStageData = {
    labels: (ch.energy_by_stage || []).map(d => d.stage),
    datasets: [{ label: 'Energy (J)', data: (ch.energy_by_stage || []).map(d => d.joules), backgroundColor: PALETTE.slice(0, (ch.energy_by_stage || []).length), borderRadius: 6, borderSkipped: false }],
  };

  const energyOverTimeData = {
    labels: (ch.energy_over_time || []).map(d => d.date),
    datasets: [
      { label: 'Energy (J)', data: (ch.energy_over_time || []).map(d => d.joules), borderColor: '#F59E0B', backgroundColor: 'rgba(245,158,11,0.1)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6 },
      { label: 'Requests', data: (ch.energy_over_time || []).map(d => d.count), borderColor: '#8B5CF6', backgroundColor: 'rgba(139,92,246,0.1)', fill: false, tension: 0.4, yAxisID: 'y1', pointRadius: 3 },
    ],
  };

  const catDistData = {
    labels: (ch.category_distribution || []).map(d => d.category),
    datasets: [{ data: (ch.category_distribution || []).map(d => d.count), backgroundColor: PALETTE.slice(0, (ch.category_distribution || []).length), borderWidth: 2, borderColor: '#fff' }],
  };

  const dupVsUniqueData = {
    labels: ['Unique', 'Duplicate'],
    datasets: [{ data: [ch.duplicate_vs_unique?.unique || 0, ch.duplicate_vs_unique?.duplicate || 0], backgroundColor: ['#10B981', '#EF4444'], borderWidth: 2, borderColor: '#fff' }],
  };

  const radarData = {
    labels: ch.stage_bottleneck_radar?.labels || [],
    datasets: [{ label: 'Avg Time (s)', data: ch.stage_bottleneck_radar?.avg_times || [], backgroundColor: 'rgba(99,102,241,0.15)', borderColor: '#6366F1', pointBackgroundColor: '#6366F1', pointBorderColor: '#fff', pointHoverRadius: 6 }],
  };

  const throughputData = {
    labels: (ch.throughput_over_time || []).map(d => d.hour),
    datasets: [{ label: 'Requests', data: (ch.throughput_over_time || []).map(d => d.count), borderColor: '#06B6D4', backgroundColor: 'rgba(6,182,212,0.1)', fill: true, tension: 0.4, pointRadius: 4 }],
  };

  const langDistData = {
    labels: (ch.language_distribution || []).map(d => d.language),
    datasets: [
      { label: 'Count', data: (ch.language_distribution || []).map(d => d.count), backgroundColor: PALETTE.slice(0, (ch.language_distribution || []).length), borderRadius: 6 },
    ],
  };

  const confHistData = {
    labels: (ch.confidence_histogram || []).map(d => d.bin),
    datasets: [{ label: 'Count', data: (ch.confidence_histogram || []).map(d => d.count), backgroundColor: CONFIDENCE_COLORS(ch.confidence_histogram || []), borderRadius: 4 }],
  };

  const entityHistData = {
    labels: (ch.entity_count_histogram || []).map(d => String(d.entities)),
    datasets: [{ label: 'Complaints', data: (ch.entity_count_histogram || []).map(d => d.count), backgroundColor: '#06B6D4', borderRadius: 6 }],
  };

  const entityTypeData = {
    labels: (ch.entity_type_breakdown || []).map(d => d.type),
    datasets: [{ data: (ch.entity_type_breakdown || []).map(d => d.count), backgroundColor: ['#0284C7','#8B5CF6','#10B981','#F59E0B','#EF4444'].slice(0, (ch.entity_type_breakdown || []).length), borderWidth: 2, borderColor: '#fff' }],
  };

  const scatterData = {
    datasets: [{ label: 'Audio Duration vs Processing Time', data: (ch.audio_duration_vs_time || []).map(d => ({ x: d.duration_s, y: d.processing_time_s })), backgroundColor: 'rgba(99,102,241,0.6)', pointRadius: 5, pointHoverRadius: 8 }],
  };

  const errorStageData = {
    labels: (ch.error_rate_by_stage || []).map(d => d.stage),
    datasets: [{ label: 'Error %', data: (ch.error_rate_by_stage || []).map(d => d.rate_percent), backgroundColor: '#EF4444', borderRadius: 6 }],
  };

  const votesData = {
    labels: (ch.votes_per_complaint || []).map(d => `#${d.complaint_id}`),
    datasets: [{ label: 'Votes', data: (ch.votes_per_complaint || []).map(d => d.votes), backgroundColor: '#EC4899', borderRadius: 6 }],
  };

  const zsFallbackData = {
    labels: ['Primary Classifier', 'Zero-shot Fallback'],
    datasets: [{ data: [Math.max(0, 100 - (ns.zero_shot_fallback_rate || 0)), ns.zero_shot_fallback_rate || 0], backgroundColor: ['#10B981', '#F59E0B'], borderWidth: 2, borderColor: '#fff' }],
  };

  const dupClusterData = {
    labels: (ch.duplicate_cluster_sizes || []).map(d => `${d.cluster_size} votes`),
    datasets: [{ label: 'Complaints', data: (ch.duplicate_cluster_sizes || []).map(d => d.count), backgroundColor: '#8B5CF6', borderRadius: 6 }],
  };

  const severityDistData = {
    labels: (ch.severity_distribution || []).map(d => d.severity),
    datasets: [{
      data: (ch.severity_distribution || []).map(d => d.count),
      backgroundColor: (ch.severity_distribution || []).map(d => {
        if (d.severity === 'Clear') return '#CBD5E1';
        if (d.severity === 'Low') return '#10B981';
        if (d.severity === 'Medium') return '#F59E0B';
        if (d.severity === 'High') return '#F97316';
        if (d.severity === 'Severe') return '#EF4444';
        return '#CBD5E1';
      }),
      borderWidth: 2,
      borderColor: '#fff'
    }],
  };

  /* ── Heatmap ───────────────────────────────────────────────── */
  const heatmapData = ch.category_language_heatmap || [];
  const hmLanguages = [...new Set(heatmapData.map(d => d.language))].sort();
  const hmCategories = [...new Set(heatmapData.map(d => d.category))].sort();
  const hmMax = Math.max(...heatmapData.map(d => d.count), 1);
  const hmLookup = {};
  heatmapData.forEach(d => { hmLookup[`${d.category}_${d.language}`] = d.count; });
  const getIntensity = (v) => {
    if (!v) return 0;
    const r = v / hmMax;
    if (r > 0.8) return 5;
    if (r > 0.6) return 4;
    if (r > 0.4) return 3;
    if (r > 0.2) return 2;
    if (r > 0) return 1;
    return 0;
  };

  const energyOverTimeOpts = {
    ...CHART_BASE,
    scales: {
      ...CHART_BASE.scales,
      y1: { position: 'right', ticks: { font: { family: 'Inter', size: 10 }, color: '#8B5CF6' }, grid: { display: false } },
    },
  };

  const radarOpts = {
    responsive: true, maintainAspectRatio: false,
    scales: { r: { ticks: { font: { family: 'Inter', size: 9 }, backdropColor: 'transparent' }, grid: { color: 'rgba(0,0,0,0.06)' }, pointLabels: { font: { family: 'Inter', size: 11 } } } },
    plugins: { ...CHART_BASE.plugins },
  };

  const scatterOpts = {
    ...CHART_BASE,
    scales: {
      x: { ...CHART_BASE.scales.x, title: { display: true, text: 'Audio Duration (s)', font: { family: 'Inter', size: 11 } } },
      y: { ...CHART_BASE.scales.y, title: { display: true, text: 'Processing Time (s)', font: { family: 'Inter', size: 11 } } },
    },
  };

  /* ── Verification Table ────────────────────────────────────── */
  const verificationRows = [
    ['Total Complaints Processed', 'nlp_metrics COUNT(*)', 'Direct DB query on nlp_metrics table'],
    ['Unique Complaints', 'complaints COUNT(*)', 'Direct DB query on complaints table'],
    ['Duplicate Complaints', 'nlp_metrics WHERE is_duplicate=True', 'Boolean flag set by duplicate detection'],
    ['Total Votes', 'complaints SUM(votes)', 'Direct DB aggregate query'],
    ['Avg Processing Time', 'nlp_metrics AVG(total_processing_time)', 'Measured via time.perf_counter()'],
    ['Total Energy (J)', 'nlp_metrics SUM(total_energy_joules)', 'CPU TDP × measured processing time'],
    ['Classifier Confidence', 'nlp_metrics.classifier_confidence', 'sklearn predict_proba() per request'],
    ['Entity Count', 'nlp_metrics.entity_count', 'len(spacy_doc.ents) per request'],
    ['Audio Duration', 'nlp_metrics.audio_duration_seconds', 'pydub AudioSegment.duration_seconds'],
    ['Zero-shot Rate', 'nlp_metrics zero_shot_triggered', 'Boolean flag per request'],
    ['Error Rate', 'nlp_metrics.error_stage', 'Exception handler captures stage name'],
  ];

  return (
    <div className="gravless-container" id="analytics-dashboard">
      {/* Header */}
      <div className="analytics-header">
        <div>
          <div className="analytics-hero-badge">📊 NLP Analytics & Energy Monitoring</div>
          <h2>Analytics Dashboard</h2>
          <p className="analytics-subtitle">Real-time NLP processing metrics, energy consumption, and system health</p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary btn-sm" onClick={fetchData} disabled={loading}>
            {loading ? <><span className="spinner" /> Refreshing…</> : '🔄 Refresh'}
          </button>
          <button className="btn btn-ghost btn-sm" onClick={handleLogout}>Logout</button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="analytics-filter-bar">
        <label>From</label>
        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
        <label>To</label>
        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
        <label>Language</label>
        <select value={langFilter} onChange={e => setLangFilter(e.target.value)}>
          <option value="">All Languages</option>
          <option value="en">English</option>
          <option value="kn">Kannada</option>
          <option value="hi">Hindi</option>
        </select>
        <div className="filter-spacer" />
        <span className="metric-unit">Showing {cs.total_complaints_processed || 0} NLP requests</span>
      </div>

      {/* Section 1: Key Metrics */}
      <div className="analytics-section">
        <div className="analytics-section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
          Key Metrics
        </div>
        <div className="metrics-grid">
          <div className="metric-card complaint"><div className="metric-icon">📋</div><div className="metric-value">{fmt(cs.total_complaints_processed, 0)}</div><div className="metric-label">Total Processed</div><div className="metric-unit">NLP requests</div></div>
          <div className="metric-card complaint"><div className="metric-icon">✅</div><div className="metric-value">{fmt(cs.unique_complaints, 0)}</div><div className="metric-label">Unique Complaints</div><div className="metric-unit">In database</div></div>
          <div className="metric-card complaint"><div className="metric-icon">🔁</div><div className="metric-value">{fmt(cs.duplicate_complaints, 0)}</div><div className="metric-label">Duplicates Detected</div><div className="metric-unit">Merged automatically</div></div>
          <div className="metric-card vote"><div className="metric-icon">👍</div><div className="metric-value">{fmt(cs.total_votes, 0)}</div><div className="metric-label">Total Votes</div><div className="metric-unit">Avg {fmt(cs.average_votes_per_complaint)}/complaint</div></div>
          <div className="metric-card nlp"><div className="metric-icon">🧠</div><div className="metric-value">{fmt(ns.total_requests, 0)}</div><div className="metric-label">NLP Requests</div><div className="metric-unit">All pipeline runs</div></div>
          <div className="metric-card nlp"><div className="metric-icon">⏱️</div><div className="metric-value">{fmt(ns.avg_processing_time_seconds)}s</div><div className="metric-label">Avg Processing Time</div><div className="metric-unit">Per request</div></div>
          <div className="metric-card energy"><div className="metric-icon">⚡</div><div className="metric-value">{fmt(es.total_energy_joules)}</div><div className="metric-label">Total Energy (J)</div><div className="metric-unit">Joules consumed</div></div>
          <div className="metric-card energy"><div className="metric-icon">📊</div><div className="metric-value">{fmt(es.avg_energy_per_complaint)}</div><div className="metric-label">Avg Energy/Request (J)</div><div className="metric-unit">Per NLP pipeline</div></div>
          <div className="metric-card energy"><div className="metric-icon">🌱</div><div className="metric-value">{fmt(es.energy_saved_by_dedup)}</div><div className="metric-label">Energy Saved (Dedup)</div><div className="metric-unit">Joules saved</div></div>
        </div>
      </div>

      {/* Section 2: Energy & Performance */}
      <div className="analytics-section">
        <div className="analytics-section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
          Energy & Performance
        </div>
        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-card-title"><span>⚡</span> Energy Consumption by NLP Stage</div>
            <div className="chart-container"><Bar data={energyByStageData} options={CHART_BASE} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>📈</span> Energy & Throughput Over Time</div>
            <div className="chart-container"><Line data={energyOverTimeData} options={energyOverTimeOpts} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>🎯</span> Stage Bottleneck Analysis</div>
            <div className="chart-container"><Radar data={radarData} options={radarOpts} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>📊</span> Throughput Over Time</div>
            <div className="chart-container"><Line data={throughputData} options={CHART_BASE} /></div>
          </div>
        </div>
      </div>

      {/* Section 3: Classification & Language */}
      <div className="analytics-section">
        <div className="analytics-section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/></svg>
          Classification & Language
        </div>
        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-card-title"><span>🏷️</span> Category Distribution</div>
            <div className="chart-container"><Doughnut data={catDistData} options={DOUGHNUT_BASE} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>🌐</span> Source Language Distribution</div>
            <div className="chart-container"><Bar data={langDistData} options={CHART_BASE} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>🔬</span> Classifier Confidence Histogram</div>
            <div className="chart-container"><Bar data={confHistData} options={{...CHART_BASE, plugins: {...CHART_BASE.plugins, legend: {display: false}}}} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>🗺️</span> Category × Language Heatmap</div>
            {hmCategories.length > 0 && hmLanguages.length > 0 ? (
              <div className="heatmap-grid" style={{ gridTemplateColumns: `120px repeat(${hmLanguages.length}, 1fr)` }}>
                <div className="heatmap-header" />
                {hmLanguages.map(l => <div key={l} className="heatmap-header">{l}</div>)}
                {hmCategories.map(cat => (
                  <>
                    <div key={`lbl-${cat}`} className="heatmap-row-label" title={cat}>{cat.length > 14 ? cat.slice(0,12)+'…' : cat}</div>
                    {hmLanguages.map(lang => {
                      const v = hmLookup[`${cat}_${lang}`] || 0;
                      return <div key={`${cat}-${lang}`} className={`heatmap-cell intensity-${getIntensity(v)}`}>{v || ''}</div>;
                    })}
                  </>
                ))}
              </div>
            ) : (
              <div className="analytics-empty"><p>No cross-tab data yet</p></div>
            )}
          </div>
        </div>
      </div>

      {/* Section 4: NER & Quality */}
      <div className="analytics-section">
        <div className="analytics-section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
          NER & Quality
        </div>
        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-card-title"><span>📍</span> NER Entity Count Distribution</div>
            <div className="chart-container"><Bar data={entityHistData} options={{...CHART_BASE, plugins: {...CHART_BASE.plugins, legend: {display: false}}}} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>🏛️</span> Entity Type Breakdown</div>
            <div className="chart-container"><Doughnut data={entityTypeData} options={DOUGHNUT_BASE} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>🎤</span> Audio Duration vs Processing Time</div>
            <div className="chart-container"><Scatter data={scatterData} options={scatterOpts} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>❌</span> Error Rate by Stage</div>
            <div className="chart-container"><Bar data={errorStageData} options={{...CHART_BASE, plugins: {...CHART_BASE.plugins, legend: {display: false}}}} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>🚧</span> Pothole Severity Distribution</div>
            <div className="chart-container"><Doughnut data={severityDistData} options={DOUGHNUT_BASE} /></div>
          </div>
        </div>
      </div>

      {/* Section 5: Complaints & Duplicates */}
      <div className="analytics-section">
        <div className="analytics-section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
          Complaints & Duplicates
        </div>
        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-card-title"><span>🥧</span> Duplicate vs Unique</div>
            <div className="chart-container"><Pie data={dupVsUniqueData} options={DOUGHNUT_BASE} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>👍</span> Votes per Complaint (Top 20)</div>
            <div className="chart-container"><Bar data={votesData} options={{...CHART_BASE, plugins: {...CHART_BASE.plugins, legend: {display: false}}}} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>🔄</span> Zero-shot Fallback Rate</div>
            <div className="chart-container"><Doughnut data={zsFallbackData} options={DOUGHNUT_BASE} /></div>
          </div>
          <div className="chart-card">
            <div className="chart-card-title"><span>📦</span> Duplicate Cluster Sizes</div>
            <div className="chart-container"><Bar data={dupClusterData} options={{...CHART_BASE, plugins: {...CHART_BASE.plugins, legend: {display: false}}}} /></div>
          </div>
        </div>
      </div>

      {/* Section 6: Verification Panel */}
      <div className="analytics-section verification-panel">
        <button className="verification-toggle" onClick={() => setVerifyOpen(!verifyOpen)}>
          {verifyOpen ? '▼' : '▶'} Data Source Verification ({verificationRows.length} metrics)
        </button>
        {verifyOpen && (
          <div className="verification-content">
            <table className="verification-table">
              <thead>
                <tr><th>Metric</th><th>Data Source</th><th>Calculation</th><th>Status</th></tr>
              </thead>
              <tbody>
                {verificationRows.map(([metric, source, calc], i) => (
                  <tr key={i}>
                    <td><strong>{metric}</strong></td>
                    <td><code style={{fontSize:'0.75rem'}}>{source}</code></td>
                    <td>{calc}</td>
                    <td><span className="source-badge">✓ Live</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {ds.calculation_method || es.calculation_method ? (
              <p style={{marginTop:'0.75rem', fontSize:'0.78rem', color:'var(--color-text-muted)'}}>
                <strong>Energy Method:</strong> {es.calculation_method || ds.energy || ''}
              </p>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
