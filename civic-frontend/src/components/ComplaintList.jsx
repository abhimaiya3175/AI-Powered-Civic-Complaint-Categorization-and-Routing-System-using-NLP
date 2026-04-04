import { useState, useEffect } from 'react';
import { getComplaints, verifyComplaint, getStats, loginAdmin, getAudioUrl } from '../services/api';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from 'react-leaflet';
import L from 'leaflet';
import '../styles/ComplaintList.css';

// Fix Leaflet default marker icon for bundlers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

 

export default function ComplaintList() {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [stats, setStats] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('bbmp_token') || '');
  const [loggedIn, setLoggedIn] = useState(!!localStorage.getItem('bbmp_token'));
  const [loginError, setLoginError] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const pageSize = 10;

  const [playingId, setPlayingId] = useState(null);
  const [audioSources, setAudioSources] = useState({});
  const [audioLoading, setAudioLoading] = useState({});

  /* ── Auth ─────────────────────────────────────────────────────── */
  const handleLogin = async (e) => {
    e.preventDefault();
    const form = e.target;
    setLoginError('');
    setLoginLoading(true);
    try {
      const data = await loginAdmin(form.username.value, form.password.value);
      setToken(data.access_token);
      localStorage.setItem('bbmp_token', data.access_token);
      setLoggedIn(true);
    } catch (err) {
      setLoginError(err.message);
    }
    setLoginLoading(false);
  };

  const handleLogout = () => {
    Object.values(audioSources).forEach((url) => URL.revokeObjectURL(url));
    setToken('');
    localStorage.removeItem('bbmp_token');
    setLoggedIn(false);
    setComplaints([]);
    setStats(null);
    setAudioSources({});
    setAudioLoading({});
  };

  /* ── Fetch Data ──────────────────────────────────────────────── */
  const fetchComplaints = async () => {
    setLoading(true);
    try {
      const data = await getComplaints(token, page, pageSize);
      setComplaints(data.items || []);
      setTotalPages(data.pages || 1);
      setTotalItems(data.total || 0);
    } catch (err) {
      if (err.message.includes('401') || err.message.includes('token')) handleLogout();
    }
    setLoading(false);
  };

  const fetchStats = async () => {
    try {
      const data = await getStats(token);
      setStats(data);
    } catch { /* ignore */ }
  };

  useEffect(() => {
    if (loggedIn) { fetchComplaints(); fetchStats(); }
  }, [loggedIn, page]);

 

  /* ── HITL: Verify ────────────────────────────────────────────── */
  const handleVerify = async (id) => {
    try {
      await verifyComplaint(token, id);
      fetchComplaints();
      fetchStats();
    } catch { alert('Failed to verify complaint'); }
  };

  /* ── Helpers ─────────────────────────────────────────────────── */
  const toggleAudio = async (complaint) => {
    if (playingId === complaint.id) {
      setPlayingId(null);
      return;
    }

    setPlayingId(complaint.id);
    if (!complaint.audio_path || audioSources[complaint.id]) {
      return;
    }

    setAudioLoading((prev) => ({ ...prev, [complaint.id]: true }));
    try {
      const response = await fetch(getAudioUrl(complaint.audio_path), {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        throw new Error('Unable to load audio file');
      }
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      setAudioSources((prev) => ({ ...prev, [complaint.id]: blobUrl }));
    } catch (err) {
      alert(err.message || 'Failed to load complaint audio');
    } finally {
      setAudioLoading((prev) => ({ ...prev, [complaint.id]: false }));
    }
  };

  const filtered = complaints.filter(
    (c) => filter === 'all' || (c.status || '').toLowerCase() === filter
  );

  const getStatusBadge = (status) => {
    if (status === 'Verified') return 'badge badge-verified';
    return 'badge badge-pending';
  };

  const getMarkerColor = (status) => {
    if (status === 'Verified') return '#10B981';
    return '#F59E0B';
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
  };

  /* ── Login Screen ────────────────────────────────────────────── */
  if (!loggedIn) {
    return (
      <div className="login-wrapper">
        <div className="login-card card">
          <div className="login-header">
            <div className="login-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect width="18" height="18" x="3" y="3" rx="2"/>
                <path d="M3 9h18"/>
                <path d="M9 21V9"/>
              </svg>
            </div>
            <h2>Admin Dashboard</h2>
            <p className="login-subtext">Sign in to manage and verify complaints</p>
          </div>
          <form onSubmit={handleLogin} className="login-form">
            <div className="form-group">
              <label htmlFor="login-username" className="form-label">Username</label>
              <input
                id="login-username"
                name="username"
                className="input"
                placeholder="Enter username"
                required
                autoComplete="username"
              />
            </div>
            <div className="form-group">
              <label htmlFor="login-password" className="form-label">Password</label>
              <input
                id="login-password"
                name="password"
                type="password"
                className="input"
                placeholder="Enter password"
                required
                autoComplete="current-password"
              />
            </div>
            {loginError && (
              <div className="login-error">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" x2="12" y1="8" y2="12"/>
                  <line x1="12" x2="12.01" y1="16" y2="16"/>
                </svg>
                {loginError}
              </div>
            )}
            <button type="submit" className="btn btn-primary btn-lg login-btn" disabled={loginLoading}>
              {loginLoading ? (
                <><span className="spinner"></span> Signing in…</>
              ) : 'Sign In'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  /* ── Loading State ───────────────────────────────────────────── */
  if (loading && complaints.length === 0) {
    return (
      <div className="dashboard-loading">
        <div className="loading-spinner"></div>
        <p>Loading dashboard…</p>
      </div>
    );
  }

  /* ── Dashboard ───────────────────────────────────────────────── */
  return (
    <div className="dashboard" id="admin-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div>
          <h2>Complaints Dashboard</h2>
          <p className="dashboard-subtitle">Manage, verify, and track all civic complaints</p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary btn-sm" onClick={() => { fetchComplaints(); fetchStats(); }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="1 4 1 10 7 10"/>
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
            </svg>
            Refresh
          </button>
          <button className="btn btn-ghost btn-sm" onClick={handleLogout}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
              <polyline points="16 17 21 12 16 7"/>
              <line x1="21" x2="9" y1="12" y2="12"/>
            </svg>
            Logout
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      {stats && (
        <div className="stats-grid">
          <div className="stat-card stat-total">
            <div className="stat-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
            </div>
            <span className="stat-value">{stats.total}</span>
            <span className="stat-label">Total Complaints</span>
          </div>
          <div className="stat-card stat-pending">
            <div className="stat-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
            </div>
            <span className="stat-value">{stats.pending}</span>
            <span className="stat-label">Pending Review</span>
          </div>
          <div className="stat-card stat-verified">
            <div className="stat-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
              </svg>
            </div>
            <span className="stat-value">{stats.verified}</span>
            <span className="stat-label">Verified</span>
          </div>
          {(stats.by_category || []).slice(0, 3).map((item) => (
            <div key={item.category} className="stat-card stat-category">
              <span className="stat-value">{item.count}</span>
              <span className="stat-label">{item.category}</span>
            </div>
          ))}
        </div>
      )}

      {/* Map */}
      <div className="map-card card">
        <div className="map-header">
          <h3>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/>
              <line x1="9" x2="9" y1="3" y2="18"/>
              <line x1="15" x2="15" y1="6" y2="21"/>
            </svg>
            Complaint Map
          </h3>
        </div>
        <div className="map-container">
          <MapContainer center={[12.9716, 77.5946]} zoom={11} style={{ height: '100%', width: '100%' }}>
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            />
            {complaints.filter(c => c.status !== 'Verified' && c.live_latitude && c.live_longitude).map((c) => (
              <CircleMarker
                key={`marker-${c.id}`}
                center={[c.live_latitude, c.live_longitude]}
                radius={8}
                pathOptions={{
                  fillColor: '#EF4444', 
                  fillOpacity: 0.8,
                  color: '#FFFFFF',
                  weight: 2,
                }}
              >
                <Popup>
                  <div className="map-popup">
                    <strong>{c.category}</strong>
                    <span>{c.location}</span>
                    <span className={getStatusBadge(c.status)}>{c.status}</span>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>
      </div>

      {/* Filters */}
      <div className="filters-bar">
        <div className="filter-pills">
          {['all', 'pending', 'verified'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
            >
              {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
              {f === 'pending' && stats ? ` (${stats.pending})` : ''}
              {f === 'verified' && stats ? ` (${stats.verified})` : ''}
            </button>
          ))}
        </div>
        <span className="results-count">{totalItems} total complaints</span>
      </div>

      {/* Complaint Cards */}
      {filtered.length === 0 ? (
        <div className="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#CBD5E1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <p>No complaints found matching your filter</p>
        </div>
      ) : (
        <div className="complaints-grid">
          {filtered.map((c) => (
            <div key={c.id} className="complaint-card card" id={`complaint-${c.id}`}>
              {/* Card Header */}
              <div className="ccard-header">
                <span className="ccard-id mono">#{c.id}</span>
                <span className={getStatusBadge(c.status)}>{c.status}</span>
              </div>

              {/* Card Body */}
              <div className="ccard-body">
                <div className="ccard-field">
                  <span className="ccard-field-label">Category</span>
                  <span className="ccard-field-value badge badge-info">{c.category}</span>
                </div>
                <div className="ccard-field">
                  <span className="ccard-field-label">Location</span>
                  <span className="ccard-field-value">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>
                      <circle cx="12" cy="10" r="3"/>
                    </svg>
                    {c.location}
                  </span>
                </div>
                <div className="ccard-field">
                  <span className="ccard-field-label">Language</span>
                  <span className="ccard-field-value ccard-lang">{c.language}</span>
                </div>
                {c.created_at && (
                  <div className="ccard-field">
                    <span className="ccard-field-label">Date</span>
                    <span className="ccard-field-value ccard-date">{formatDate(c.created_at)}</span>
                  </div>
                )}
                {c.translated_text && (
                  <div className="ccard-transcript">
                    <p>{c.translated_text}</p>
                  </div>
                )}
              </div>

              {/* Audio */}
              {c.audio_path && (
                <div className="ccard-audio">
                  <button className="btn btn-secondary btn-sm" onClick={() => toggleAudio(c)}>
                    {playingId === c.id ? (
                      <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Hide Player</>
                    ) : (
                      <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Play Audio</>
                    )}
                  </button>
                  {playingId === c.id && (
                    audioLoading[c.id] ? (
                      <p className="ccard-audio-loading">Loading audio...</p>
                    ) : (
                      <audio controls autoPlay src={audioSources[c.id] || ''} className="ccard-audio-player" />
                    )
                  )}
                </div>
              )}

              {/* Footer */}
              {c.status !== 'Verified' && (
                <div className="ccard-footer">
                  <button className="btn btn-success btn-sm" onClick={() => handleVerify(c.id)}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                    Verify
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination">
          <button className="btn btn-secondary btn-sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6"/>
            </svg>
            Previous
          </button>
          <span className="page-info mono">
            Page {page} of {totalPages}
          </span>
          <button className="btn btn-secondary btn-sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>
            Next
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6"/>
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}
