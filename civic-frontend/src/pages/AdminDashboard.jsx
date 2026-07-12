import { useCallback, useState, useEffect, useRef } from 'react';
import { useAuth } from '../hooks/useAuth';
import { getComplaints, verifyComplaint, getComplaintTimeline, getMapComplaints } from '../services/complaintService';
import { getStats } from '../services/analyticsService';
import { getAudioUrl } from '../utils/helpers';
import { reanalyzeImage } from '../services/uploadService';
import { MapContainer, TileLayer, Popup, CircleMarker } from 'react-leaflet';
import L from 'leaflet';
import '../styles/ComplaintList.css';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const CATEGORY_OPTIONS = [
  'Street Light','Garbage / Sanitation','Road Repair','Drainage / SWD',
  'Water Supply','Health / Sanitation','Parks','Parks / Forest',
  'Town Planning','Veterinary','Advertisement','Revenue','Traffic','Others',
];

const CAT_ICONS = {
  'Street Light':'💡','Garbage / Sanitation':'🗑️','Road Repair':'🛣️',
  'Drainage / SWD':'🌊','Water Supply':'💧','Health / Sanitation':'🏥',
  'Parks':'🌳','Parks / Forest':'🌲','Town Planning':'🏙️',
  'Veterinary':'🐾','Advertisement':'📢','Revenue':'💰','Traffic':'🚦','Others':'📋',
};

const STATUS_MAP = {
  pending:       { cls: 'pending',    icon: '⏳', label: 'Pending' },
  Verified:      { cls: 'verified',   icon: '✅', label: 'Verified' },
  Rejected:      { cls: 'rejected',   icon: '❌', label: 'Rejected' },
  'In Progress': { cls: 'inprogress', icon: '🔄', label: 'In Progress' },
  Resolved:      { cls: 'resolved',   icon: '🎉', label: 'Resolved' },
};

const isAuthError = (err) => {
  const m = String(err?.message || '').toLowerCase();
  return m.includes('401') || m.includes('token') || m.includes('log in') || m.includes('unauthorized');
};

function StatusBadge({ status }) {
  const s = STATUS_MAP[status] || STATUS_MAP.pending;
  return (
    <span className={`cc-status-badge s-${s.cls}`}>
      <span className="cc-dot" />
      {s.label}
    </span>
  );
}

function DetectionOverlay({ imageUrl, detections, alt }) {
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const [imgLoaded, setImgLoaded] = useState(false);

  // Re-draw when image loads or window resizes
  const drawDetections = useCallback(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !detections || detections.length === 0) return;

    // Match canvas size to rendered image size
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const severityColors = {
      Low: '#10B981',     // Green
      Medium: '#F59E0B',  // Yellow
      High: '#F97316',    // Orange
      Severe: '#EF4444'   // Red
    };

    detections.forEach(d => {
      const color = severityColors[d.severity] || '#06B6D4';
      
      // Draw Bounding Box (normalized coords)
      if (d.bbox && d.bbox.length === 4) {
        const [x1, y1, x2, y2] = d.bbox;
        const px1 = x1 * canvas.width;
        const py1 = y1 * canvas.height;
        const px2 = x2 * canvas.width;
        const py2 = y2 * canvas.height;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);

        // Label background
        const label = `${d.class} ${(d.confidence * 100).toFixed(0)}%`;
        ctx.font = '10px Inter, sans-serif';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(px1, py1 - 16, tw + 8, 16);

        // Label text
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, px1 + 4, py1 - 4);
      }

      // Draw Polygon Mask
      if (d.mask_polygon && d.mask_polygon.length > 0) {
        ctx.beginPath();
        d.mask_polygon.forEach((pt, i) => {
          const px = pt[0] * canvas.width;
          const py = pt[1] * canvas.height;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fillStyle = `${color}40`; // 25% opacity fill
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
  }, [detections]);

  useEffect(() => {
    if (imgLoaded) {
      drawDetections();
      window.addEventListener('resize', drawDetections);
      return () => window.removeEventListener('resize', drawDetections);
    }
  }, [imgLoaded, drawDetections]);

  return (
    <div className="detection-overlay-container">
      <img 
        ref={imgRef}
        src={imageUrl} 
        alt={alt} 
        className="ccard-image-preview" 
        loading="lazy" 
        onLoad={() => setImgLoaded(true)}
      />
      <canvas ref={canvasRef} className="detection-canvas" />
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="admin-skel-card">
      <div style={{ display:'flex', justifyContent:'space-between' }}>
        <div className="admin-skel-line w30" />
        <div className="admin-skel-pill" />
      </div>
      <div className="admin-skel-line w80" />
      <div className="admin-skel-line w60" />
      <div className="admin-skel-line w50" />
      <div style={{ display:'flex', gap:'0.5rem', paddingTop:'0.5rem', borderTop:'1px solid rgba(0,0,0,0.05)' }}>
        <div className="admin-skel-pill" />
        <div className="admin-skel-pill" />
      </div>
    </div>
  );
}

export default function AdminDashboard() {
  const { token, logout, isAuthenticated } = useAuth();
  const [complaints, setComplaints] = useState([]);
  const [mapComplaints, setMapComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [sortBy, setSortBy] = useState('most_voted');
  const [stats, setStats] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const pageSize = 10;
  const [playingId, setPlayingId] = useState(null);
  const [audioSources, setAudioSources] = useState({});
  const audioSourcesRef = useRef({});
  const [audioLoading, setAudioLoading] = useState({});
  const [adminNotes, setAdminNotes] = useState({});
  const [categoryEdits, setCategoryEdits] = useState({});
  const [timelines, setTimelines] = useState({});
  const [tlLoading, setTlLoading] = useState({});
  const [tlOpen, setTlOpen] = useState({});
  const [mismatchFilter, setMismatchFilter] = useState(false);
  const [aiPanelOpen, setAiPanelOpen] = useState({});
  const [mapOpen, setMapOpen] = useState(false);
  
  const toggleAiPanel = (id) => setAiPanelOpen(p => ({ ...p, [id]: !p[id] }));

  useEffect(() => { audioSourcesRef.current = audioSources; }, [audioSources]);

  /* ── Auth ───────────────────────────────────────────────────── */
  const handleLogout = useCallback(() => {
    Object.values(audioSourcesRef.current).forEach((url) => URL.revokeObjectURL(url));
    logout();
    setComplaints([]); setStats(null);
    setAudioSources({}); setAudioLoading({});
  }, [logout]);

  /* ── Fetch ──────────────────────────────────────────────────── */
  const fetchComplaints = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getComplaints(token, page, pageSize, mismatchFilter);
      setComplaints(data.items || []);
      setTotalPages(data.pages || 1);
      setTotalItems(data.total || 0);
    } catch (err) { if (isAuthError(err)) handleLogout(); }
    setLoading(false);
  }, [handleLogout, page, token, mismatchFilter]);

  const fetchMapComplaints = useCallback(async () => {
    try {
      const data = await getMapComplaints(token);
      setMapComplaints(data.items || []);
    } catch (err) { if (isAuthError(err)) handleLogout(); }
  }, [handleLogout, token]);

  const fetchStats = useCallback(async () => {
    try { const data = await getStats(token); setStats(data); }
    catch (err) { if (isAuthError(err)) handleLogout(); }
  }, [handleLogout, token]);

  useEffect(() => {
    if (isAuthenticated) { fetchComplaints(); fetchMapComplaints(); fetchStats(); }
  }, [fetchComplaints, fetchMapComplaints, fetchStats, isAuthenticated]);

  /* ── HITL ───────────────────────────────────────────────────── */
  const handleStatusUpdate = async (id, newStatus) => {
    try {
      await verifyComplaint(token, id, { status: newStatus, note: (adminNotes[id] || '').trim() || undefined });
      setAdminNotes(p => ({ ...p, [id]: '' }));
      if (tlOpen[id]) await loadTimeline(id);
      fetchComplaints(); fetchMapComplaints(); fetchStats();
    } catch (e) {
      if (isAuthError(e)) handleLogout();
      else alert(e.message || 'Failed to update complaint');
    }
  };

  const handleCategoryUpdate = async (complaint) => {
    const next = categoryEdits[complaint.id] || complaint.category;
    if (!next || next === complaint.category) return;
    try {
      await verifyComplaint(token, complaint.id, {
        category: next, status: complaint.status,
        note: (adminNotes[complaint.id] || '').trim() || undefined,
      });
      setAdminNotes(p => ({ ...p, [complaint.id]: '' }));
      if (tlOpen[complaint.id]) await loadTimeline(complaint.id);
      fetchComplaints(); fetchStats();
    } catch (e) {
      if (isAuthError(e)) handleLogout();
      else alert(e.message || 'Failed to update category');
    }
  };
  const handleReanalyze = async (id) => {
    try {
      await reanalyzeImage(id, token);
      alert('Image re-analysis started in the background. It will update in a few seconds.');
      fetchComplaints();
    } catch (e) {
      if (isAuthError(e)) handleLogout();
      else alert(e.message || 'Failed to start re-analysis');
    }
  };

  /* ── Timeline ───────────────────────────────────────────────── */
  const loadTimeline = async (id) => {
    setTlLoading(p => ({ ...p, [id]: true }));
    try {
      const data = await getComplaintTimeline(id);
      setTimelines(p => ({ ...p, [id]: data.timeline || [] }));
    } catch { setTimelines(p => ({ ...p, [id]: [] })); }
    setTlLoading(p => ({ ...p, [id]: false }));
  };

  const toggleTimeline = async (id) => {
    if (!tlOpen[id]) {
      setTlOpen({ [id]: true });
      if (!timelines[id]) await loadTimeline(id);
    } else { setTlOpen({}); }
  };

  /* ── Audio ──────────────────────────────────────────────────── */
  const toggleAudio = async (c) => {
    if (playingId === c.id) { setPlayingId(null); return; }
    setPlayingId(c.id);
    if (!c.audio_path || audioSources[c.id]) return;
    setAudioLoading(p => ({ ...p, [c.id]: true }));
    try {
      const r = await fetch(getAudioUrl(c.audio_path), { headers: { Authorization: `Bearer ${token}` } });
      if (!r.ok) throw new Error('Unable to load audio file');
      const blob = await r.blob();
      setAudioSources(p => ({ ...p, [c.id]: URL.createObjectURL(blob) }));
    } catch (err) { alert(err.message || 'Failed to load audio'); }
    finally { setAudioLoading(p => ({ ...p, [c.id]: false })); }
  };

  /* ── Derived data ───────────────────────────────────────────── */
  // Compute global max from ALL loaded complaints (not just filtered page)
  const globalMaxVotes = Math.max(...complaints.map(c => c.votes || 0), 1);

  const filtered = complaints
    .filter(c => filter === 'all' || (c.status || '').toLowerCase() === filter.toLowerCase())
    .filter(c => categoryFilter === 'all' || c.category === categoryFilter)
    .filter(c => !mismatchFilter || c.category_mismatch)
    .sort((a, b) => {
      if (sortBy === 'most_voted') {
        const voteDiff = (b.votes || 0) - (a.votes || 0);
        if (voteDiff !== 0) return voteDiff;
      }
      return new Date(b.created_at) - new Date(a.created_at);
    });

  // Tiered priority thresholds (relative to globalMaxVotes)
  const CRITICAL_THRESHOLD = Math.max(5, Math.round(globalMaxVotes * 0.75));
  const HIGH_THRESHOLD     = Math.max(3, Math.round(globalMaxVotes * 0.4));
  const MEDIUM_THRESHOLD   = Math.max(2, Math.round(globalMaxVotes * 0.2));

  const getVotePriority = (votes) => {
    const v = votes || 0;
    if (v >= CRITICAL_THRESHOLD) return 'critical';
    if (v >= HIGH_THRESHOLD)     return 'high';
    if (v >= MEDIUM_THRESHOLD)   return 'medium';
    return 'none';
  };

  // Top-voted leaderboard (top 3 by votes, at least 2 votes)
  const topVoted = [...complaints]
    .filter(c => (c.votes || 0) >= 2)
    .sort((a, b) => (b.votes || 0) - (a.votes || 0))
    .slice(0, 3);


  const fmtDate = (d) => d ? new Date(d).toLocaleDateString('en-IN',{day:'numeric',month:'short',year:'numeric'}) : '—';
  const statusCls = (s) => (STATUS_MAP[s] || STATUS_MAP.pending).cls;

  /* ── Login Screen ───────────────────────────────────────────── */
  if (!isAuthenticated) {
    return null; // ProtectedRoute will handle redirection to /admin/login
  }

  /* ── Loading ────────────────────────────────────────────────── */
  if (loading && complaints.length === 0) {
    return (
      <div className="dashboard gravless-container" id="admin-dashboard">
        <div className="dashboard-header">
          <div>
            <div className="admin-hero-badge">🛡️ Admin Portal</div>
            <h2>Complaints Dashboard</h2>
          </div>
        </div>
        <div className="complaints-grid">
          {Array.from({length:6}).map((_,i) => <SkeletonCard key={i} />)}
        </div>
      </div>
    );
  }

  /* ── Dashboard ──────────────────────────────────────────────── */
  return (
    <div className="dashboard gravless-container" id="admin-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div>
          <div className="admin-hero-badge">🛡️ Admin Portal</div>
          <h2>Complaints Dashboard</h2>
          <p className="dashboard-subtitle">Manage, verify, and track all civic complaints</p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary btn-sm" onClick={() => { fetchComplaints(); fetchStats(); }}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
            Refresh
          </button>
          <button className="btn btn-ghost btn-sm" onClick={handleLogout}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/></svg>
            Logout
          </button>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="stats-grid">
          <div className="stat-card stat-total">
            <div className="stat-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
            </div>
            <span className="stat-value">{stats.total}</span>
            <span className="stat-label">Total Complaints</span>
          </div>
          <div className="stat-card stat-pending">
            <div className="stat-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            </div>
            <span className="stat-value">{stats.pending}</span>
            <span className="stat-label">Pending Review</span>
          </div>
          <div className="stat-card stat-verified">
            <div className="stat-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </div>
            <span className="stat-value">{stats.verified}</span>
            <span className="stat-label">Verified</span>
          </div>
          <div className="stat-card stat-votes">
            <div className="stat-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
              </svg>
            </div>
            <span className="stat-value">{stats.total_votes || 0}</span>
            <span className="stat-label">Total Votes</span>
          </div>
          {(stats.by_category || []).slice(0,3).map(item => (
            <div key={item.category} className="stat-card stat-category">
              <span className="stat-value">{item.count}</span>
              <span className="stat-label">{item.category}</span>
            </div>
          ))}
        </div>
      )}

      {/* Map */}
      <div className="map-card">
        <div className="map-header">
          <h3>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" x2="9" y1="3" y2="18"/><line x1="15" x2="15" y1="6" y2="21"/></svg>
            Complaint Map
          </h3>
          <span style={{fontSize:'0.75rem',color:'#64748B',marginLeft:'auto'}}>
            {mapComplaints.length} active complaint{mapComplaints.length !== 1 ? 's' : ''} on map
          </span>
        </div>
        <div className="map-container">
          <MapContainer center={[12.9716, 77.5946]} zoom={11} className="map-container">
            <TileLayer attribution='&copy; <a href="https://carto.com/">CARTO</a>' url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" />
            {mapComplaints.map(c => {
              const statusColors = {
                'pending':     { fill: '#F59E0B', stroke: '#B45309' },
                'Verified':    { fill: '#0284C7', stroke: '#0369A1' },
                'In Progress': { fill: '#8B5CF6', stroke: '#6D28D9' },
                'Rejected':    { fill: '#6B7280', stroke: '#374151' },
              };
              const col = statusColors[(c.status || 'pending')] || statusColors['pending'];
              // Radius scales with vote count: base 8, max 22
              const voteRadius = Math.min(22, 8 + Math.round(((c.votes || 0) / Math.max(globalMaxVotes, 1)) * 14));
              return (
                <CircleMarker
                  key={`marker-${c.id}`}
                  center={[c.live_latitude, c.live_longitude]}
                  radius={voteRadius}
                  pathOptions={{ fillColor: col.fill, fillOpacity: 0.88, color: col.stroke, weight: (c.votes || 0) >= HIGH_THRESHOLD ? 3 : 2 }}
                >
                  <Popup>
                    <div className="map-popup">
                      <strong>{c.category}</strong>
                      <span style={{display:'block',marginTop:'2px',color:'#64748B',fontSize:'0.78rem'}}>{c.location}</span>
                      <span style={{display:'inline-block',marginTop:'4px',padding:'1px 7px',borderRadius:'999px',fontSize:'0.72rem',fontWeight:600,background: col.fill,color:'#fff'}}>{c.status || 'pending'}</span>
                      {c.votes > 0 && <span style={{marginLeft:'6px',fontSize:'0.72rem',color:'#64748B'}}>👍 {c.votes} votes</span>}
                      {(c.votes || 0) >= CRITICAL_THRESHOLD && <span style={{display:'block',marginTop:'4px',fontSize:'0.72rem',fontWeight:700,color:'#DC2626'}}>🔥 Critical Priority</span>}
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>
        </div>
      </div>

      {/* Filters */}
      <div className="filters-bar">
        <div className="filter-pills">
          {['all','pending','Verified','Rejected','In Progress','Resolved'].map(f => (
            <button key={f} className={`filter-pill ${filter === f ? 'active' : ''}`} onClick={() => setFilter(f)}>
              {f === 'all' ? 'All' : f}
            </button>
          ))}
        </div>
        <div className="filter-right">
          <select className="admin-filter-select" value={categoryFilter} onChange={e => setCategoryFilter(e.target.value)}>
            <option value="all">All Categories</option>
            {CATEGORY_OPTIONS.map(cat => <option key={cat} value={cat}>{cat}</option>)}
          </select>
          <select className="admin-filter-select" value={sortBy} onChange={e => setSortBy(e.target.value)}>
            <option value="most_voted">🔥 Most Voted (Priority)</option>
            <option value="latest">🕐 Latest First</option>
          </select>
          <label className="mismatch-filter-label" style={{display:'flex',alignItems:'center',gap:'0.3rem',fontSize:'0.85rem',cursor:'pointer'}}>
            <input type="checkbox" checked={mismatchFilter} onChange={e => setMismatchFilter(e.target.checked)} />
            ⚠️ Mismatches Only
          </label>
          <span className="results-count"><strong>{totalItems}</strong> total</span>
        </div>
      </div>

      {/* Top-Voted Priority Leaderboard */}
      {topVoted.length > 0 && (
        <div className="priority-leaderboard">
          <div className="leaderboard-header">
            <span className="leaderboard-title">🔥 Top Priority Complaints</span>
            <span className="leaderboard-subtitle">Sorted by community votes — highest urgency</span>
          </div>
          <div className="leaderboard-list">
            {topVoted.map((c, rank) => {
              const pct = Math.round(((c.votes || 0) / globalMaxVotes) * 100);
              const rankColors = ['#DC2626','#EA580C','#D97706'];
              return (
                <div key={c.id} className="leaderboard-item" onClick={() => document.getElementById(`complaint-${c.id}`)?.scrollIntoView({behavior:'smooth', block:'center'})}>
                  <div className="lb-rank" style={{background: rankColors[rank] || '#64748B'}}>#{rank + 1}</div>
                  <div className="lb-info">
                    <span className="lb-cat">{CAT_ICONS[c.category] || '📋'} {c.category}</span>
                    <span className="lb-loc">{c.location}</span>
                  </div>
                  <div className="lb-votes">
                    <div className="lb-vote-bar" style={{width:`${pct}%`, background: rankColors[rank] || '#64748B'}} />
                    <span className="lb-vote-count">👍 {c.votes}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Cards */}
      {filtered.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">📭</div>
          <h3>No complaints found</h3>
          <p>Try adjusting your filters or check back later.</p>
        </div>
      ) : (
        <div className="complaints-grid">
          {filtered.map(c => {
            const priority = getVotePriority(c.votes);
            const votePct  = Math.round(((c.votes || 0) / globalMaxVotes) * 100);
            const sCls = statusCls(c.status);
            return (
              <div key={c.id}
                className={`complaint-card ${sCls}-glow${priority === 'critical' ? ' priority-critical-card' : priority === 'high' ? ' priority-high-card' : ''}`}
                id={`complaint-${c.id}`}>
                <div className="cc-inner">
                  {/* Header */}
                  <div className="ccard-header">
                    <div className="cc-head-left">
                      <div className={`cc-id-icon ${sCls}-icon`}>{CAT_ICONS[c.category] || '📋'}</div>
                      <div>
                        <div className="ccard-id">#{c.id}</div>
                        <div style={{fontSize:'0.72rem',color:'var(--color-text-muted)',marginTop:'0.1rem'}}>{c.category}</div>
                      </div>
                    </div>
                    <div className="cc-head-right">
                      {priority === 'critical' && <span className="badge-priority badge-critical">🔥 Critical</span>}
                      {priority === 'high'     && <span className="badge-priority badge-high">⚠️ High Priority</span>}
                      {priority === 'medium'   && <span className="badge-priority badge-medium">📌 Medium</span>}
                      {(c.votes||0) > 0 && (
                        <span className={`vote-count-badge vote-${priority}`}>👍 {c.votes}</span>
                      )}
                      {c.category_mismatch && (
                        <span className="badge-warning tooltip" title={`Image suggests: ${c.image_suggested_category}`}>
                          ⚠️ Mismatch
                        </span>
                      )}
                      {c.pothole_severity && c.pothole_severity !== "Clear" && (
                        <span className={`severity-badge severity-${c.pothole_severity.toLowerCase()}`}>
                          {c.pothole_severity} Damage
                        </span>
                      )}
                      <StatusBadge status={c.status} />
                    </div>
                  </div>

                  {/* Vote Priority Bar */}
                  {(c.votes || 0) > 0 && (
                    <div className="vote-priority-row">
                      <span className="vote-priority-label">Community Votes</span>
                      <div className="vote-bar-track">
                        <div
                          className={`vote-bar-fill vote-fill-${priority}`}
                          style={{ width: `${votePct}%` }}
                        />
                      </div>
                      <span className="vote-bar-pct">{votePct}%</span>
                    </div>
                  )}

                  {/* Body */}
                  <div className="ccard-body">
                    <div className="ccard-field">
                      <span className="ccard-field-label">Category</span>
                      <div className="ccard-category-edit">
                        <select className="admin-filter-select" value={categoryEdits[c.id] || c.category || ''}
                          onChange={e => setCategoryEdits(p => ({...p,[c.id]:e.target.value}))}>
                          {CATEGORY_OPTIONS.map(cat => <option key={cat} value={cat}>{cat}</option>)}
                        </select>
                        <button className="btn btn-secondary btn-sm" onClick={() => handleCategoryUpdate(c)}
                          disabled={(categoryEdits[c.id] || c.category) === c.category}>Save</button>
                      </div>
                    </div>
                    <div className="ccard-field">
                      <span className="ccard-field-label">Location</span>
                      <span className="ccard-field-value">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>
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
                        <span className="ccard-field-value ccard-date">{fmtDate(c.created_at)}</span>
                      </div>
                    )}
                    {c.translated_text && <div className="ccard-transcript"><p>{c.translated_text}</p></div>}
                  </div>

                  {c.image_path && (
                    <div className="ccard-ai-section">
                      <div className="ai-panel-header" onClick={() => toggleAiPanel(c.id)}>
                        <span className="ai-panel-title">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2a10 10 0 1 0 10 10H12V2z"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
                          AI Analysis {aiPanelOpen[c.id] ? '▼' : '▶'}
                        </span>
                        <div className="ai-panel-badges">
                          {c.florence_analysis?.status === 'processing' && <span className="badge-info">⏳ Processing Image...</span>}
                          {c.cross_modal?.verification_result === 'mismatch' && <span className="badge-warning">⚠️ Cross-Modal Mismatch</span>}
                          {c.cross_modal?.verification_result === 'match' && <span className="badge-success">✓ Verified Match</span>}
                          {c.cross_modal?.verification_result === 'image_unclear' && <span className="badge-neutral">ℹ️ Image Unclear</span>}
                        </div>
                      </div>

                      {aiPanelOpen[c.id] && (
                        <div className="ai-panel-content">
                          <div className="ai-split-view">
                            <div className="ai-image-col">
                              <DetectionOverlay 
                                imageUrl={getAudioUrl(c.image_path, token)} 
                                alt={`Evidence #${c.id}`}
                                detections={c.detected_objects ? JSON.parse(c.detected_objects) : null}
                              />
                            </div>
                            <div className="ai-data-col">
                              <div className="ai-data-group">
                                <h4>Florence-2 Visual Understanding</h4>
                                {c.florence_analysis?.status === 'processing' ? (
                                  <p className="text-muted">Analysis running in background...</p>
                                ) : c.florence_analysis?.status === 'success' ? (
                                  <>
                                    <p><strong>Caption:</strong> {c.florence_analysis.caption}</p>
                                    <p><strong>Detected Object:</strong> {c.florence_analysis.damaged_object || 'None'}</p>
                                    <p><strong>Problem Type:</strong> {c.florence_analysis.problem_type || 'None'}</p>
                                    <p><strong>Severity:</strong> <span className={`severity-badge severity-${c.florence_analysis.severity?.toLowerCase()}`}>{c.florence_analysis.severity}</span></p>
                                    <p><strong>Evidence Text:</strong> <em>{c.florence_analysis.supporting_evidence}</em></p>
                                    {c.florence_analysis.all_suggested_categories?.length > 0 && (
                                      <div style={{ marginTop: '0.5rem' }}>
                                        <strong>Detected Categories:</strong>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginTop: '0.3rem' }}>
                                          {c.florence_analysis.all_suggested_categories.map((cat, idx) => (
                                            <span key={idx} className={`badge-info`} style={{
                                              padding: '0.2rem 0.5rem',
                                              borderRadius: '0.25rem',
                                              fontSize: '0.75rem',
                                              opacity: idx === 0 ? 1 : 0.75,
                                              fontWeight: idx === 0 ? '600' : '400',
                                            }}>
                                              {cat.category} ({cat.score})
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                    <p className="text-muted text-xs">Processed in {c.florence_analysis.processing_time}s</p>
                                  </>
                                ) : c.florence_analysis?.status === 'error' || c.florence_analysis?.status === 'timeout' ? (
                                  <>
                                    <p className="text-danger">Analysis failed ({c.florence_analysis.status})</p>
                                    <button className="btn btn-secondary btn-sm mt-2" onClick={() => handleReanalyze(c.id)}>
                                      ↻ Retry Analysis
                                    </button>
                                  </>
                                ) : (
                                  <>
                                    <p className="text-muted">No Florence-2 data available.</p>
                                    <button className="btn btn-secondary btn-sm mt-2" onClick={() => handleReanalyze(c.id)}>
                                      ↻ Run Florence-2
                                    </button>
                                  </>
                                )}
                              </div>
                              <div className="ai-data-group">
                                <h4>Cross-Modal Verification</h4>
                                <p><strong>NLP Category:</strong> {c.cross_modal?.nlp_category}</p>
                                <p><strong>Image Category:</strong> {c.cross_modal?.image_category || 'None'}</p>
                                <p><strong>System Trust Level:</strong> <span className={`trust-badge trust-${c.cross_modal?.trust_level?.replace('_', '-')}`}>{c.cross_modal?.trust_level}</span></p>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {c.audio_path && (
                    <div className="ccard-audio">
                      <button className="btn btn-secondary btn-sm" onClick={() => toggleAudio(c)}>
                        {playingId === c.id ? '⏸ Hide Player' : '▶ Play Audio'}
                      </button>
                      {playingId === c.id && (audioLoading[c.id]
                        ? <p className="ccard-audio-loading">Loading audio...</p>
                        : <audio controls autoPlay src={audioSources[c.id] || ''} className="ccard-audio-player" />
                      )}
                    </div>
                  )}

                  <div className="admin-note-block">
                    <textarea className="admin-note-input" placeholder="Add a note before updating status (optional)" rows={2}
                      value={adminNotes[c.id] || ''} onChange={e => setAdminNotes(p => ({...p,[c.id]:e.target.value}))} />
                  </div>
                </div>

                {/* Footer actions */}
                <div className="ccard-footer">
                  {c.status !== 'Verified' && c.status !== 'Rejected' && c.status !== 'In Progress' && c.status !== 'Resolved' && (
                    <button className="btn btn-success btn-sm" onClick={() => handleStatusUpdate(c.id,'Verified')}>✓ Verify</button>
                  )}
                  {c.status !== 'Rejected' && c.status !== 'In Progress' && c.status !== 'Resolved' && (
                    <button className="btn btn-danger btn-sm" onClick={() => handleStatusUpdate(c.id,'Rejected')}>✕ Reject</button>
                  )}
                  {c.status !== 'Rejected' && c.status !== 'In Progress' && c.status !== 'Resolved' && (
                    <button className="btn btn-info btn-sm" onClick={() => handleStatusUpdate(c.id,'In Progress')}>⏱ In Progress</button>
                  )}
                  {c.status !== 'Rejected' && c.status !== 'Resolved' && (
                    <button className="btn btn-resolved btn-sm" onClick={() => handleStatusUpdate(c.id,'Resolved')}>✔ Resolved</button>
                  )}
                  <button className="btn btn-ghost btn-sm" style={{marginLeft:'auto'}} onClick={() => toggleTimeline(c.id)}>
                    🕐 Timeline {tlOpen[c.id] ? '▲' : '▼'}
                  </button>
                </div>

                {/* Timeline */}
                {tlOpen[c.id] && (
                  <div className="admin-timeline-panel">
                    {tlLoading[c.id] ? (
                      <p className="ccard-audio-loading">Loading timeline…</p>
                    ) : (timelines[c.id]||[]).length === 0 ? (
                      <p className="ccard-audio-loading">No timeline entries yet.</p>
                    ) : (
                      <div className="admin-timeline">
                        {(timelines[c.id]||[]).map((entry,i) => (
                          <div key={i} className="admin-tl-entry">
                            <span className="admin-tl-status">{entry.status}</span>
                            <span className="admin-tl-time">{entry.created_at ? new Date(entry.created_at).toLocaleString('en-IN') : ''}</span>
                            {entry.note && <span className="admin-tl-note">{entry.note}</span>}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination">
          <button className="btn btn-secondary btn-sm" disabled={page <= 1} onClick={() => setPage(page-1)}>← Prev</button>
          <span className="page-info mono">Page {page} of {totalPages}</span>
          <button className="btn btn-secondary btn-sm" disabled={page >= totalPages} onClick={() => setPage(page+1)}>Next →</button>
        </div>
      )}
    </div>
  );
}
