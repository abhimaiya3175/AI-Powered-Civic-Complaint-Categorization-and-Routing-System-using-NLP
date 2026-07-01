import { useState, useEffect, useCallback } from 'react';
import {
  getPublicComplaints,
  getResolvedComplaints,
  voteComplaint,
  getComplaintTimeline,
  getVoterFingerprint,
} from '../services/api';
import '../styles/UserComplaints.css';

/* ── Constants ───────────────────────────────────────────────── */
const CATEGORIES = [
  'all', 'Street Light', 'Garbage / Sanitation', 'Road Repair',
  'Drainage / SWD', 'Water Supply', 'Health / Sanitation',
  'Parks', 'Parks / Forest', 'Town Planning', 'Veterinary',
  'Advertisement', 'Revenue', 'Traffic', 'Others',
];

const CATEGORY_ICONS = {
  'Street Light': '💡',
  'Garbage / Sanitation': '🗑️',
  'Road Repair': '🛣️',
  'Drainage / SWD': '🌊',
  'Water Supply': '💧',
  'Health / Sanitation': '🏥',
  'Parks': '🌳',
  'Parks / Forest': '🌲',
  'Town Planning': '🏙️',
  'Veterinary': '🐾',
  'Advertisement': '📢',
  'Revenue': '💰',
  'Traffic': '🚦',
  'Others': '📋',
};

const STATUS_CONFIG = {
  pending:      { bg: 'rgba(239,68,68,0.1)',  color: '#DC2626', border: 'rgba(239,68,68,0.3)',  label: 'Pending',     dot: '#EF4444', glow: 'rgba(239,68,68,0.4)'  },
  Verified:     { bg: 'rgba(245,158,11,0.1)', color: '#D97706', border: 'rgba(245,158,11,0.3)', label: 'Verified',    dot: '#F59E0B', glow: 'rgba(245,158,11,0.4)' },
  'In Progress':{ bg: 'rgba(59,130,246,0.1)', color: '#2563EB', border: 'rgba(59,130,246,0.3)', label: 'In Progress', dot: '#3B82F6', glow: 'rgba(59,130,246,0.4)' },
  Resolved:     { bg: 'rgba(16,185,129,0.1)', color: '#059669', border: 'rgba(16,185,129,0.3)', label: 'Resolved',    dot: '#10B981', glow: 'rgba(16,185,129,0.4)' },
};

const TIMELINE_STEPS = ['Reported', 'Verified', 'In Progress', 'Resolved'];

/* ── Sub-Components ──────────────────────────────────────────── */

function StatusBadge({ status }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG['pending'];
  return (
    <span
      className="uc-badge"
      style={{
        background: cfg.bg,
        color: cfg.color,
        border: `1px solid ${cfg.border}`,
        boxShadow: `0 0 8px ${cfg.glow}`,
      }}
    >
      <span className="uc-badge-dot" style={{ background: cfg.dot, boxShadow: `0 0 4px ${cfg.dot}` }} />
      {cfg.label}
    </span>
  );
}

function HorizontalTimeline({ entries, currentStatus }) {
  const activeIdx = TIMELINE_STEPS.indexOf(currentStatus);
  return (
    <div className="uc-h-timeline">
      {TIMELINE_STEPS.map((step, idx) => {
        const entry = entries.find(e => e.status === step);
        const isDone = idx <= activeIdx;
        const isCurrent = step === currentStatus;
        return (
          <div key={step} className={`uc-h-step ${isDone ? 'done' : ''} ${isCurrent ? 'current' : ''}`}>
            {/* Connector line before */}
            {idx > 0 && (
              <div className={`uc-h-connector ${isDone ? 'done' : ''}`} />
            )}
            <div className="uc-h-node">
              <div className="uc-h-dot">
                {isDone && (
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </div>
              <div className="uc-h-label">{step}</div>
              {entry && (
                <div className="uc-h-time">
                  {new Date(entry.created_at).toLocaleDateString('en-IN', {
                    day: 'numeric', month: 'short',
                  })}
                </div>
              )}
              {!entry && <div className="uc-h-time pending-time">Pending</div>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="uc-skeleton-card">
      <div className="uc-skel-head">
        <div className="uc-skel-line short" />
        <div className="uc-skel-pill" />
      </div>
      <div className="uc-skel-line medium" />
      <div className="uc-skel-line long" />
      <div className="uc-skel-line medium" />
      <div className="uc-skel-foot">
        <div className="uc-skel-pill small" />
        <div className="uc-skel-line xs" />
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, color }) {
  return (
    <div className="uc-stat-card" style={{ '--stat-color': color }}>
      <div className="uc-stat-icon">{icon}</div>
      <div className="uc-stat-body">
        <div className="uc-stat-value">{value}</div>
        <div className="uc-stat-label">{label}</div>
      </div>
    </div>
  );
}

function ComplaintCard({ complaint, voterFp, onVote, showTimeline, onToggleTimeline }) {
  const [votes, setVotes] = useState(complaint.votes || 0);
  const [voted, setVoted] = useState(() => 
    Boolean(localStorage.getItem(`_bbmp_voted_${complaint.id}`)) || Boolean(complaint.voted)
  );
  const [voting, setVoting] = useState(false);
  const [timeline, setTimeline] = useState(null);
  const [tlLoading, setTlLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setVotes(complaint.votes || 0);
  }, [complaint.votes]);

  useEffect(() => {
    setVoted(Boolean(localStorage.getItem(`_bbmp_voted_${complaint.id}`)) || Boolean(complaint.voted));
  }, [complaint.id, complaint.voted]);

  useEffect(() => {
    if (complaint.voted && !localStorage.getItem(`_bbmp_voted_${complaint.id}`)) {
      localStorage.setItem(`_bbmp_voted_${complaint.id}`, '1');
    }
  }, [complaint.id, complaint.voted]);

  useEffect(() => {
    if (showTimeline && !timeline) {
      const fetchTimeline = async () => {
        setTlLoading(true);
        try {
          const data = await getComplaintTimeline(complaint.id);
          setTimeline(data.timeline || []);
        } catch { setTimeline([]); }
        setTlLoading(false);
      };
      fetchTimeline();
    }
  }, [showTimeline, timeline, complaint.id]);

  const handleVote = async () => {
    if (voted || voting) return;
    setVoting(true);
    try {
      const res = await voteComplaint(complaint.id, voterFp);
      setVotes(res.votes);
      localStorage.setItem(`_bbmp_voted_${complaint.id}`, '1');
      setVoted(true);
      if (onVote) onVote(complaint.id, res.votes);
    } catch { /* ignore */ }
    setVoting(false);
  };

  const formatDate = (d) => {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
  };

  const catIcon = CATEGORY_ICONS[complaint.category] || '📋';
  const statusCfg = STATUS_CONFIG[complaint.status] || STATUS_CONFIG['pending'];
  const isResolved = complaint.status === 'Resolved';
  const text = complaint.translated_text || '';
  const truncated = text.length > 120 && !expanded;

  return (
    <div className={`uc-card ${isResolved ? 'uc-resolved' : ''}`}>
      {/* Shimmer accent bar */}
      <div className="uc-card-accent" style={{ background: `linear-gradient(90deg, ${statusCfg.dot}, ${statusCfg.glow || statusCfg.dot})` }} />

      {/* Card Header */}
      <div className="uc-card-head">
        <div className="uc-card-head-left">
          <span className="uc-card-cat-icon">{catIcon}</span>
          <div>
            <div className="uc-card-cat-name">{complaint.category}</div>
            <div className="uc-card-id">Complaint #{complaint.id}</div>
          </div>
        </div>
        <StatusBadge status={complaint.status} />
      </div>

      {/* Location & Date row */}
      <div className="uc-card-meta-row">
        {complaint.location && (
          <div className="uc-meta-chip">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>
            </svg>
            {complaint.location}
          </div>
        )}
        <div className="uc-meta-chip date-chip">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>
          </svg>
          {formatDate(complaint.created_at)}
        </div>
      </div>

      {/* Complaint Text */}
      {text && (
        <div className="uc-card-text-wrap">
          <p className="uc-card-text">
            {truncated ? text.slice(0, 120) + '…' : text}
          </p>
          {text.length > 120 && (
            <button className="uc-read-more" onClick={() => setExpanded(e => !e)}>
              {expanded ? 'Show less ↑' : 'Read more ↓'}
            </button>
          )}
        </div>
      )}

      {/* Footer */}
      <div className="uc-card-foot">
        <button
          className={`uc-vote-btn ${voted ? 'voted' : ''} ${voting ? 'voting' : ''}`}
          onClick={handleVote}
          disabled={voted || voting}
          title={voted ? 'Already upvoted' : 'Upvote this issue'}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill={voted ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/>
          </svg>
          <span className="uc-vote-count">{votes}</span>
          {voted && <span className="uc-voted-label">Voted!</span>}
        </button>

        <button
          className={`uc-tl-toggle ${showTimeline ? 'open' : ''}`}
          onClick={() => onToggleTimeline(complaint.id)}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          Timeline
          <svg className="uc-toggle-caret" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points={showTimeline ? '18 15 12 9 6 15' : '6 9 12 15 18 9'} />
          </svg>
        </button>
      </div>

      {/* Timeline Panel */}
      {showTimeline && (
        <div className="uc-tl-panel">
          {tlLoading ? (
            <div className="uc-tl-skeleton">
              <div className="uc-skel-line short" style={{ margin: '0 auto' }} />
            </div>
          ) : (
            <HorizontalTimeline entries={timeline || []} currentStatus={complaint.status} />
          )}
        </div>
      )}
    </div>
  );
}

/* ── Main Component ──────────────────────────────────────────── */
export default function UserComplaints() {
  const [tab, setTab] = useState('active');
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [category, setCategory] = useState('all');
  const [status, setStatus] = useState('all');
  const [sort, setSort] = useState('latest');
  const [openTimelineId, setOpenTimelineId] = useState(null);
  const [stats, setStats] = useState({ total: 0, active: 0, resolved: 0, totalVotes: 0 });
  const voterFp = getVoterFingerprint();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      let data;
      if (tab === 'active') {
        data = await getPublicComplaints({ page, size: 12, category, status, sort, voterFingerprint: voterFp });
      } else {
        data = await getResolvedComplaints({ page, size: 12, category, voterFingerprint: voterFp });
      }
      setComplaints(data.items || []);
      setPages(data.pages || 1);
      setTotal(data.total || 0);
    } catch {
      setComplaints([]);
    }
    setLoading(false);
  }, [tab, page, category, status, sort, voterFp]);

  useEffect(() => {
    queueMicrotask(() => { load(); });
  }, [load]);

  // Load stats from both endpoints
  useEffect(() => {
    const loadStats = async () => {
      try {
        const [activeData, resolvedData] = await Promise.all([
          getPublicComplaints({ page: 1, size: 1, voterFingerprint: voterFp }),
          getResolvedComplaints({ page: 1, size: 1, voterFingerprint: voterFp }),
        ]);
        const activeTotal = activeData.total || 0;
        const resolvedTotal = resolvedData.total || 0;
        const activeVotes = activeData.total_votes || 0;
        const resolvedVotes = resolvedData.total_votes || 0;
        setStats({
          total: activeTotal + resolvedTotal,
          active: activeTotal,
          resolved: resolvedTotal,
          totalVotes: activeVotes + resolvedVotes,
        });
      } catch { /* ignore */ }
    };
    loadStats();
  }, [voterFp]);

  useEffect(() => {
    queueMicrotask(() => setPage(1));
  }, [tab, category, status, sort]);

  const handleVote = (id, newVotes) => {
    setComplaints(prev => prev.map(c => c.id === id ? { ...c, votes: newVotes } : c));
    // Proactively refresh stats to reflect the new vote immediately
    const loadStats = async () => {
      try {
        const [activeData, resolvedData] = await Promise.all([
          getPublicComplaints({ page: 1, size: 1, voterFingerprint: voterFp }),
          getResolvedComplaints({ page: 1, size: 1, voterFingerprint: voterFp }),
        ]);
        const activeTotal = activeData.total || 0;
        const resolvedTotal = resolvedData.total || 0;
        const activeVotes = activeData.total_votes || 0;
        const resolvedVotes = resolvedData.total_votes || 0;
        setStats({
          total: activeTotal + resolvedTotal,
          active: activeTotal,
          resolved: resolvedTotal,
          totalVotes: activeVotes + resolvedVotes,
        });
      } catch { /* ignore */ }
    };
    loadStats();
  };

  return (
    <div className="uc-page gravless-container">

      {/* ── Page Hero ── */}
      <div className="uc-hero">
        <div className="uc-hero-content">
          <div className="uc-hero-badge">🏙️ BBMP Civic Portal</div>
          <h1 className="uc-hero-title">Civic Issues in Bengaluru</h1>
          <p className="uc-hero-sub">Browse complaints, track progress, and upvote issues in your neighbourhood</p>
        </div>

        {/* Stats Row */}
        <div className="uc-stats-row">
          <StatCard icon="📊" label="Total Complaints" value={stats.total} color="#0284C7" />
          <StatCard icon="🔄" label="Active Issues" value={stats.active} color="#F59E0B" />
          <StatCard icon="✅" label="Resolved" value={stats.resolved} color="#10B981" />
          <StatCard icon="👍" label="Total Votes" value={stats.totalVotes} color="#8B5CF6" />
        </div>
      </div>

      {/* ── Controls Bar ── */}
      <div className="uc-controls">
        {/* Tabs */}
        <div className="uc-tabs">
          <button
            className={`uc-tab ${tab === 'active' ? 'active' : ''}`}
            onClick={() => setTab('active')}
            id="tab-active"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>
            </svg>
            Active Issues
          </button>
          <button
            className={`uc-tab ${tab === 'resolved' ? 'active' : ''}`}
            onClick={() => setTab('resolved')}
            id="tab-resolved"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
            Resolved
          </button>
        </div>

        {/* Filters */}
        <div className="uc-filters">
          <div className="uc-filter-group">
            <svg className="uc-filter-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
            <select
              className="uc-select"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              id="filter-category"
            >
              {CATEGORIES.map(c => (
                <option key={c} value={c}>{c === 'all' ? 'All Categories' : `${CATEGORY_ICONS[c] || ''} ${c}`}</option>
              ))}
            </select>
          </div>

          {tab === 'active' && (
            <div className="uc-filter-group">
              <svg className="uc-filter-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              <select
                className="uc-select"
                value={status}
                onChange={(e) => setStatus(e.target.value)}
                id="filter-status"
              >
                <option value="all">All Statuses</option>
                <option value="pending">Pending</option>
                <option value="Verified">Verified</option>
                <option value="In Progress">In Progress</option>
              </select>
            </div>
          )}

          {tab === 'active' && (
            <div className="uc-filter-group">
              <svg className="uc-filter-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M7 12h10M11 18h2"/></svg>
              <select
                className="uc-select"
                value={sort}
                onChange={(e) => setSort(e.target.value)}
                id="sort-select"
              >
                <option value="latest">Latest First</option>
                <option value="most_voted">Most Voted</option>
              </select>
            </div>
          )}

          <span className="uc-total">
            <strong>{total}</strong> complaints
          </span>
        </div>
      </div>

      {/* ── Grid ── */}
      {loading ? (
        <div className="uc-grid">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      ) : complaints.length === 0 ? (
        <div className="uc-empty">
          <div className="uc-empty-icon">📭</div>
          <h3>No complaints found</h3>
          <p>Try adjusting your filters or check back later.</p>
        </div>
      ) : (
        <div className="uc-grid">
          {complaints.map(c => (
            <ComplaintCard
              key={c.id}
              complaint={c}
              voterFp={voterFp}
              onVote={handleVote}
              showTimeline={openTimelineId === c.id}
              onToggleTimeline={(id) => setOpenTimelineId(prev => prev === id ? null : id)}
            />
          ))}
        </div>
      )}

      {/* ── Pagination ── */}
      {pages > 1 && (
        <div className="uc-pagination">
          <button
            className="uc-page-btn"
            disabled={page <= 1}
            onClick={() => setPage(page - 1)}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="15 18 9 12 15 6"/></svg>
            Prev
          </button>

          <div className="uc-page-dots">
            {Array.from({ length: Math.min(pages, 7) }, (_, i) => {
              const p = i + 1;
              return (
                <button
                  key={p}
                  className={`uc-page-dot ${p === page ? 'active' : ''}`}
                  onClick={() => setPage(p)}
                >
                  {p}
                </button>
              );
            })}
            {pages > 7 && <span className="uc-page-ellipsis">…</span>}
          </div>

          <button
            className="uc-page-btn"
            disabled={page >= pages}
            onClick={() => setPage(page + 1)}
          >
            Next
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="9 18 15 12 9 6"/></svg>
          </button>
        </div>
      )}
    </div>
  );
}
