import { useState, useEffect, useCallback } from 'react';
import {
  getPublicComplaints,
  getResolvedComplaints,
  voteComplaint,
  getComplaintTimeline,
  getVoterFingerprint,
} from '../services/api';
import '../styles/UserComplaints.css';

const CATEGORIES = [
  'all', 'Street Light', 'Garbage / Sanitation', 'Road Repair',
  'Drainage / SWD', 'Water Supply', 'Health / Sanitation',
  'Parks', 'Parks / Forest', 'Town Planning', 'Veterinary',
  'Advertisement', 'Revenue', 'Traffic', 'Others',
];

const STATUS_COLORS = {
  pending: { bg: '#FEE2E2', color: '#DC2626', label: 'Pending' },
  Verified: { bg: '#FEF9C3', color: '#D97706', label: 'Verified' },
  'In Progress': { bg: '#DBEAFE', color: '#2563EB', label: 'In Progress' },
  Resolved: { bg: '#DCFCE7', color: '#16A34A', label: 'Resolved' },
};

const TIMELINE_STATUSES = ['Reported', 'Verified', 'In Progress', 'Resolved'];

function StatusBadge({ status }) {
  const s = STATUS_COLORS[status] || STATUS_COLORS['pending'];
  return (
    <span className="uc-badge" style={{ background: s.bg, color: s.color }}>
      {s.label || status}
    </span>
  );
}

function Timeline({ entries, currentStatus }) {
  const activeIdx = TIMELINE_STATUSES.indexOf(currentStatus);
  return (
    <div className="uc-timeline">
      {TIMELINE_STATUSES.map((step, idx) => {
        const entry = entries.find((e) => e.status === step);
        const isDone = idx <= activeIdx;
        const isCurrent = step === currentStatus;
        return (
          <div key={step} className={`uc-tl-step ${isDone ? 'done' : ''} ${isCurrent ? 'current' : ''}`}>
            <div className="uc-tl-dot-col">
              <div className="uc-tl-dot" />
              {idx < TIMELINE_STATUSES.length - 1 && <div className="uc-tl-line" />}
            </div>
            <div className="uc-tl-content">
              <span className="uc-tl-label">{step}</span>
              {entry && (
                <>
                  <span className="uc-tl-time">
                    {new Date(entry.created_at).toLocaleString('en-IN', {
                      day: 'numeric', month: 'short', year: 'numeric',
                      hour: '2-digit', minute: '2-digit',
                    })}
                  </span>
                  {entry.note && <span className="uc-tl-note">{entry.note}</span>}
                </>
              )}
              {!entry && <span className="uc-tl-time" style={{ color: '#CBD5E1' }}>Not yet</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ComplaintCard({ complaint, voterFp, onVote, showTimeline, onToggleTimeline }) {
  const [votes, setVotes] = useState(complaint.votes || 0);
  const [voted, setVoted] = useState(false);
  const [voting, setVoting] = useState(false);
  const [timeline, setTimeline] = useState(null);
  const [tlLoading, setTlLoading] = useState(false);

  // Check local storage for existing vote
  useEffect(() => {
    const key = `_bbmp_voted_${complaint.id}`;
    if (localStorage.getItem(key)) setVoted(true);
  }, [complaint.id]);

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
      if (!res.already_voted) {
        localStorage.setItem(`_bbmp_voted_${complaint.id}`, '1');
        setVoted(true);
      }
      if (onVote) onVote(complaint.id, res.votes);
    } catch { /* ignore */ }
    setVoting(false);
  };

  const formatDate = (d) => {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
  };

  return (
    <div className={`uc-card ${complaint.status === 'Resolved' ? 'uc-resolved' : ''}`}>
      {/* Card Header */}
      <div className="uc-card-head">
        <span className="uc-card-id">#{complaint.id}</span>
        <StatusBadge status={complaint.status} />
      </div>

      {/* Category + Location */}
      <div className="uc-card-meta">
        <span className="uc-cat-badge">{complaint.category}</span>
        {complaint.location && (
          <span className="uc-location">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>
            </svg>
            {complaint.location}
          </span>
        )}
      </div>

      {/* Complaint text */}
      {complaint.translated_text && (
        <p className="uc-card-text">{complaint.translated_text}</p>
      )}

      {/* Footer: vote + date + timeline toggle */}
      <div className="uc-card-foot">
        <div className="uc-vote-row">
          <button
            className={`uc-vote-btn ${voted ? 'voted' : ''}`}
            onClick={handleVote}
            disabled={voted || voting}
            title={voted ? 'Already voted' : 'Upvote this complaint'}
          >
            👍 {votes}
          </button>
          {voted && <span className="uc-voted-label">Voted</span>}
        </div>
        <div className="uc-foot-right">
          <span className="uc-date">{formatDate(complaint.created_at)}</span>
          <button className="uc-tl-toggle" onClick={() => onToggleTimeline(complaint.id)}>
            {showTimeline ? 'Hide Timeline' : 'Timeline'}
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points={showTimeline ? '18 15 12 9 6 15' : '6 9 12 15 18 9'} />
            </svg>
          </button>
        </div>
      </div>

      {/* Timeline panel */}
      {showTimeline && (
        <div className="uc-tl-panel">
          {tlLoading ? (
            <p className="uc-tl-loading">Loading timeline…</p>
          ) : (
            <Timeline entries={timeline || []} currentStatus={complaint.status} />
          )}
        </div>
      )}
    </div>
  );
}

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
  const voterFp = getVoterFingerprint();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      let data;
      if (tab === 'active') {
        data = await getPublicComplaints({ page, size: 12, category, status, sort });
      } else {
        data = await getResolvedComplaints({ page, size: 12, category });
      }
      setComplaints(data.items || []);
      setPages(data.pages || 1);
      setTotal(data.total || 0);
    } catch {
      setComplaints([]);
    }
    setLoading(false);
  }, [tab, page, category, status, sort]);

  useEffect(() => { load(); }, [load]);

  // Reset page when filters change
  useEffect(() => { setPage(1); }, [tab, category, status, sort]);

  const handleVote = (id, newVotes) => {
    setComplaints((prev) =>
      prev.map((c) => c.id === id ? { ...c, votes: newVotes } : c)
    );
  };

  return (
    <div className="uc-page gravless-container">
      {/* Page header */}
      <div className="uc-header">
        <div>
          <h2>Civic Issues in Bengaluru</h2>
          <p className="uc-subtitle">Browse complaints, track progress, and upvote issues near you</p>
        </div>
        <div className="uc-tabs">
          <button
            className={`uc-tab ${tab === 'active' ? 'active' : ''}`}
            onClick={() => setTab('active')}
            id="tab-active"
          >
            Active Issues
          </button>
          <button
            className={`uc-tab ${tab === 'resolved' ? 'active' : ''}`}
            onClick={() => setTab('resolved')}
            id="tab-resolved"
          >
            ✓ Resolved
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="uc-filters">
        <select
          className="uc-select"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          id="filter-category"
        >
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>{c === 'all' ? 'All Categories' : c}</option>
          ))}
        </select>

        {tab === 'active' && (
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
        )}

        {tab === 'active' && (
          <select
            className="uc-select"
            value={sort}
            onChange={(e) => setSort(e.target.value)}
            id="sort-select"
          >
            <option value="latest">Latest First</option>
            <option value="most_voted">Most Voted</option>
          </select>
        )}

        <span className="uc-total">{total} complaints</span>
      </div>

      {/* Grid */}
      {loading ? (
        <div className="uc-loading">
          <div className="uc-spinner" />
          <p>Loading complaints…</p>
        </div>
      ) : complaints.length === 0 ? (
        <div className="uc-empty">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#CBD5E1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <p>No complaints found</p>
        </div>
      ) : (
        <div className="uc-grid">
          {complaints.map((c) => (
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

      {/* Pagination */}
      {pages > 1 && (
        <div className="uc-pagination">
          <button className="btn btn-secondary btn-sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>← Prev</button>
          <span className="uc-page-info">Page {page} of {pages}</span>
          <button className="btn btn-secondary btn-sm" disabled={page >= pages} onClick={() => setPage(page + 1)}>Next →</button>
        </div>
      )}
    </div>
  );
}
