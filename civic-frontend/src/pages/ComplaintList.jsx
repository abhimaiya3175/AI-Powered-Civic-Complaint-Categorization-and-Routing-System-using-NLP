import { useState, useEffect, useCallback } from 'react';
import { useComplaint } from '../hooks/useComplaint';
import { ComplaintCard } from '../components/complaint/ComplaintCard';
import { getPublicComplaints, getResolvedComplaints } from '../services/complaintService';
import { getVoterFingerprint } from '../utils/storage';
import '../styles/UserComplaints.css';

const CATEGORIES = [
  'all', 'Street Light', 'Garbage / Sanitation', 'Road Repair',
  'Drainage / SWD', 'Water Supply', 'Health / Sanitation',
  'Parks', 'Parks / Forest', 'Town Planning', 'Veterinary',
  'Advertisement', 'Revenue', 'Traffic', 'Others',
];

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

export default function ComplaintList() {
  const [tab, setTab] = useState('active');
  const [page, setPage] = useState(1);
  const [category, setCategory] = useState('all');
  const [status, setStatus] = useState('all');
  const [sort, setSort] = useState('most_voted');
  const [openTimelineId, setOpenTimelineId] = useState(null);
  const [stats, setStats] = useState({ total: 0, active: 0, resolved: 0, totalVotes: 0 });

  const {
    complaints,
    loading,
    totalPages,
    fetchPublicComplaints,
    fetchResolvedComplaints,
    handleVote
  } = useComplaint();

  const voterFp = getVoterFingerprint();

  const load = useCallback(() => {
    if (tab === 'active') {
      fetchPublicComplaints({ page, size: 12, category, status, sort });
    } else {
      fetchResolvedComplaints({ page, size: 12, category });
    }
  }, [tab, page, category, status, sort, fetchPublicComplaints, fetchResolvedComplaints]);

  useEffect(() => {
    queueMicrotask(() => { load(); });
  }, [load]);

  const loadStats = useCallback(async () => {
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
  }, [voterFp]);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  useEffect(() => {
    queueMicrotask(() => setPage(1));
  }, [tab, category, status, sort]);

  const onVote = async (id) => {
    const res = await handleVote(id);
    loadStats();
    return res;
  };

  return (
    <div className="uc-page gravless-container">
      <div className="uc-hero">
        <div className="uc-hero-content">
          <div className="uc-hero-badge">🏙️ BBMP Civic Portal</div>
          <h1 className="uc-hero-title">Civic Issues in Bengaluru</h1>
          <p className="uc-hero-sub">Browse complaints, track progress, and upvote issues in your neighbourhood</p>
        </div>
        <div className="uc-stats-row">
          <StatCard icon="📊" label="Total Complaints" value={stats.total} color="#0284C7" />
          <StatCard icon="🔄" label="Active Issues" value={stats.active} color="#F59E0B" />
          <StatCard icon="✅" label="Resolved" value={stats.resolved} color="#10B981" />
          <StatCard icon="👍" label="Total Votes" value={stats.totalVotes} color="#8B5CF6" />
        </div>
      </div>

      <div className="uc-controls">
        <div className="uc-tabs">
          <button className={`uc-tab ${tab === 'active' ? 'active' : ''}`} onClick={() => setTab('active')}>Active Issues</button>
          <button className={`uc-tab ${tab === 'resolved' ? 'active' : ''}`} onClick={() => setTab('resolved')}>Resolved</button>
        </div>
        <div className="uc-filters">
          <select value={category} onChange={e => setCategory(e.target.value)} className="uc-select">
            {CATEGORIES.map(c => <option key={c} value={c}>{c === 'all' ? 'All Categories' : c}</option>)}
          </select>
          {tab === 'active' && (
            <select value={status} onChange={e => setStatus(e.target.value)} className="uc-select">
              <option value="all">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="Verified">Verified</option>
              <option value="In Progress">In Progress</option>
            </select>
          )}
          {tab === 'active' && (
            <select value={sort} onChange={e => setSort(e.target.value)} className="uc-select">
              <option value="latest">Latest First</option>
              <option value="most_voted">Most Voted</option>
            </select>
          )}
        </div>
      </div>

      {loading ? (
        <div className="uc-grid">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      ) : complaints.length === 0 ? (
        <div className="uc-empty">
          <div className="uc-empty-icon">📭</div>
          <h3>No complaints found</h3>
          <p>Try adjusting your filters or category.</p>
        </div>
      ) : (
        <div className="uc-grid">
          {complaints.map(c => (
            <ComplaintCard
              key={c.id}
              complaint={c}
              onVote={onVote}
              showTimeline={openTimelineId === c.id}
              onToggleTimeline={(id) => setOpenTimelineId(prev => prev === id ? null : id)}
            />
          ))}
        </div>
      )}

      {totalPages > 1 && (
        <div className="uc-pagination">
          <button disabled={page === 1} onClick={() => setPage(p => p - 1)} className="uc-page-btn">← Prev</button>
          <span className="uc-page-info">Page {page} of {totalPages}</span>
          <button disabled={page >= totalPages} onClick={() => setPage(p => p + 1)} className="uc-page-btn">Next →</button>
        </div>
      )}
    </div>
  );
}
