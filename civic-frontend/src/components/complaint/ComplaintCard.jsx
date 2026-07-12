import { useState, useEffect } from 'react';
import { getComplaintTimeline } from '../../services/complaintService';
import { ComplaintTimeline } from './ComplaintTimeline';
import { VoteButton } from './VoteButton';
import { formatDate } from '../../utils/formatters';

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

export function StatusBadge({ status }) {
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

export function ComplaintCard({ complaint, onVote, showTimeline, onToggleTimeline }) {
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
      if (onVote) {
        const res = await onVote(complaint.id);
        setVotes(res.votes);
        localStorage.setItem(`_bbmp_voted_${complaint.id}`, '1');
        setVoted(true);
      }
    } catch { /* ignore */ }
    setVoting(false);
  };

  const catIcon = CATEGORY_ICONS[complaint.category] || '📋';
  const statusCfg = STATUS_CONFIG[complaint.status] || STATUS_CONFIG['pending'];
  const isResolved = complaint.status === 'Resolved';
  const text = complaint.translated_text || '';
  const truncated = text.length > 120 && !expanded;

  const getVotePriority = (v) => {
    if (v >= 5) return 'critical';
    if (v >= 3) return 'high';
    if (v >= 2) return 'medium';
    return 'none';
  };
  const priority = getVotePriority(votes);

  return (
    <div className={`uc-card ${isResolved ? 'uc-resolved' : ''} ${priority === 'critical' ? 'uc-card-critical' : priority === 'high' ? 'uc-card-high' : ''}`}>
      <div className="uc-card-accent" style={{ background: `linear-gradient(90deg, ${statusCfg.dot}, ${statusCfg.glow || statusCfg.dot})` }} />

      <div className="uc-card-head">
        <div className="uc-card-head-left">
          <span className="uc-card-cat-icon">{catIcon}</span>
          <div>
            <div className="uc-card-cat-name">{complaint.category}</div>
            <div className="uc-card-id">Complaint #{complaint.id}</div>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.4rem' }}>
          {priority === 'critical' && <span className="uc-priority-badge critical">🔥 Critical</span>}
          {priority === 'high'     && <span className="uc-priority-badge high">⚠️ High Priority</span>}
          {priority === 'medium'   && <span className="uc-priority-badge medium">📌 Medium</span>}
          <StatusBadge status={complaint.status} />
        </div>
      </div>

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

      <div className="uc-card-foot">
        <VoteButton voted={voted} voting={voting} votes={votes} onVote={handleVote} />

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

      {showTimeline && (
        <div className="uc-tl-panel">
          {tlLoading ? (
            <div className="uc-tl-skeleton">
              <div className="uc-skel-line short" style={{ margin: '0 auto' }} />
            </div>
          ) : (
            <ComplaintTimeline entries={timeline || []} currentStatus={complaint.status} />
          )}
        </div>
      )}
    </div>
  );
}
