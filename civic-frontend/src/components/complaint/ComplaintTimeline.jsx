import React from 'react';

const TIMELINE_STEPS = ['Reported', 'Verified', 'In Progress', 'Resolved'];

export function ComplaintTimeline({ entries, currentStatus }) {
  const activeIdx = TIMELINE_STEPS.indexOf(currentStatus);
  return (
    <div className="uc-h-timeline">
      {TIMELINE_STEPS.map((step, idx) => {
        const entry = entries.find(e => e.status === step);
        const isDone = idx <= activeIdx;
        const isCurrent = step === currentStatus;
        return (
          <div key={step} className={`uc-h-step ${isDone ? 'done' : ''} ${isCurrent ? 'current' : ''}`}>
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
              {entry ? (
                <div className="uc-h-time">
                  {new Date(entry.created_at).toLocaleDateString('en-IN', {
                    day: 'numeric', month: 'short',
                  })}
                </div>
              ) : (
                <div className="uc-h-time pending-time">Pending</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
