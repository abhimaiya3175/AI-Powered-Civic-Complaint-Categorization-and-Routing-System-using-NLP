import { API_BASE } from './api';

/**
 * Fetch paginated complaints (requires JWT token).
 */
export const getComplaints = async (token, page = 1, size = 10, categoryMismatch = false) => {
  let url = `${API_BASE}/complaints?page=${page}&size=${size}`;
  if (categoryMismatch) {
    url += `&category_mismatch=true`;
  }
  const response = await fetch(
    url,
    {
      headers: { Authorization: `Bearer ${token}` },
    }
  );

  if (!response.ok) {
    if (response.status === 401) throw new Error('401 Unauthorized');
    throw new Error('Failed to fetch complaints');
  }

  return response.json();
};

/**
 * Fetch all non-resolved complaints with GPS coordinates for the map view (requires JWT token).
 */
export const getMapComplaints = async (token) => {
  const response = await fetch(`${API_BASE}/complaints/map`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    if (response.status === 401) throw new Error('401 Unauthorized');
    throw new Error('Failed to fetch map complaints');
  }
  return response.json();
};

/**
 * Verify / edit a complaint (HITL — requires JWT token).
 */
export const verifyComplaint = async (token, complaintId, data = {}) => {
  const response = await fetch(
    `${API_BASE}/complaints/${complaintId}/verify`,
    {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        status: 'Verified',
        ...data,
      }),
    }
  );

  if (!response.ok) {
    if (response.status === 401) throw new Error('401 Unauthorized');
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to verify complaint');
  }

  return response.json();
};

/**
 * Get public (unauthenticated) complaint listing.
 */
export const getPublicComplaints = async ({ page = 1, size = 12, category = '', status = '', sort = 'latest', voterFingerprint = '' } = {}) => {
  const params = new URLSearchParams({ page, size, sort });
  if (category && category !== 'all') params.set('category', category);
  if (status && status !== 'all') params.set('status', status);
  if (voterFingerprint) params.set('voter_fingerprint', voterFingerprint);
  const response = await fetch(`${API_BASE}/complaints/public?${params}`);
  if (!response.ok) throw new Error('Failed to load complaints');
  return response.json();
};

/**
 * Get resolved complaints archive (unauthenticated).
 */
export const getResolvedComplaints = async ({ page = 1, size = 12, category = '', voterFingerprint = '' } = {}) => {
  const params = new URLSearchParams({ page, size });
  if (category && category !== 'all') params.set('category', category);
  if (voterFingerprint) params.set('voter_fingerprint', voterFingerprint);
  const response = await fetch(`${API_BASE}/complaints/resolved?${params}`);
  if (!response.ok) throw new Error('Failed to load resolved complaints');
  return response.json();
};

/**
 * Upvote a complaint (unauthenticated, fingerprint-based dedup).
 */
export const voteComplaint = async (id, voterFingerprint) => {
  const response = await fetch(`${API_BASE}/complaints/${id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ voter_fingerprint: voterFingerprint }),
  });
  if (!response.ok) throw new Error('Failed to vote');
  return response.json();
};

/**
 * Fetch the complaint status timeline.
 */
export const getComplaintTimeline = async (id) => {
  const response = await fetch(`${API_BASE}/complaints/${id}/timeline`);
  if (!response.ok) throw new Error('Failed to load timeline');
  return response.json();
};
