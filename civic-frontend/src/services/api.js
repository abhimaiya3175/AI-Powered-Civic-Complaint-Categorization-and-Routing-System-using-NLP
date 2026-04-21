const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

/**
 * Submit a complaint with mandatory live location and optional media evidence.
 */
export const submitComplaint = async ({
  audioFile,
  imageFile,
  liveLatitude,
  liveLongitude,
  liveLocationTimestamp,
  textNote,
  language,
  targetLanguage,
  voterFingerprint,
}) => {
  const formData = new FormData();
  if (audioFile) {
    formData.append('file', audioFile);
  }
  if (imageFile) {
    formData.append('image', imageFile);
  }
  if (textNote) {
    formData.append('text_note', textNote);
  }
  if (voterFingerprint) {
    formData.append('voter_fingerprint', voterFingerprint);
  }
  formData.append('live_latitude', String(liveLatitude));
  formData.append('live_longitude', String(liveLongitude));
  formData.append('live_location_timestamp', liveLocationTimestamp);
  formData.append('language', language || 'en');
  formData.append('target_language', targetLanguage || 'en');

  const response = await fetch(`${API_BASE}/submit-complaint`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let detail = '';
    const contentType = (response.headers.get('content-type') || '').toLowerCase();

    if (contentType.includes('application/json')) {
      const err = await response.json().catch(() => ({}));
      detail = err.detail || '';
    } else {
      detail = (await response.text().catch(() => '')).trim();
    }

    if (!detail) {
      detail = response.status >= 500
        ? `Server error (${response.status}). Please try again.`
        : `Request failed (${response.status}).`;
    }

    throw new Error(detail);
  }

  return response.json();
};

/**
 * Fetch paginated complaints (requires JWT token).
 */
export const getComplaints = async (token, page = 1, size = 10) => {
  const response = await fetch(
    `${API_BASE}/complaints?page=${page}&size=${size}`,
    {
      headers: { Authorization: `Bearer ${token}` },
    }
  );

  if (!response.ok) {
    throw new Error('Failed to fetch complaints');
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
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to verify complaint');
  }

  return response.json();
};

/**
 * Fetch complaint statistics (requires JWT token).
 */
export const getStats = async (token) => {
  const response = await fetch(`${API_BASE}/complaints/stats`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    throw new Error('Failed to fetch statistics');
  }

  return response.json();
};

/**
 * Login and get JWT access token.
 */
export const loginAdmin = async (username, password) => {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const response = await fetch(`${API_BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Login failed');
  }

  return response.json();
};

/**
 * Build the full URL for an audio file so <audio> can play it.
 */
export const getAudioUrl = (audioPath, token) => {
  if (!audioPath) return '';
  const filename = audioPath.split('/').pop();
  const baseUrl = `${API_BASE}/uploads/${filename}`;
  if (!token) return baseUrl;
  return `${baseUrl}?token=${encodeURIComponent(token)}`;
};

/**
 * Get public (unauthenticated) complaint listing.
 */
export const getPublicComplaints = async ({ page = 1, size = 12, category = '', status = '', sort = 'latest' } = {}) => {
  const params = new URLSearchParams({ page, size, sort });
  if (category && category !== 'all') params.set('category', category);
  if (status && status !== 'all') params.set('status', status);
  const response = await fetch(`${API_BASE}/complaints/public?${params}`);
  if (!response.ok) throw new Error('Failed to load complaints');
  return response.json();
};

/**
 * Get resolved complaints archive (unauthenticated).
 */
export const getResolvedComplaints = async ({ page = 1, size = 12, category = '' } = {}) => {
  const params = new URLSearchParams({ page, size });
  if (category && category !== 'all') params.set('category', category);
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

/**
 * Get or create a stable browser fingerprint for voting.
 */
export const getVoterFingerprint = () => {
  let fp = localStorage.getItem('_bbmp_fp');
  if (!fp) {
    fp = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now();
    localStorage.setItem('_bbmp_fp', fp);
  }
  return fp;
};

