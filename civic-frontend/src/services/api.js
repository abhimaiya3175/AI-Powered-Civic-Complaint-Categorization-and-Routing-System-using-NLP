const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

/**
 * Submit a voice complaint (audio file upload).
 */
export const submitComplaint = async (audioFile) => {
  const formData = new FormData();
  formData.append('file', audioFile);

  const response = await fetch(`${API_BASE}/submit-complaint`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to submit complaint');
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
    throw new Error('Failed to verify complaint');
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
export const getAudioUrl = (audioPath) => {
  return `${API_BASE}/${audioPath}`;
};
