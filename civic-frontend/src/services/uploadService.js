import { API_BASE } from './api';

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
 * Re-analyze an image using Florence-2.
 */
export const reanalyzeImage = async (id, token) => {
  const response = await fetch(`${API_BASE}/complaints/${id}/reanalyze`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Re-analysis failed');
  }

  return response.json();
};
