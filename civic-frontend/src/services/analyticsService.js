import { API_BASE } from './api';

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
 * Fetch NLP Analytics Dashboard data (requires JWT token).
 * Supports optional date range and language filters.
 */
export const getAnalyticsDashboard = async (token, { startDate, endDate, language } = {}) => {
  const params = new URLSearchParams();
  if (startDate) params.set('start_date', startDate);
  if (endDate) params.set('end_date', endDate);
  if (language) params.set('language', language);
  const response = await fetch(`${API_BASE}/analytics/dashboard?${params}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    if (response.status === 401) throw new Error('401 Unauthorized');
    throw new Error('Failed to fetch analytics');
  }
  return response.json();
};
