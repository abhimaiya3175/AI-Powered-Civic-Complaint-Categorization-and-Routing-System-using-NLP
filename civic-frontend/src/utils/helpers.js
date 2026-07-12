import { API_BASE } from '../services/api';

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
 * Calculate distance between two coordinates in meters
 */
export const getDistance = (lat1, lon1, lat2, lon2) => {
  if (!lat1 || !lon1 || !lat2 || !lon2) return null;
  const R = 6371e3; // metres
  const φ1 = lat1 * Math.PI / 180;
  const φ2 = lat2 * Math.PI / 180;
  const Δφ = (lat2 - lat1) * Math.PI / 180;
  const Δλ = (lon2 - lon1) * Math.PI / 180;

  const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
    Math.cos(φ1) * Math.cos(φ2) *
    Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return Math.round(R * c);
};
