import { useState, useCallback } from 'react';
import { getAnalyticsDashboard } from '../services/analyticsService';
import { useAuth } from './useAuth';

export const useAnalytics = () => {
  const { token } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchAnalytics = useCallback(async (filters = {}) => {
    if (!token) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await getAnalyticsDashboard(token, filters);
      setData(response);
    } catch (err) {
      setError(err.message || 'Failed to fetch analytics data');
    } finally {
      setLoading(false);
    }
  }, [token]);

  return {
    data,
    loading,
    error,
    fetchAnalytics
  };
};
