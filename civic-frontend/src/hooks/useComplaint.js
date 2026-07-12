import { useState, useCallback } from 'react';
import { getPublicComplaints, getResolvedComplaints, voteComplaint } from '../services/complaintService';
import { getVoterFingerprint } from '../utils/storage';

export const useComplaint = () => {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);

  const fetchPublicComplaints = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    try {
      const fp = getVoterFingerprint();
      const data = await getPublicComplaints({ ...params, voterFingerprint: fp });
      setComplaints(data.items || []);
      setTotalPages(data.pages || 1);
      setTotalItems(data.total || 0);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchResolvedComplaints = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    try {
      const fp = getVoterFingerprint();
      const data = await getResolvedComplaints({ ...params, voterFingerprint: fp });
      setComplaints(data.items || []);
      setTotalPages(data.pages || 1);
      setTotalItems(data.total || 0);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const handleVote = async (id) => {
    try {
      const fp = getVoterFingerprint();
      const res = await voteComplaint(id, fp);
      if (!res.already_voted) {
        setComplaints(prev => prev.map(c => c.id === id ? { ...c, voted: true, votes: res.votes } : c));
      }
      return res;
    } catch (err) {
      console.error('Failed to vote:', err);
      throw err;
    }
  };

  return {
    complaints,
    loading,
    error,
    totalPages,
    totalItems,
    fetchPublicComplaints,
    fetchResolvedComplaints,
    handleVote,
  };
};
