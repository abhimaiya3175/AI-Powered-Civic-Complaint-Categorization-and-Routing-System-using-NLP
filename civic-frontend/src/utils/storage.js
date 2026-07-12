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

export const getToken = () => {
  return localStorage.getItem('bbmp_token');
};

export const setToken = (token) => {
  localStorage.setItem('bbmp_token', token);
};

export const removeToken = () => {
  localStorage.removeItem('bbmp_token');
};
