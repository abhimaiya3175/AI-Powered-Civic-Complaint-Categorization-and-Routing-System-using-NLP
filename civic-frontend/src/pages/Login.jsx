import { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { useNavigate } from 'react-router-dom';
import '../styles/ComplaintList.css'; // Uses Admin dashboard styles

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [loginError, setLoginError] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    const form = e.target;
    setLoginError('');
    setLoginLoading(true);
    try {
      await login(form.username.value, form.password.value);
      navigate('/admin');
    } catch (err) {
      setLoginError(err.message);
    }
    setLoginLoading(false);
  };

  return (
    <div className="login-wrapper gravless-container">
      <div className="login-card">
        <div className="login-header">
          <div className="login-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0284C7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/>
            </svg>
          </div>
          <h2>Admin Dashboard</h2>
          <p className="login-subtext">Sign in to manage and verify complaints</p>
        </div>
        <form onSubmit={handleLogin} className="login-form">
          <div className="form-group">
            <label htmlFor="login-username" className="form-label">Username</label>
            <input id="login-username" name="username" className="input" placeholder="Enter username" required autoComplete="username" />
          </div>
          <div className="form-group">
            <label htmlFor="login-password" className="form-label">Password</label>
            <input id="login-password" name="password" type="password" className="input" placeholder="Enter password" required autoComplete="current-password" />
          </div>
          {loginError && (
            <div className="login-error">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>
              {loginError}
            </div>
          )}
          <button type="submit" className="btn btn-primary btn-lg login-btn" disabled={loginLoading}>
            {loginLoading ? <><span className="spinner" /> Signing in…</> : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
}
