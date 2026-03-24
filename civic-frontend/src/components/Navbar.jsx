import { NavLink } from 'react-router-dom';
import '../styles/Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar-inner container">
        {/* Brand */}
        <NavLink to="/" className="navbar-brand">
          <div className="brand-icon" aria-hidden="true">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="8" fill="#1D4ED8"/>
              <path d="M8 22V12l8-5 8 5v10H8z" fill="#fff" opacity="0.9"/>
              <rect x="13" y="17" width="6" height="5" rx="1" fill="#1D4ED8"/>
              <circle cx="16" cy="13" r="2" fill="#1D4ED8"/>
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-title">BBMP</span>
            <span className="brand-subtitle">Civic Complaints</span>
          </div>
        </NavLink>

        {/* Navigation Links */}
        <div className="navbar-nav">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            id="nav-citizen"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" x2="12" y1="19" y2="22"/>
            </svg>
            Submit Complaint
          </NavLink>
          <NavLink
            to="/admin"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            id="nav-admin"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect width="18" height="18" x="3" y="3" rx="2"/>
              <path d="M3 9h18"/>
              <path d="M9 21V9"/>
            </svg>
            Admin Dashboard
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
