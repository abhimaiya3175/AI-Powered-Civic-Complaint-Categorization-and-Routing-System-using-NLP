import { NavLink } from 'react-router-dom';
import '../styles/Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar-inner container">
        {/* Brand */}
        <NavLink to="/" className="navbar-brand">
          <div className="brand-icon" aria-hidden="true">
            <div className="levitation-ring">
              <div className="ring-dot"></div>
              <div className="ring-base"></div>
            </div>
          </div>
          <div className="brand-text">
            <span className="brand-title">GravLess</span>
            <span className="brand-subtitle">Civic Portal</span>
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
            to="/complaints"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            id="nav-complaints"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" x2="8" y1="13" y2="13"/>
              <line x1="16" x2="8" y1="17" y2="17"/>
              <line x1="10" x2="8" y1="9" y2="9"/>
            </svg>
            View Issues
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
