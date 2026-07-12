import { useState } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ChevronDown, Mic, Map as MapIcon, BarChart3, ShieldCheck, Sun, Moon } from 'lucide-react';
import '../../styles/Navbar.css';

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [featuresOpen, setFeaturesOpen] = useState(false);
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');
  const location = useLocation();

  const toggleMenu = () => setIsOpen(!isOpen);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  const closeMenus = () => {
    setIsOpen(false);
    setFeaturesOpen(false);
  };

  const featureCards = [
    {
      title: "Submit Complaint",
      description: "Report civic issues using voice or text.",
      icon: <Mic size={22} color="#2563eb" />, // blue-600
      path: "/",
    },
    {
      title: "View Issues",
      description: "Track and browse reported complaints.",
      icon: <MapIcon size={22} color="#4f46e5" />, // indigo-600
      path: "/complaints",
    },
    {
      title: "Analytics Dashboard",
      description: "View AI analytics and processing metrics.",
      icon: <BarChart3 size={22} color="#9333ea" />, // purple-600
      path: "/analytics",
    },
    {
      title: "Admin Dashboard",
      description: "Manage complaints and perform verification.",
      icon: <ShieldCheck size={22} color="#059669" />, // emerald-600
      path: "/admin",
    }
  ];

  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar-inner container">
        {/* Brand - Left */}
        <Link to="/" className="navbar-brand" onClick={closeMenus}>
          <div className="brand-icon" aria-hidden="true">
            <div className="levitation-ring">
              <div className="ring-dot"></div>
              <div className="ring-base"></div>
            </div>
          </div>
          <div className="brand-text">
            <span className="brand-title">BBMP AI</span>
            <span className="brand-subtitle">Civic Portal</span>
          </div>
        </Link>

        {/* Desktop Navigation - Center */}
        <div className="navbar-desktop-nav">
          <div className="nav-links-wrapper">
            <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Home
              {location.pathname === '/' && <motion.div layoutId="nav-indicator" className="nav-indicator" />}
            </NavLink>
            
            {/* Features Mega Menu */}
            <div 
              className="nav-dropdown-container"
              onMouseEnter={() => setFeaturesOpen(true)}
              onMouseLeave={() => setFeaturesOpen(false)}
            >
              <button className={`nav-link dropdown-trigger ${featuresOpen ? 'open' : ''} ${['/complaints', '/analytics', '/admin'].includes(location.pathname) ? 'active' : ''}`}>
                Features 
                <ChevronDown 
                  size={14} 
                  style={{ 
                    marginLeft: '4px', 
                    transition: 'transform 0.2s ease',
                    transform: featuresOpen ? 'rotate(180deg)' : 'rotate(0deg)' 
                  }} 
                />
                {['/complaints', '/analytics', '/admin'].includes(location.pathname) && <motion.div layoutId="nav-indicator" className="nav-indicator" />}
              </button>
              
              <AnimatePresence>
                {featuresOpen && (
                  <motion.div 
                    initial={{ opacity: 0, y: 12, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 8, scale: 0.98 }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                    className="mega-menu"
                  >
                    <div className="mega-menu-grid">
                      {featureCards.map((card, idx) => (
                        <Link to={card.path} className="mega-menu-card" key={idx} onClick={closeMenus}>
                          <div className="mega-menu-icon">
                            {card.icon}
                          </div>
                          <div className="mega-menu-content">
                            <h3 className="mega-menu-title">{card.title}</h3>
                            <p className="mega-menu-desc">{card.description}</p>
                          </div>
                        </Link>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Theme Toggle, Admin Login & Mobile Toggle - Right */}
        <div className="navbar-actions">
          <button className="btn-theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
            {theme === 'light' ? <Moon size={18} color="#475569" /> : <Sun size={18} color="#e2e8f0" />}
          </button>

          <Link to="/admin/login" className="btn-admin-login" onClick={closeMenus}>
            Admin Login
          </Link>
          
          <button className="mobile-menu-btn" onClick={toggleMenu} aria-label="Toggle menu">
            {isOpen ? <X size={24} color="#334155" /> : <Menu size={24} color="#334155" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="mobile-nav"
          >
            <div className="mobile-nav-inner container">
              <NavLink to="/" end className="mobile-nav-link" onClick={closeMenus}>Home</NavLink>
              
              <div className="mobile-nav-group">
                <span className="mobile-nav-group-title">Features</span>
                <div className="mobile-nav-sublinks">
                  {featureCards.map((card, idx) => (
                    <Link to={card.path} className="mobile-sublink" key={idx} onClick={closeMenus}>
                      <div className="mobile-sublink-icon">
                        {card.icon}
                      </div>
                      <div className="mobile-sublink-text">
                        <span className="mobile-sublink-title">{card.title}</span>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>

              <Link to="/admin/login" className="mobile-nav-btn" onClick={closeMenus}>Admin Login</Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
