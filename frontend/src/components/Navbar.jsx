import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Sprout, LayoutDashboard, TestTube, BarChart3, HelpCircle, Info, Menu, X, Layers } from 'lucide-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  return (
    <nav className="navbar">
      <div className="nav-content">
        <NavLink to="/" className="brand-logo" onClick={closeMenu}>
          <div className="brand-icon">
            <Sprout size={22} />
          </div>
          <div>
            <span>SMART CROP ADVISOR</span>
            <span className="brand-tag">GA + RF</span>
          </div>
        </NavLink>

        <button className="menu-btn" onClick={toggleMenu} aria-label="Toggle navigation">
          {isOpen ? <X size={24} /> : <Menu size={24} />}
        </button>

        <ul className={`nav-links ${isOpen ? 'open' : ''}`}>
          <li>
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={closeMenu} end>
              <LayoutDashboard size={18} />
              <span>Dashboard</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/predict" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={closeMenu}>
              <TestTube size={18} />
              <span>Crop Prediction</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/performance" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={closeMenu}>
              <BarChart3 size={18} />
              <span>Model Performance</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/explainability" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={closeMenu}>
              <HelpCircle size={18} />
              <span>Explainable AI</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/about" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={closeMenu}>
              <Info size={18} />
              <span>About</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/research" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={closeMenu}>
              <Layers size={18} />
              <span>Research</span>
            </NavLink>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
