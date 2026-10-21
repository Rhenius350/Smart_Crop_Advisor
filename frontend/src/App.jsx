import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Dashboard from './pages/Dashboard';
import Prediction from './pages/Prediction';
import Performance from './pages/Performance';
import Explainability from './pages/Explainability';
import About from './pages/About';
import Research from './pages/Research';

function App() {
  return (
    <Router>
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Prediction />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/about" element={<About />} />
            <Route path="/research" element={<Research />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
