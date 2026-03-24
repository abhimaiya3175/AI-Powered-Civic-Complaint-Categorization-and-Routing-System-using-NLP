import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import RecordComplaint from './components/RecordComplaint';
import ComplaintList from './components/ComplaintList';

function App() {
  return (
    <Router>
      <div id="app-shell">
        <Navbar />
        <main className="page">
          <div className="container">
            <Routes>
              <Route path="/" element={<RecordComplaint />} />
              <Route path="/admin" element={<ComplaintList />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
