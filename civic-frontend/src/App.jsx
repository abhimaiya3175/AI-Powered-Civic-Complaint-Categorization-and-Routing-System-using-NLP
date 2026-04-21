import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import RecordComplaint from './components/RecordComplaint';
import ComplaintList from './components/ComplaintList';
import UserComplaints from './components/UserComplaints';

function App() {
  return (
    <Router>
      <div id="app-shell">
        <Navbar />
        <main className="page">
          <div className="container">
            <Routes>
              <Route path="/" element={<RecordComplaint />} />
              <Route path="/complaints" element={<UserComplaints />} />
              <Route path="/admin" element={<ComplaintList />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
