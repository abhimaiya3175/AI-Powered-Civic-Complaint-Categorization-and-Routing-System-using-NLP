import { BrowserRouter as Router } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Navbar from './components/common/Navbar';
import { AppRoutes } from './routes/AppRoutes';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div id="app-shell">
          <Navbar />
          <main className="page">
            <div className="container">
              <AppRoutes />
            </div>
          </main>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;
