import { Routes, Route } from 'react-router-dom';
import { ProtectedRoute } from '../components/common/ProtectedRoute';

import RecordComplaint from '../pages/RecordComplaint';
import ComplaintList from '../pages/ComplaintList';
import AdminDashboard from '../pages/AdminDashboard';
import Login from '../pages/Login';
import AnalyticsDashboard from '../pages/AnalyticsDashboard';

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<RecordComplaint />} />
      <Route path="/complaints" element={<ComplaintList />} />
      <Route path="/admin/login" element={<Login />} />
      <Route 
        path="/admin" 
        element={
          <ProtectedRoute>
            <AdminDashboard />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/analytics" 
        element={
          <ProtectedRoute>
            <AnalyticsDashboard />
          </ProtectedRoute>
        } 
      />
    </Routes>
  );
};
