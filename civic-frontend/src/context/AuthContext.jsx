import { createContext, useState, useEffect } from 'react';
import { getToken, setToken, removeToken } from '../utils/storage';
import { loginAdmin } from '../services/authService';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [token, setTokenState] = useState(getToken());

  useEffect(() => {
    if (token) {
      setToken(token);
    } else {
      removeToken();
    }
  }, [token]);

  const login = async (username, password) => {
    const data = await loginAdmin(username, password);
    setTokenState(data.access_token);
    return data;
  };

  const logout = () => {
    setTokenState(null);
  };

  return (
    <AuthContext.Provider value={{ token, login, logout, isAuthenticated: !!token }}>
      {children}
    </AuthContext.Provider>
  );
};
