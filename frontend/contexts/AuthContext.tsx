'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User, login as apiLogin, register as apiRegister, logout as apiLogout, getStoredUser, storeUser, getCurrentUser } from '@/lib/auth';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (email: string, username: string, password: string, fullName?: string, role?: User['role']) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  hasRole: (role: User['role']) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load user from storage on mount
    const storedUser = getStoredUser();
    if (storedUser) {
      setUser(storedUser);
    }
    setLoading(false);
  }, []);

  const login = async (username: string, password: string) => {
    try {
      // Login and get tokens
      await apiLogin({ username, password });

      // Fetch user data from /me endpoint
      const userData = await getCurrentUser();

      setUser(userData);
    } catch (error) {
      throw error;
    }
  };

  const register = async (
    email: string,
    username: string,
    password: string,
    fullName?: string,
    role?: User['role']
  ) => {
    try {
      const userData = await apiRegister({
        email,
        username,
        password,
        full_name: fullName,
        role: role || 'viewer',
      });

      storeUser(userData);
      setUser(userData);
    } catch (error) {
      throw error;
    }
  };

  const logout = () => {
    apiLogout();
    setUser(null);
  };

  const hasRole = (requiredRole: User['role']): boolean => {
    if (!user) return false;

    const roleHierarchy: Record<User['role'], number> = {
      viewer: 1,
      data_scientist: 2,
      admin: 3,
    };

    return roleHierarchy[user.role] >= roleHierarchy[requiredRole];
  };

  const value = {
    user,
    loading,
    login,
    register,
    logout,
    isAuthenticated: !!user,
    hasRole,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
