'use client';

import { useAuth } from '@/contexts/AuthContext';
import APIKeyManagement from '@/components/APIKeyManagement';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function SettingsPage() {
  const { user, loading, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login');
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="mt-2 text-gray-600">
            Manage your account settings and API keys
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <div className="h-16 w-16 rounded-full bg-blue-600 flex items-center justify-center text-white text-2xl font-bold">
                    {user.username.charAt(0).toUpperCase()}
                  </div>
                </div>
                <div>
                  <h2 className="text-lg font-medium text-gray-900">
                    {user.full_name || user.username}
                  </h2>
                  <p className="text-sm text-gray-500">{user.email}</p>
                  <p className="text-xs text-gray-400 mt-1 capitalize">
                    {user.role.replace('_', ' ')}
                  </p>
                </div>
              </div>

              <div className="mt-6 space-y-3">
                <div className="border-t border-gray-200 pt-4">
                  <p className="text-sm text-gray-500">Account created</p>
                  <p className="text-sm font-medium text-gray-900">
                    {new Date(user.created_at).toLocaleDateString()}
                  </p>
                </div>

                <div className="border-t border-gray-200 pt-4">
                  <p className="text-sm text-gray-500">Status</p>
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      user.is_active
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {user.is_active ? 'Active' : 'Inactive'}
                  </span>
                </div>

                {user.is_superuser && (
                  <div className="border-t border-gray-200 pt-4">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                      Superuser
                    </span>
                  </div>
                )}
              </div>

              <button
                onClick={logout}
                className="mt-6 w-full px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
              >
                Sign Out
              </button>
            </div>
          </div>

          {/* Main content */}
          <div className="lg:col-span-2 space-y-6">
            {/* API Keys Section */}
            <div className="bg-white shadow rounded-lg p-6">
              <APIKeyManagement />
            </div>

            {/* Account Information */}
            <div className="bg-white shadow rounded-lg p-6">
              <h3 className="text-lg font-medium text-gray-900">
                Account Information
              </h3>
              <div className="mt-4 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Username
                  </label>
                  <p className="mt-1 text-sm text-gray-900">{user.username}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Email
                  </label>
                  <p className="mt-1 text-sm text-gray-900">{user.email}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Role
                  </label>
                  <p className="mt-1 text-sm text-gray-900 capitalize">
                    {user.role.replace('_', ' ')}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
