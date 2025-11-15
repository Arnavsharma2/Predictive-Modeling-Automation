'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import type { User } from '@/lib/auth';
import BackendStartupNotice from '@/components/BackendStartupNotice';

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();

  const [formData, setFormData] = useState({
    email: '',
    username: '',
    password: '',
    confirmPassword: '',
    fullName: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    // Validate passwords match
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    // Validate password length
    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters long');
      return;
    }

    setLoading(true);

    try {
      await register(
        formData.email,
        formData.username,
        formData.password,
        formData.fullName || undefined,
        'viewer' // All users get basic viewer access by default
      );
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center animated-bg py-12 px-4 sm:px-6 lg:px-8 relative">
      {/* Animated pastel background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-pastel-blue/30 rounded-full blur-3xl float"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pastel-mint/25 rounded-full blur-3xl float" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-pastel-green/20 rounded-full blur-3xl float" style={{ animationDelay: '4s' }}></div>
      </div>

      <BackendStartupNotice />
      <div className="max-w-md w-full space-y-8 relative z-10">
        <div className="glass rounded-2xl p-8 shadow-lg border-pastel-blue/40">
          <div>
            <h2 className="text-center text-3xl font-bold text-gray-soft-700 mb-2">
              Create your account
            </h2>
            <p className="mt-2 text-center text-sm text-gray-soft-600">
              Already have an account?{' '}
              <Link
                href="/auth/login"
                className="font-medium text-pastel-blue hover:text-pastel-powder transition-colors"
              >
                Sign in
              </Link>
            </p>
          </div>

          <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
            {error && (
              <div className="rounded-md bg-red-50 p-4 border border-red-200">
                <div className="flex">
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">{error}</h3>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label
                  htmlFor="email"
                  className="block text-sm font-medium text-gray-soft-700 mb-1"
                >
                  Email address
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  required
                  className="appearance-none relative block w-full px-4 py-3 border border-pastel-blue/40 placeholder-gray-soft-500 text-gray-soft-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-pastel-blue focus:border-pastel-blue bg-white/80 backdrop-blur-sm sm:text-sm transition-all"
                  placeholder="you@example.com"
                  value={formData.email}
                  onChange={handleChange}
                  disabled={loading}
                />
              </div>

              <div>
                <label
                  htmlFor="username"
                  className="block text-sm font-medium text-gray-soft-700 mb-1"
                >
                  Username
                </label>
                <input
                  id="username"
                  name="username"
                  type="text"
                  required
                  className="appearance-none relative block w-full px-4 py-3 border border-pastel-blue/40 placeholder-gray-soft-500 text-gray-soft-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-pastel-blue focus:border-pastel-blue bg-white/80 backdrop-blur-sm sm:text-sm transition-all"
                  placeholder="johndoe"
                  value={formData.username}
                  onChange={handleChange}
                  disabled={loading}
                />
              </div>

              <div>
                <label
                  htmlFor="fullName"
                  className="block text-sm font-medium text-gray-soft-700 mb-1"
                >
                  Full Name (optional)
                </label>
                <input
                  id="fullName"
                  name="fullName"
                  type="text"
                  className="appearance-none relative block w-full px-4 py-3 border border-pastel-blue/40 placeholder-gray-soft-500 text-gray-soft-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-pastel-blue focus:border-pastel-blue bg-white/80 backdrop-blur-sm sm:text-sm transition-all"
                  placeholder="John Doe"
                  value={formData.fullName}
                  onChange={handleChange}
                  disabled={loading}
                />
              </div>

              <div>
                <label
                  htmlFor="password"
                  className="block text-sm font-medium text-gray-soft-700 mb-1"
                >
                  Password
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  className="appearance-none relative block w-full px-4 py-3 border border-pastel-blue/40 placeholder-gray-soft-500 text-gray-soft-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-pastel-blue focus:border-pastel-blue bg-white/80 backdrop-blur-sm sm:text-sm transition-all"
                  placeholder="••••••••"
                  value={formData.password}
                  onChange={handleChange}
                  disabled={loading}
                />
                <p className="mt-1 text-xs text-gray-soft-500">
                  Must be at least 8 characters
                </p>
              </div>

              <div>
                <label
                  htmlFor="confirmPassword"
                  className="block text-sm font-medium text-gray-soft-700 mb-1"
                >
                  Confirm Password
                </label>
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type="password"
                  required
                  className="appearance-none relative block w-full px-4 py-3 border border-pastel-blue/40 placeholder-gray-soft-500 text-gray-soft-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-pastel-blue focus:border-pastel-blue bg-white/80 backdrop-blur-sm sm:text-sm transition-all"
                  placeholder="••••••••"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  disabled={loading}
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={loading}
                className="btn-primary hover-glow handcrafted-outline w-full flex justify-center items-center py-3 px-4 text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <span className="flex items-center">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Creating account...
                  </span>
                ) : (
                  'Create account'
                )}
              </button>
            </div>
          </form>

          <div className="text-center mt-6">
            <p className="text-xs text-gray-soft-500">
              By creating an account, you agree to our Terms of Service and
              Privacy Policy
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
