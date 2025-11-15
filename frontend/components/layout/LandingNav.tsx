'use client'

import Link from 'next/link'
import { useState } from 'react'
import { usePathname } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'

export default function LandingNav() {
  const pathname = usePathname()
  const { user, isAuthenticated, logout } = useAuth()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Don't render nav on dashboard routes or auth routes
  if (pathname?.startsWith('/dashboard') || pathname?.startsWith('/auth')) {
    return null
  }

  const navigation: Array<{ name: string; href: string }> = []

  return (
    <nav className="glass-strong sticky top-0 z-50 border-b border-pastel-blue/30 backdrop-blur-xl shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          <div className="flex items-center">
            <Link href="/" className="flex items-center group">
              <span className="text-2xl font-bold">
                <span className="gradient-text">Predict</span><span className="text-gray-soft-700">Lab</span>
              </span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:space-x-8">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className="text-gray-soft-600 hover:text-pastel-blue px-3 py-2 text-sm font-medium transition-all duration-300 relative group"
              >
                {item.name}
                <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-pastel-blue transition-all duration-300 group-hover:w-full"></span>
              </Link>
            ))}
            {isAuthenticated ? (
              <div className="flex items-center space-x-4">
                <Link
                  href="/dashboard"
                  className="text-gray-soft-600 hover:text-pastel-blue px-3 py-2 text-sm font-medium transition-colors"
                >
                  Dashboard
                </Link>
                <Link
                  href="/settings"
                  className="text-gray-soft-600 hover:text-pastel-blue px-3 py-2 text-sm font-medium transition-colors"
                >
                  Settings
                </Link>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-soft-600">
                    {user?.username}
                  </span>
                  <button
                    onClick={logout}
                    className="text-sm text-red-600 hover:text-red-700 font-medium"
                  >
                    Sign Out
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex items-center space-x-3">
                <Link
                  href="/auth/login"
                  className="text-gray-soft-600 hover:text-pastel-blue px-4 py-2 text-sm font-medium transition-colors"
                >
                  Sign In
                </Link>
                <Link
                  href="/auth/register"
                  className="bg-pastel-blue text-gray-soft-700 px-6 py-2.5 rounded-xl text-sm font-semibold hover:bg-pastel-powder transition-all duration-300 shadow-lg hover-glow border border-pastel-blue/40"
                >
                  Get Started
                </Link>
              </div>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="text-gray-soft-600 hover:text-pastel-blue p-2 transition-colors"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {mobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-pastel-blue/30 glass">
          <div className="px-2 pt-2 pb-3 space-y-1">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className="text-gray-soft-600 hover:text-pastel-blue block px-3 py-2 text-base font-medium transition-colors rounded-lg hover:bg-pastel-blue/10"
              >
                {item.name}
              </Link>
            ))}
            {isAuthenticated ? (
              <>
                <Link
                  href="/dashboard"
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-gray-soft-600 hover:text-pastel-blue block px-3 py-2 text-base font-medium transition-colors rounded-lg hover:bg-pastel-blue/10"
                >
                  Dashboard
                </Link>
                <Link
                  href="/settings"
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-gray-soft-600 hover:text-pastel-blue block px-3 py-2 text-base font-medium transition-colors rounded-lg hover:bg-pastel-blue/10"
                >
                  Settings
                </Link>
                <div className="px-3 py-2 text-sm text-gray-soft-600">
                  Signed in as {user?.username}
                </div>
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    logout();
                  }}
                  className="w-full text-left text-red-600 hover:text-red-700 block px-3 py-2 text-base font-medium transition-colors rounded-lg hover:bg-red-50"
                >
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <Link
                  href="/auth/login"
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-gray-soft-600 hover:text-pastel-blue block px-3 py-2 text-base font-medium transition-colors rounded-lg hover:bg-pastel-blue/10"
                >
                  Sign In
                </Link>
                <Link
                  href="/auth/register"
                  onClick={() => setMobileMenuOpen(false)}
                  className="block bg-pastel-blue text-gray-soft-700 px-3 py-2 rounded-xl text-base font-semibold hover:bg-pastel-powder transition-all duration-300 mt-2 text-center"
                >
                  Get Started
                </Link>
              </>
            )}
          </div>
        </div>
      )}
    </nav>
  )
}

