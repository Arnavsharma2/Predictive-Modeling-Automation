'use client'

import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'

export default function Header() {
  const { user } = useAuth()
  const username = user?.username || 'User'
  const initial = username.charAt(0).toUpperCase()

  return (
    <header className="glass-strong border-b border-pastel-blue/30 sticky top-0 z-50 backdrop-blur-xl">
      <div className="px-6 lg:px-8 py-5">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">
            <span className="gradient-text">Predict</span><span className="text-gray-soft-700">Lab</span>
          </h1>
          <div className="flex items-center gap-2">
            <Link 
              href="/settings"
              className="w-8 h-8 bg-pastel-blue rounded-full flex items-center justify-center shadow-sm hover:bg-pastel-powder transition-colors cursor-pointer"
            >
              <span className="text-gray-soft-700 text-sm font-semibold">{initial}</span>
            </Link>
            <span className="text-sm text-gray-soft-700 font-medium hidden sm:block">{username}</span>
          </div>
        </div>
      </div>
    </header>
  )
}

