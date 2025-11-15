'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const navigation = [
  { name: 'Dashboard', href: '/dashboard' },
  { name: 'Data Ingestion', href: '/dashboard/ingestion' },
  { name: 'Data Exploration', href: '/dashboard/data' },
  { name: 'Data Quality', href: '/dashboard/data-quality' },
  { name: 'ML Models', href: '/dashboard/models' },
  { name: 'Experiments', href: '/dashboard/experiments' },
  { name: 'Optimization', href: '/dashboard/optimization' },
  { name: 'Predictions', href: '/dashboard/predictions' },
  { name: 'Batch Predictions', href: '/dashboard/batch-predictions' },
  { name: 'Drift Detection', href: '/dashboard/drift' },
  { name: 'Explainability', href: '/dashboard/explainability' },
  { name: 'Analytics', href: '/dashboard/analytics' },
]

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <div className="w-64 glass-strong border-r border-pastel-blue/30 flex flex-col backdrop-blur-xl">
      <div className="p-6 border-b border-pastel-blue/30">
        <Link href="/" className="block hover:opacity-80 transition-opacity">
          <h2 className="text-2xl font-bold">
            <span className="gradient-text">Predict</span><span className="text-gray-soft-700">Lab</span>
          </h2>
          <p className="text-xs text-gray-soft-500 mt-1 font-medium">Platform</p>
        </Link>
      </div>
      <nav className="flex-1 mt-6 px-3">
        {navigation.map((item, index) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`group block px-4 py-3 mb-1 text-sm font-semibold transition-all duration-300 rounded-xl animate-slide-in ${
                isActive
                  ? 'bg-pastel-blue text-gray-soft-700 shadow-lg'
                  : 'text-gray-soft-600 hover:text-pastel-blue hover:bg-pastel-blue/20'
              }`}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <span className="flex items-center gap-3">
                {isActive && (
                  <span className="w-1.5 h-1.5 bg-gray-soft-700 rounded-full animate-pulse"></span>
                )}
                <span>{item.name}</span>
              </span>
            </Link>
          )
        })}
      </nav>
    </div>
  )
}

