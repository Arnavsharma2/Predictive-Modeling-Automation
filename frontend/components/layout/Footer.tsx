'use client'

import { usePathname } from 'next/navigation'

export default function Footer() {
  const pathname = usePathname()
  
  // Don't render footer on dashboard routes, main page, or auth pages
  if (pathname?.startsWith('/dashboard') || pathname === '/' || pathname?.startsWith('/auth')) {
    return null
  }

  return (
    <footer className="glass-strong border-t border-pastel-blue/30 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
          <div className="col-span-1 md:col-span-2">
            <h3 className="text-2xl font-bold mb-4">
              <span className="gradient-text">Predict</span><span className="text-gray-soft-700">Lab</span>
            </h3>
            <p className="text-gray-soft-600 mb-6 leading-relaxed max-w-md">
              Automated Predictive Modeling Platform. Transform your data into actionable insights with our AI-powered analytics solution.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-6 text-slate-800">Quick Links</h4>
            <ul className="space-y-3 text-slate-600">
              <li><a href="#about" className="hover:text-pastel-blue transition-colors duration-300 inline-block hover:translate-x-1 transform">About</a></li>
              <li><a href="#process" className="hover:text-pastel-blue transition-colors duration-300 inline-block hover:translate-x-1 transform">Process</a></li>
              <li><a href="#features" className="hover:text-pastel-blue transition-colors duration-300 inline-block hover:translate-x-1 transform">Features</a></li>
              <li><a href="#contact" className="hover:text-pastel-blue transition-colors duration-300 inline-block hover:translate-x-1 transform">Contact</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-6 text-gray-soft-700">Contact</h4>
            <ul className="space-y-3 text-gray-soft-600">
              <li className="hover:text-pastel-blue transition-colors duration-300">Email: info@predictlab.com</li>
              <li className="hover:text-pastel-blue transition-colors duration-300">Phone: (555) 123-4567</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-pastel-blue/30 mt-12 pt-8 text-center text-gray-soft-500">
          <p>&copy; {new Date().getFullYear()} PredictLab. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}

