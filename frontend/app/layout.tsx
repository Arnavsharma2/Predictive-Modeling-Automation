import type { Metadata } from 'next'
import './globals.css'
import LandingNav from '@/components/layout/LandingNav'
import Footer from '@/components/layout/Footer'
import { AuthProvider } from '@/contexts/AuthContext'

export const metadata: Metadata = {
  title: 'PredictLab - AI-Powered Predictive Modeling',
  description: 'PredictLab - Automated Predictive Modeling Platform - Transform data into insights',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased animated-bg text-slate-700">
        <AuthProvider>
          <div className="relative z-10">
            <LandingNav />
            <main className="min-h-screen relative">
              {children}
            </main>
            <Footer />
          </div>
        </AuthProvider>
      </body>
    </html>
  )
}

