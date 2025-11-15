'use client'

import { useEffect, useRef } from 'react'
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const mainRef = useRef<HTMLElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const main = mainRef.current
    const content = contentRef.current
    if (!main || !content) return

    const adjustSpacing = () => {
      const viewportHeight = window.innerHeight
      const headerHeight = 80 // Approximate header height
      const padding = 32 * 2 // Top and bottom padding
      const availableHeight = viewportHeight - headerHeight - padding
      const contentHeight = content.scrollHeight
      const difference = availableHeight - contentHeight

      // If content is within 100px of fitting, reduce spacing
      if (difference > 0 && difference < 100) {
        main.style.padding = '1rem'
        content.style.gap = '0.75rem'
        content.style.paddingBottom = '0.5rem'
      } else if (difference > 0 && difference < 200) {
        main.style.padding = '1.25rem'
        content.style.gap = '1rem'
        content.style.paddingBottom = '1rem'
      } else {
        main.style.padding = ''
        content.style.gap = ''
        content.style.paddingBottom = ''
      }
    }

    // Check on mount and resize
    adjustSpacing()
    window.addEventListener('resize', adjustSpacing)

    // Use ResizeObserver to watch for content changes
    const resizeObserver = new ResizeObserver(adjustSpacing)
    resizeObserver.observe(content)

    return () => {
      window.removeEventListener('resize', adjustSpacing)
      resizeObserver.disconnect()
    }
  }, [])

  return (
    <div className="flex h-screen animated-bg text-slate-700 overflow-hidden">
      <div className="relative z-10 flex w-full h-full">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <main ref={mainRef} className="flex-1 overflow-x-hidden overflow-y-auto scrollbar-hide dashboard-main-content">
            <div ref={contentRef} className="animate-fade-in relative z-10 min-h-full flex flex-col">
              <div className="flex-1">
                {children}
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}

