'use client'

import { useEffect, useRef, useState } from 'react'

interface AdaptiveSpacingProps {
  children: React.ReactNode
}

export default function AdaptiveSpacing({ children }: AdaptiveSpacingProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [spacing, setSpacing] = useState<'normal' | 'compact'>('normal')

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const checkSpacing = () => {
      const viewportHeight = window.innerHeight
      const headerHeight = 80 // Approximate header height
      const availableHeight = viewportHeight - headerHeight
      const contentHeight = container.scrollHeight
      const difference = availableHeight - contentHeight

      // If content is within 50px of fitting, use compact spacing
      if (difference > 0 && difference < 50) {
        setSpacing('compact')
      } else {
        setSpacing('normal')
      }
    }

    // Check on mount and resize
    checkSpacing()
    window.addEventListener('resize', checkSpacing)

    // Use ResizeObserver to watch for content changes
    const resizeObserver = new ResizeObserver(checkSpacing)
    resizeObserver.observe(container)

    return () => {
      window.removeEventListener('resize', checkSpacing)
      resizeObserver.disconnect()
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className={spacing === 'compact' ? 'space-y-4' : 'space-y-6'}
      style={{
        paddingBottom: spacing === 'compact' ? '1rem' : '1.5rem',
      }}
    >
      {children}
    </div>
  )
}

