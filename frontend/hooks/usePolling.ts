'use client'

import { useEffect, useRef, useState } from 'react'

interface UsePollingOptions {
  enabled?: boolean
  interval?: number // in milliseconds
  onPoll?: () => void | Promise<void>
}

export function usePolling({ enabled = true, interval = 5000, onPoll }: UsePollingOptions) {
  const [isPolling, setIsPolling] = useState(false)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const onPollRef = useRef(onPoll)

  // Keep onPoll ref up to date
  useEffect(() => {
    onPollRef.current = onPoll
  }, [onPoll])

  useEffect(() => {
    if (!enabled || !onPollRef.current) {
      return
    }

    setIsPolling(true)
    
    // Initial poll
    onPollRef.current()

    // Set up interval
    intervalRef.current = setInterval(async () => {
      try {
        if (onPollRef.current) {
          await onPollRef.current()
        }
      } catch (error) {
        console.error('Polling error:', error)
      }
    }, interval)

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      setIsPolling(false)
    }
  }, [enabled, interval])

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setIsPolling(false)
  }

  const startPolling = () => {
    if (!onPoll) return
    setIsPolling(true)
    onPoll()
    intervalRef.current = setInterval(async () => {
      try {
        await onPoll()
      } catch (error) {
        console.error('Polling error:', error)
      }
    }, interval)
  }

  return { isPolling, stopPolling, startPolling }
}

