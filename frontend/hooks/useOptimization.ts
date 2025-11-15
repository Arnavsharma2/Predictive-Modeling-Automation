'use client'

import { useState, useEffect, useCallback } from 'react'
import { mlEndpoints } from '@/lib/api/endpoints'

export function useHyperparameterOptimization() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  const optimize = async (config: {
    model_id?: number
    algorithm: string
    search_space?: Record<string, any>
    metric_name?: string
    direction?: string
    n_trials?: number
    timeout?: number
    study_name?: string
  }) => {
    try {
      setLoading(true)
      setError(null)
      setResult(null)
      
      // Validate required fields
      if (!config.algorithm) {
        throw new Error('Algorithm is required')
      }
      
      if (config.n_trials && (config.n_trials < 1 || config.n_trials > 1000)) {
        throw new Error('Number of trials must be between 1 and 1000')
      }
      
      if (config.timeout && config.timeout < 0) {
        throw new Error('Timeout must be non-negative')
      }
      
      const response = await mlEndpoints.optimizeHyperparameters(config)
      setResult(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to start optimization'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return { optimize, result, loading, error }
}

export function useStudySummary(studyName: string | null, autoRefresh: boolean = false, refreshInterval: number = 5000) {
  const [summary, setSummary] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isPolling, setIsPolling] = useState(false)

  const fetchSummary = useCallback(async () => {
    if (!studyName || !studyName.trim()) {
      setSummary(null)
      setError(null)
      return
    }

    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.getStudySummary(studyName)
      setSummary(response.data)
      
      // Stop polling if study is complete (has trials and best value)
      if (response.data.n_trials > 0 && response.data.best_value !== 0) {
        setIsPolling(false)
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to fetch study summary'
      // Don't set error if study doesn't exist yet (might still be starting)
      if (err.response?.status === 404) {
        setError(null) // Clear error, study might not exist yet
        setSummary(null)
      } else {
        setError(errorMsg)
      }
    } finally {
      setLoading(false)
    }
  }, [studyName])

  // Auto-refresh when study is running
  useEffect(() => {
    if (!studyName || !autoRefresh || !isPolling) return

    fetchSummary()
    
    const interval = setInterval(() => {
      fetchSummary()
    }, refreshInterval)

    return () => clearInterval(interval)
  }, [studyName, autoRefresh, isPolling, fetchSummary, refreshInterval])

  // Start polling when study name is set
  useEffect(() => {
    if (studyName && autoRefresh) {
      setIsPolling(true)
      fetchSummary()
    }
  }, [studyName, autoRefresh, fetchSummary])

  return { summary, loading, error, refetch: fetchSummary, isPolling, setIsPolling }
}

