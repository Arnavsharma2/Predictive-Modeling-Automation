'use client'

import { useState, useEffect } from 'react'
import { mlEndpoints } from '@/lib/api/endpoints'

export function useExperiments() {
  const [experiments, setExperiments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchExperiments = async () => {
    try {
      setLoading(true)
      const response = await mlEndpoints.listExperiments()
      setExperiments(response.data || [])
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch experiments')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchExperiments()
  }, [])

  return { experiments, loading, error, refetch: fetchExperiments }
}

export function useExperiment(experimentName: string | null) {
  const [experiment, setExperiment] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!experimentName) {
      setExperiment(null)
      return
    }

    const fetchExperiment = async () => {
      try {
        setLoading(true)
        const response = await mlEndpoints.getExperiment(experimentName)
        setExperiment(response.data)
        setError(null)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch experiment')
      } finally {
        setLoading(false)
      }
    }

    fetchExperiment()
  }, [experimentName])

  return { experiment, loading, error }
}

export function useRunSearch(experimentName: string | null, filterString?: string) {
  const [runs, setRuns] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const search = async (maxResults: number = 100) => {
    if (!experimentName) return

    try {
      setLoading(true)
      const response = await mlEndpoints.searchRuns({
        experiment_name: experimentName,
        filter_string: filterString,
        max_results: maxResults
      })
      setRuns(response.data.runs || [])
      setTotal(response.data.total || 0)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to search runs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (experimentName) {
      search()
    }
  }, [experimentName, filterString])

  return { runs, loading, error, total, refetch: search }
}

export function useRunComparison(experimentName: string | null, metricName: string, topN: number = 10) {
  const [comparison, setComparison] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const compare = async () => {
    if (!experimentName || !metricName) return

    try {
      setLoading(true)
      const response = await mlEndpoints.compareRuns(experimentName, metricName, topN)
      setComparison(response.data)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to compare runs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (experimentName && metricName) {
      compare()
    }
  }, [experimentName, metricName, topN])

  return { comparison, loading, error, refetch: compare }
}

