'use client'

import { useState, useEffect } from 'react'
import { dataEndpoints, ingestionEndpoints } from '@/lib/api/endpoints'

export function useDataSources() {
  const [sources, setSources] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchSources = async () => {
      try {
        setLoading(true)
        const response = await dataEndpoints.sources()
        setSources(response.data.sources || [])
        setError(null)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch data sources')
      } finally {
        setLoading(false)
      }
    }

    fetchSources()
  }, [])

  return { sources, loading, error, refetch: () => {
    const fetchSources = async () => {
      try {
        setLoading(true)
        const response = await dataEndpoints.sources()
        setSources(response.data.sources || [])
        setError(null)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch data sources')
      } finally {
        setLoading(false)
      }
    }
    fetchSources()
  } }
}

export function useDataPreview(sourceId: number | null, limit: number = 100, offset: number = 0) {
  const [data, setData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    if (!sourceId) {
      setData([])
      return
    }

    try {
      setLoading(true)
      const response = await dataEndpoints.preview(sourceId, limit, offset)
      setData(response.data.data || [])
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch data preview')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [sourceId, limit, offset])

  return { data, loading, error, refetch: fetchData }
}

export function useDataStats(sourceId: number | null) {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!sourceId) {
      setStats(null)
      return
    }

    const fetchStats = async () => {
      try {
        setLoading(true)
        const response = await dataEndpoints.stats(sourceId)
        setStats(response.data)
        setError(null)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch statistics')
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
  }, [sourceId])

  return { stats, loading, error }
}

export function useETLJobs(sourceId?: number) {
  const [jobs, setJobs] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        setLoading(true)
        const response = await ingestionEndpoints.getETLJobs(sourceId?.toString())
        setJobs(response.data.jobs || [])
        setError(null)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch ETL jobs')
      } finally {
        setLoading(false)
      }
    }

    fetchJobs()
  }, [sourceId])

  return { jobs, loading, error }
}

