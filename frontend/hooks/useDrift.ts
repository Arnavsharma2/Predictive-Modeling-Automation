'use client'

import { useState, useEffect } from 'react'
import { mlEndpoints } from '@/lib/api/endpoints'

export function useDriftReports(modelId: number | null, limit: number = 50, offset: number = 0) {
  const [reports, setReports] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const fetchReports = async () => {
    if (!modelId) {
      setReports([])
      return
    }

    try {
      setLoading(true)
      const response = await mlEndpoints.getDriftReports(modelId, limit, offset)
      setReports(response.data.reports || [])
      setTotal(response.data.total || 0)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch drift reports')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchReports()
  }, [modelId, limit, offset])

  return { reports, loading, error, total, refetch: fetchReports }
}

export function useLatestDriftReport(modelId: number | null) {
  const [report, setReport] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchReport = async () => {
    if (!modelId) {
      setReport(null)
      return
    }

    try {
      setLoading(true)
      const response = await mlEndpoints.getLatestDriftReport(modelId)
      setReport(response.data)
      setError(null)
    } catch (err: any) {
      if (err.response?.status === 404) {
        setReport(null)
        setError(null)
      } else {
        setError(err.response?.data?.detail || 'Failed to fetch drift report')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchReport()
  }, [modelId])

  return { report, loading, error, refetch: fetchReport }
}

export function useCheckDrift() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const checkDrift = async (data: { model_id: number; current_data_source_id?: number; features?: string[] }) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.checkDrift(data)
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to check drift'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return { checkDrift, loading, error }
}

