'use client'

import { useState, useEffect } from 'react'
import { dataQualityEndpoints } from '@/lib/api/endpoints'

export function useDataQualityReports(dataSourceId: number | null, limit: number = 50, offset: number = 0) {
  const [reports, setReports] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const fetchReports = async () => {
    if (!dataSourceId) {
      setReports([])
      return
    }

    try {
      setLoading(true)
      const response = await dataQualityEndpoints.getQualityReports(dataSourceId, limit, offset)
      setReports(response.data.reports || [])
      setTotal(response.data.total || 0)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch quality reports')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchReports()
  }, [dataSourceId, limit, offset])

  return { reports, loading, error, total, refetch: fetchReports }
}

export function useDataProfile(dataSourceId: number | null, sampleSize: number = 10000) {
  const [profile, setProfile] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchProfile = async () => {
    if (!dataSourceId) {
      setProfile(null)
      return
    }

    try {
      setLoading(true)
      const response = await dataQualityEndpoints.getDataProfile(dataSourceId, sampleSize)
      setProfile(response.data)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch data profile')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchProfile()
  }, [dataSourceId, sampleSize])

  return { profile, loading, error, refetch: fetchProfile }
}

export function useDataLineage(dataSourceId: number | null) {
  const [lineage, setLineage] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchLineage = async () => {
    if (!dataSourceId) {
      setLineage([])
      return
    }

    try {
      setLoading(true)
      const response = await dataQualityEndpoints.getDataLineage(dataSourceId)
      setLineage(response.data || [])
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch data lineage')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLineage()
  }, [dataSourceId])

  return { lineage, loading, error, refetch: fetchLineage }
}

export function useCheckDataQuality() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const checkQuality = async (dataSourceId: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await dataQualityEndpoints.checkQuality({ data_source_id: dataSourceId })
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to check data quality'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return { checkQuality, loading, error }
}

