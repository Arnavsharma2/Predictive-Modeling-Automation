'use client'

import { useState, useEffect, useCallback } from 'react'
import { mlEndpoints } from '@/lib/api/endpoints'

export interface BatchJob {
  id: number
  model_id: number
  data_source_id: number | null
  status: string
  job_name: string | null
  input_type: string
  progress: number
  total_records: number | null
  processed_records: number | null
  failed_records: number | null
  result_path: string | null
  result_format: string
  error_message: string | null
  scheduled_at: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
}

export function useBatchJobs(params?: { skip?: number; limit?: number; model_id?: number; status_filter?: string }) {
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const fetchJobs = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.listBatchJobs(params)
      setJobs(response.data.jobs || [])
      setTotal(response.data.total || 0)
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to fetch batch jobs'
      setError(errorMsg)
      console.error('Error fetching batch jobs:', err)
    } finally {
      setLoading(false)
    }
  }, [params?.skip, params?.limit, params?.model_id, params?.status_filter])

  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

  return { jobs, loading, error, total, refetch: fetchJobs }
}

export function useBatchJob(jobId: number | null) {
  const [job, setJob] = useState<BatchJob | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchJob = useCallback(async () => {
    if (!jobId) {
      setJob(null)
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.getBatchJob(jobId)
      setJob(response.data)
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to fetch batch job'
      setError(errorMsg)
      console.error('Error fetching batch job:', err)
    } finally {
      setLoading(false)
    }
  }, [jobId])

  useEffect(() => {
    fetchJob()
  }, [fetchJob])

  return { job, loading, error, refetch: fetchJob }
}

export function useCreateBatchJob() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const createJob = useCallback(async (data: {
    model_id: number
    input_type?: string
    data_source_id?: number
    input_config?: any
    job_name?: string
    result_format?: string
    scheduled_at?: string
  }) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.createBatchJob(data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to create batch job'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }, [])

  return { createJob, loading, error }
}

export function useCancelBatchJob() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const cancelJob = useCallback(async (jobId: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.cancelBatchJob(jobId)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to cancel batch job'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }, [])

  return { cancelJob, loading, error }
}

export function useDownloadBatchResults() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const downloadResults = useCallback(async (jobId: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.downloadBatchResults(jobId)
      
      // Create a blob and download it
      const blob = new Blob([response.data])
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `batch_results_${jobId}.csv`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to download results'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }, [])

  return { downloadResults, loading, error }
}

