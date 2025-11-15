'use client'

import { useState, useEffect } from 'react'
import { mlEndpoints } from '@/lib/api/endpoints'

export function useModels(params?: { skip?: number; limit?: number; model_type?: string; status?: string }) {
  const [models, setModels] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const fetchModels = async () => {
    try {
      setLoading(true)
      const response = await mlEndpoints.models(params)
      setModels(response.data.models || [])
      setTotal(response.data.total || 0)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch models')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [params?.skip, params?.limit, params?.model_type, params?.status])

  return { models, loading, error, total, refetch: fetchModels }
}

export function useModelDetails(modelId: number | null) {
  const [model, setModel] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) {
      setModel(null)
      return
    }

    const fetchModel = async () => {
      try {
        setLoading(true)
        const response = await mlEndpoints.modelDetails(modelId)
        setModel(response.data)
        setError(null)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch model details')
      } finally {
        setLoading(false)
      }
    }

    fetchModel()
  }, [modelId])

  return { model, loading, error }
}

export function useTrainingJobs(params?: { skip?: number; limit?: number; model_id?: number; status?: string }) {
  const [jobs, setJobs] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const fetchJobs = async () => {
    try {
      setLoading(true)
      const response = await mlEndpoints.trainHistory(params)
      setJobs(response.data.jobs || [])
      setTotal(response.data.total || 0)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch training jobs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchJobs()
  }, [params?.skip, params?.limit, params?.model_id, params?.status])

  return { jobs, loading, error, total, refetch: fetchJobs }
}

export function useModelVersions(modelId: number | null, includeArchived: boolean = false) {
  const [versions, setVersions] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) {
      setVersions([])
      return
    }

    const fetchVersions = async () => {
      try {
        setLoading(true)
        const response = await mlEndpoints.modelVersions(modelId, includeArchived)
        setVersions(response.data.versions || [])
        setError(null)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch model versions')
      } finally {
        setLoading(false)
      }
    }

    fetchVersions()
  }, [modelId, includeArchived])

  return { versions, loading, error }
}

