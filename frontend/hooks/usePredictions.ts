'use client'

import { useState } from 'react'
import { mlEndpoints } from '@/lib/api/endpoints'

export function usePrediction() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  const predict = async (modelId: number, features: Record<string, any>) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.predict({ model_id: modelId, features })
      setResult(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to make prediction'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const predictBatch = async (modelId: number, featuresList: Record<string, any>[]) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.predictBatch({ 
        model_id: modelId, 
        features_list: featuresList 
      })
      setResult(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to make batch predictions'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return { predict, predictBatch, loading, error, result }
}

export function useAnomalyDetection() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  const detect = async (modelId: number, features: Record<string, any>) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.detectAnomaly({ model_id: modelId, features })
      setResult(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to detect anomaly'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const detectBatch = async (modelId: number, featuresList: Record<string, any>[]) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.detectAnomaliesBatch({ 
        model_id: modelId, 
        features_list: featuresList 
      })
      setResult(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to detect anomalies'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return { detect, detectBatch, loading, error, result }
}

export function useClassification() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  const classify = async (modelId: number, features: Record<string, any>) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.classify({ model_id: modelId, features })
      setResult(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to classify'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const classifyBatch = async (modelId: number, featuresList: Record<string, any>[]) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.classifyBatch({ 
        model_id: modelId, 
        features_list: featuresList 
      })
      setResult(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to classify batch'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return { classify, classifyBatch, loading, error, result }
}

