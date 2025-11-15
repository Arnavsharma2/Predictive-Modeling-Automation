'use client'

import { useState } from 'react'
import { mlEndpoints } from '@/lib/api/endpoints'

export function useShapExplanation() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<any>(null)

  const explain = async (modelId: number, features?: Record<string, any>, backgroundSamples?: number, sampleSize?: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.explainShap({
        model_id: modelId,
        features,
        background_samples: backgroundSamples,
        sample_size: sampleSize
      })
      setData(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to generate SHAP explanation'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return { explain, data, loading, error }
}

export function useLimeExplanation() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<any>(null)

  const explain = async (modelId: number, features?: Record<string, any>, numFeatures?: number, numSamples?: number, sampleSize?: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.explainLime({
        model_id: modelId,
        features,
        num_features: numFeatures,
        num_samples: numSamples,
        sample_size: sampleSize
      })
      setData(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to generate LIME explanation'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return { explain, data, loading, error }
}

export function useFeatureImportance() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<any>(null)

  const getImportance = async (modelId: number, method: string = 'model', nRepeats?: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.getFeatureImportance({
        model_id: modelId,
        method,
        n_repeats: nRepeats
      })
      setData(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to get feature importance'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return { getImportance, data, loading, error }
}

export function useWaterfallData() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<any>(null)

  const getWaterfall = async (modelId: number, features: Record<string, any>, maxDisplay?: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await mlEndpoints.getWaterfallData({
        model_id: modelId,
        features,
        max_display: maxDisplay,
      })
      setData(response.data)
      return response.data
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to get waterfall data'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return { getWaterfall, data, loading, error }
}

