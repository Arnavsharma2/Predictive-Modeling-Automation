import axios from 'axios'
import { getAccessToken } from '@/lib/auth'

// Normalize API base URL - remove trailing slashes to prevent double slashes
const getApiBaseUrl = () => {
  const url = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  return url.replace(/\/+$/, '') // Remove trailing slashes
}

const API_BASE_URL = getApiBaseUrl()

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add request interceptor to include auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = getAccessToken()
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// ML Endpoints
export const mlEndpoints = {
  models: (params?: { skip?: number; limit?: number; model_type?: string; status?: string }) => {
    return apiClient.get('/ml/models', { params })
  },
  
  modelDetails: (modelId: number) => {
    return apiClient.get(`/ml/models/${modelId}`)
  },
  
  train: (data: any) => {
    return apiClient.post('/ml/training', data)
  },
  
  trainStatus: (jobId: number) => {
    return apiClient.get(`/ml/training/${jobId}/status`)
  },
  
  trainHistory: (params?: { skip?: number; limit?: number; model_id?: number; status?: string }) => {
    return apiClient.get('/ml/training', { params })
  },
  
  deleteModel: (modelId: number) => {
    return apiClient.delete(`/ml/models/${modelId}`)
  },
  
  predict: (data: { model_id: number; features: Record<string, any> }) => {
    return apiClient.post('/ml/predictions', data)
  },
  
  predictBatch: (data: { model_id: number; features_list: Record<string, any>[] }) => {
    return apiClient.post('/ml/predictions/batch', data)
  },

  detectAnomaly: (data: { model_id: number; features: Record<string, any> }) => {
    return apiClient.post('/ml/anomaly/detect', data)
  },

  detectAnomaliesBatch: (data: { model_id: number; features_list: Record<string, any>[] }) => {
    return apiClient.post('/ml/anomaly/detect/batch', data)
  },

  classify: (data: { model_id: number; features: Record<string, any> }) => {
    return apiClient.post('/ml/classification', data)
  },

  classifyBatch: (data: { model_id: number; features_list: Record<string, any>[] }) => {
    return apiClient.post('/ml/classification/batch', data)
  },

  uploadBatchFile: (formData: FormData) => {
    return apiClient.post('/ml/batch-predictions/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },
  
  listBatchJobs: (params?: { skip?: number; limit?: number; status?: string }) => {
    return apiClient.get('/ml/batch-predictions', { params })
  },

  createBatchJob: (data: any) => {
    return apiClient.post('/ml/batch-predictions', data)
  },

  batchJobStatus: (jobId: number) => {
    return apiClient.get(`/ml/batch-predictions/${jobId}`)
  },

  getBatchJob: (jobId: number) => {
    return apiClient.get(`/ml/batch-predictions/${jobId}`)
  },

  cancelBatchJob: (jobId: number) => {
    return apiClient.post(`/ml/batch-predictions/${jobId}/cancel`)
  },

  downloadBatchResults: (jobId: number) => {
    return apiClient.get(`/ml/batch-predictions/${jobId}/download`, {
      responseType: 'blob',
    })
  },

  modelVersions: (modelId: number, includeArchived: boolean = false) => {
    return apiClient.get(`/ml/models/${modelId}/versions`, {
      params: { include_archived: includeArchived },
    })
  },
  
  shapExplanation: (data: { model_id: number; features?: Record<string, any>; background_samples?: any; sample_size?: number }) => {
    return apiClient.post('/ml/explain/shap', data)
  },

  explainShap: (data: { model_id: number; features?: Record<string, any>; background_samples?: any; sample_size?: number }) => {
    return apiClient.post('/ml/explain/shap', data)
  },

  limeExplanation: (data: { model_id: number; features?: Record<string, any>; num_features?: number; num_samples?: number; sample_size?: number }) => {
    return apiClient.post('/ml/explain/lime', data)
  },

  explainLime: (data: { model_id: number; features?: Record<string, any>; num_features?: number; num_samples?: number; sample_size?: number }) => {
    return apiClient.post('/ml/explain/lime', data)
  },

  featureImportance: (modelId: number) => {
    return apiClient.get(`/ml/explain/feature-importance/${modelId}`)
  },

  getFeatureImportance: (data: { model_id: number; method?: string; n_repeats?: number }) => {
    return apiClient.post('/ml/explain/feature-importance', data)
  },

  getWaterfallData: (data: { model_id: number; features?: Record<string, any>; max_display?: number }) => {
    return apiClient.post('/ml/explain/waterfall', data)
  },
  
  optimize: (data: any) => {
    return apiClient.post('/ml/optimize', data)
  },

  optimizeHyperparameters: (data: any) => {
    return apiClient.post('/ml/optimize', data)
  },

  studySummary: (studyName: string) => {
    return apiClient.get(`/ml/optimize/${encodeURIComponent(studyName)}/summary`)
  },

  getStudySummary: (studyName: string) => {
    return apiClient.get(`/ml/optimize/${encodeURIComponent(studyName)}/summary`)
  },
  
  experiments: (params?: { skip?: number; limit?: number }) => {
    return apiClient.get('/ml/experiments', { params })
  },

  listExperiments: () => {
    return apiClient.get('/ml/experiments')
  },

  experimentDetails: (experimentName: string) => {
    return apiClient.get(`/ml/experiments/${encodeURIComponent(experimentName)}`)
  },

  getExperiment: (experimentName: string) => {
    return apiClient.get(`/ml/experiments/${encodeURIComponent(experimentName)}`)
  },

  searchRuns: (data: any) => {
    return apiClient.post('/ml/experiments/runs/search', data)
  },

  compareRuns: (experimentName: string, metricName: string, topN?: number) => {
    return apiClient.get(`/ml/experiments/${encodeURIComponent(experimentName)}/compare`, {
      params: { metric_name: metricName, top_n: topN },
    })
  },

  driftReports: (modelId: number, params?: { skip?: number; limit?: number }) => {
    return apiClient.get(`/ml/drift/${modelId}/reports`, { params })
  },

  getDriftReports: (modelId: number, limit?: number, offset?: number) => {
    return apiClient.get(`/ml/drift/${modelId}/reports`, {
      params: { limit, skip: offset },
    })
  },

  getLatestDriftReport: (modelId: number) => {
    return apiClient.get(`/ml/drift/${modelId}/latest`)
  },

  checkDrift: (data: any) => {
    return apiClient.post('/ml/drift/check', data)
  },
  
  retrain: (modelId: number, data?: any) => {
    return apiClient.post(`/ml/models/${modelId}/retrain`, data)
  },
}

// Data Endpoints
export const dataEndpoints = {
  sources: () => {
    return apiClient.get('/data/sources')
  },

  preview: (sourceId: number, limit: number = 100, offset: number = 0) => {
    return apiClient.get(`/data/sources/${sourceId}/preview`, {
      params: { limit, offset },
    })
  },

  stats: (sourceId: number) => {
    return apiClient.get(`/data/sources/${sourceId}/stats`)
  },

  deletePoint: (pointId: number) => {
    return apiClient.delete(`/data/points/${pointId}`)
  },
}

// Ingestion Endpoints
export const ingestionEndpoints = {
  uploadCSV: (formData: FormData) => {
    return apiClient.post('/ingestion/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },
  
  deleteSource: (sourceId: number) => {
    return apiClient.delete(`/ingestion/sources/${sourceId}`)
  },
  
  getETLJobs: (sourceId?: string) => {
    const params = sourceId ? { source_id: sourceId } : {}
    return apiClient.get('/ingestion/etl-jobs', { params })
  },
}

// Data Quality Endpoints
export const dataQualityEndpoints = {
  reports: (dataSourceId: number, params?: { skip?: number; limit?: number }) => {
    return apiClient.get(`/data/quality/${dataSourceId}/reports`, { params })
  },

  getQualityReports: (dataSourceId: number, limit?: number, offset?: number) => {
    return apiClient.get(`/data/quality/${dataSourceId}/reports`, {
      params: { limit, skip: offset },
    })
  },

  getDataProfile: (dataSourceId: number, sampleSize?: number) => {
    return apiClient.get(`/data/quality/${dataSourceId}/profile`, {
      params: { sample_size: sampleSize },
    })
  },

  getDataLineage: (dataSourceId: number) => {
    return apiClient.get(`/data/quality/${dataSourceId}/lineage`)
  },

  checkQuality: (data: { data_source_id: number }) => {
    return apiClient.post('/data/quality/check', data)
  },
}

