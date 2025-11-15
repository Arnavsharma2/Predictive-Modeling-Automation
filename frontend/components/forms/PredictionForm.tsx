'use client'

import { useState, useEffect } from 'react'
import { useModels } from '@/hooks/useModels'
import { usePrediction, useAnomalyDetection, useClassification } from '@/hooks/usePredictions'

interface PredictionFormProps {
  modelType?: 'regression' | 'classification' | 'anomaly_detection'
}

export default function PredictionForm({ modelType }: PredictionFormProps) {
  const { models, loading: modelsLoading } = useModels()
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null)
  const [features, setFeatures] = useState<Record<string, string>>({})
  const [featureKeys, setFeatureKeys] = useState<string[]>([])
  
  const prediction = usePrediction()
  const anomaly = useAnomalyDetection()
  const classification = useClassification()

  const selectedModel = models.find((m: any) => m.id === selectedModelId)
  // Use original_columns if available (new models), otherwise fall back to features (legacy models)
  const modelColumns = selectedModel?.original_columns || selectedModel?.features || []
  const featureNameMapping = selectedModel?.feature_name_mapping || {}

  useEffect(() => {
    if (selectedModel && modelColumns.length > 0) {
      const initialFeatures: Record<string, string> = {}
      modelColumns.forEach((key: string) => {
        initialFeatures[key] = ''
      })
      setFeatures(initialFeatures)
      setFeatureKeys(modelColumns)
    }
  }, [selectedModel, modelColumns])

  // Helper function to get readable feature name
  const getReadableFeatureName = (columnName: string): string => {
    // For original columns, format them nicely
    // First check if there's a mapping
    if (featureNameMapping[columnName]) {
      return featureNameMapping[columnName]
    }
    // Otherwise format the column name (capitalize, replace underscores)
    return columnName
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  const handleFeatureChange = (key: string, value: string) => {
    setFeatures(prev => ({ ...prev, [key]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedModelId) return

    // For models with original_columns, send mixed types (string/number)
    // For legacy models with transformed features, convert to numbers
    const processedFeatures: Record<string, string | number> = {}
    for (const [key, value] of Object.entries(features)) {
      if (selectedModel?.original_columns) {
        // New models: keep string values for categorical columns, parse numbers for numeric
        const numValue = parseFloat(value)
        processedFeatures[key] = isNaN(numValue) ? value : numValue
      } else {
        // Legacy models: convert everything to numbers
        processedFeatures[key] = parseFloat(value) || 0
      }
    }

    try {
      if (selectedModel?.type === 'classification') {
        await classification.classify(selectedModelId, processedFeatures)
      } else if (selectedModel?.type === 'anomaly_detection') {
        await anomaly.detect(selectedModelId, processedFeatures)
      } else {
        await prediction.predict(selectedModelId, processedFeatures)
      }
    } catch (err) {
      // Error handled by hook
    }
  }

  const result = selectedModel?.type === 'classification' 
    ? classification.result 
    : selectedModel?.type === 'anomaly_detection'
    ? anomaly.result
    : prediction.result

  const loading = selectedModel?.type === 'classification'
    ? classification.loading
    : selectedModel?.type === 'anomaly_detection'
    ? anomaly.loading
    : prediction.loading

  const error = selectedModel?.type === 'classification'
    ? classification.error
    : selectedModel?.type === 'anomaly_detection'
    ? anomaly.error
    : prediction.error

  // Filter models by type if specified
  const filteredModels = modelType 
    ? models.filter((m: any) => m.type === modelType && (m.status === 'trained' || m.status === 'deployed'))
    : models.filter((m: any) => m.status === 'trained' || m.status === 'deployed')

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label className="block text-sm font-semibold text-gray-soft-700 mb-2 uppercase tracking-wider">
          Select Model
        </label>
        <select
          value={selectedModelId || ''}
          onChange={(e) => setSelectedModelId(parseInt(e.target.value) || null)}
          className="w-full px-4 py-3 bg-white border border-gray-soft-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-pastel-green focus:border-pastel-green text-gray-soft-700 transition-all"
          required
        >
          <option value="">Choose a model...</option>
          {filteredModels.map((model: any) => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.type})
            </option>
          ))}
        </select>
      </div>

      {selectedModel && featureKeys.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-gray-soft-700">
            {selectedModel.original_columns ? 'Input Data' : 'Features'}
          </h3>
          <p className="text-sm text-gray-soft-500 mb-4">
            {selectedModel.original_columns
              ? `Enter values for each column from your dataset. The model will automatically handle preprocessing and transformations.`
              : 'Enter values for each feature to make a prediction. Features are shown using their original dataset names where possible.'}
          </p>
          {featureKeys.map((key) => {
            const readableName = getReadableFeatureName(key)

            return (
              <div key={key}>
                <label
                  className="block text-sm font-semibold text-gray-soft-700 mb-2 uppercase tracking-wider"
                >
                  {readableName}
                </label>
                <input
                  type="text"
                  value={features[key] || ''}
                  onChange={(e) => handleFeatureChange(key, e.target.value)}
                  className="w-full px-4 py-3 bg-white border border-gray-soft-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-pastel-green focus:border-pastel-green text-gray-soft-700 placeholder-gray-soft-400 transition-all"
                  placeholder={`Enter ${readableName.toLowerCase()}`}
                  required
                />
              </div>
            )
          })}
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-accent-50 border border-accent-100 text-accent-500 px-4 py-3 rounded-lg">
          <h4 className="font-semibold mb-2">Prediction Result:</h4>
          {selectedModel?.type === 'classification' && result.predicted_class && (
            <div>
              <p className="font-medium">Predicted Class: {result.predicted_class}</p>
              {result.probabilities && (
                <div className="mt-2">
                  <p className="text-sm font-medium">Probabilities:</p>
                  <ul className="list-disc list-inside text-sm">
                    {Object.entries(result.probabilities).map(([cls, prob]: [string, any]) => (
                      <li key={cls}>{cls}: {(prob * 100).toFixed(2)}%</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          {selectedModel?.type === 'anomaly_detection' && (
            <div>
              <p className="font-medium">
                {result.is_anomaly ? 'Anomaly Detected' : 'Normal'}
              </p>
              <p className="text-sm">Score: {result.score?.toFixed(4)}</p>
              <p className="text-sm">Probability: {(result.probability * 100).toFixed(2)}%</p>
            </div>
          )}
          {selectedModel?.type === 'regression' && (
            <div>
              <p className="font-medium">Prediction: {result.prediction?.toFixed(4)}</p>
              {result.confidence && (
                <p className="text-sm">
                  Confidence: {result.confidence.lower_bound?.toFixed(4)} - {result.confidence.upper_bound?.toFixed(4)}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      <button
        type="submit"
        disabled={loading || !selectedModelId || featureKeys.length === 0}
        className="w-full bg-pastel-green text-gray-soft-700 py-3 px-4 rounded-lg font-semibold transition-all duration-300 hover:bg-pastel-mint disabled:bg-gray-soft-300 disabled:text-gray-soft-500 disabled:cursor-not-allowed shadow-sm hover:shadow-md"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Predicting...
          </span>
        ) : (
          'Make Prediction'
        )}
      </button>
    </form>
  )
}

