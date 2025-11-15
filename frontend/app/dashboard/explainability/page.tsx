'use client'

import { useState, useEffect } from 'react'
import { useModels } from '@/hooks/useModels'
import { useShapExplanation, useLimeExplanation, useFeatureImportance, useWaterfallData } from '@/hooks/useExplainability'
import FeatureImportanceChart from '@/components/charts/FeatureImportanceChart'
import ShapWaterfallChart from '@/components/charts/ShapWaterfallChart'

export default function ExplainabilityPage() {
  const { models } = useModels()
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState<'importance' | 'shap' | 'lime'>('importance')
  const [features, setFeatures] = useState<Record<string, string>>({})
  const [featureKeys, setFeatureKeys] = useState<string[]>([])
  const [explainMode, setExplainMode] = useState<'global' | 'instance'>('global')
  
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

  const { getImportance, data: importanceData, loading: importanceLoading } = useFeatureImportance()
  const { explain: explainShap, data: shapData, loading: shapLoading } = useShapExplanation()
  const { explain: explainLime, data: limeData, loading: limeLoading } = useLimeExplanation()
  const { getWaterfall, data: waterfallData, loading: waterfallLoading } = useWaterfallData()

  const handleGetImportance = async () => {
    if (!selectedModelId) return
    await getImportance(selectedModelId, 'model')
  }

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

  const handleExplainShap = async () => {
    if (!selectedModelId) return
    
    try {
      if (explainMode === 'global') {
        // Global explanation - no features needed
        await explainShap(selectedModelId, undefined, undefined, 100)
      } else {
        // Instance explanation - process features
        const processedFeatures: Record<string, string | number> = {}
        for (const [key, value] of Object.entries(features)) {
          if (selectedModel?.original_columns) {
            const numValue = parseFloat(value)
            processedFeatures[key] = isNaN(numValue) ? value : numValue
          } else {
            processedFeatures[key] = parseFloat(value) || 0
          }
        }
        await explainShap(selectedModelId, processedFeatures)
        // Try to get waterfall data for instance explanations
        try {
          await getWaterfall(selectedModelId, processedFeatures)
        } catch (err) {
          // Waterfall is optional, continue without it
          console.debug('Waterfall data not available:', err)
        }
      }
    } catch (error) {
      console.error('Error explaining with SHAP:', error)
    }
  }

  const handleExplainLime = async () => {
    if (!selectedModelId) return
    
    try {
      if (explainMode === 'global') {
        // Global explanation - no features needed
        await explainLime(selectedModelId, undefined, 10, 5000, 100)
      } else {
        // Instance explanation - process features
        const processedFeatures: Record<string, string | number> = {}
        for (const [key, value] of Object.entries(features)) {
          if (selectedModel?.original_columns) {
            const numValue = parseFloat(value)
            processedFeatures[key] = isNaN(numValue) ? value : numValue
          } else {
            processedFeatures[key] = parseFloat(value) || 0
          }
        }
        await explainLime(selectedModelId, processedFeatures)
      }
    } catch (error) {
      console.error('Error explaining with LIME:', error)
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-soft-700 mb-2">Model Explainability</h1>
        <p className="text-gray-soft-600">Understand how your models make predictions</p>
      </div>

      <div className="glass rounded-2xl border-pastel-green/40 p-6 mb-6">
        <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
          Select Model
        </label>
        <select
          value={selectedModelId || ''}
          onChange={(e) => setSelectedModelId(e.target.value ? parseInt(e.target.value) : null)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
        >
          <option value="">Choose a model...</option>
          {models.map((model: any) => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.type})
            </option>
          ))}
        </select>
        {selectedModel && (
          <div className="mt-4 text-sm text-gray-soft-600">
            <p>Original Columns: {modelColumns.length}</p>
            <p>Status: {selectedModel.status}</p>
          </div>
        )}
      </div>

      {selectedModelId && (
        <>
          <div className="mb-6">
            <div className="flex gap-2 border-b border-gray-200">
              <button
                onClick={() => setActiveTab('importance')}
                className={`px-4 py-2 font-semibold transition-colors ${
                  activeTab === 'importance'
                    ? 'text-pastel-green border-b-2 border-pastel-green'
                    : 'text-gray-soft-600 hover:text-gray-soft-700'
                }`}
              >
                Feature Importance
              </button>
              <button
                onClick={() => setActiveTab('shap')}
                className={`px-4 py-2 font-semibold transition-colors ${
                  activeTab === 'shap'
                    ? 'text-pastel-green border-b-2 border-pastel-green'
                    : 'text-gray-soft-600 hover:text-gray-soft-700'
                }`}
              >
                SHAP Explanation
              </button>
              <button
                onClick={() => setActiveTab('lime')}
                className={`px-4 py-2 font-semibold transition-colors ${
                  activeTab === 'lime'
                    ? 'text-pastel-green border-b-2 border-pastel-green'
                    : 'text-gray-soft-600 hover:text-gray-soft-700'
                }`}
              >
                LIME Explanation
              </button>
            </div>
          </div>

          {activeTab === 'importance' && (
            <div className="glass rounded-2xl border-pastel-green/40 p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-soft-700">Feature Importance</h2>
                <button
                  onClick={handleGetImportance}
                  disabled={importanceLoading}
                  className="px-4 py-2 bg-pastel-green text-gray-soft-700 rounded-lg hover:bg-pastel-green/90 disabled:opacity-50"
                >
                  {importanceLoading ? 'Loading...' : 'Get Importance'}
                </button>
              </div>
              {importanceData && (
                <FeatureImportanceChart
                  importance={importanceData.importance || {}}
                  title="Model Feature Importance"
                />
              )}
            </div>
          )}

          {activeTab === 'shap' && (
            <div className="glass rounded-2xl border-pastel-green/40 p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-soft-700">SHAP Explanation</h2>
                <div className="flex gap-2">
                  <button
                    onClick={() => setExplainMode('global')}
                    className={`px-3 py-1 text-sm rounded-lg ${
                      explainMode === 'global'
                        ? 'bg-pastel-green text-white'
                        : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    Global
                  </button>
                  <button
                    onClick={() => setExplainMode('instance')}
                    className={`px-3 py-1 text-sm rounded-lg ${
                      explainMode === 'instance'
                        ? 'bg-pastel-green text-white'
                        : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    Instance
                  </button>
                </div>
              </div>
              {explainMode === 'instance' && (
                <div className="mb-4">
                  <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
                    Enter Feature Values (Original Dataset Columns)
                  </label>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {featureKeys.map((key: string) => (
                    <div key={key}>
                      <label className="block text-xs font-medium text-gray-soft-600 mb-1">
                        {getReadableFeatureName(key)}
                      </label>
                      <input
                        type="text"
                        value={features[key] || ''}
                        onChange={(e) => handleFeatureChange(key, e.target.value)}
                        placeholder={`Enter value for ${key}`}
                        className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
                      />
                    </div>
                  ))}
                </div>
                  <p className="text-xs text-gray-soft-600 mt-2">
                    Enter values for the original dataset columns. The system will automatically apply preprocessing.
                  </p>
                </div>
              )}
              {explainMode === 'global' && (
                <div className="mb-4 p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-gray-soft-700">
                    Global explanation will analyze the training data to show overall feature importance across all instances.
                  </p>
                </div>
              )}
              <button
                onClick={handleExplainShap}
                disabled={shapLoading || (explainMode === 'instance' && Object.values(features).every(v => !v))}
                className="px-4 py-2 bg-pastel-green text-white rounded-lg hover:bg-pastel-green/90 disabled:opacity-50 mb-4"
              >
                {shapLoading ? 'Generating...' : explainMode === 'global' ? 'Generate Global SHAP Explanation' : 'Explain with SHAP'}
              </button>
              {shapData && (
                <div className="mt-6">
                  <div className="mb-4 p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm"><strong>Prediction:</strong> {shapData.prediction?.toFixed(4)}</p>
                    <p className="text-sm"><strong>Base Value:</strong> {shapData.base_value?.toFixed(4)}</p>
                  </div>
                  
                  {/* Debug: Show raw data structure */}
                  {process.env.NODE_ENV === 'development' && (
                    <details className="mb-4 text-xs">
                      <summary className="cursor-pointer text-gray-600">Debug: SHAP Data Structure</summary>
                      <pre className="mt-2 p-2 bg-gray-100 rounded overflow-auto max-h-40">
                        {JSON.stringify(shapData, null, 2)}
                      </pre>
                    </details>
                  )}
                  
                  {/* Show feature contributions - check multiple possible structures */}
                  {(() => {
                    const contributions = shapData.feature_contributions || shapData.contributions || []
                    if (contributions.length > 0) {
                      return (
                        <div className="mb-6">
                          <h3 className="text-lg font-semibold text-gray-soft-700 mb-4">Feature Contributions</h3>
                          <div className="space-y-2 max-h-96 overflow-y-auto">
                            {contributions.map((contrib: any, idx: number) => {
                              const value = contrib.shap_value || contrib.value || 0
                              const featureName = contrib.feature || contrib.name || `Feature ${idx}`
                              const isPositive = value >= 0
                              const absValue = Math.abs(value)
                              const maxAbs = Math.max(...contributions.map((c: any) => Math.abs(c.shap_value || c.value || 0)))
                              const percentage = maxAbs > 0 ? (absValue / maxAbs) * 100 : 0
                              
                              return (
                                <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                                  <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-gray-soft-700">
                                      {featureName}
                                    </span>
                                    <span className={`text-sm font-semibold ${
                                      isPositive ? 'text-pastel-green' : 'text-red-500'
                                    }`}>
                                      {isPositive ? '+' : ''}{value.toFixed(4)}
                                    </span>
                                  </div>
                                  <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                                    <div
                                      className={`h-full transition-all duration-300 ${
                                        isPositive ? 'bg-pastel-green' : 'bg-red-500'
                                      }`}
                                      style={{ width: `${percentage}%` }}
                                    />
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      )
                    }
                    return null
                  })()}
                  
                  {/* Show waterfall chart for instance explanations */}
                  {waterfallData && explainMode === 'instance' && (
                    <div className="mt-6">
                      <ShapWaterfallChart
                        baseValue={waterfallData.base_value}
                        prediction={waterfallData.prediction}
                        contributions={waterfallData.contributions || []}
                      />
                    </div>
                  )}
                  
                  {/* Show waterfall chart using SHAP data if waterfallData not available */}
                  {!waterfallData && explainMode === 'instance' && (() => {
                    const contributions = shapData.feature_contributions || shapData.contributions || []
                    if (contributions.length > 0) {
                      return (
                        <div className="mt-6">
                          <ShapWaterfallChart
                            baseValue={shapData.base_value}
                            prediction={shapData.prediction}
                            contributions={contributions.map((c: any) => ({
                              feature: c.feature || c.name,
                              shap_value: c.shap_value || c.value || 0
                            }))}
                          />
                        </div>
                      )
                    }
                    return null
                  })()}
                  
                  {/* Show waterfall chart for global explanations too */}
                  {explainMode === 'global' && (() => {
                    const contributions = shapData.feature_contributions || shapData.contributions || []
                    if (contributions.length > 0) {
                      return (
                        <div className="mt-6">
                          <h3 className="text-lg font-semibold text-gray-soft-700 mb-4">SHAP Waterfall (Global)</h3>
                          <ShapWaterfallChart
                            baseValue={shapData.base_value}
                            prediction={shapData.prediction}
                            contributions={contributions.map((c: any) => ({
                              feature: c.feature || c.name,
                              shap_value: c.shap_value || c.value || 0
                            }))}
                          />
                        </div>
                      )
                    }
                    return null
                  })()}
                </div>
              )}
            </div>
          )}

          {activeTab === 'lime' && (
            <div className="glass rounded-2xl border-pastel-green/40 p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-soft-700">LIME Explanation</h2>
                <div className="flex gap-2">
                  <button
                    onClick={() => setExplainMode('global')}
                    className={`px-3 py-1 text-sm rounded-lg ${
                      explainMode === 'global'
                        ? 'bg-pastel-green text-white'
                        : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    Global
                  </button>
                  <button
                    onClick={() => setExplainMode('instance')}
                    className={`px-3 py-1 text-sm rounded-lg ${
                      explainMode === 'instance'
                        ? 'bg-pastel-green text-white'
                        : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    Instance
                  </button>
                </div>
              </div>
              {explainMode === 'instance' && (
                <div className="mb-4">
                  <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
                    Enter Feature Values (Original Dataset Columns)
                  </label>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {featureKeys.map((key: string) => (
                    <div key={key}>
                      <label className="block text-xs font-medium text-gray-soft-600 mb-1">
                        {getReadableFeatureName(key)}
                      </label>
                      <input
                        type="text"
                        value={features[key] || ''}
                        onChange={(e) => handleFeatureChange(key, e.target.value)}
                        placeholder={`Enter value for ${key}`}
                        className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
                      />
                    </div>
                  ))}
                </div>
                  <p className="text-xs text-gray-soft-600 mt-2">
                    Enter values for the original dataset columns. The system will automatically apply preprocessing.
                  </p>
                </div>
              )}
              {explainMode === 'global' && (
                <div className="mb-4 p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-gray-soft-700">
                    Global explanation will analyze multiple training instances to show overall feature importance.
                  </p>
                </div>
              )}
              <button
                onClick={handleExplainLime}
                disabled={limeLoading || (explainMode === 'instance' && Object.values(features).every(v => !v))}
                className="px-4 py-2 bg-pastel-green text-white rounded-lg hover:bg-pastel-green/90 disabled:opacity-50 mb-4"
              >
                {limeLoading ? 'Generating...' : explainMode === 'global' ? 'Generate Global LIME Explanation' : 'Explain with LIME'}
              </button>
              {limeData && (
                <div className="mt-6">
                  <div className="mb-4 p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm"><strong>Prediction:</strong> {Array.isArray(limeData.prediction) ? limeData.prediction.join(', ') : limeData.prediction?.toFixed(4)}</p>
                    {limeData.local_prediction && (
                      <p className="text-sm"><strong>Local Prediction:</strong> {limeData.local_prediction.toFixed(4)}</p>
                    )}
                    {limeData.score && (
                      <p className="text-sm"><strong>Local Model Score:</strong> {limeData.score.toFixed(4)}</p>
                    )}
                  </div>
                  {limeData.feature_contributions && (
                    <div>
                      <h3 className="text-lg font-semibold mb-4">Feature Contributions</h3>
                      <div className="space-y-2">
                        {limeData.feature_contributions.map((contrib: any, idx: number) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                            <div className="flex justify-between items-center">
                              <span className="font-medium">{contrib.feature}</span>
                              <span className={`font-semibold ${contrib.value >= 0 ? 'text-pastel-green' : 'text-red-500'}`}>
                                {contrib.value >= 0 ? '+' : ''}{contrib.value.toFixed(4)}
                              </span>
                            </div>
                            {contrib.description && (
                              <p className="text-xs text-gray-soft-600 mt-1">{contrib.description}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

