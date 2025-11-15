'use client'

import { useState } from 'react'
import { useModels } from '@/hooks/useModels'
import { useHyperparameterOptimization } from '@/hooks/useOptimization'

interface HyperparameterOptimizationFormProps {
  onOptimize?: (result: any) => void
}

export default function HyperparameterOptimizationForm({ onOptimize }: HyperparameterOptimizationFormProps) {
  const { models } = useModels()
  const { optimize, loading, error, result } = useHyperparameterOptimization()
  
  const [formData, setFormData] = useState({
    model_id: '',
    algorithm: 'random_forest',
    metric_name: 'rmse',
    direction: 'minimize',
    n_trials: 50,
    timeout: undefined as number | undefined,
    study_name: '',
  })

  const [formError, setFormError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setFormError(null)

    // Validate model selection
    if (!formData.model_id) {
      setFormError('Please select a model to optimize')
      return
    }

    try {
      const config = {
        ...formData,
        model_id: parseInt(formData.model_id),
        n_trials: formData.n_trials,
        timeout: formData.timeout || undefined,
        study_name: formData.study_name || `${formData.algorithm}_optimization_${Date.now()}`,
      }
      const optimizationResult = await optimize(config)
      if (onOptimize) {
        onOptimize(optimizationResult)
      }
    } catch (err) {
      console.error('Optimization error:', err)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
          Model *
        </label>
        <select
          value={formData.model_id}
          onChange={(e) => setFormData({ ...formData, model_id: e.target.value })}
          required
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
        >
          <option value="">Select a model to optimize</option>
          {models.map((model: any) => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.type})
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500 mt-1">
          Select an existing model to optimize its hyperparameters
        </p>
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
          Algorithm *
        </label>
        <select
          value={formData.algorithm}
          onChange={(e) => setFormData({ ...formData, algorithm: e.target.value })}
          required
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
        >
          <option value="random_forest">Random Forest</option>
          <option value="xgboost">XGBoost</option>
          <option value="lightgbm">LightGBM</option>
          <option value="catboost">CatBoost</option>
          <option value="gradient_boosting">Gradient Boosting</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
            Metric to Optimize
          </label>
          <select
            value={formData.metric_name}
            onChange={(e) => setFormData({ ...formData, metric_name: e.target.value })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
          >
            <option value="rmse">RMSE</option>
            <option value="mae">MAE</option>
            <option value="r2_score">R² Score</option>
            <option value="accuracy">Accuracy</option>
            <option value="f1_score">F1 Score</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
            Direction
          </label>
          <select
            value={formData.direction}
            onChange={(e) => setFormData({ ...formData, direction: e.target.value })}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
          >
            <option value="minimize">Minimize</option>
            <option value="maximize">Maximize</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
            Number of Trials
          </label>
          <input
            type="number"
            value={formData.n_trials}
            onChange={(e) => {
              const value = e.target.value;
              const parsed = parseInt(value, 10);
              // Only update if we have a valid number, otherwise keep current value
              if (!isNaN(parsed) && parsed >= 1 && parsed <= 1000) {
                setFormData({ ...formData, n_trials: parsed });
              }
            }}
            onBlur={(e) => {
              const value = e.target.value;
              const parsed = parseInt(value, 10);
              // Only reset to default if invalid on blur
              if (isNaN(parsed) || parsed < 1 || parsed > 1000) {
                setFormData({ ...formData, n_trials: 50 });
              }
            }}
            min={1}
            max={1000}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
            Timeout (seconds, optional)
          </label>
          <input
            type="number"
            value={formData.timeout || ''}
            onChange={(e) => setFormData({ ...formData, timeout: e.target.value ? parseInt(e.target.value) : undefined })}
            min={0}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
          Study Name (optional)
        </label>
        <input
          type="text"
          value={formData.study_name}
          onChange={(e) => setFormData({ ...formData, study_name: e.target.value })}
          placeholder="Auto-generated if not provided"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-green focus:border-transparent"
        />
      </div>

      {(error || formError) && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700 font-semibold text-sm mb-1">Error</p>
          <p className="text-red-600 text-sm">{formError || error}</p>
        </div>
      )}

      <button
        type="submit"
        disabled={loading || !formData.algorithm || !formData.model_id}
        className="w-full px-6 py-3 bg-pastel-green text-white rounded-lg hover:bg-pastel-green/90 disabled:opacity-50 disabled:cursor-not-allowed font-semibold transition-all"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            Starting Optimization...
          </span>
        ) : (
          'Start Optimization'
        )}
      </button>

      {result && (
        <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-start gap-2">
            <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-green-800 mb-1">Optimization Started</p>
              <p className="text-xs text-green-700 mb-1">
                <span className="font-medium">Study:</span> {result.study_name}
              </p>
              <p className="text-xs text-green-700">
                <span className="font-medium">Status:</span> {result.status || 'running'}
              </p>
              <p className="text-xs text-green-600 mt-2">
                The optimization is running in the background. Check the study summary panel for progress.
              </p>
            </div>
          </div>
        </div>
      )}
    </form>
  )
}

