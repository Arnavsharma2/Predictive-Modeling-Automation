'use client'

import { useState, useEffect } from 'react'
import HyperparameterOptimizationForm from '@/components/forms/HyperparameterOptimizationForm'
import { useStudySummary } from '@/hooks/useOptimization'

export default function OptimizationPage() {
  const [studyName, setStudyName] = useState<string | null>(null)
  const { summary, loading, error, refetch, isPolling, setIsPolling } = useStudySummary(studyName, true, 3000)

  const handleOptimize = (result: any) => {
    if (result?.study_name) {
      setStudyName(result.study_name)
      setIsPolling(true)
      // Initial fetch after a short delay
      setTimeout(() => refetch(), 1000)
    }
  }

  // Determine status
  const getStatus = () => {
    if (!studyName) return null
    if (loading && !summary) return 'starting'
    if (summary && summary.n_trials === 0) return 'starting'
    if (summary && summary.n_trials > 0 && summary.best_value !== 0) return 'completed'
    if (isPolling) return 'running'
    return 'unknown'
  }

  const status = getStatus()

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-soft-700 mb-2">Hyperparameter Optimization</h1>
        <p className="text-gray-soft-500">Optimize model hyperparameters using Optuna</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Optimization Form */}
        <div className="glass rounded-2xl border-pastel-purple/40 p-6">
          <h2 className="text-xl font-bold text-gray-soft-700 mb-6">Start Optimization</h2>
          <HyperparameterOptimizationForm onOptimize={handleOptimize} />
        </div>

        {/* Study Summary */}
        <div className="glass rounded-2xl border-pastel-purple/40 p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-gray-soft-700">Study Summary</h2>
            {isPolling && (
              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                Auto-refreshing...
              </span>
            )}
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
              Study Name
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={studyName || ''}
                onChange={(e) => {
                  const newName = e.target.value || null
                  setStudyName(newName)
                  if (newName) {
                    setIsPolling(true)
                  }
                }}
                placeholder="Enter study name to track progress"
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-purple focus:border-transparent"
              />
              <button
                onClick={() => {
                  if (studyName) {
                    refetch()
                    setIsPolling(true)
                  }
                }}
                disabled={loading || !studyName}
                className="px-4 py-2 bg-pastel-purple text-white rounded-lg hover:bg-pastel-purple/90 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-semibold"
              >
                {loading ? 'Loading...' : 'Refresh'}
              </button>
            </div>
          </div>

          {studyName ? (
            <>
              
              {loading && !summary ? (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <p className="text-blue-700 text-sm">Loading study summary...</p>
                  </div>
                </div>
              ) : error ? (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-700 text-sm font-semibold mb-1">Error</p>
                  <p className="text-red-600 text-sm">{error}</p>
                  <p className="text-red-500 text-xs mt-2">
                    The study may still be starting. Try refreshing in a few moments.
                  </p>
                </div>
              ) : summary ? (
                <div className="space-y-4">
                  {/* Status Badge */}
                  {status && (
                    <div className={`p-3 rounded-lg ${
                      status === 'completed' ? 'bg-green-50 border border-green-200' :
                      status === 'running' ? 'bg-blue-50 border border-blue-200' :
                      'bg-yellow-50 border border-yellow-200'
                    }`}>
                      <div className="flex items-center gap-2">
                        {status === 'running' && (
                          <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
                        )}
                        <span className={`text-sm font-semibold ${
                          status === 'completed' ? 'text-green-800' :
                          status === 'running' ? 'text-blue-800' :
                          'text-yellow-800'
                        }`}>
                          {status === 'completed' ? 'Completed' :
                           status === 'running' ? 'Running...' :
                           'Starting...'}
                        </span>
                      </div>
                    </div>
                  )}
                  
                  <div className="p-4 bg-gray-50 rounded-lg space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 font-medium">Study:</span>
                      <span className="font-semibold text-gray-soft-700">{summary.study_name}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 font-medium">Trials:</span>
                      <span className="font-semibold text-gray-soft-700">{summary.n_trials}</span>
                    </div>
                    {summary.best_value !== 0 && (
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 font-medium">Best Value:</span>
                        <span className="font-semibold text-green-600">
                          {typeof summary.best_value === 'number' 
                            ? summary.best_value.toFixed(4) 
                            : summary.best_value}
                        </span>
                      </div>
                    )}
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 font-medium">Direction:</span>
                      <span className="font-semibold text-gray-soft-700 capitalize">{summary.direction}</span>
                    </div>
                  </div>
                  
                  {summary.best_params && Object.keys(summary.best_params).length > 0 && (
                    <div className="p-4 bg-white border border-gray-200 rounded-lg">
                      <h3 className="text-sm font-semibold text-gray-soft-700 mb-3">Best Parameters</h3>
                      <div className="space-y-2">
                        {Object.entries(summary.best_params).map(([key, value]) => (
                          <div key={key} className="flex justify-between items-center text-sm py-1 border-b border-gray-100 last:border-0">
                            <span className="text-gray-600 font-mono text-xs">{key}:</span>
                            <span className="font-semibold text-gray-soft-700">{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {summary.n_trials === 0 && (
                    <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <p className="text-yellow-700 text-xs">
                        Optimization is starting. Trials will appear here once they begin.
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-gray-500 text-sm text-center">
                    Study not found. It may still be starting or the name may be incorrect.
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <div className="mb-4">
                <svg className="w-16 h-16 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <p className="text-gray-500 mb-2 font-medium">No study selected</p>
              <p className="text-gray-400 text-sm">Start an optimization to view results, or enter a study name above to track an existing study</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

