'use client'

import { useState } from 'react'
import { useExperiments, useRunSearch, useRunComparison } from '@/hooks/useExperiments'

export default function ExperimentsPage() {
  const { experiments, loading: experimentsLoading, refetch: refetchExperiments } = useExperiments()
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null)
  const [selectedMetric, setSelectedMetric] = useState<string>('test_rmse')

  const { runs, loading: runsLoading } = useRunSearch(selectedExperiment)
  const { comparison, loading: comparisonLoading } = useRunComparison(selectedExperiment, selectedMetric)

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-soft-700 mb-2">MLflow Experiments</h1>
        <p className="text-gray-soft-500">Track and compare your ML experiments</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Experiments List */}
        <div className="lg:col-span-1">
          <div className="glass rounded-2xl border-pastel-blue/40 p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-soft-700">Experiments</h2>
              <button
                onClick={refetchExperiments}
                className="text-sm text-pastel-blue hover:text-pastel-blue/80"
              >
                Refresh
              </button>
            </div>
            {experimentsLoading ? (
              <p className="text-gray-500">Loading experiments...</p>
            ) : experiments.length === 0 ? (
              <p className="text-gray-500">No experiments found</p>
            ) : (
              <div className="space-y-2">
                {experiments.map((exp: any) => (
                  <button
                    key={exp.experiment_id}
                    onClick={() => setSelectedExperiment(exp.name)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedExperiment === exp.name
                        ? 'bg-pastel-blue text-white'
                        : 'bg-gray-50 hover:bg-gray-100 text-gray-soft-700'
                    }`}
                  >
                    <div className="font-semibold">{exp.name}</div>
                    <div className="text-xs opacity-75 mt-1">ID: {exp.experiment_id}</div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Runs and Comparison */}
        <div className="lg:col-span-2">
          {selectedExperiment ? (
            <>
              <div className="glass rounded-2xl border-pastel-blue/40 p-6 mb-6">
                <h2 className="text-xl font-bold text-gray-soft-700 mb-4">
                  Runs: {selectedExperiment}
                </h2>
                <div className="mb-4">
                  <label className="block text-sm font-semibold text-gray-soft-700 mb-2">
                    Compare by Metric
                  </label>
                  <select
                    value={selectedMetric}
                    onChange={(e) => setSelectedMetric(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pastel-blue focus:border-transparent"
                  >
                    <optgroup label="Regression Metrics">
                      <option value="test_rmse">Test RMSE</option>
                      <option value="test_mae">Test MAE</option>
                      <option value="test_r2">Test R² Score</option>
                    </optgroup>
                    <optgroup label="Classification Metrics">
                      <option value="test_accuracy">Test Accuracy</option>
                      <option value="test_f1_score">Test F1 Score</option>
                      <option value="test_precision">Test Precision</option>
                      <option value="test_recall">Test Recall</option>
                    </optgroup>
                  </select>
                </div>
                {runsLoading ? (
                  <p className="text-gray-500">Loading runs...</p>
                ) : runs.length === 0 ? (
                  <p className="text-gray-500">No runs found for this experiment</p>
                ) : (
                  <div className="space-y-2">
                    {runs.slice(0, 10).map((run: any) => (
                      <div key={run.run_id} className="p-4 bg-gray-50 rounded-lg">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <div className="font-semibold text-gray-soft-700">
                              {run.run_name || run.run_id}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              Status: {run.status}
                            </div>
                          </div>
                          {run.metrics && run.metrics[selectedMetric] !== undefined && (
                            <div className="text-right">
                              <div className="text-sm font-semibold text-pastel-blue">
                                {selectedMetric}: {run.metrics[selectedMetric].toFixed(4)}
                              </div>
                            </div>
                          )}
                        </div>
                        {run.params && Object.keys(run.params).length > 0 && (
                          <div className="mt-2 text-xs text-gray-600">
                            <strong>Params:</strong> {Object.entries(run.params).slice(0, 3).map(([k, v]) => `${k}=${v}`).join(', ')}
                            {Object.keys(run.params).length > 3 && '...'}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {comparison && (
                <div className="glass rounded-2xl border-pastel-blue/40 p-6">
                  <h2 className="text-xl font-bold text-gray-soft-700 mb-4">Run Comparison</h2>
                  {comparison.best_run_id && (
                    <div className="mb-4 p-4 bg-pastel-blue/10 rounded-lg">
                      <p className="text-sm">
                        <strong>Best Run:</strong> {comparison.best_run_id}
                      </p>
                      <p className="text-sm">
                        <strong>Best {selectedMetric}:</strong> {comparison.best_metric_value?.toFixed(4)}
                      </p>
                    </div>
                  )}
                  <div className="space-y-2">
                    {comparison.runs.slice(0, 5).map((run: any) => (
                      <div key={run.run_id} className="p-3 bg-gray-50 rounded-lg">
                        <div className="flex justify-between items-center">
                          <span className="font-medium text-sm">{run.run_name || run.run_id}</span>
                          {run.metrics && run.metrics[selectedMetric] !== undefined && (
                            <span className="text-sm font-semibold text-pastel-blue">
                              {run.metrics[selectedMetric].toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="glass rounded-2xl border-pastel-blue/40 p-6">
              <p className="text-gray-500 text-center py-8">
                Select an experiment to view runs and comparisons
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

