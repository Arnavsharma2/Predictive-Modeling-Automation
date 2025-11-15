'use client'

import { useState } from 'react'
import { useModels } from '@/hooks/useModels'
import { useDriftReports, useLatestDriftReport, useCheckDrift } from '@/hooks/useDrift'
import { useDataSources } from '@/hooks/useData'
import DriftChart from '@/components/charts/DriftChart'

export default function DriftPage() {
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null)
  const [selectedDataSourceId, setSelectedDataSourceId] = useState<number | null>(null)
  const [checking, setChecking] = useState(false)

  const { models, loading: modelsLoading } = useModels()
  const { sources } = useDataSources()
  const { reports, loading: reportsLoading, refetch: refetchReports } = useDriftReports(selectedModelId)
  const { report: latestReport, loading: latestLoading, refetch: refetchLatest } = useLatestDriftReport(selectedModelId)
  const { checkDrift, loading: checkLoading, error: checkError } = useCheckDrift()

  const handleCheckDrift = async () => {
    if (!selectedModelId) return

    try {
      setChecking(true)
      await checkDrift({
        model_id: selectedModelId,
        current_data_source_id: selectedDataSourceId || undefined,
      })
      // Refresh reports after checking
      await refetchReports()
      await refetchLatest()
    } catch (err) {
      console.error('Error checking drift:', err)
    } finally {
      setChecking(false)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-300'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      case 'low':
        return 'bg-blue-100 text-blue-800 border-blue-300'
      default:
        return 'bg-green-100 text-green-800 border-green-300'
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-soft-700">Model Drift Detection</h1>
          <p className="text-gray-soft-500 mt-2">Monitor and detect data drift in your ML models</p>
        </div>
      </div>

      {/* Model Selection */}
      <div className="glass-strong p-6 rounded-xl border border-pastel-blue/30">
        <h2 className="text-xl font-semibold text-gray-soft-700 mb-4">Select Model</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-soft-700 mb-2">Model</label>
            <select
              value={selectedModelId || ''}
              onChange={(e) => setSelectedModelId(e.target.value ? parseInt(e.target.value) : null)}
              className="w-full px-4 py-2 border border-pastel-blue/30 rounded-lg focus:ring-2 focus:ring-pastel-blue focus:border-transparent"
            >
              <option value="">Select a model</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.type})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-soft-700 mb-2">Current Data Source (Optional)</label>
            <select
              value={selectedDataSourceId || ''}
              onChange={(e) => setSelectedDataSourceId(e.target.value ? parseInt(e.target.value) : null)}
              className="w-full px-4 py-2 border border-pastel-blue/30 rounded-lg focus:ring-2 focus:ring-pastel-blue focus:border-transparent"
            >
              <option value="">Use model's training data</option>
              {sources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="mt-4">
          <button
            onClick={handleCheckDrift}
            disabled={!selectedModelId || checkLoading || checking}
            className="px-6 py-2 bg-pastel-blue text-white rounded-lg hover:bg-pastel-blue/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {checkLoading || checking ? 'Checking...' : 'Check for Drift'}
          </button>
          {checkError && (
            <div className="mt-2 text-sm text-red-600">{checkError}</div>
          )}
        </div>
      </div>

      {/* Latest Drift Report */}
      {selectedModelId && (
        <div className="glass-strong p-6 rounded-xl border border-pastel-blue/30">
          <h2 className="text-xl font-semibold text-gray-soft-700 mb-4">Latest Drift Report</h2>
          {latestLoading ? (
            <div className="text-center py-8 text-gray-500">Loading...</div>
          ) : latestReport ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${getSeverityColor(latestReport.severity)}`}>
                    {latestReport.severity?.toUpperCase() || 'NONE'}
                  </span>
                  <span className="ml-4 text-sm text-gray-600">
                    Drift Detected: {latestReport.drift_detected ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="text-sm text-gray-500">
                  {new Date(latestReport.created_at).toLocaleString()}
                </div>
              </div>
              
              {/* Summary Statistics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
                <div>
                  <div className="text-xs text-gray-500 mb-1">Reference Samples</div>
                  <div className="text-lg font-semibold text-gray-700">
                    {latestReport.reference_samples?.toLocaleString() || 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Current Samples</div>
                  <div className="text-lg font-semibold text-gray-700">
                    {latestReport.current_samples?.toLocaleString() || 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Features Checked</div>
                  <div className="text-lg font-semibold text-gray-700">
                    {latestReport.features_checked || 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Detection Method</div>
                  <div className="text-sm font-semibold text-gray-700">
                    {latestReport.detection_method || 'N/A'}
                  </div>
                </div>
              </div>

              {/* Drift Summary */}
              {latestReport.drift_detected && latestReport.feature_results && (
                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <div className="text-sm font-semibold text-yellow-800 mb-2">Drift Summary</div>
                  <div className="text-sm text-yellow-700">
                    {Object.values(latestReport.feature_results).filter((r: any) => r.drift_detected).length} out of{' '}
                    {Object.keys(latestReport.feature_results).length} features show drift.
                    {latestReport.drift_results?.drift_severity && (
                      <span className="ml-2">
                        Overall severity: <strong>{latestReport.drift_results.drift_severity.toUpperCase()}</strong>
                      </span>
                    )}
                  </div>
                </div>
              )}

              {latestReport.drift_detected && latestReport.feature_results && (
                <DriftChart featureResults={latestReport.feature_results} />
              )}
              
              {/* Threshold Information */}
              {latestReport.threshold_used && (
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <div className="text-sm font-semibold text-gray-700 mb-2">Detection Thresholds</div>
                  <div className="grid grid-cols-2 gap-4 text-xs text-gray-600">
                    {latestReport.threshold_used.psi_threshold !== undefined && (
                      <div>
                        <span className="text-gray-500">PSI Threshold:</span>{' '}
                        <span className="font-semibold">{latestReport.threshold_used.psi_threshold}</span>
                      </div>
                    )}
                    {latestReport.threshold_used.significance_level !== undefined && (
                      <div>
                        <span className="text-gray-500">Significance Level:</span>{' '}
                        <span className="font-semibold">{latestReport.threshold_used.significance_level}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {latestReport.retraining_triggered && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-sm text-blue-800">
                    <strong>Retraining Triggered:</strong> Automatic retraining was triggered due to high drift severity.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No drift reports available. Click "Check for Drift" to generate a report.
            </div>
          )}
        </div>
      )}

      {/* Drift History */}
      {selectedModelId && (
        <div className="glass-strong p-6 rounded-xl border border-pastel-blue/30">
          <h2 className="text-xl font-semibold text-gray-soft-700 mb-4">Drift History</h2>
          {reportsLoading ? (
            <div className="text-center py-8 text-gray-500">Loading...</div>
          ) : reports.length > 0 ? (
            <div className="space-y-3">
              {reports.map((report) => (
                <div
                  key={report.id}
                  className="p-4 rounded-lg border border-pastel-blue/30 bg-white/50 backdrop-blur-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getSeverityColor(report.severity)}`}>
                        {report.severity?.toUpperCase() || 'NONE'}
                      </span>
                      <span className="text-sm text-gray-600">
                        {report.drift_detected ? 'Drift Detected' : 'No Drift'}
                      </span>
                      {report.features_checked && (
                        <span className="text-sm text-gray-500">
                          {report.features_checked} features checked
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-500">
                      {new Date(report.created_at).toLocaleString()}
                    </div>
                  </div>
                  {report.retraining_triggered && (
                    <div className="mt-2 text-xs text-blue-600">
                      Automatic retraining was triggered
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No drift history available
            </div>
          )}
        </div>
      )}
    </div>
  )
}

