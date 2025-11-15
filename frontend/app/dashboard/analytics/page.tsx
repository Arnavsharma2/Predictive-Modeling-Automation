'use client'

import { useState, useMemo } from 'react'
import { useModels, useTrainingJobs } from '@/hooks/useModels'
import { useDataSources } from '@/hooks/useData'
import { useDataPreview } from '@/hooks/useData'
import TimeSeriesChart from '@/components/charts/TimeSeriesChart'
import PerformanceChart from '@/components/charts/PerformanceChart'
import ConfusionMatrix from '@/components/charts/ConfusionMatrix'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function AnalyticsPage() {
  const { models } = useModels()
  const { sources } = useDataSources()
  const [selectedSourceId, setSelectedSourceId] = useState<number | null>(null)
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null)
  
  const { data: timeSeriesData } = useDataPreview(selectedSourceId, 100, 0)
  const { jobs: trainingJobs } = useTrainingJobs(
    selectedModelId ? { model_id: selectedModelId, limit: 100 } : undefined
  )
  
  const selectedModel = models.find((m: any) => m.id === selectedModelId)
  
  // Prepare time series data for chart
  // Data structure: { id, data: {...fields...}, timestamp, metadata }
  const chartData = useMemo(() => {
    if (!timeSeriesData || timeSeriesData.length === 0) return []
    
    // Extract numeric fields from the first data point to determine series
    const firstPoint = timeSeriesData[0]
    const dataFields = firstPoint?.data || {}
    const numericFields = Object.entries(dataFields)
      .filter(([_, value]) => typeof value === 'number' && !isNaN(value))
      .map(([key]) => key)
    
    if (numericFields.length === 0) return []
    
    // Map data points to chart format
    const mapped = timeSeriesData
      .filter((d: any) => d.timestamp && d.data)
      .map((d: any) => {
        const point: any = {
          timestamp: d.timestamp
        }
        // Add all numeric fields as separate series
        numericFields.forEach(field => {
          point[field] = d.data[field]
        })
        return point
      })
      .sort((a: any, b: any) => {
        // Sort by timestamp (ascending for time series)
        return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      })
    
    return mapped
  }, [timeSeriesData])
  
  // Get numeric field names for series
  const timeSeriesFields = useMemo(() => {
    if (chartData.length === 0) return []
    const firstPoint = chartData[0]
    return Object.keys(firstPoint).filter(key => key !== 'timestamp')
  }, [chartData])
  
  // Prepare comprehensive performance data from training jobs
  const performanceData = useMemo(() => {
    if (!selectedModelId) return []
    
    if (!trainingJobs || trainingJobs.length === 0) {
      if (selectedModel) {
        const hasValidMetrics = 
          (selectedModel.rmse !== undefined && selectedModel.rmse !== null && selectedModel.rmse > 0) ||
          (selectedModel.r2_score !== undefined && selectedModel.r2_score !== null && !isNaN(selectedModel.r2_score)) ||
          (selectedModel.accuracy !== undefined && selectedModel.accuracy !== null && selectedModel.accuracy > 0)
        
        if (hasValidMetrics) {
          return [{
            version: 'Current',
            date: selectedModel.created_at || new Date().toISOString(),
            accuracy: selectedModel.accuracy || undefined,
            precision: selectedModel.precision || undefined,
            recall: selectedModel.recall || undefined,
            f1_score: selectedModel.f1_score || undefined,
            rmse: selectedModel.rmse || undefined,
            mae: selectedModel.mae || undefined,
            r2_score: selectedModel.r2_score || undefined,
            training_duration: undefined
          }]
        }
      }
      return []
    }
    
    const completedJobs = trainingJobs
      .filter((job: any) => job.status === 'completed' && job.metrics)
      .map((job: any, idx: number) => {
        const metrics = job.metrics || {}
        const duration = job.completed_at && job.started_at 
          ? (new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000 / 60 // minutes
          : undefined
        
        return {
          version: `v${idx + 1}`,
          date: job.completed_at || job.created_at || new Date().toISOString(),
          accuracy: metrics.accuracy !== undefined && metrics.accuracy !== null ? metrics.accuracy : undefined,
          precision: metrics.precision !== undefined && metrics.precision !== null ? metrics.precision : undefined,
          recall: metrics.recall !== undefined && metrics.recall !== null ? metrics.recall : undefined,
          f1_score: metrics.f1_score !== undefined && metrics.f1_score !== null ? metrics.f1_score : undefined,
          rmse: metrics.rmse !== undefined && metrics.rmse !== null ? metrics.rmse : undefined,
          mae: metrics.mae !== undefined && metrics.mae !== null ? metrics.mae : undefined,
          r2_score: metrics.r2_score !== undefined && metrics.r2_score !== null ? metrics.r2_score : undefined,
          mape: metrics.mape !== undefined && metrics.mape !== null ? metrics.mape : undefined,
          training_duration: duration
        }
      })
      .filter((item: any) => {
        return item.accuracy !== undefined || item.rmse !== undefined || item.r2_score !== undefined ||
               item.precision !== undefined || item.recall !== undefined || item.f1_score !== undefined
      })
      .sort((a: any, b: any) => new Date(a.date).getTime() - new Date(b.date).getTime())
    
    return completedJobs
  }, [trainingJobs, selectedModel, selectedModelId])
  
  // Calculate performance statistics
  const performanceStats = useMemo(() => {
    if (performanceData.length === 0) return null
    
    const isClassification = selectedModel?.type === 'classification'
    
    if (isClassification) {
      const accuracies = performanceData.map((d: any) => d.accuracy).filter((v: any) => v !== undefined)
      const precisions = performanceData.map((d: any) => d.precision).filter((v: any) => v !== undefined)
      const recalls = performanceData.map((d: any) => d.recall).filter((v: any) => v !== undefined)
      const f1s = performanceData.map((d: any) => d.f1_score).filter((v: any) => v !== undefined)
      
      const bestAccuracy = accuracies.length > 0 ? Math.max(...accuracies) : null
      const worstAccuracy = accuracies.length > 0 ? Math.min(...accuracies) : null
      const avgAccuracy = accuracies.length > 0 ? accuracies.reduce((a: number, b: number) => a + b, 0) / accuracies.length : null
      
      const bestF1 = f1s.length > 0 ? Math.max(...f1s) : null
      const avgF1 = f1s.length > 0 ? f1s.reduce((a: number, b: number) => a + b, 0) / f1s.length : null
      
      // Calculate trend (improvement/regression)
      let trend = 'stable'
      if (accuracies.length >= 2) {
        const first = accuracies[0]
        const last = accuracies[accuracies.length - 1]
        const change = ((last - first) / first) * 100
        if (change > 2) trend = 'improving'
        else if (change < -2) trend = 'declining'
      }
      
      return {
        bestAccuracy,
        worstAccuracy,
        avgAccuracy,
        bestF1,
        avgF1,
        trend,
        totalVersions: performanceData.length
      }
    } else {
      const r2s = performanceData.map((d: any) => d.r2_score).filter((v: any) => v !== undefined)
      const rmses = performanceData.map((d: any) => d.rmse).filter((v: any) => v !== undefined)
      const maes = performanceData.map((d: any) => d.mae).filter((v: any) => v !== undefined)
      
      const bestR2 = r2s.length > 0 ? Math.max(...r2s) : null
      const worstR2 = r2s.length > 0 ? Math.min(...r2s) : null
      const avgR2 = r2s.length > 0 ? r2s.reduce((a: number, b: number) => a + b, 0) / r2s.length : null
      
      const bestRMSE = rmses.length > 0 ? Math.min(...rmses) : null
      const worstRMSE = rmses.length > 0 ? Math.max(...rmses) : null
      const avgRMSE = rmses.length > 0 ? rmses.reduce((a: number, b: number) => a + b, 0) / rmses.length : null
      
      let trend = 'stable'
      if (r2s.length >= 2) {
        const first = r2s[0]
        const last = r2s[r2s.length - 1]
        const change = ((last - first) / Math.abs(first + 0.001)) * 100
        if (change > 5) trend = 'improving'
        else if (change < -5) trend = 'declining'
      }
      
      return {
        bestR2,
        worstR2,
        avgR2,
        bestRMSE,
        worstRMSE,
        avgRMSE,
        trend,
        totalVersions: performanceData.length
      }
    }
  }, [performanceData, selectedModel])

  // Get metrics for chart based on model type
  const chartMetrics = useMemo(() => {
    if (!selectedModel) return []
    
    if (selectedModel.type === 'classification') {
      return [
        { key: 'accuracy', name: 'Accuracy', color: '#9DD5B8' },
        { key: 'precision', name: 'Precision', color: '#8FC4D4' },
        { key: 'recall', name: 'Recall', color: '#A8DFF5' },
        { key: 'f1_score', name: 'F1 Score', color: '#B0D9B1' }
      ]
    } else {
      return [
        { key: 'r2_score', name: 'R² Score', color: '#8FC4D4' },
        { key: 'rmse', name: 'RMSE', color: '#A8DFF5' },
        { key: 'mae', name: 'MAE', color: '#9BD0D9' }
      ]
    }
  }, [selectedModel])

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold text-gray-soft-700 mb-2">Analytics Dashboard</h1>
      <p className="text-gray-soft-600 mb-8 font-medium">Visualize and analyze your data and models</p>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="glass rounded-2xl border-pastel-blue/40 p-6">
          <h2 className="text-xl font-bold text-gray-soft-700 mb-4 flex items-center gap-2">
            <span className="w-1 h-6 bg-pastel-blue rounded-full"></span>
            Time Series Data
          </h2>
          <div className="mb-4">
            <select
              value={selectedSourceId || ''}
              onChange={(e) => setSelectedSourceId(parseInt(e.target.value) || null)}
              className="w-full px-3 py-2 border border-gray-soft-300 rounded-lg bg-white text-gray-soft-700"
            >
              <option value="">Select data source...</option>
              {sources.map((source: any) => (
                <option key={source.id} value={source.id}>{source.name}</option>
              ))}
            </select>
          </div>
          {chartData.length > 0 && timeSeriesFields.length > 0 ? (
            <TimeSeriesChart
              data={chartData}
              series={timeSeriesFields.map((field, idx) => {
                const pastelColors = ['#8FC4D4', '#9DD5B8', '#A8DFF5', '#B0D9B1', '#9BD0D9', '#C4A8D4', '#D4B8A8', '#B8D4C4']
                return {
                  key: field,
                  name: field,
                  color: pastelColors[idx % pastelColors.length]
                }
              })}
              height={300}
            />
          ) : selectedSourceId ? (
            <p className="text-gray-soft-600 text-center py-8">
              {timeSeriesData.length === 0 
                ? 'No data available for this source'
                : 'No numeric fields found in data'}
            </p>
          ) : (
            <p className="text-gray-soft-600 text-center py-8">Select a data source to view time series</p>
          )}
        </div>

        <div className="glass rounded-2xl border-pastel-mint/40 p-6">
          <h2 className="text-xl font-bold text-gray-soft-700 mb-4 flex items-center gap-2">
            <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
            Model Performance
          </h2>
          <div className="mb-4">
            <select
              value={selectedModelId || ''}
              onChange={(e) => setSelectedModelId(parseInt(e.target.value) || null)}
              className="w-full px-3 py-2 border border-gray-soft-300 rounded-lg bg-white text-gray-soft-700"
            >
              <option value="">Select model...</option>
              {models.filter((m: any) => m.status === 'trained' || m.status === 'deployed').map((model: any) => (
                <option key={model.id} value={model.id}>{model.name}</option>
              ))}
            </select>
          </div>
          {selectedModelId && performanceData.length > 0 ? (
            <PerformanceChart
              data={performanceData}
              metrics={chartMetrics.filter(m => performanceData.some((d: any) => d[m.key] !== undefined))}
              height={300}
            />
          ) : selectedModelId ? (
            <p className="text-gray-soft-600 text-center py-8">
              No performance data available for this model
            </p>
          ) : (
            <p className="text-gray-soft-600 text-center py-8">Select a model to view performance</p>
          )}
        </div>
      </div>

      {/* Enhanced Performance Statistics */}
      {selectedModelId && performanceStats && performanceData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Performance Trend Chart */}
          <div className="glass rounded-2xl border-pastel-mint/40 p-6">
            <h2 className="text-xl font-bold text-gray-soft-700 mb-4 flex items-center gap-2">
              <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
              Performance Trend Over Time
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#9DD5B8" strokeOpacity={0.2} />
                <XAxis 
                  dataKey="version" 
                  tick={{ fontSize: 11, fill: '#5C5C5C' }}
                  stroke="#9DD5B8"
                  strokeOpacity={0.4}
                />
                <YAxis 
                  tick={{ fontSize: 11, fill: '#5C5C5C' }}
                  stroke="#9DD5B8"
                  strokeOpacity={0.4}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                    border: '1px solid rgba(157, 213, 184, 0.4)',
                    borderRadius: '8px',
                    padding: '8px'
                  }}
                />
                <Legend />
                {selectedModel?.type === 'classification' ? (
                  <>
                    <Line type="monotone" dataKey="accuracy" stroke="#9DD5B8" strokeWidth={2} name="Accuracy" dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="f1_score" stroke="#8FC4D4" strokeWidth={2} name="F1 Score" dot={{ r: 4 }} />
                  </>
                ) : (
                  <>
                    <Line type="monotone" dataKey="r2_score" stroke="#8FC4D4" strokeWidth={2} name="R² Score" dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="rmse" stroke="#A8DFF5" strokeWidth={2} name="RMSE" dot={{ r: 4 }} yAxisId="right" />
                  </>
                )}
                {selectedModel?.type !== 'classification' && (
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: '#5C5C5C' }} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Performance Statistics Summary */}
          <div className="glass rounded-2xl border-pastel-blue/40 p-6">
            <h2 className="text-xl font-bold text-gray-soft-700 mb-4 flex items-center gap-2">
              <span className="w-1 h-6 bg-pastel-blue rounded-full"></span>
              Performance Summary
            </h2>
            <div className="space-y-4">
              {selectedModel?.type === 'classification' ? (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-pastel-mint/10 rounded-lg p-3">
                      <p className="text-xs text-gray-soft-600 mb-1">Best Accuracy</p>
                      <p className="text-2xl font-bold text-gray-soft-700">
                        {performanceStats?.bestAccuracy !== null && performanceStats?.bestAccuracy !== undefined ? (performanceStats.bestAccuracy * 100).toFixed(2) + '%' : 'N/A'}
                      </p>
                    </div>
                    <div className="bg-pastel-blue/10 rounded-lg p-3">
                      <p className="text-xs text-gray-soft-600 mb-1">Average Accuracy</p>
                      <p className="text-2xl font-bold text-gray-soft-700">
                        {performanceStats?.avgAccuracy !== null && performanceStats?.avgAccuracy !== undefined ? (performanceStats.avgAccuracy * 100).toFixed(2) + '%' : 'N/A'}
                      </p>
                    </div>
                    <div className="bg-pastel-green/10 rounded-lg p-3">
                      <p className="text-xs text-gray-soft-600 mb-1">Best F1 Score</p>
                      <p className="text-2xl font-bold text-gray-soft-700">
                        {performanceStats?.bestF1 !== null && performanceStats?.bestF1 !== undefined ? performanceStats.bestF1.toFixed(3) : 'N/A'}
                      </p>
                    </div>
                    <div className="bg-pastel-powder/10 rounded-lg p-3">
                      <p className="text-xs text-gray-soft-600 mb-1">Average F1 Score</p>
                      <p className="text-2xl font-bold text-gray-soft-700">
                        {performanceStats?.avgF1 !== null && performanceStats?.avgF1 !== undefined ? performanceStats.avgF1.toFixed(3) : 'N/A'}
                      </p>
                    </div>
                  </div>
                  <div className="bg-pastel-mint/10 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-gray-soft-600">Performance Trend</p>
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        performanceStats.trend === 'improving' ? 'bg-green-100 text-green-700' :
                        performanceStats.trend === 'declining' ? 'bg-red-100 text-red-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                        {performanceStats.trend === 'improving' ? '↗ Improving' :
                         performanceStats.trend === 'declining' ? '↘ Declining' :
                         '→ Stable'}
                      </span>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-pastel-mint/10 rounded-lg p-3">
                      <p className="text-xs text-gray-soft-600 mb-1">Best R² Score</p>
                      <p className="text-2xl font-bold text-gray-soft-700">
                        {performanceStats?.bestR2 !== null && performanceStats?.bestR2 !== undefined ? performanceStats.bestR2.toFixed(4) : 'N/A'}
                      </p>
                    </div>
                    <div className="bg-pastel-blue/10 rounded-lg p-3">
                      <p className="text-xs text-gray-soft-600 mb-1">Average R² Score</p>
                      <p className="text-2xl font-bold text-gray-soft-700">
                        {performanceStats?.avgR2 !== null && performanceStats?.avgR2 !== undefined ? performanceStats.avgR2.toFixed(4) : 'N/A'}
                      </p>
                    </div>
                    <div className="bg-pastel-green/10 rounded-lg p-3">
                      <p className="text-xs text-gray-soft-600 mb-1">Best RMSE</p>
                      <p className="text-2xl font-bold text-gray-soft-700">
                        {performanceStats?.bestRMSE !== null && performanceStats?.bestRMSE !== undefined ? performanceStats.bestRMSE.toFixed(2) : 'N/A'}
                      </p>
                    </div>
                    <div className="bg-pastel-powder/10 rounded-lg p-3">
                      <p className="text-xs text-gray-soft-600 mb-1">Average RMSE</p>
                      <p className="text-2xl font-bold text-gray-soft-700">
                        {performanceStats?.avgRMSE !== null && performanceStats?.avgRMSE !== undefined ? performanceStats.avgRMSE.toFixed(2) : 'N/A'}
                      </p>
                    </div>
                  </div>
                  <div className="bg-pastel-mint/10 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-gray-soft-600">Performance Trend</p>
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        performanceStats?.trend === 'improving' ? 'bg-green-100 text-green-700' :
                        performanceStats?.trend === 'declining' ? 'bg-red-100 text-red-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                        {performanceStats?.trend === 'improving' ? '↗ Improving' :
                         performanceStats?.trend === 'declining' ? '↘ Declining' :
                         '→ Stable'}
                      </span>
                    </div>
                  </div>
                </>
              )}
              <div className="bg-pastel-blue/10 rounded-lg p-3">
                <p className="text-xs text-gray-soft-600 mb-1">Total Versions</p>
                <p className="text-2xl font-bold text-gray-soft-700">{performanceStats?.totalVersions ?? 0}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedModel && selectedModel.type === 'classification' && trainingJobs && trainingJobs.length > 0 && (
        (() => {
          const latestJob = trainingJobs.find((job: any) => job.status === 'completed' && job.metrics?.confusion_matrix) || trainingJobs[0]
          return latestJob?.metrics?.confusion_matrix ? (
            <div className="glass rounded-2xl border-pastel-mint/40 p-6 mb-6">
              <h2 className="text-xl font-bold text-gray-soft-700 mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
                Confusion Matrix
              </h2>
              <ConfusionMatrix
                matrix={latestJob.metrics.confusion_matrix}
                labels={latestJob.metrics.classes || ['Class 0', 'Class 1']}
              />
            </div>
          ) : null
        })()
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
        <div className="glass rounded-2xl border-pastel-mint/40 p-6">
          <h3 className="text-sm font-medium text-gray-soft-600 mb-2">Total Models</h3>
          <p className="text-2xl font-bold text-gray-soft-700">{models.length}</p>
        </div>
        <div className="glass rounded-2xl border-pastel-green/40 p-6">
          <h3 className="text-sm font-medium text-gray-soft-600 mb-2">Trained Models</h3>
          <p className="text-2xl font-bold text-gray-soft-700">
            {models.filter((m: any) => m.status === 'trained' || m.status === 'deployed').length}
          </p>
        </div>
        <div className="glass rounded-2xl border-pastel-blue/40 p-6">
          <h3 className="text-sm font-medium text-gray-soft-600 mb-2">Data Sources</h3>
          <p className="text-2xl font-bold text-gray-soft-700">{sources.length}</p>
        </div>
      </div>
    </div>
  )
}

