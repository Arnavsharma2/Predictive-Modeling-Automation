'use client'

import { useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts'

interface DriftChartProps {
  featureResults: Record<string, any>
  height?: number
}

const PASTEL_COLORS = [
  '#8FC4D4', // Pastel blue
  '#9DD5B8', // Pastel green
  '#F4A5AE', // Pastel pink
  '#FFD3A5', // Pastel orange
  '#C7A8E8', // Pastel purple
]

const getSeverityColor = (severity: string) => {
  switch (severity?.toLowerCase()) {
    case 'high':
      return '#ef4444'
    case 'medium':
      return '#f59e0b'
    case 'low':
      return '#eab308'
    default:
      return '#9DD5B8'
  }
}

export default function DriftChart({ featureResults, height = 400 }: DriftChartProps) {
  const chartData = useMemo(() => {
    return Object.entries(featureResults || {}).map(([feature, result]) => ({
      feature,
      psi: result.metrics?.psi || 0,
      ks_pvalue: result.metrics?.ks_pvalue || 1,
      chi2_pvalue: result.metrics?.chi2_pvalue || 1,
      drift_detected: result.drift_detected ? 1 : 0,
      severity: result.severity || 'none',
    }))
  }, [featureResults])

  const driftFeatures = useMemo(() => {
    return Object.entries(featureResults || {})
      .filter(([, result]) => result.drift_detected)
      .map(([feature, result]) => ({
        feature,
        ...result,
      }))
  }, [featureResults])

  if (!featureResults || Object.keys(featureResults).length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        No drift data available
      </div>
    )
  }

  return (
    <div className="w-full space-y-6">
      {/* PSI Values Chart */}
      <div>
        <h3 className="text-lg font-semibold text-gray-soft-700 mb-4">PSI Values by Feature</h3>
        <ResponsiveContainer width="100%" height={height}>
          <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#9DD5B8" strokeOpacity={0.2} />
            <XAxis
              dataKey="feature"
              tick={{ fontSize: 11, fill: '#5C5C5C' }}
              angle={-45}
              textAnchor="end"
              height={80}
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
                border: '1px solid #9DD5B8',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Bar
              dataKey="psi"
              name="PSI"
              fill="#8FC4D4"
              radius={[4, 4, 0, 0]}
              stroke="#8FC4D4"
              strokeWidth={1}
              fillOpacity={0.8}
            />
            <Bar
              dataKey="drift_detected"
              name="Drift Detected"
              fill="#ef4444"
              radius={[4, 4, 0, 0]}
              stroke="#ef4444"
              strokeWidth={1}
              fillOpacity={0.6}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Drift Features Summary */}
      {driftFeatures.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-soft-700 mb-4">Features with Drift Detected</h3>
          <div className="space-y-3">
            {driftFeatures.map((item) => (
              <div
                key={item.feature}
                className="p-4 rounded-lg border border-pastel-blue/30 bg-white/50 backdrop-blur-sm"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-gray-soft-700">{item.feature}</span>
                  <span
                    className="px-3 py-1 rounded-full text-xs font-semibold text-white"
                    style={{ backgroundColor: getSeverityColor(item.severity) }}
                  >
                    {item.severity?.toUpperCase() || 'NONE'}
                  </span>
                </div>
                {item.drift_reasons && item.drift_reasons.length > 0 && (
                  <div className="text-sm text-gray-600 mb-2">
                    <strong>Reasons:</strong> {item.drift_reasons.join(', ')}
                  </div>
                )}
                {item.metrics && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                      {item.metrics.psi !== null && item.metrics.psi !== undefined && (
                        <div>
                          <span className="text-gray-500">PSI:</span>{' '}
                          <span className="font-semibold">{item.metrics.psi.toFixed(4)}</span>
                        </div>
                      )}
                      {item.metrics.ks_pvalue !== null && item.metrics.ks_pvalue !== undefined && (
                        <div>
                          <span className="text-gray-500">KS p-value:</span>{' '}
                          <span className="font-semibold">{item.metrics.ks_pvalue.toFixed(4)}</span>
                        </div>
                      )}
                      {item.metrics.chi2_pvalue !== null && item.metrics.chi2_pvalue !== undefined && (
                        <div>
                          <span className="text-gray-500">Chi² p-value:</span>{' '}
                          <span className="font-semibold">{item.metrics.chi2_pvalue.toFixed(4)}</span>
                        </div>
                      )}
                      {item.metrics.js_divergence !== null && item.metrics.js_divergence !== undefined && (
                        <div>
                          <span className="text-gray-500">JS Divergence:</span>{' '}
                          <span className="font-semibold">{item.metrics.js_divergence.toFixed(4)}</span>
                        </div>
                      )}
                      {item.metrics.wasserstein_distance !== null && item.metrics.wasserstein_distance !== undefined && (
                        <div>
                          <span className="text-gray-500">Wasserstein:</span>{' '}
                          <span className="font-semibold">{item.metrics.wasserstein_distance.toFixed(4)}</span>
                        </div>
                      )}
                      {item.metrics.ks_statistic !== null && item.metrics.ks_statistic !== undefined && (
                        <div>
                          <span className="text-gray-500">KS Statistic:</span>{' '}
                          <span className="font-semibold">{item.metrics.ks_statistic.toFixed(4)}</span>
                        </div>
                      )}
                      {item.metrics.chi2_statistic !== null && item.metrics.chi2_statistic !== undefined && (
                        <div>
                          <span className="text-gray-500">Chi² Statistic:</span>{' '}
                          <span className="font-semibold">{item.metrics.chi2_statistic.toFixed(4)}</span>
                        </div>
                      )}
                    </div>
                    {item.statistics && (
                      <div className="border-t border-gray-200 pt-2">
                        <div className="text-xs font-semibold text-gray-700 mb-2">Distribution Statistics</div>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-xs font-semibold text-gray-600 mb-1">Reference Data</div>
                            <div className="space-y-1 text-xs">
                              {item.statistics.reference?.mean !== null && item.statistics.reference?.mean !== undefined && (
                                <div>
                                  <span className="text-gray-500">Mean:</span>{' '}
                                  <span className="font-semibold">{item.statistics.reference.mean.toFixed(4)}</span>
                                </div>
                              )}
                              {item.statistics.reference?.std !== null && item.statistics.reference?.std !== undefined && (
                                <div>
                                  <span className="text-gray-500">Std:</span>{' '}
                                  <span className="font-semibold">{item.statistics.reference.std.toFixed(4)}</span>
                                </div>
                              )}
                              {item.statistics.reference?.min !== null && item.statistics.reference?.min !== undefined && (
                                <div>
                                  <span className="text-gray-500">Min:</span>{' '}
                                  <span className="font-semibold">{item.statistics.reference.min.toFixed(4)}</span>
                                </div>
                              )}
                              {item.statistics.reference?.max !== null && item.statistics.reference?.max !== undefined && (
                                <div>
                                  <span className="text-gray-500">Max:</span>{' '}
                                  <span className="font-semibold">{item.statistics.reference.max.toFixed(4)}</span>
                                </div>
                              )}
                              {item.statistics.reference?.count !== null && item.statistics.reference?.count !== undefined && (
                                <div>
                                  <span className="text-gray-500">Count:</span>{' '}
                                  <span className="font-semibold">{item.statistics.reference.count}</span>
                                </div>
                              )}
                            </div>
                          </div>
                          <div>
                            <div className="text-xs font-semibold text-gray-600 mb-1">Current Data</div>
                            <div className="space-y-1 text-xs">
                              {item.statistics.current?.mean !== null && item.statistics.current?.mean !== undefined && (
                                <div>
                                  <span className="text-gray-500">Mean:</span>{' '}
                                  <span className="font-semibold">{item.statistics.current.mean.toFixed(4)}</span>
                                </div>
                              )}
                              {item.statistics.current?.std !== null && item.statistics.current?.std !== undefined && (
                                <div>
                                  <span className="text-gray-500">Std:</span>{' '}
                                  <span className="font-semibold">{item.statistics.current.std.toFixed(4)}</span>
                                </div>
                              )}
                              {item.statistics.current?.min !== null && item.statistics.current?.min !== undefined && (
                                <div>
                                  <span className="text-gray-500">Min:</span>{' '}
                                  <span className="font-semibold">{item.statistics.current.min.toFixed(4)}</span>
                                </div>
                              )}
                              {item.statistics.current?.max !== null && item.statistics.current?.max !== undefined && (
                                <div>
                                  <span className="text-gray-500">Max:</span>{' '}
                                  <span className="font-semibold">{item.statistics.current.max.toFixed(4)}</span>
                                </div>
                              )}
                              {item.statistics.current?.count !== null && item.statistics.current?.count !== undefined && (
                                <div>
                                  <span className="text-gray-500">Count:</span>{' '}
                                  <span className="font-semibold">{item.statistics.current.count}</span>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

