'use client'

import { useMemo } from 'react'

interface FeatureImportanceChartProps {
  importance: Record<string, number>
  title?: string
  maxFeatures?: number
  height?: number
}

export default function FeatureImportanceChart({
  importance,
  title = 'Feature Importance',
  maxFeatures = 20,
  height = 400
}: FeatureImportanceChartProps) {
  const sortedFeatures = useMemo(() => {
    return Object.entries(importance)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, maxFeatures)
  }, [importance, maxFeatures])

  const maxValue = useMemo(() => {
    return Math.max(...sortedFeatures.map(([, value]) => Math.abs(value)))
  }, [sortedFeatures])

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-lg font-semibold text-gray-soft-700 mb-4">{title}</h3>
      )}
      <div className="space-y-2" style={{ height: `${height}px`, overflowY: 'auto' }}>
        {sortedFeatures.map(([feature, value], index) => {
          const percentage = (Math.abs(value) / maxValue) * 100
          const isPositive = value >= 0
          
          return (
            <div key={feature} className="flex items-center gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-gray-soft-700 truncate">
                    {feature}
                  </span>
                  <span className={`text-sm font-semibold ml-2 ${
                    isPositive ? 'text-pastel-green' : 'text-red-500'
                  }`}>
                    {value.toFixed(4)}
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
            </div>
          )
        })}
      </div>
      {sortedFeatures.length === 0 && (
        <div className="text-center text-gray-500 py-8">
          No feature importance data available
        </div>
      )}
    </div>
  )
}

