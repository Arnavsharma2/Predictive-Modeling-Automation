'use client'

import { useMemo } from 'react'

interface ShapWaterfallChartProps {
  baseValue: number
  prediction: number
  contributions: Array<{ feature: string; shap_value: number }>
  featureNames?: string[]
  maxDisplay?: number
  height?: number
}

export default function ShapWaterfallChart({
  baseValue,
  prediction,
  contributions,
  featureNames,
  maxDisplay = 10,
  height = 500
}: ShapWaterfallChartProps) {
  // Calculate spacing based on available height first
  const headerHeight = 60 // Space for labels and top padding
  const axisHeight = 50 // Space for x-axis
  const availableHeight = height - headerHeight - axisHeight
  const barHeight = 32
  const minBarSpacing = 8
  
  // Calculate how many features can fit in the available space
  const maxFeaturesThatFit = useMemo(() => {
    // For n features, we need: n * barHeight + (n + 1) * minBarSpacing <= availableHeight
    // Solving: n * (barHeight + minBarSpacing) + minBarSpacing <= availableHeight
    // n <= (availableHeight - minBarSpacing) / (barHeight + minBarSpacing)
    const maxFit = Math.floor((availableHeight - minBarSpacing) / (barHeight + minBarSpacing))
    return Math.max(1, Math.min(maxFit, maxDisplay))
  }, [availableHeight, barHeight, minBarSpacing, maxDisplay])

  const sortedContributions = useMemo(() => {
    return [...contributions]
      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
      .slice(0, maxFeaturesThatFit)
  }, [contributions, maxFeaturesThatFit])

  const cumulativeValues = useMemo(() => {
    const values = [baseValue]
    sortedContributions.forEach((contrib) => {
      values.push(values[values.length - 1] + contrib.shap_value)
    })
    return values
  }, [baseValue, sortedContributions])

  const minValue = Math.min(...cumulativeValues, prediction, baseValue)
  const maxValue = Math.max(...cumulativeValues, prediction, baseValue)
  const range = maxValue - minValue || 1

  // Add padding to range for better visualization
  const paddedMin = minValue - range * 0.1
  const paddedMax = maxValue + range * 0.1
  const paddedRange = paddedMax - paddedMin || 1

  const getXPosition = (value: number) => {
    return ((value - paddedMin) / paddedRange) * 100
  }

  // Calculate spacing based on actual number of features to display
  const barSpacing = Math.max(minBarSpacing, (availableHeight - sortedContributions.length * barHeight) / (sortedContributions.length + 1))
  const rowHeight = barHeight + barSpacing

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold text-gray-soft-700 mb-4">SHAP Waterfall Explanation</h3>
      <div className="relative bg-gray-50/50 rounded-lg p-4" style={{ minHeight: `${height}px` }}>
        {/* Header with reference lines */}
        <div className="relative mb-2" style={{ height: `${headerHeight}px` }}>
          {/* Prediction line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-pastel-green z-10"
            style={{ left: `${getXPosition(prediction)}%` }}
          >
            <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full mb-1 whitespace-nowrap">
              <div className="text-xs font-semibold text-pastel-green bg-white px-1 rounded">
                Prediction: {prediction.toFixed(4)}
              </div>
            </div>
          </div>

          {/* Base value line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-gray-400 z-10"
            style={{ left: `${getXPosition(baseValue)}%` }}
          >
            <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full mb-1 whitespace-nowrap">
              <div className="text-xs font-semibold text-gray-600 bg-white px-1 rounded">
                Base: {baseValue.toFixed(4)}
              </div>
            </div>
          </div>
        </div>

        {/* Feature contributions area */}
        <div className="relative overflow-hidden" style={{ height: `${availableHeight}px`, marginBottom: `${axisHeight}px` }}>
          {sortedContributions.map((contrib, index) => {
            const startValue = cumulativeValues[index]
            const endValue = cumulativeValues[index + 1]
            const isPositive = contrib.shap_value >= 0
            const startX = getXPosition(startValue)
            const endX = getXPosition(endValue)
            const width = Math.max(2, Math.abs(endX - startX)) // Minimum width to ensure visibility
            const left = Math.min(startX, endX)
            const top = index * rowHeight + barSpacing

            // Ensure the bar doesn't go beyond the available height
            if (top + barHeight > availableHeight) {
              return null
            }

            return (
              <div
                key={contrib.feature}
                className="absolute flex items-center"
                style={{
                  top: `${top}px`,
                  left: `${Math.max(0, Math.min(100 - width, left))}%`,
                  width: `${Math.min(100, width)}%`,
                  height: `${barHeight}px`,
                }}
              >
                <div
                  className={`h-full rounded ${
                    isPositive ? 'bg-pastel-green/40' : 'bg-red-500/40'
                  } border-l-2 ${
                    isPositive ? 'border-pastel-green' : 'border-red-500'
                  } flex items-center px-2 shadow-sm`}
                  style={{ width: '100%', minWidth: '120px' }}
                >
                  <span className="text-xs font-medium text-gray-soft-700 truncate flex-1 min-w-0">
                    {contrib.feature}
                  </span>
                  <span className={`text-xs font-semibold ml-2 whitespace-nowrap ${
                    isPositive ? 'text-pastel-green' : 'text-red-500'
                  }`}>
                    {isPositive ? '+' : ''}{contrib.shap_value.toFixed(4)}
                  </span>
                </div>
              </div>
            )
          })}
        </div>

        {/* X-axis scale */}
        <div className="absolute bottom-0 left-0 right-0 border-t-2 border-gray-300 bg-white/80 rounded-b-lg" style={{ height: `${axisHeight}px` }}>
          <div className="relative h-full pt-2">
            {[0, 25, 50, 75, 100].map((percent) => {
              const value = paddedMin + (percent / 100) * paddedRange
              return (
                <div
                  key={percent}
                  className="absolute top-0 flex flex-col items-center"
                  style={{ left: `${percent}%`, transform: 'translateX(-50%)' }}
                >
                  <div className="w-0.5 h-3 bg-gray-400" />
                  <span className="text-xs text-gray-600 mt-1 whitespace-nowrap">
                    {value.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

