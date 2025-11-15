'use client'

import { useState, useEffect } from 'react'
import { usePredictionWebSocket } from '@/hooks/usePredictionWebSocket'
import { getAccessToken } from '@/lib/auth'

interface LivePredictionsProps {
  modelId: number
  enabled?: boolean
}

export default function LivePredictions({ modelId, enabled = true }: LivePredictionsProps) {
  const token = typeof window !== 'undefined' ? getAccessToken() : null
  const [predictions, setPredictions] = useState<any[]>([])
  const [maxPredictions] = useState(50) // Keep last 50 predictions

  const { isConnected, lastPrediction } = usePredictionWebSocket({
    modelId,
    token: token || '',
    enabled: enabled && !!token,
    onPrediction: (update) => {
      setPredictions(prev => {
        const newPredictions = [update, ...prev]
        // Keep only the last N predictions
        return newPredictions.slice(0, maxPredictions)
      })
    }
  })

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <div className="flex justify-between items-center mb-4">
        <h3 className="font-semibold text-lg">Live Predictions</h3>
        <div className="flex items-center gap-2">
          {isConnected ? (
            <span className="text-xs text-green-600">● Connected</span>
          ) : (
            <span className="text-xs text-gray-500">○ Disconnected</span>
          )}
        </div>
      </div>

      {predictions.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          {isConnected 
            ? 'Waiting for predictions...' 
            : 'Not connected. Predictions will appear here when made via API.'}
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {predictions.map((pred, idx) => (
            <div
              key={idx}
              className="p-3 bg-gray-50 rounded border border-gray-200 animate-fade-in"
            >
              <div className="flex justify-between items-start mb-2">
                <div className="font-medium">
                  Prediction: {typeof pred.prediction === 'number' 
                    ? pred.prediction.toFixed(4) 
                    : String(pred.prediction)}
                </div>
                <div className="text-xs text-gray-500">
                  {new Date(pred.timestamp).toLocaleTimeString()}
                </div>
              </div>
              {pred.confidence && (
                <div className="text-sm text-gray-600">
                  Confidence: {pred.confidence.std ? `±${pred.confidence.std.toFixed(4)}` : 'N/A'}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

