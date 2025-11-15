'use client'

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'

interface AnomalyChartProps {
  data: Array<{ x: number; y: number; isAnomaly: boolean; score?: number }>
  xLabel?: string
  yLabel?: string
  height?: number
}

export default function AnomalyChart({ data, xLabel = 'Feature 1', yLabel = 'Feature 2', height = 400 }: AnomalyChartProps) {
  const normalData = data.filter(d => !d.isAnomaly)
  const anomalyData = data.filter(d => d.isAnomaly)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          type="number" 
          dataKey="x" 
          name={xLabel}
          tick={{ fontSize: 12 }}
        />
        <YAxis 
          type="number" 
          dataKey="y" 
          name={yLabel}
          tick={{ fontSize: 12 }}
        />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Legend />
        <Scatter name="Normal" data={normalData} fill="#10b981">
          {normalData.map((entry, index) => (
            <Cell key={`normal-${index}`} fill="#10b981" />
          ))}
        </Scatter>
        <Scatter name="Anomaly" data={anomalyData} fill="#ef4444">
          {anomalyData.map((entry, index) => (
            <Cell key={`anomaly-${index}`} fill="#ef4444" />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  )
}

