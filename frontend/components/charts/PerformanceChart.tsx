'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface PerformanceChartProps {
  data: Array<{ version: string; [key: string]: number | string | undefined }>
  metrics: Array<{ key: string; name: string; color?: string }>
  height?: number
}

// Pastel color palette matching the theme
const PASTEL_COLORS = [
  '#8FC4D4', // pastel-blue
  '#9DD5B8', // pastel-mint
  '#A8DFF5', // pastel-baby
  '#B0D9B1', // pastel-green
  '#9BD0D9', // pastel-powder
  '#C4A8D4', // soft purple
  '#D4B8A8', // soft peach
  '#B8D4C4', // soft teal
]

// Custom tooltip styling
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white/95 backdrop-blur-sm border border-pastel-mint/40 rounded-lg shadow-lg p-3">
        <p className="text-sm font-semibold text-gray-soft-700 mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            <span className="font-medium">{entry.name}:</span>{' '}
            <span className="font-semibold">{typeof entry.value === 'number' ? entry.value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : entry.value}</span>
          </p>
        ))}
      </div>
    )
  }
  return null
}

export default function PerformanceChart({ data, metrics, height = 300 }: PerformanceChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
        <CartesianGrid 
          strokeDasharray="3 3" 
          stroke="#9DD5B8"
          strokeOpacity={0.2}
        />
        <XAxis 
          dataKey="version" 
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
        <Tooltip content={<CustomTooltip />} />
        <Legend 
          wrapperStyle={{ paddingTop: '20px' }}
          formatter={(value) => <span style={{ color: '#3D3D3D', fontSize: '12px' }}>{value}</span>}
        />
        {metrics.map((metric, index) => (
          <Bar
            key={metric.key}
            dataKey={metric.key}
            name={metric.name}
            fill={metric.color || PASTEL_COLORS[index % PASTEL_COLORS.length]}
            radius={[4, 4, 0, 0]}
            stroke={metric.color || PASTEL_COLORS[index % PASTEL_COLORS.length]}
            strokeWidth={1}
            fillOpacity={0.8}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

