'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface TimeSeriesChartProps {
  data: Array<{ timestamp: string; [key: string]: any }>
  series: Array<{ key: string; name: string; color?: string }>
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
      <div className="bg-white/95 backdrop-blur-sm border border-pastel-blue/40 rounded-lg shadow-lg p-3">
        <p className="text-sm font-semibold text-gray-soft-700 mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            <span className="font-medium">{entry.name}:</span>{' '}
            <span className="font-semibold">{typeof entry.value === 'number' ? entry.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : entry.value}</span>
          </p>
        ))}
      </div>
    )
  }
  return null
}

export default function TimeSeriesChart({ data, series, height = 300 }: TimeSeriesChartProps) {
  // Format data for Recharts
  const chartData = data.map(item => {
    const date = new Date(item.timestamp)
    // Format timestamp based on data range
    // If data spans multiple days, show date; otherwise show date + time
    let timestampLabel: string
    if (data.length > 0) {
      const firstDate = new Date(data[0].timestamp)
      const lastDate = new Date(data[data.length - 1].timestamp)
      const daysDiff = (lastDate.getTime() - firstDate.getTime()) / (1000 * 60 * 60 * 24)
      
      if (daysDiff > 1) {
        // Multiple days: show date only
        timestampLabel = date.toLocaleDateString()
      } else {
        // Same day: show date and time
        timestampLabel = date.toLocaleString(undefined, { 
          month: 'short', 
          day: 'numeric', 
          hour: '2-digit', 
          minute: '2-digit' 
        })
      }
    } else {
      timestampLabel = date.toLocaleDateString()
    }
    
    const formatted: any = {
      timestamp: timestampLabel,
      timestampRaw: date.getTime() // Keep raw timestamp for sorting
    }
    series.forEach(s => {
      // Ensure values are numbers, not strings
      const value = item[s.key]
      formatted[s.key] = typeof value === 'number' ? value : parseFloat(value) || 0
    })
    return formatted
  })

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
        <CartesianGrid 
          strokeDasharray="3 3" 
          stroke="#8FC4D4"
          strokeOpacity={0.2}
        />
        <XAxis 
          dataKey="timestamp" 
          tick={{ fontSize: 11, fill: '#5C5C5C' }}
          angle={-45}
          textAnchor="end"
          height={80}
          stroke="#8FC4D4"
          strokeOpacity={0.4}
        />
        <YAxis 
          tick={{ fontSize: 11, fill: '#5C5C5C' }}
          stroke="#8FC4D4"
          strokeOpacity={0.4}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend 
          wrapperStyle={{ paddingTop: '20px' }}
          iconType="line"
          formatter={(value) => <span style={{ color: '#3D3D3D', fontSize: '12px' }}>{value}</span>}
        />
        {series.map((s, index) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.name}
            stroke={s.color || PASTEL_COLORS[index % PASTEL_COLORS.length]}
            strokeWidth={2.5}
            dot={{ r: 4, fill: s.color || PASTEL_COLORS[index % PASTEL_COLORS.length], strokeWidth: 2, stroke: '#fff' }}
            activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }}
            strokeOpacity={0.8}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

