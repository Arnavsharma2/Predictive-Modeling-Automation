'use client'

import { useMemo } from 'react'

interface ConfusionMatrixProps {
  matrix: number[][]
  labels: string[]
}

export default function ConfusionMatrix({ matrix, labels }: ConfusionMatrixProps) {
  const maxValue = useMemo(() => {
    return Math.max(...matrix.flat())
  }, [matrix])

  const getColor = (value: number, max: number) => {
    const intensity = value / max
    if (intensity > 0.7) return 'bg-red-600'
    if (intensity > 0.4) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-collapse">
        <thead>
          <tr>
            <th className="border border-gray-300 p-2 bg-gray-100"></th>
            {labels.map((label, idx) => (
              <th key={idx} className="border border-gray-300 p-2 bg-gray-100 text-sm font-semibold">
                Predicted: {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIdx) => (
            <tr key={rowIdx}>
              <td className="border border-gray-300 p-2 bg-gray-100 text-sm font-semibold">
                Actual: {labels[rowIdx]}
              </td>
              {row.map((cell, colIdx) => (
                <td
                  key={colIdx}
                  className={`border border-gray-300 p-3 text-center text-white font-semibold ${getColor(cell, maxValue)}`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

