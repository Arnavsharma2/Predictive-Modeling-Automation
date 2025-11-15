'use client'

import { useEffect, useState } from 'react'
import { useBatchJob } from '@/hooks/useBatchPredictions'
import { useBatchWebSocket } from '@/hooks/useBatchWebSocket'
import { getAccessToken } from '@/lib/auth'

interface BatchJobProgressProps {
  jobId: number
  onComplete?: () => void
}

export default function BatchJobProgress({ jobId, onComplete }: BatchJobProgressProps) {
  const { job, loading, refetch } = useBatchJob(jobId)
  const token = typeof window !== 'undefined' ? getAccessToken() : null
  const [isExpanded, setIsExpanded] = useState(true)

  const { isConnected, lastUpdate } = useBatchWebSocket({
    jobId,
    token: token || '',
    enabled: !!token && (job?.status === 'running' || job?.status === 'pending' || job?.status === 'queued'),
    onUpdate: (update) => {
      // Refetch job details when update is received
      refetch()
      
      // Call onComplete if job is finished
      if (update.status === 'completed' || update.status === 'failed' || update.status === 'cancelled') {
        onComplete?.()
      }
    }
  })

  // Auto-refresh if not connected via WebSocket
  useEffect(() => {
    if (!isConnected && job && (job.status === 'running' || job.status === 'pending' || job.status === 'queued')) {
      const interval = setInterval(() => {
        refetch()
      }, 5000) // Poll every 5 seconds
      return () => clearInterval(interval)
    }
  }, [isConnected, job, refetch])

  if (loading || !job) {
    return (
      <div className="p-4 bg-white rounded-lg shadow">
        <div className="text-center py-4">Loading job details...</div>
      </div>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'text-green-600'
      case 'running':
        return 'text-blue-600'
      case 'pending':
      case 'queued':
        return 'text-yellow-600'
      case 'failed':
        return 'text-red-600'
      case 'cancelled':
        return 'text-gray-600'
      default:
        return 'text-gray-600'
    }
  }

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="font-semibold text-lg">
            {job.job_name || `Batch Job #${job.id}`}
          </h3>
          <div className="flex items-center gap-2 mt-1">
            <span className={`text-sm font-medium ${getStatusColor(job.status)}`}>
              {job.status.toUpperCase()}
            </span>
            {isConnected && (job.status === 'running' || job.status === 'pending' || job.status === 'queued') && (
              <span className="text-xs text-green-600">● Live</span>
            )}
          </div>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-gray-500 hover:text-gray-700"
        >
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      {isExpanded && (
        <>
          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600">Progress</span>
              <span className="font-medium">{job.progress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${job.progress}%` }}
              />
            </div>
          </div>

          {job.total_records !== null && job.processed_records !== null && (
            <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
              <div>
                <div className="text-gray-600">Processed</div>
                <div className="font-medium">
                  {job.processed_records.toLocaleString()} / {job.total_records.toLocaleString()}
                </div>
              </div>
              {job.failed_records !== null && job.failed_records > 0 && (
                <div>
                  <div className="text-gray-600">Failed</div>
                  <div className="font-medium text-red-600">
                    {job.failed_records.toLocaleString()}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="text-sm text-gray-600 space-y-1">
            {job.started_at && (
              <div>Started: {new Date(job.started_at).toLocaleString()}</div>
            )}
            {job.completed_at && (
              <div>Completed: {new Date(job.completed_at).toLocaleString()}</div>
            )}
            {job.error_message && (
              <div className="text-red-600 mt-2 p-2 bg-red-50 rounded">
                Error: {job.error_message}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

