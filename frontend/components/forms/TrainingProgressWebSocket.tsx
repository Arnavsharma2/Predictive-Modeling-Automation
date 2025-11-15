'use client'

import { useEffect, useState, useRef } from 'react'
import { useTrainingWebSocket } from '@/hooks/useTrainingWebSocket'
import { mlEndpoints } from '@/lib/api/endpoints'
import { getAccessToken } from '@/lib/auth'

interface TrainingProgressProps {
  jobId: number
  modelName: string
  onComplete?: () => void
}

interface TrainingStep {
  name: string
  description: string
  progressThreshold: number
}

const TRAINING_STEPS: TrainingStep[] = [
  {
    name: 'Loading Data',
    description: 'Loading training data from data source',
    progressThreshold: 10
  },
  {
    name: 'Feature Engineering',
    description: 'Creating and engineering features',
    progressThreshold: 20
  },
  {
    name: 'Data Preprocessing',
    description: 'Preprocessing and cleaning the data',
    progressThreshold: 40
  },
  {
    name: 'Training Model',
    description: 'Training the machine learning model',
    progressThreshold: 80
  },
  {
    name: 'Saving Model',
    description: 'Saving model to storage and logging experiments',
    progressThreshold: 87
  },
  {
    name: 'Creating Model Record',
    description: 'Creating model record in database',
    progressThreshold: 95
  },
  {
    name: 'Finalizing',
    description: 'Completing training job',
    progressThreshold: 100
  }
]

export default function TrainingProgressWebSocket({ jobId, modelName, onComplete }: TrainingProgressProps) {
  const token = getAccessToken()
  const [jobStatus, setJobStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const onCompleteRef = useRef(onComplete)

  // Keep onComplete ref up to date
  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  // Fetch initial status (fallback if WebSocket hasn't connected yet)
  useEffect(() => {
    const fetchInitialStatus = async () => {
      try {
        const response = await mlEndpoints.trainStatus(jobId)
        const job = response.data
        const progressValue = typeof job.progress === 'number' ? job.progress : parseFloat(job.progress) || 0

        setJobStatus({
          ...job,
          progress: progressValue,
        })
        setLoading(false)

        // If already completed, call callback
        const jobStatusLower = job.status ? String(job.status).toLowerCase() : ''
        if (jobStatusLower === 'completed' || jobStatusLower === 'failed' || jobStatusLower === 'cancelled') {
          if (onCompleteRef.current) {
            onCompleteRef.current()
          }
        }
      } catch (err) {
        console.error('Failed to fetch initial training status:', err)
        setLoading(false)
      }
    }

    fetchInitialStatus()
  }, [jobId])

  // WebSocket connection for real-time updates
  const { isConnected, lastUpdate, error } = useTrainingWebSocket({
    jobId,
    token: token || '',
    enabled: !!token,
    onUpdate: (update) => {
      console.log(`[TrainingProgress] WebSocket update:`, update)

      const progressValue = typeof update.progress === 'number' ? update.progress : parseFloat(String(update.progress)) || 0

      setJobStatus({
        id: update.job_id,
        status: update.status,
        progress: progressValue,
        current_epoch: update.current_epoch,
        total_epochs: update.total_epochs,
        metrics: update.metrics,
        error_message: update.error_message,
        updated_at: update.updated_at
      })
      setLoading(false)

      // If completed or failed, call callback
      const statusLower = update.status ? String(update.status).toLowerCase() : ''
      if (statusLower === 'completed' || statusLower === 'failed' || statusLower === 'cancelled') {
        if (onCompleteRef.current) {
          onCompleteRef.current()
        }
      }
    },
    onError: (err) => {
      console.error('[TrainingProgress] WebSocket error:', err)
    }
  })

  if (loading && !jobStatus) {
    return (
      <div className="glass rounded-2xl border-pastel-mint/40 p-6">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin h-8 w-8 border-4 border-pastel-blue border-t-transparent rounded-full"></div>
        </div>
      </div>
    )
  }

  if (!jobStatus) {
    return (
      <div className="glass rounded-2xl border-pastel-mint/40 p-6">
        <div className="text-center py-8 text-gray-500">
          No training status available
        </div>
      </div>
    )
  }

  const progress = jobStatus.progress || 0
  const status = jobStatus.status || 'PENDING'
  const statusLower = String(status).toLowerCase()

  // Determine current step based on progress
  let currentStepIndex = 0
  for (let i = 0; i < TRAINING_STEPS.length; i++) {
    if (progress < TRAINING_STEPS[i].progressThreshold) {
      currentStepIndex = Math.max(0, i - 1)
      break
    }
  }
  if (progress >= 100) {
    currentStepIndex = TRAINING_STEPS.length - 1
  }

  const getStatusColor = () => {
    if (statusLower === 'completed') return 'text-green-600'
    if (statusLower === 'failed') return 'text-red-600'
    if (statusLower === 'running') return 'text-blue-600'
    return 'text-gray-600'
  }

  const getStatusBadgeColor = () => {
    if (statusLower === 'completed') return 'bg-green-100 text-green-800'
    if (statusLower === 'failed') return 'bg-red-100 text-red-800'
    if (statusLower === 'running') return 'bg-blue-100 text-blue-800'
    return 'bg-gray-100 text-gray-800'
  }

  return (
    <div className="glass rounded-2xl border-pastel-mint/40 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">{modelName}</h3>
          <p className="text-sm text-gray-500">Job ID: {jobId}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusBadgeColor()}`}>
            {status}
          </span>
          {isConnected && (
            <span className="flex items-center gap-1 text-xs text-green-600">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              Live
            </span>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Training Progress</span>
          <span className={`text-sm font-semibold ${getStatusColor()}`}>{progress.toFixed(1)}%</span>
        </div>
        <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-pastel-blue to-pastel-purple transition-all duration-300 ease-out"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
      </div>

      {/* Current Step */}
      {statusLower === 'running' && (
        <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center gap-2 mb-2">
            <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
            <span className="font-medium text-blue-900">{TRAINING_STEPS[currentStepIndex]?.name}</span>
          </div>
          <p className="text-sm text-blue-700">{TRAINING_STEPS[currentStepIndex]?.description}</p>
        </div>
      )}

      {/* Epoch Progress (if available) */}
      {jobStatus.current_epoch != null && jobStatus.total_epochs != null && (
        <div className="mb-4">
          <div className="text-sm text-gray-600">
            Epoch {jobStatus.current_epoch} of {jobStatus.total_epochs}
          </div>
        </div>
      )}

      {/* Metrics (if available) */}
      {jobStatus.metrics && Object.keys(jobStatus.metrics).length > 0 && (
        <div className="mb-4">
          <div className="text-sm font-medium text-gray-700 mb-2">Training Metrics</div>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(jobStatus.metrics).slice(0, 4).map(([key, value]) => (
              <div key={key} className="p-2 bg-gray-50 rounded-lg">
                <div className="text-xs text-gray-500 capitalize">{key.replace(/_/g, ' ')}</div>
                <div className="text-sm font-semibold text-gray-800">
                  {typeof value === 'number' ? value.toFixed(4) : String(value)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Message */}
      {statusLower === 'failed' && jobStatus.error_message && (
        <div className="p-4 bg-red-50 rounded-lg border border-red-200">
          <div className="font-medium text-red-900 mb-1">Error</div>
          <p className="text-sm text-red-700">{jobStatus.error_message}</p>
        </div>
      )}

      {/* Completed Message */}
      {statusLower === 'completed' && (
        <div className="p-4 bg-green-50 rounded-lg border border-green-200">
          <div className="flex items-center gap-2 text-green-900">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <span className="font-medium">Training completed successfully!</span>
          </div>
        </div>
      )}

      {/* WebSocket Connection Error */}
      {error && !isConnected && (
        <div className="mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
          <div className="text-xs text-yellow-800">
            WebSocket disconnected. Updates may be delayed.
          </div>
        </div>
      )}
    </div>
  )
}
