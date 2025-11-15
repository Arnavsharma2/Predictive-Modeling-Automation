'use client'

import { useEffect, useState, useCallback, useRef } from 'react'
import { mlEndpoints } from '@/lib/api/endpoints'

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

export default function TrainingProgress({ jobId, modelName, onComplete }: TrainingProgressProps) {
  const [jobStatus, setJobStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdateTime, setLastUpdateTime] = useState<number>(Date.now())
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const onCompleteRef = useRef(onComplete)

  // Log jobId on mount/update
  useEffect(() => {
    console.log(`[TrainingProgress] Component mounted/updated with jobId: ${jobId}, modelName: ${modelName}`)
  }, [jobId, modelName])

  // Keep onComplete ref up to date
  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  const fetchStatus = useCallback(async () => {
    try {
      // Add timestamp to bypass any potential caching
      console.log(`[TrainingProgress] Fetching status for job ${jobId}...`)
      const response = await mlEndpoints.trainStatus(jobId)
      const job = response.data
      console.log(`[TrainingProgress] Received response for job ${jobId}:`, job)

      // Ensure progress is a number
      const progressValue = typeof job.progress === 'number' ? job.progress : parseFloat(job.progress) || 0
      console.log(`[TrainingProgress] Progress value: ${progressValue}`)

      // Always create a new object to force React to detect the change
      const newJobStatus = {
        ...job,
        progress: progressValue, // Ensure progress is always a number
        _timestamp: Date.now(), // Add timestamp to ensure uniqueness
        _fetchTime: Date.now() // Track when we fetched this data
      }
      
      // Check if anything actually changed
      const prevProgress = typeof jobStatus?.progress === 'number' ? jobStatus.progress : parseFloat(jobStatus?.progress) || 0
      const progressChanged = prevProgress !== progressValue
      const statusChanged = jobStatus?.status !== newJobStatus.status

      if (!jobStatus || progressChanged || statusChanged) {
        console.log(`[TrainingProgress] Status update: ${progressValue}% - ${newJobStatus.status}`, {
          prevProgress,
          newProgress: progressValue,
          prevStatus: jobStatus?.status,
          newStatus: newJobStatus.status,
          progressChanged,
          statusChanged
        })
      } else {
        // Even if nothing changed, log that we're polling (for debugging)
        console.log(`[TrainingProgress] Polling: No change (${progressValue}% - ${newJobStatus.status})`)
      }

      // Always update status to ensure UI updates
      // This forces React to see it as a new object even if values are the same
      setJobStatus(newJobStatus)
      
      // Update last update time to force re-render - this ensures the component re-renders
      setLastUpdateTime(Date.now())
      setLoading(false)

      // If completed or failed, stop polling
      const jobStatusLower = job.status ? String(job.status).toLowerCase() : ''
      if (jobStatusLower === 'completed' || jobStatusLower === 'failed' || jobStatusLower === 'cancelled') {
        // Stop polling
        if (intervalRef.current) {
          clearInterval(intervalRef.current)
          intervalRef.current = null
        }
        // Call onComplete callback
        if (onCompleteRef.current) {
          onCompleteRef.current()
        }
      }
    } catch (err) {
      console.error('Failed to fetch training status:', err)
      setLoading(false)
      // Don't stop polling on error - keep trying
      // This ensures we continue polling even if there's a temporary network issue
    }
  }, [jobId])

  useEffect(() => {
    console.log(`[TrainingProgress] Setting up polling for job ${jobId}`)

    // Reset state when jobId changes
    setJobStatus(null)
    setLoading(true)

    // Clear any existing interval
    if (intervalRef.current) {
      console.log(`[TrainingProgress] Clearing existing interval`)
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    // Fetch immediately
    console.log(`[TrainingProgress] Starting immediate fetch for job ${jobId}`)
    fetchStatus()

    // Set up polling interval - poll every 1 second for more real-time updates
    intervalRef.current = setInterval(() => {
      fetchStatus()
    }, 1000) // Poll every 1 second for real-time updates

    console.log(`[TrainingProgress] Polling interval set up for job ${jobId}`)

    // Cleanup on unmount or jobId change
    return () => {
      console.log(`[TrainingProgress] Cleaning up polling for job ${jobId}`)
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [jobId, fetchStatus])

  if (loading && !jobStatus) {
    return (
      <div className="glass rounded-2xl border-pastel-mint/40 p-6">
        <div className="flex items-center justify-center py-8">
          <div className="w-8 h-8 border-4 border-pastel-mint/30 border-t-pastel-mint rounded-full animate-spin"></div>
        </div>
      </div>
    )
  }

  if (!jobStatus) {
    return null
  }

  // Ensure progress is always a number and within valid range
  const progress = typeof jobStatus.progress === 'number' 
    ? Math.max(0, Math.min(100, jobStatus.progress))
    : Math.max(0, Math.min(100, parseFloat(jobStatus.progress) || 0))
  const status = jobStatus.status ? String(jobStatus.status).toLowerCase() : ''
  
  // Determine active step based on progress ranges:
  // 0-10%: Loading Data (threshold 10)
  // 10-20%: Feature Engineering (threshold 20)
  // 20-40%: Data Preprocessing (threshold 40)
  // 40-80%: Training Model (threshold 80)
  // 80-87%: Saving Model (threshold 87)
  // 87-95%: Creating Model Record (threshold 95)
  // 95-100%: Finalizing (threshold 100)
  
  let activeStepIndex = 0 // Default to first step
  
  if (status === 'completed') {
    // All steps are done - show last step as completed
    activeStepIndex = TRAINING_STEPS.length - 1
  } else if (status === 'running' || status === 'pending') {
    // Find the first step whose threshold hasn't been reached yet
    // That step is the active one
    for (let i = 0; i < TRAINING_STEPS.length; i++) {
      const threshold = TRAINING_STEPS[i].progressThreshold
      if (progress < threshold) {
        // This step's threshold hasn't been reached - it's the active step
        activeStepIndex = i
        break
      }
      // If we've reached this threshold, continue to next step
      // If we've reached all thresholds, activeStepIndex will remain at last step
      if (i === TRAINING_STEPS.length - 1) {
        // All thresholds reached, but not completed - show last step as active
        activeStepIndex = i
      }
    }
  }
  
  // Debug logging for step determination
  if (process.env.NODE_ENV === 'development') {
    console.log(`Progress: ${progress}%, Status: ${status}, Active Step: ${activeStepIndex} (${TRAINING_STEPS[activeStepIndex]?.name})`)
  }

  const getStepStatus = (stepIndex: number) => {
    if (status === 'failed' || status === 'cancelled') {
      // Show error on the step where it failed
      if (stepIndex < activeStepIndex) {
        return 'completed'
      }
      if (stepIndex === activeStepIndex) {
        return 'error'
      }
      return 'pending'
    }
    if (status === 'completed') {
      // All steps are completed when training is done
      return 'completed'
    }
    // For running/pending status
    if (stepIndex < activeStepIndex) {
      return 'completed'
    }
    if (stepIndex === activeStepIndex) {
      return 'active'
    }
    return 'pending'
  }

  return (
    <div className="glass rounded-2xl border-pastel-mint/40 p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-bold text-gray-soft-700 flex items-center gap-2">
            <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
            Training Progress
          </h2>
          {(status === 'running' || status === 'pending') && (
            <div className="flex items-center gap-2 text-xs text-pastel-mint">
              <div className="w-2 h-2 bg-pastel-mint rounded-full animate-pulse"></div>
              <span>Updating...</span>
            </div>
          )}
        </div>
        <p className="text-sm text-gray-soft-600">{modelName}</p>
      </div>

      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-semibold text-gray-soft-700">Overall Progress</span>
          <span className="text-sm font-semibold text-pastel-mint">{Math.round(progress)}%</span>
        </div>
        <div className="w-full bg-gray-soft-200 rounded-full h-2.5 overflow-hidden relative">
          <div
            className={`h-2.5 rounded-full transition-all duration-500 ease-out ${
              status === 'failed' || status === 'cancelled'
                ? 'bg-red-500'
                : status === 'completed'
                ? 'bg-accent-500'
                : 'bg-pastel-mint'
            }`}
            style={{ 
              width: `${Math.max(0, Math.min(100, progress))}%`,
              transition: 'width 0.5s ease-out',
              willChange: 'width'
            }}
          ></div>
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-gray-soft-500">
          <span>
            Status: <span className="font-semibold capitalize">{status}</span>
          </span>
          {(status === 'running' || status === 'pending') && (
            <span className="flex items-center gap-1 text-pastel-mint">
              <span className="w-1.5 h-1.5 bg-pastel-mint rounded-full animate-pulse"></span>
              Live
            </span>
          )}
        </div>
      </div>

      <div className="space-y-4">
        {TRAINING_STEPS.map((step, index) => {
          const stepStatus = getStepStatus(index)
          const isActive = stepStatus === 'active'
          const isCompleted = stepStatus === 'completed'
          const isError = stepStatus === 'error' && index === activeStepIndex

          return (
            <div key={`${step.name}-${index}`} className="flex items-start gap-4">
              <div className="flex flex-col items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all ${
                    isCompleted
                      ? 'bg-accent-500 border-accent-500 text-white'
                      : isActive
                      ? 'bg-pastel-mint border-pastel-mint text-gray-soft-700'
                      : isError
                      ? 'bg-red-500 border-red-500 text-white'
                      : 'bg-gray-soft-100 border-gray-soft-300 text-gray-soft-400'
                  }`}
                >
                  {isCompleted ? (
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  ) : isActive ? (
                    <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                  ) : isError ? (
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                        clipRule="evenodd"
                      />
                    </svg>
                  ) : (
                    <div className="w-2 h-2 bg-gray-soft-400 rounded-full"></div>
                  )}
                </div>
                {index < TRAINING_STEPS.length - 1 && (
                  <div
                    className={`w-0.5 h-12 mt-2 ${
                      isCompleted ? 'bg-accent-500' : 'bg-gray-soft-200'
                    }`}
                  ></div>
                )}
              </div>
              <div className="flex-1 pt-1">
                <h3
                  className={`font-semibold ${
                    isActive
                      ? 'text-pastel-mint'
                      : isCompleted
                      ? 'text-accent-500'
                      : isError
                      ? 'text-red-600'
                      : 'text-gray-soft-500'
                  }`}
                >
                  {step.name}
                </h3>
                <p className="text-sm text-gray-soft-600 mt-1">{step.description}</p>
                {isActive && status === 'running' && (
                  <div className="mt-2 flex items-center gap-2 text-xs text-pastel-mint">
                    <div className="w-2 h-2 bg-pastel-mint rounded-full animate-pulse"></div>
                    In progress...
                  </div>
                )}
                {isError && jobStatus.error_message && (
                  <div className="mt-2 text-xs text-red-600 bg-red-50 p-2 rounded">
                    Error: {jobStatus.error_message}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {status === 'completed' && jobStatus.metrics && (
        <div className="mt-6 pt-6 border-t border-gray-soft-200">
          <h3 className="text-sm font-semibold text-gray-soft-700 mb-3">Training Results</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            {jobStatus.metrics.rmse && (
              <div>
                <span className="text-gray-soft-600">RMSE:</span>
                <span className="ml-2 font-semibold text-gray-soft-700">
                  {jobStatus.metrics.rmse.toFixed(4)}
                </span>
              </div>
            )}
            {jobStatus.metrics.r2_score !== undefined && (
              <div>
                <span className="text-gray-soft-600">R² Score:</span>
                <span className="ml-2 font-semibold text-gray-soft-700">
                  {jobStatus.metrics.r2_score.toFixed(4)}
                </span>
              </div>
            )}
            {jobStatus.metrics.accuracy !== undefined && (
              <div>
                <span className="text-gray-soft-600">Accuracy:</span>
                <span className="ml-2 font-semibold text-gray-soft-700">
                  {(jobStatus.metrics.accuracy * 100).toFixed(2)}%
                </span>
              </div>
            )}
            {jobStatus.metrics.mae && (
              <div>
                <span className="text-gray-soft-600">MAE:</span>
                <span className="ml-2 font-semibold text-gray-soft-700">
                  {jobStatus.metrics.mae.toFixed(4)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

