'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { useModels, useModelDetails, useModelVersions, useTrainingJobs } from '@/hooks/useModels'
import { mlEndpoints } from '@/lib/api/endpoints'
import { useDataSources } from '@/hooks/useData'
import { usePolling } from '@/hooks/usePolling'
import Link from 'next/link'
import TrainingProgressWebSocket from '@/components/forms/TrainingProgressWebSocket'

interface SelectionBox {
  startX: number
  startY: number
  endX: number
  endY: number
}

export default function ModelsPage() {
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null)
  const [showTrainingForm, setShowTrainingForm] = useState(false)
  const [activeTrainingJob, setActiveTrainingJob] = useState<{ jobId: number; modelName: string } | null>(null)
  const [trainingConfig, setTrainingConfig] = useState({
    model_name: '',
    model_type: 'regression',
    data_source_id: '',
    target_column: '',
    algorithm: 'random_forest',
    description: ''
  })

  const { models, loading: modelsLoading, refetch } = useModels()
  const { model: modelDetails, loading: detailsLoading } = useModelDetails(selectedModelId)
  const { versions, loading: versionsLoading } = useModelVersions(selectedModelId)
  const { jobs: trainingJobs } = useTrainingJobs({ model_id: selectedModelId ?? undefined })
  const { jobs: allTrainingJobs, refetch: refetchAllJobs } = useTrainingJobs({ limit: 10 })
  const { sources } = useDataSources()

  // Restore active training job from localStorage on mount (before API data is available)
  useEffect(() => {
    try {
      const saved = localStorage.getItem('activeTrainingJob')
      if (saved) {
        const parsed = JSON.parse(saved)
        if (parsed && parsed.jobId && parsed.modelName) {
          setActiveTrainingJob(parsed)
        }
      }
    } catch (err) {
      console.error('Failed to restore active training job from localStorage:', err)
    }
  }, []) // Only run on mount

  // Save active training job to localStorage whenever it changes
  useEffect(() => {
    if (activeTrainingJob) {
      try {
        localStorage.setItem('activeTrainingJob', JSON.stringify(activeTrainingJob))
      } catch (err) {
        console.error('Failed to save active training job to localStorage:', err)
      }
    } else {
      // Remove from localStorage when cleared
      try {
        localStorage.removeItem('activeTrainingJob')
      } catch (err) {
        console.error('Failed to remove active training job from localStorage:', err)
      }
    }
  }, [activeTrainingJob])

  // Poll for training jobs updates when there's an active training job (only to verify status, not to reload models)
  usePolling({
    enabled: !!activeTrainingJob,
    interval: 10000, // Poll every 10 seconds (less frequent, just to verify job status)
    onPoll: async () => {
      await refetchAllJobs()
      // Don't refetch models - TrainingProgress component handles its own status updates
    }
  })

  // Verify and restore active training jobs from API data
  useEffect(() => {
    console.log(`[ModelsPage] Verifying active training jobs. allTrainingJobs count: ${allTrainingJobs.length}, activeTrainingJob:`, activeTrainingJob)

    // Only verify if we have API data loaded
    if (allTrainingJobs.length === 0) {
      console.log(`[ModelsPage] No training jobs loaded yet, waiting...`)
      return // Wait for API data to load
    }

    // If we have an activeTrainingJob from localStorage, verify it's still running
    if (activeTrainingJob) {
      console.log(`[ModelsPage] Checking if activeTrainingJob ${activeTrainingJob.jobId} is still running...`)
      const currentJob = allTrainingJobs.find((job: any) => job.id === activeTrainingJob.jobId)
      if (currentJob) {
        // Job found in API
        console.log(`[ModelsPage] Found job ${activeTrainingJob.jobId} in API. Status: ${currentJob.status}, Progress: ${currentJob.progress}`)
        if (currentJob.status) {
          const status = String(currentJob.status).toLowerCase()
          if (status !== 'running' && status !== 'pending') {
            // Job is no longer running, clear it
            console.log(`[ModelsPage] Job ${activeTrainingJob.jobId} is no longer running (status: ${status}). Clearing.`)
            setActiveTrainingJob(null)
            localStorage.removeItem('activeTrainingJob')
          } else {
            // Job is still running, update model name if available
            console.log(`[ModelsPage] Job ${activeTrainingJob.jobId} is still ${status}`)
            const modelName = currentJob.model_id 
              ? models.find((m: any) => m.id === currentJob.model_id)?.name || activeTrainingJob.modelName
              : activeTrainingJob.modelName
            if (modelName !== activeTrainingJob.modelName) {
              setActiveTrainingJob({
                jobId: activeTrainingJob.jobId,
                modelName
              })
            }
          }
        }
      } else {
        // Job not found in API - might have been deleted or API hasn't loaded it yet
        // Don't clear it immediately - let the TrainingProgress component handle errors
        // It will show an error if the job doesn't exist
        console.warn(`Training job ${activeTrainingJob.jobId} not found in API response. Keeping it visible - TrainingProgress will handle errors.`)
      }
    } else {
      // If we don't have an activeTrainingJob, check for running jobs in API
      // Find the most recent running or pending job
      console.log(`[ModelsPage] No activeTrainingJob, searching for running jobs in ${allTrainingJobs.length} jobs...`)
      const runningJob = allTrainingJobs.find((job: any) => {
        if (!job || !job.status) return false
        const status = String(job.status).toLowerCase()
        return status === 'running' || status === 'pending'
      })
      if (runningJob) {
        console.log(`[ModelsPage] Found running job: ${runningJob.id}, status: ${runningJob.status}, progress: ${runningJob.progress}`)
        // Try to get model name from the job or use a default
        const modelName = runningJob.model_id 
          ? models.find((m: any) => m.id === runningJob.model_id)?.name || 'Training Model'
          : 'Training Model'
        setActiveTrainingJob({
          jobId: runningJob.id,
          modelName
        })
      }
    }
  }, [allTrainingJobs, models, activeTrainingJob])

  // Selection state for drag-to-select
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [isSelecting, setIsSelecting] = useState(false)
  const [selectionBox, setSelectionBox] = useState<SelectionBox | null>(null)
  const [isShiftHeld, setIsShiftHeld] = useState(false)
  const modelsContainerRef = useRef<HTMLDivElement>(null)
  const modelRefs = useRef<Map<number, HTMLDivElement>>(new Map())

  const handleTrain = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const response = await mlEndpoints.train({
        ...trainingConfig,
        data_source_id: parseInt(trainingConfig.data_source_id),
      })
      // Track the active training job
      if (response.data?.job_id) {
        const newActiveJob = {
          jobId: response.data.job_id,
          modelName: trainingConfig.model_name
        }
        setActiveTrainingJob(newActiveJob)
        // Save to localStorage immediately
        try {
          localStorage.setItem('activeTrainingJob', JSON.stringify(newActiveJob))
        } catch (err) {
          console.error('Failed to save active training job to localStorage:', err)
        }
      }
      setShowTrainingForm(false)
      setTrainingConfig({
        model_name: '',
        model_type: 'regression',
        data_source_id: '',
        target_column: '',
        algorithm: 'random_forest',
        description: ''
      })
      refetch()
    } catch (err) {
      console.error('Training failed:', err)
    }
  }

  // Reset selection when models change
  useEffect(() => {
    setSelectedIds(new Set())
    setSelectionBox(null)
  }, [models.length])

  const handleDeleteModel = async (modelId: number, modelName: string, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent triggering the model selection
    if (!confirm(`Are you sure you want to delete the model "${modelName}"? This action cannot be undone and will delete all associated versions and training jobs.`)) {
      return
    }
    
    try {
      await mlEndpoints.deleteModel(modelId)
      if (selectedModelId === modelId) {
        setSelectedModelId(null)
      }
      refetch()
    } catch (err) {
      console.error('Delete failed:', err)
      alert('Failed to delete model. Please try again.')
    }
  }

  const handleBulkDelete = useCallback(async () => {
    if (selectedIds.size === 0) return
    
    if (!confirm(`Are you sure you want to delete ${selectedIds.size} model(s)? This action cannot be undone and will delete all associated versions, training jobs, and model files.`)) {
      return
    }
    
    try {
      const idsArray = Array.from(selectedIds)
      // Delete one at a time
      for (const id of idsArray) {
        await mlEndpoints.deleteModel(id)
      }
      setSelectedIds(new Set())
      if (selectedModelId && selectedIds.has(selectedModelId)) {
        setSelectedModelId(null)
      }
      refetch()
    } catch (err) {
      console.error('Delete failed:', err)
      alert('Failed to delete some models. Please try again.')
      refetch() // Refresh to show current state
    }
  }, [selectedIds, selectedModelId, refetch])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0 || (e.target as HTMLElement).tagName === 'BUTTON') return

    const container = modelsContainerRef.current
    if (!container) return

    // Prevent text selection during drag
    e.preventDefault()

    const rect = container.getBoundingClientRect()
    const startX = e.clientX - rect.left
    const startY = e.clientY - rect.top

    setIsSelecting(true)
    setIsShiftHeld(e.shiftKey)
    setSelectionBox({ startX, startY, endX: startX, endY: startY })

    if (!e.shiftKey && !e.ctrlKey && !e.metaKey) {
      setSelectedIds(new Set())
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isSelecting || !selectionBox || !modelsContainerRef.current) return

    // Prevent text selection during drag
    e.preventDefault()

    const container = modelsContainerRef.current
    const rect = container.getBoundingClientRect()
    const endX = e.clientX - rect.left
    const endY = e.clientY - rect.top

    const updatedBox = {
      ...selectionBox,
      endX,
      endY
    }
    
    setSelectionBox(updatedBox)
    updateSelectionFromBox(updatedBox.startX, updatedBox.startY, endX, endY)
  }

  const updateSelectionFromBox = (startX: number, startY: number, endX: number, endY: number) => {
    const boxLeft = Math.min(startX, endX)
    const boxRight = Math.max(startX, endX)
    const boxTop = Math.min(startY, endY)
    const boxBottom = Math.max(startY, endY)

    const newSelected = isShiftHeld ? new Set(selectedIds) : new Set<number>()

    modelRefs.current.forEach((element: HTMLDivElement, modelId: number) => {
      if (!element) return

      const elementRect = element.getBoundingClientRect()
      const containerRect = modelsContainerRef.current?.getBoundingClientRect()
      if (!containerRect) return

      const elTop = elementRect.top - containerRect.top
      const elBottom = elementRect.bottom - containerRect.top
      const elLeft = elementRect.left - containerRect.left
      const elRight = elementRect.right - containerRect.left

      const intersects = !(
        elBottom < boxTop ||
        elTop > boxBottom ||
        elRight < boxLeft ||
        elLeft > boxRight
      )

      if (intersects) {
        newSelected.add(modelId)
      }
    })

    setSelectedIds(newSelected)
  }

  const handleMouseUp = () => {
    setIsSelecting(false)
    setIsShiftHeld(false)
    setSelectionBox(null)
  }

  useEffect(() => {
    if (isSelecting) {
      window.addEventListener('mouseup', handleMouseUp)
      return () => window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isSelecting])

  // Handle Delete key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only trigger if Delete or Backspace is pressed and items are selected
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIds.size > 0) {
        // Don't trigger if user is typing in an input field
        const target = e.target as HTMLElement
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
          return
        }
        e.preventDefault()
        handleBulkDelete()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedIds.size, handleBulkDelete])

  return (
    <div className="max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-4xl font-bold text-gray-soft-700 mb-2">ML Models</h1>
          <p className="text-gray-soft-600 font-medium">Train and manage machine learning models</p>
        </div>
        <div className="flex gap-2">
          {selectedIds.size > 0 && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-pastel-mint/20 border border-pastel-mint/40 rounded-lg">
              <svg className="w-4 h-4 text-pastel-mint" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="text-sm font-semibold text-primary-600">{selectedIds.size} selected</span>
            </div>
          )}
          <button
            onClick={() => setShowTrainingForm(!showTrainingForm)}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all duration-200 shadow-sm hover:shadow-md ${
              showTrainingForm 
                ? "bg-gray-soft-200 text-gray-soft-700 hover:bg-gray-soft-300" 
                : "bg-pastel-mint text-gray-soft-700 hover:bg-pastel-green"
            }`}
          >
            {showTrainingForm ? (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
                Cancel
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Train New Model
              </>
            )}
          </button>
        </div>
      </div>

      {showTrainingForm && (
        <div className="glass rounded-2xl border-pastel-mint/40 p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-soft-700 mb-6 flex items-center gap-2">
            <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
            Train New Model
          </h2>
          <form onSubmit={handleTrain} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wider">Model Name</label>
                <input
                  type="text"
                  value={trainingConfig.model_name}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, model_name: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-600 focus:border-primary-600 text-gray-900 placeholder-gray-400 transition-all"
                  placeholder="Enter model name"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wider">Model Type</label>
                <select
                  value={trainingConfig.model_type}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, model_type: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-600 focus:border-primary-600 text-gray-900 transition-all"
                >
                  <option value="regression">Regression</option>
                  <option value="classification">Classification</option>
                  <option value="anomaly_detection">Anomaly Detection</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wider">Data Source</label>
                <select
                  value={trainingConfig.data_source_id}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, data_source_id: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-600 focus:border-primary-600 text-gray-900 transition-all"
                  required
                >
                  <option value="">Select data source...</option>
                  {sources.map((source: any) => (
                    <option key={source.id} value={source.id}>{source.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wider">Target Column</label>
                <input
                  type="text"
                  value={trainingConfig.target_column}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, target_column: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-600 focus:border-primary-600 text-gray-900 placeholder-gray-400 transition-all"
                  placeholder="Enter target column"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wider">Algorithm</label>
                <select
                  value={trainingConfig.algorithm}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, algorithm: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-600 focus:border-primary-600 text-gray-900 transition-all"
                >
                  <option value="random_forest">Random Forest</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="lightgbm">LightGBM</option>
                  <option value="catboost">CatBoost</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wider">Description</label>
              <textarea
                value={trainingConfig.description}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, description: e.target.value })}
                className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-600 focus:border-primary-600 text-gray-900 placeholder-gray-400 transition-all resize-none"
                rows={3}
                placeholder="Add a description for this model"
              />
            </div>
            <button
              type="submit"
              className="flex items-center justify-center gap-2 bg-pastel-mint text-gray-soft-700 px-6 py-3 rounded-lg font-semibold transition-all duration-200 hover:bg-pastel-green shadow-sm hover:shadow-md"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Start Training
            </button>
          </form>
        </div>
      )}

      <div className={`grid grid-cols-1 gap-6 ${activeTrainingJob || selectedModelId ? 'lg:grid-cols-3' : 'lg:grid-cols-1'}`}>
        <div className={activeTrainingJob || selectedModelId ? 'lg:col-span-2' : 'lg:col-span-1'}>
          <div className="glass rounded-2xl border-pastel-mint/40 p-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-bold text-gray-soft-700 flex items-center gap-2">
                  <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
                  Models
                </h2>
                <p className="text-xs text-gray-soft-500 mt-1 flex items-center gap-1">
                  <svg className="w-3 h-3 text-pastel-mint" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Click and drag to select multiple models. Press Delete to remove selected items.
                </p>
              </div>
            </div>
            {modelsLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-8 h-8 border-4 border-pastel-mint/30 border-t-pastel-mint rounded-full animate-spin"></div>
                  <p className="text-sm text-gray-soft-500">Loading models...</p>
                </div>
              </div>
            ) : models.length === 0 ? (
              <div className="text-center py-12">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-16 h-16 bg-pastel-mint/20 rounded-full flex items-center justify-center">
                    <svg className="w-8 h-8 text-pastel-mint" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-soft-700">No models yet</h3>
                  <p className="text-sm text-gray-soft-500 max-w-sm">Train your first machine learning model to get started with predictions and analytics.</p>
                </div>
              </div>
            ) : (
              <div 
                ref={modelsContainerRef}
                className={`space-y-4 relative ${isSelecting ? 'select-none' : ''}`}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
              >
                {/* Selection box overlay */}
                {selectionBox && (
                  <div
                    className="absolute border-2 border-pastel-mint bg-pastel-mint/20 pointer-events-none z-10 rounded shadow-lg"
                    style={{
                      left: `${Math.min(selectionBox.startX, selectionBox.endX)}px`,
                      top: `${Math.min(selectionBox.startY, selectionBox.endY)}px`,
                      width: `${Math.abs(selectionBox.endX - selectionBox.startX)}px`,
                      height: `${Math.abs(selectionBox.endY - selectionBox.startY)}px`,
                    }}
                  />
                )}
                {models.map((model: any) => {
                  const isSelected = selectedIds.has(model.id)
                  return (
                    <div
                      key={model.id}
                      ref={(el) => {
                        if (el) {
                          modelRefs.current.set(model.id, el)
                        } else {
                          modelRefs.current.delete(model.id)
                        }
                      }}
                      onClick={() => {
                        if (!isSelecting) {
                          setSelectedModelId(model.id === selectedModelId ? null : model.id)
                        }
                      }}
                      className={`p-5 border rounded-xl cursor-pointer transition-all duration-200 hover:shadow-md ${
                        isSelected
                          ? 'border-pastel-mint bg-pastel-mint/20 shadow-md ring-2 ring-pastel-mint/30'
                          : selectedModelId === model.id 
                            ? 'border-pastel-mint bg-pastel-mint/20 shadow-md ring-2 ring-pastel-mint/30' 
                            : 'border-gray-soft-200 hover:border-pastel-mint/50 glass'
                      }`}
                    >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h3 className="font-bold text-gray-900 text-lg">{model.name}</h3>
                        <p className="text-sm text-gray-600 mt-1">
                          {model.type} • Version {model.version}
                        </p>
                        {model.description && (
                          <p className="text-sm text-gray-700 mt-2">{model.description}</p>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`px-3 py-1.5 inline-flex items-center gap-1.5 text-xs font-semibold rounded-full ${
                          model.status === 'trained' || model.status === 'deployed' ? 'bg-green-100 text-green-700 border border-green-200' :
                          model.status === 'training' ? 'bg-yellow-100 text-yellow-700 border border-yellow-200' :
                          'bg-gray-100 text-gray-700 border border-gray-200'
                        }`}>
                          {model.status === 'trained' || model.status === 'deployed' ? (
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                          ) : model.status === 'training' ? (
                            <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                          ) : null}
                          {model.status}
                        </span>
                        <button
                          onClick={(e) => handleDeleteModel(model.id, model.name, e)}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-all duration-150 border border-red-200 hover:border-red-300 hover:shadow-sm"
                          title="Delete model"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                          Delete
                        </button>
                      </div>
                    </div>
                    {model.rmse && (
                      <div className="mt-3 text-sm text-gray-700">
                        <span className="text-primary-600 font-semibold">RMSE:</span> {model.rmse.toFixed(4)} • <span className="text-primary-600 font-semibold">R²:</span> {model.r2_score?.toFixed(4) || 'N/A'}
                      </div>
                    )}
                    {model.accuracy && (
                      <div className="mt-3 text-sm text-gray-700">
                        <span className="text-primary-600 font-semibold">Accuracy:</span> {(model.accuracy * 100).toFixed(2)}%
                      </div>
                    )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        {(activeTrainingJob || selectedModelId) && (
          <div className="lg:col-span-1 space-y-6">
            {activeTrainingJob && (
              <TrainingProgressWebSocket
                jobId={activeTrainingJob.jobId}
                modelName={activeTrainingJob.modelName}
                onComplete={() => {
                  setActiveTrainingJob(null)
                  localStorage.removeItem('activeTrainingJob')
                  // Only refetch models when training completes (to show the new model)
                  refetch()
                }}
              />
            )}
            
            {selectedModelId && (
              <>
                <div className="glass rounded-2xl border-pastel-mint/40 p-6">
                  <h2 className="text-xl font-bold text-gray-soft-700 mb-6 flex items-center gap-2">
                    <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
                    Model Details
                  </h2>
                  {detailsLoading ? (
                    <p className="text-gray-500">Loading...</p>
                  ) : modelDetails ? (
                    <div className="space-y-4 text-sm">
                      <div>
                        <span className="font-semibold text-gray-600">Type:</span>
                        <span className="text-gray-900 ml-2">{modelDetails.type}</span>
                      </div>
                      <div>
                        <span className="font-semibold text-gray-600">Status:</span>
                        <span className="text-gray-900 ml-2">{modelDetails.status}</span>
                      </div>
                      {modelDetails.features && (
                        <div>
                          <span className="font-semibold text-gray-600">Features:</span>
                          <span className="text-gray-900 ml-2">{modelDetails.features.length}</span>
                        </div>
                      )}
                      {modelDetails.feature_importance && (
                        <div>
                          <span className="font-semibold text-gray-600 mb-2 block">Top Features:</span>
                          <ul className="mt-2 space-y-2">
                            {Object.entries(modelDetails.feature_importance)
                              .sort(([, a]: any, [, b]: any) => b - a)
                              .slice(0, 5)
                              .map(([feature, importance]: [string, any]) => (
                                <li key={feature} className="text-xs text-gray-700 flex justify-between items-center">
                                  <span>{feature}</span>
                                  <span className="text-primary-600 font-semibold">{importance.toFixed(4)}</span>
                                </li>
                              ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ) : null}
                </div>

                {versions.length > 0 && (
                  <div className="glass rounded-2xl border-pastel-mint/40 p-6">
                    <h2 className="text-xl font-bold text-gray-soft-700 mb-6 flex items-center gap-2">
                      <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
                      Versions
                    </h2>
                    <div className="space-y-3">
                      {versions.map((version: any) => (
                        <div key={version.id} className="text-sm p-4 border border-gray-200 rounded-lg bg-gray-50">
                          <div className="flex justify-between items-center mb-2">
                            <span className="font-bold text-gray-900">v{version.version}</span>
                            {version.is_active && (
                              <span className="text-xs bg-green-100 text-green-700 border border-green-200 px-2 py-1 rounded-full font-semibold">Active</span>
                            )}
                          </div>
                          {version.performance_metrics && (
                            <div className="mt-2 text-xs text-gray-600">
                              {version.performance_metrics.rmse && <span>RMSE: <span className="text-primary-600">{version.performance_metrics.rmse.toFixed(4)}</span></span>}
                              {version.performance_metrics.accuracy && <span> • Accuracy: <span className="text-primary-600">{(version.performance_metrics.accuracy * 100).toFixed(2)}%</span></span>}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

