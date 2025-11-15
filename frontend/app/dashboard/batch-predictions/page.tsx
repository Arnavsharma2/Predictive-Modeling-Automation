'use client'

import { useState, useEffect } from 'react'
import { useBatchJobs, useCreateBatchJob, useCancelBatchJob, useDownloadBatchResults, useBatchJob, BatchJob } from '@/hooks/useBatchPredictions'
import { useModels } from '@/hooks/useModels'
import { useDataSources } from '@/hooks/useData'
import { useBatchWebSocket } from '@/hooks/useBatchWebSocket'
import { useAuth } from '@/contexts/AuthContext'
import { getAccessToken } from '@/lib/auth'
import { mlEndpoints } from '@/lib/api/endpoints'

export default function BatchPredictionsPage() {
  const { user } = useAuth()
  const token = typeof window !== 'undefined' ? getAccessToken() : null
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [createForm, setCreateForm] = useState({
    model_id: '',
    input_type: 'data_source',
    data_source_id: '',
    job_name: '',
    result_format: 'csv',
  })
  const [file, setFile] = useState<File | null>(null)
  const [uploadingFile, setUploadingFile] = useState(false)
  const [statusFilter, setStatusFilter] = useState<string>('')

  const { jobs, loading, error, total, refetch } = useBatchJobs({
    skip: 0,
    limit: 50,
    status_filter: statusFilter || undefined
  })
  const { job: selectedJob } = useBatchJob(selectedJobId)
  const { models } = useModels()
  const { sources } = useDataSources()
  const { createJob, loading: creating } = useCreateBatchJob()
  const { cancelJob, loading: cancelling } = useCancelBatchJob()
  const { downloadResults, loading: downloading } = useDownloadBatchResults()

  // WebSocket connection for selected job
  const { isConnected, lastUpdate } = useBatchWebSocket({
    jobId: selectedJobId || 0,
    token: token || '',
    enabled: !!selectedJobId && !!token && selectedJob?.status === 'running',
    onUpdate: (update) => {
      // Refetch job when update is received
      if (update.job_id === selectedJobId) {
        refetch()
      }
    }
  })

  // Auto-refresh running jobs
  useEffect(() => {
    const interval = setInterval(() => {
      const hasRunningJobs = jobs.some(j => j.status === 'running' || j.status === 'pending')
      if (hasRunningJobs) {
        refetch()
      }
    }, 5000) // Poll every 5 seconds for running jobs

    return () => clearInterval(interval)
  }, [jobs, refetch])

  const handleCreateJob = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const jobData: any = {
        model_id: parseInt(createForm.model_id),
        input_type: createForm.input_type,
        result_format: createForm.result_format,
      }
      
      if (createForm.job_name) {
        jobData.job_name = createForm.job_name
      }
      
      if (createForm.input_type === 'data_source' && createForm.data_source_id) {
        jobData.data_source_id = parseInt(createForm.data_source_id)
      } else if (createForm.input_type === 'file') {
        // Upload file first if not already uploaded
        if (!file) {
          alert('Please select a file for batch prediction')
          return
        }
        
        setUploadingFile(true)
        try {
          const formData = new FormData()
          formData.append('file', file)
          const uploadResponse = await mlEndpoints.uploadBatchFile(formData)
          jobData.input_config = {
            file_path: uploadResponse.data.file_path
          }
        } catch (uploadErr: any) {
          console.error('Failed to upload file:', uploadErr)
          alert(uploadErr.response?.data?.detail || 'Failed to upload file')
          setUploadingFile(false)
          return
        } finally {
          setUploadingFile(false)
        }
      }

      await createJob(jobData)
      setShowCreateForm(false)
      setCreateForm({
        model_id: '',
        input_type: 'data_source',
        data_source_id: '',
        job_name: '',
        result_format: 'csv',
      })
      setFile(null)
      refetch()
    } catch (err) {
      console.error('Failed to create batch job:', err)
    }
  }

  const handleCancelJob = async (jobId: number) => {
    if (confirm('Are you sure you want to cancel this batch job?')) {
      try {
        await cancelJob(jobId)
        refetch()
      } catch (err) {
        console.error('Failed to cancel batch job:', err)
      }
    }
  }

  const handleDownloadResults = async (jobId: number) => {
    try {
      await downloadResults(jobId)
    } catch (err) {
      console.error('Failed to download results:', err)
      alert('Failed to download results. Please try again.')
    }
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'running':
        return 'bg-blue-100 text-blue-800'
      case 'pending':
      case 'queued':
        return 'bg-yellow-100 text-yellow-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      case 'cancelled':
        return 'bg-gray-100 text-gray-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const readyModels = models.filter(m => m.status === 'trained' || m.status === 'deployed')

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Batch Predictions</h1>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          {showCreateForm ? 'Cancel' : 'Create Batch Job'}
        </button>
      </div>

      {showCreateForm && (
        <div className="mb-6 p-6 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Create Batch Prediction Job</h2>
          <form onSubmit={handleCreateJob} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                required
                value={createForm.model_id}
                onChange={(e) => setCreateForm({ ...createForm, model_id: e.target.value })}
                className="w-full px-3 py-2 border rounded-lg"
              >
                <option value="">Select a model</option>
                {readyModels.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.type})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Input Type</label>
              <select
                value={createForm.input_type}
                onChange={(e) => {
                  setCreateForm({ ...createForm, input_type: e.target.value })
                  setFile(null) // Reset file when input type changes
                }}
                className="w-full px-3 py-2 border rounded-lg"
              >
                <option value="data_source">Data Source</option>
                <option value="file">File</option>
              </select>
            </div>

            {createForm.input_type === 'data_source' && (
              <div>
                <label className="block text-sm font-medium mb-1">Data Source</label>
                <select
                  required
                  value={createForm.data_source_id}
                  onChange={(e) => setCreateForm({ ...createForm, data_source_id: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  <option value="">Select a data source</option>
                  {sources.map(source => (
                    <option key={source.id} value={source.id}>
                      {source.name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {createForm.input_type === 'file' && (
              <div>
                <label className="block text-sm font-medium mb-1">Input File</label>
                <input
                  type="file"
                  accept=".csv,.json"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setFile(e.target.files[0])
                    }
                  }}
                  className="w-full px-3 py-2 border rounded-lg"
                  required
                />
                {file && (
                  <p className="mt-2 text-sm text-gray-600">
                    Selected: {file.name} ({(file.size / 1024).toFixed(2)} KB)
                  </p>
                )}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium mb-1">Job Name (Optional)</label>
              <input
                type="text"
                value={createForm.job_name}
                onChange={(e) => setCreateForm({ ...createForm, job_name: e.target.value })}
                className="w-full px-3 py-2 border rounded-lg"
                placeholder="My Batch Prediction Job"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Result Format</label>
              <select
                value={createForm.result_format}
                onChange={(e) => setCreateForm({ ...createForm, result_format: e.target.value })}
                className="w-full px-3 py-2 border rounded-lg"
              >
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
                <option value="parquet">Parquet</option>
              </select>
            </div>

            <div className="flex gap-2">
              <button
                type="submit"
                disabled={creating || uploadingFile}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {uploadingFile ? 'Uploading...' : creating ? 'Creating...' : 'Create Job'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowCreateForm(false)
                  setFile(null)
                }}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="mb-4 flex gap-2">
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-3 py-2 border rounded-lg"
        >
          <option value="">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="queued">Queued</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="cancelled">Cancelled</option>
        </select>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-100 text-red-800 rounded-lg">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-center py-8">Loading batch jobs...</div>
      ) : jobs.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No batch jobs found. Create one to get started.
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Job</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Progress</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Records</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {jobs.map((job) => {
                const model = models.find(m => m.id === job.model_id)
                const isRunning = job.status === 'running' || job.status === 'pending' || job.status === 'queued'
                const isSelected = selectedJobId === job.id

                return (
                  <tr
                    key={job.id}
                    className={isSelected ? 'bg-blue-50' : 'hover:bg-gray-50'}
                    onClick={() => setSelectedJobId(job.id)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="font-medium">{job.job_name || `Job #${job.id}`}</div>
                      {isConnected && isSelected && isRunning && (
                        <div className="text-xs text-green-600">● Live</div>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {model?.name || `Model #${job.model_id}`}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-600">{job.progress.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {job.processed_records !== null && job.total_records !== null
                        ? `${job.processed_records.toLocaleString()} / ${job.total_records.toLocaleString()}`
                        : '-'}
                      {job.failed_records && job.failed_records > 0 && (
                        <div className="text-xs text-red-600">{job.failed_records} failed</div>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {new Date(job.created_at).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex gap-2">
                        {isRunning && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleCancelJob(job.id)
                            }}
                            disabled={cancelling}
                            className="text-red-600 hover:text-red-800 disabled:opacity-50"
                          >
                            Cancel
                          </button>
                        )}
                        {job.status === 'completed' && job.result_path && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDownloadResults(job.id)
                            }}
                            disabled={downloading}
                            className="text-blue-600 hover:text-blue-800 disabled:opacity-50"
                          >
                            Download
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {selectedJob && (
        <div className="mt-6 p-6 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Job Details</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-gray-600">Job ID</div>
              <div className="font-medium">{selectedJob.id}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Status</div>
              <div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedJob.status)}`}>
                  {selectedJob.status}
                </span>
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Progress</div>
              <div className="font-medium">{selectedJob.progress.toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Records</div>
              <div className="font-medium">
                {selectedJob.processed_records !== null && selectedJob.total_records !== null
                  ? `${selectedJob.processed_records.toLocaleString()} / ${selectedJob.total_records.toLocaleString()}`
                  : '-'}
              </div>
            </div>
            {selectedJob.started_at && (
              <div>
                <div className="text-sm text-gray-600">Started</div>
                <div className="font-medium">{new Date(selectedJob.started_at).toLocaleString()}</div>
              </div>
            )}
            {selectedJob.completed_at && (
              <div>
                <div className="text-sm text-gray-600">Completed</div>
                <div className="font-medium">{new Date(selectedJob.completed_at).toLocaleString()}</div>
              </div>
            )}
            {selectedJob.error_message && (
              <div className="col-span-2">
                <div className="text-sm text-gray-600">Error</div>
                <div className="text-red-600">{selectedJob.error_message}</div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="mt-4 text-sm text-gray-600">
        Total: {total} batch jobs
      </div>
    </div>
  )
}

