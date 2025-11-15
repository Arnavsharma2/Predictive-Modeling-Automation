'use client'

import { useState } from 'react'
import { ingestionEndpoints } from '@/lib/api/endpoints'

interface DataUploadFormProps {
  onSuccess?: () => void
}

export default function DataUploadForm({ onSuccess }: DataUploadFormProps) {
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      if (!name) {
        setName(e.target.files[0].name.replace('.csv', ''))
      }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) {
      setError('Please select a file')
      return
    }

    setUploading(true)
    setError(null)
    setSuccess(false)
    setProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('name', name)
      if (description) {
        formData.append('description', description)
      }
      formData.append('auto_process', 'true')

      const response = await ingestionEndpoints.uploadCSV(formData)
      
      setProgress(100)
      setSuccess(true)
      
      if (onSuccess) {
        onSuccess()
      }
      
      // Reset form
      setTimeout(() => {
        setFile(null)
        setName('')
        setDescription('')
        setSuccess(false)
        setProgress(0)
      }, 2000)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload file')
    } finally {
      setUploading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label className="block text-sm font-semibold text-gray-soft-700 mb-2 uppercase tracking-wider">
          CSV File
        </label>
        <div className="relative">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-soft-600 file:mr-4 file:py-3 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-pastel-blue file:text-gray-soft-700 hover:file:bg-pastel-powder file:transition-colors file:cursor-pointer cursor-pointer"
            required
          />
        </div>
        {file && (
          <p className="mt-3 text-sm text-gray-soft-600 flex items-center gap-2">
            <svg className="w-4 h-4 text-pastel-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Selected: <span className="text-gray-soft-700 font-medium">{file.name}</span> ({(file.size / 1024).toFixed(2)} KB)
          </p>
        )}
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-soft-700 mb-2 uppercase tracking-wider">
          Name
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full px-4 py-3 bg-pastel-blue/20 border-2 border-gray-soft-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-pastel-blue focus:border-pastel-blue text-gray-900 placeholder-gray-soft-400 transition-all"
          placeholder="Enter data source name"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-soft-700 mb-2 uppercase tracking-wider">
          Description <span className="text-gray-soft-500 normal-case">(Optional)</span>
        </label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          className="w-full px-4 py-3 bg-white border-2 border-gray-soft-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-pastel-blue focus:border-pastel-blue text-gray-soft-700 placeholder-gray-soft-400 transition-all resize-none"
          placeholder="Add a description for this data source"
        />
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          File uploaded successfully!
        </div>
      )}

      {uploading && (
        <div className="w-full bg-gray-soft-200 rounded-full h-3 overflow-hidden">
          <div
            className="bg-pastel-blue h-3 rounded-full transition-all duration-300 flex items-center justify-end pr-2"
            style={{ width: `${progress}%` }}
          >
            {progress > 10 && (
              <span className="text-xs text-gray-soft-700 font-semibold">{progress}%</span>
            )}
          </div>
        </div>
      )}

      <button
        type="submit"
        disabled={uploading || !file}
        className="w-full bg-pastel-blue text-gray-soft-700 px-6 py-3 rounded-lg font-semibold transition-all duration-300 hover:bg-pastel-powder disabled:bg-gray-soft-300 disabled:text-gray-soft-500 disabled:cursor-not-allowed disabled:hover:bg-gray-soft-300"
      >
        {uploading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Uploading...
          </span>
        ) : (
          'Upload CSV'
        )}
      </button>
    </form>
  )
}

