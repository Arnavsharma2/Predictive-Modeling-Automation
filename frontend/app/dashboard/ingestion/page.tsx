'use client'

import { useState, useRef, useEffect } from 'react'
import DataUploadForm from '@/components/forms/DataUploadForm'
import { useDataSources } from '@/hooks/useData'
import { ingestionEndpoints } from '@/lib/api/endpoints'

interface SelectionBox {
  startX: number
  startY: number
  endX: number
  endY: number
}

export default function IngestionPage() {
  const [activeTab, setActiveTab] = useState<'upload' | 'sources'>('upload')
  const { sources, loading: sourcesLoading, refetch } = useDataSources()

  // Selection state for drag-to-select
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [isSelecting, setIsSelecting] = useState(false)
  const [selectionBox, setSelectionBox] = useState<SelectionBox | null>(null)
  const [isShiftHeld, setIsShiftHeld] = useState(false)
  const tableContainerRef = useRef<HTMLDivElement>(null)
  const rowRefs = useRef<Map<number, HTMLTableRowElement>>(new Map())

  // Reset selection when sources change
  useEffect(() => {
    setSelectedIds(new Set())
    setSelectionBox(null)
  }, [sources.length])

  const handleDeleteSource = async (sourceId: number, sourceName: string) => {
    if (!confirm(`Are you sure you want to delete the data source "${sourceName}"? This action cannot be undone and will delete all associated data points and ETL jobs.`)) {
      return
    }
    
    try {
      await ingestionEndpoints.deleteSource(sourceId)
      refetch()
    } catch (err) {
      console.error('Delete failed:', err)
      alert('Failed to delete data source. Please try again.')
    }
  }

  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) return
    
    if (!confirm(`Are you sure you want to delete ${selectedIds.size} data source(s)? This action cannot be undone and will delete all associated data points, ETL jobs, and source files.`)) {
      return
    }
    
    try {
      const idsArray = Array.from(selectedIds)
      // Delete one at a time
      for (const id of idsArray) {
        await ingestionEndpoints.deleteSource(id)
      }
      setSelectedIds(new Set())
      refetch()
    } catch (err) {
      console.error('Delete failed:', err)
      alert('Failed to delete some data sources. Please try again.')
      refetch() // Refresh to show current state
    }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0 || (e.target as HTMLElement).tagName === 'BUTTON') return

    const container = tableContainerRef.current
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
    if (!isSelecting || !selectionBox || !tableContainerRef.current) return

    // Prevent text selection during drag
    e.preventDefault()

    const container = tableContainerRef.current
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

    rowRefs.current.forEach((element: HTMLTableRowElement, sourceId: number) => {
      if (!element) return

      const elementRect = element.getBoundingClientRect()
      const containerRect = tableContainerRef.current?.getBoundingClientRect()
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
        newSelected.add(sourceId)
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
      <h1 className="text-4xl font-bold text-gray-soft-700 mb-2">Data Ingestion</h1>
      <p className="text-gray-soft-600 mb-8 font-medium">Upload and manage your data sources</p>
      
      <div className="mb-8">
        <div className="border-b border-pastel-blue/30">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-4 px-1 border-b-2 font-semibold text-sm transition-all duration-200 flex items-center gap-2 ${
                activeTab === 'upload'
                  ? 'border-pastel-blue text-pastel-blue'
                  : 'border-transparent text-gray-soft-500 hover:text-gray-soft-700 hover:border-pastel-blue/30'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Upload CSV
            </button>
            <button
              onClick={() => setActiveTab('sources')}
              className={`py-4 px-1 border-b-2 font-semibold text-sm transition-all duration-200 flex items-center gap-2 ${
                activeTab === 'sources'
                  ? 'border-pastel-blue text-pastel-blue'
                  : 'border-transparent text-gray-soft-500 hover:text-gray-soft-700 hover:border-pastel-blue/30'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
              Data Sources
              {sources.length > 0 && (
                <span className={`ml-1 px-2 py-0.5 text-xs rounded-full ${
                  activeTab === 'sources' 
                    ? 'bg-pastel-blue/30 text-primary-600' 
                    : 'bg-gray-soft-100 text-gray-soft-600'
                }`}>
                  {sources.length}
                </span>
              )}
            </button>
          </nav>
        </div>
      </div>

      {activeTab === 'upload' && (
        <div className="glass rounded-2xl border-pastel-blue/40 p-8">
          <h2 className="text-2xl font-bold text-gray-soft-700 mb-6 flex items-center gap-2">
            <span className="w-1 h-6 bg-pastel-blue rounded-full"></span>
            Upload CSV File
          </h2>
          <DataUploadForm onSuccess={refetch} />
        </div>
      )}

      {activeTab === 'sources' && (
        <div className="space-y-6">
          <div className="glass rounded-2xl border-pastel-blue/40 p-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-bold text-gray-soft-700 flex items-center gap-2">
                  <span className="w-1 h-6 bg-pastel-blue rounded-full"></span>
                  Data Sources
                </h2>
                <p className="text-xs text-gray-soft-500 mt-1 flex items-center gap-1">
                  <svg className="w-3 h-3 text-pastel-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Click and drag to select multiple data sources. Press Delete to remove selected items.
                </p>
              </div>
              {selectedIds.size > 0 && (
                <div className="flex items-center gap-2 px-3 py-1.5 bg-pastel-blue/20 border border-pastel-blue/40 rounded-lg">
                  <svg className="w-4 h-4 text-pastel-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-sm font-semibold text-primary-600">{selectedIds.size} selected</span>
                </div>
              )}
            </div>
            {sourcesLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-8 h-8 border-4 border-pastel-blue/30 border-t-pastel-blue rounded-full animate-spin"></div>
                  <p className="text-sm text-gray-soft-500">Loading data sources...</p>
                </div>
              </div>
            ) : sources.length === 0 ? (
              <div className="text-center py-12">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-16 h-16 bg-pastel-blue/20 rounded-full flex items-center justify-center">
                    <svg className="w-8 h-8 text-pastel-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-soft-700">No data sources yet</h3>
                  <p className="text-sm text-gray-soft-500 max-w-sm">Upload a CSV file to get started with data ingestion and processing.</p>
                </div>
              </div>
            ) : (
              <div 
                ref={tableContainerRef}
                className={`overflow-x-auto relative ${isSelecting ? 'select-none' : ''}`}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
              >
                {/* Selection box overlay */}
                {selectionBox && (
                  <div
                    className="absolute border-2 border-pastel-blue bg-pastel-blue/20 pointer-events-none z-10 rounded shadow-lg"
                    style={{
                      left: `${Math.min(selectionBox.startX, selectionBox.endX)}px`,
                      top: `${Math.min(selectionBox.startY, selectionBox.endY)}px`,
                      width: `${Math.abs(selectionBox.endX - selectionBox.startX)}px`,
                      height: `${Math.abs(selectionBox.endY - selectionBox.startY)}px`,
                    }}
                  />
                )}
                <table className="min-w-full divide-y divide-gray-soft-200">
                  <thead className="bg-pastel-white">
                    <tr>
                      <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-soft-700 uppercase tracking-wider">
                        Name
                      </th>
                      <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-soft-700 uppercase tracking-wider">
                        Type
                      </th>
                      <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-soft-700 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-soft-700 uppercase tracking-wider">
                        Created
                      </th>
                      <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-soft-700 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-soft-200 bg-pastel-white">
                    {sources.map((source: any) => {
                      const isSelected = selectedIds.has(source.id)
                      return (
                        <tr
                          key={source.id}
                          ref={(el) => {
                            if (el) {
                              rowRefs.current.set(source.id, el)
                            } else {
                              rowRefs.current.delete(source.id)
                            }
                          }}
                          className={`transition-all duration-150 ${
                            isSelected
                              ? 'bg-pastel-blue/20 border-l-4 border-pastel-blue shadow-sm'
                              : 'hover:bg-pastel-blue/10 border-l-4 border-transparent'
                          }`}
                        >
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-soft-700">
                            {source.name}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-soft-600">
                            {source.type}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-3 py-1.5 inline-flex items-center gap-1.5 text-xs font-semibold rounded-full ${
                              source.status === 'active' ? 'bg-accent-50 text-accent-500 border border-accent-100' :
                              source.status === 'processing' ? 'bg-pastel-mint/50 text-primary-600 border border-pastel-mint' :
                              'bg-gray-soft-100 text-gray-soft-600 border border-gray-soft-200'
                            }`}>
                              {source.status === 'active' && (
                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                              )}
                              {source.status === 'processing' && (
                                <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                              )}
                              {source.status}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-soft-600">
                            {new Date(source.created_at).toLocaleDateString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm">
                            <button
                              onClick={() => handleDeleteSource(source.id, source.name)}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-all duration-150 border border-red-200 hover:border-red-300 hover:shadow-sm"
                              title="Delete data source"
                            >
                              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                              Delete
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

