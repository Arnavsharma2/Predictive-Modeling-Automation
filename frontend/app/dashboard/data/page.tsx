'use client'

import { useState, useRef, useEffect } from 'react'
import { useDataSources } from '@/hooks/useData'
import { useDataPreview, useDataStats } from '@/hooks/useData'
import { dataEndpoints } from '@/lib/api/endpoints'

interface SelectionBox {
  startX: number
  startY: number
  endX: number
  endY: number
}

export default function DataPage() {
  const { sources, refetch: refetchSources } = useDataSources()
  const [selectedSourceId, setSelectedSourceId] = useState<number | null>(null)
  const [limit, setLimit] = useState(100)
  const [offset, setOffset] = useState(0)
  
  const { data, loading: dataLoading, refetch: refetchData } = useDataPreview(selectedSourceId, limit, offset)
  const { stats, loading: statsLoading, error: statsError } = useDataStats(selectedSourceId)

  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [isSelecting, setIsSelecting] = useState(false)
  const [selectionBox, setSelectionBox] = useState<SelectionBox | null>(null)
  const [isShiftHeld, setIsShiftHeld] = useState(false)
  const tableContainerRef = useRef<HTMLDivElement>(null)
  const rowRefs = useRef<Map<number, HTMLTableRowElement>>(new Map())

  const selectedSource = sources.find((s: any) => s.id === selectedSourceId)

  // Reset selection when data changes
  useEffect(() => {
    setSelectedIds(new Set())
    setSelectionBox(null)
  }, [selectedSourceId, offset])

  const handleExport = (format: 'csv' | 'json') => {
    if (!data || data.length === 0) return

    if (format === 'csv') {
      const headers = Object.keys(data[0].data || {})
      const csv = [
        headers.join(','),
        ...data.map((row: any) => headers.map(h => JSON.stringify(row.data?.[h] || '')).join(','))
      ].join('\n')
      
      const blob = new Blob([csv], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `data_${selectedSourceId}_${Date.now()}.csv`
      a.click()
    } else {
      const json = JSON.stringify(data, null, 2)
      const blob = new Blob([json], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `data_${selectedSourceId}_${Date.now()}.json`
      a.click()
    }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    // Only handle left mouse button and ignore clicks on interactive elements
    if (e.button !== 0 || (e.target as HTMLElement).tagName === 'BUTTON' || (e.target as HTMLElement).tagName === 'INPUT') {
      return
    }

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

    // If not holding shift/ctrl/cmd, clear previous selection
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

    // Calculate which rows intersect with the selection box
    updateSelectionFromBox(updatedBox.startX, updatedBox.startY, endX, endY)
  }

  const updateSelectionFromBox = (startX: number, startY: number, endX: number, endY: number) => {
    const boxLeft = Math.min(startX, endX)
    const boxRight = Math.max(startX, endX)
    const boxTop = Math.min(startY, endY)
    const boxBottom = Math.max(startY, endY)

    // Start with existing selection if shift is held, otherwise start fresh
    const newSelected = isShiftHeld ? new Set(selectedIds) : new Set<number>()

    // Check each row for intersection
    rowRefs.current.forEach((rowElement: HTMLTableRowElement, pointId: number) => {
      if (!rowElement) return

      const rowRect = rowElement.getBoundingClientRect()
      const containerRect = tableContainerRef.current?.getBoundingClientRect()
      if (!containerRect) return

      const rowTop = rowRect.top - containerRect.top
      const rowBottom = rowRect.bottom - containerRect.top
      const rowLeft = rowRect.left - containerRect.left
      const rowRight = rowRect.right - containerRect.left

      // Check if row intersects with selection box
      const intersects = !(
        rowBottom < boxTop ||
        rowTop > boxBottom ||
        rowRight < boxLeft ||
        rowLeft > boxRight
      )

      if (intersects) {
        newSelected.add(pointId)
      }
    })

    setSelectedIds(newSelected)
  }

  const handleMouseUp = () => {
    setIsSelecting(false)
    setIsShiftHeld(false)
    setSelectionBox(null)
  }

  // Global mouse up listener
  useEffect(() => {
    if (isSelecting) {
      window.addEventListener('mouseup', handleMouseUp)
      return () => window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isSelecting])

  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0) return
    
    if (!confirm(`Are you sure you want to delete ${selectedIds.size} data point(s)? This action cannot be undone.`)) {
      return
    }
    
    try {
      const idsArray = Array.from(selectedIds)
      // Delete one at a time
      for (const id of idsArray) {
        await dataEndpoints.deletePoint(id)
      }
      setSelectedIds(new Set())
      refetchData()
      refetchSources()
    } catch (err) {
      console.error('Delete failed:', err)
      alert('Failed to delete some data points. Please try again.')
      refetchData() // Refresh to show current state
      refetchSources()
    }
  }

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
        handleDeleteSelected()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedIds.size, handleDeleteSelected])

  // Get column keys from first data point
  const columnKeys = data.length > 0 && data[0].data ? Object.keys(data[0].data) : []

  // Calculate selection box dimensions
  const boxStyle = selectionBox ? {
    left: `${Math.min(selectionBox.startX, selectionBox.endX)}px`,
    top: `${Math.min(selectionBox.startY, selectionBox.endY)}px`,
    width: `${Math.abs(selectionBox.endX - selectionBox.startX)}px`,
    height: `${Math.abs(selectionBox.endY - selectionBox.startY)}px`,
  } : {}

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold text-gray-soft-700 mb-2">Data Exploration</h1>
      <p className="text-gray-soft-600 mb-8 font-medium">Explore and analyze your data sources</p>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
        <div className="lg:col-span-1">
          <div className="glass rounded-2xl border-pastel-blue/40 p-6">
            <h2 className="text-lg font-bold text-gray-soft-700 mb-4 flex items-center gap-2">
              <span className="w-1 h-5 bg-pastel-blue rounded-full"></span>
              Select Data Source
            </h2>
            <select
              value={selectedSourceId || ''}
              onChange={(e) => {
                setSelectedSourceId(parseInt(e.target.value) || null)
                setOffset(0)
              }}
              className="w-full px-3 py-2 border border-gray-soft-300 rounded-lg shadow-sm focus:outline-none focus:ring-pastel-blue focus:border-pastel-blue bg-white text-gray-soft-700"
            >
              <option value="">Choose a source...</option>
              {sources.map((source: any) => (
                <option key={source.id} value={source.id}>
                  {source.name}
                </option>
              ))}
            </select>
          </div>

          {selectedSourceId && (
            <div className="glass rounded-2xl border-pastel-blue/40 p-6 mt-6">
              <h2 className="text-lg font-bold text-gray-soft-700 mb-4 flex items-center gap-2">
                <span className="w-1 h-5 bg-pastel-blue rounded-full"></span>
                Statistics
              </h2>
              {statsLoading ? (
                <p className="text-sm text-gray-soft-500">Loading statistics...</p>
              ) : statsError ? (
                <p className="text-sm text-red-600">Error loading statistics</p>
              ) : stats ? (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-soft-600">Total Records:</span>
                    <span className="font-medium text-gray-soft-700">{stats.total_records || 0}</span>
                  </div>
                  {stats.columns && (
                    <div className="flex justify-between">
                      <span className="text-gray-soft-600">Columns:</span>
                      <span className="font-medium text-gray-soft-700">{stats.columns.length}</span>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-gray-soft-500">No statistics available</p>
              )}
            </div>
          )}
        </div>

        <div className="lg:col-span-3">
          {selectedSourceId ? (
            <div className="glass rounded-2xl border-pastel-blue/40 p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-soft-700 flex items-center gap-2">
                  <span className="w-1 h-6 bg-pastel-blue rounded-full"></span>
                  Data Preview {selectedSource && `- ${selectedSource.name}`}
                </h2>
                <div className="flex gap-2">
                  {selectedIds.size > 0 && (
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-pastel-blue/20 border border-pastel-blue/40 rounded-lg">
                      <svg className="w-4 h-4 text-pastel-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span className="text-sm font-semibold text-primary-600">{selectedIds.size} selected</span>
                    </div>
                  )}
                  <button
                    onClick={() => handleExport('csv')}
                    disabled={!data || data.length === 0}
                    className="px-4 py-2 text-sm bg-pastel-blue text-gray-soft-700 rounded-lg hover:bg-pastel-powder disabled:bg-gray-soft-100 disabled:text-gray-soft-400 font-medium transition-colors"
                  >
                    Export CSV
                  </button>
                  <button
                    onClick={() => handleExport('json')}
                    disabled={!data || data.length === 0}
                    className="px-4 py-2 text-sm bg-pastel-blue text-gray-soft-700 rounded-lg hover:bg-pastel-powder disabled:bg-gray-soft-100 disabled:text-gray-soft-400 font-medium transition-colors"
                  >
                    Export JSON
                  </button>
                </div>
              </div>

              {dataLoading ? (
                <p className="text-gray-soft-500">Loading data...</p>
              ) : data.length === 0 ? (
                <p className="text-gray-soft-500">No data available for this source.</p>
              ) : (
                <>
                  <div className="mb-2 text-sm text-gray-soft-600">
                    <p>Hold click and drag to select multiple rows (macOS-style selection). Press Delete to remove selected items.</p>
                  </div>
                  <div 
                    ref={tableContainerRef}
                    className={`overflow-x-auto mb-4 relative ${isSelecting ? 'select-none' : ''}`}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                  >
                    {/* Selection box overlay */}
                    {selectionBox && (
                      <div
                        className="absolute border-2 border-pastel-blue bg-pastel-blue/20 pointer-events-none z-10 rounded"
                        style={boxStyle}
                      />
                    )}
                    
                    <table className="min-w-full divide-y divide-gray-soft-200">
                      <thead className="bg-pastel-white">
                        <tr>
                          {columnKeys.map((key) => (
                            <th
                              key={key}
                              className="px-4 py-3 text-left text-xs font-semibold text-gray-soft-700 uppercase tracking-wider"
                            >
                              {key}
                            </th>
                          ))}
                          <th className="px-4 py-3 text-left text-xs font-semibold text-gray-soft-700 uppercase tracking-wider">
                            Timestamp
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-pastel-white divide-y divide-gray-soft-200">
                        {data.map((row: any, idx: number) => {
                          const isSelected = selectedIds.has(row.id)
                          return (
                            <tr
                              key={row.id || idx}
                              ref={(el) => {
                                if (el && row.id) {
                                  rowRefs.current.set(row.id, el)
                                } else if (row.id) {
                                  rowRefs.current.delete(row.id)
                                }
                              }}
                              className={`
                                ${isSelected ? 'bg-pastel-blue/20 border-l-4 border-pastel-blue' : 'hover:bg-pastel-blue/10'}
                                transition-colors
                              `}
                            >
                              {columnKeys.map((key) => (
                                <td
                                  key={key}
                                  className="px-4 py-3 whitespace-nowrap text-sm text-gray-soft-700"
                                >
                                  {typeof row.data?.[key] === 'object' 
                                    ? JSON.stringify(row.data[key]) 
                                    : String(row.data?.[key] ?? '')}
                                </td>
                              ))}
                              <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-soft-500">
                                {row.timestamp ? new Date(row.timestamp).toLocaleString() : '-'}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div className="flex justify-between items-center">
                    <div className="flex gap-2">
                      <button
                        onClick={() => setOffset(Math.max(0, offset - limit))}
                        disabled={offset === 0}
                        className="px-4 py-2 text-sm bg-pastel-blue text-gray-soft-700 rounded-lg hover:bg-pastel-powder disabled:bg-gray-soft-100 disabled:text-gray-soft-400 font-medium transition-colors"
                      >
                        Previous
                      </button>
                      <button
                        onClick={() => setOffset(offset + limit)}
                        disabled={data.length < limit}
                        className="px-4 py-2 text-sm bg-pastel-blue text-gray-soft-700 rounded-lg hover:bg-pastel-powder disabled:bg-gray-soft-100 disabled:text-gray-soft-400 font-medium transition-colors"
                      >
                        Next
                      </button>
                    </div>
                    <p className="text-sm text-gray-soft-600">
                      Showing {offset + 1} - {offset + data.length} of {stats?.total_records || 0} records
                      {selectedIds.size > 0 && ` • ${selectedIds.size} selected`}
                    </p>
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="glass rounded-2xl border-pastel-blue/40 p-6">
              <p className="text-gray-soft-500 text-center py-8">Please select a data source to explore.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
