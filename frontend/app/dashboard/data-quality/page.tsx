'use client'

import { useState } from 'react'
import { useDataSources } from '@/hooks/useData'
import {
  useDataQualityReports,
  useDataProfile,
  useDataLineage,
  useCheckDataQuality,
} from '@/hooks/useDataQuality'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function DataQualityPage() {
  const [selectedDataSourceId, setSelectedDataSourceId] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState<'quality' | 'profile' | 'lineage'>('quality')

  const { sources } = useDataSources()
  const { reports, loading: reportsLoading, refetch: refetchReports } = useDataQualityReports(selectedDataSourceId)
  const { profile, loading: profileLoading, error: profileError, refetch: refetchProfile } = useDataProfile(selectedDataSourceId)
  const { lineage, loading: lineageLoading, error: lineageError, refetch: refetchLineage } = useDataLineage(selectedDataSourceId)
  const { checkQuality, loading: checkLoading, error: checkError } = useCheckDataQuality()

  const handleCheckQuality = async () => {
    if (!selectedDataSourceId) return

    try {
      await checkQuality(selectedDataSourceId)
      await refetchReports()
    } catch (err) {
      console.error('Error checking quality:', err)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'excellent':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'good':
        return 'bg-blue-100 text-blue-800 border-blue-300'
      case 'fair':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      case 'poor':
        return 'bg-orange-100 text-orange-800 border-orange-300'
      case 'critical':
        return 'bg-red-100 text-red-800 border-red-300'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  const qualityChartData = reports.length > 0 && reports[0]?.quality_metrics?.summary
    ? [
        {
          metric: 'Completeness',
          score: reports[0].quality_metrics.summary.completeness_score,
        },
        {
          metric: 'Uniqueness',
          score: reports[0].quality_metrics.summary.uniqueness_score,
        },
        {
          metric: 'Validity',
          score: reports[0].quality_metrics.summary.validity_score,
        },
        {
          metric: 'Consistency',
          score: reports[0].quality_metrics.summary.consistency_score,
        },
      ]
    : []

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-soft-700">Data Quality Monitoring</h1>
          <p className="text-gray-soft-500 mt-2">Monitor data quality, profiling, and lineage</p>
        </div>
      </div>

      {/* Data Source Selection */}
      <div className="glass-strong p-6 rounded-xl border border-pastel-blue/30">
        <h2 className="text-xl font-semibold text-gray-soft-700 mb-4">Select Data Source</h2>
        <div className="flex gap-4">
          <select
            value={selectedDataSourceId || ''}
            onChange={(e) => setSelectedDataSourceId(e.target.value ? parseInt(e.target.value) : null)}
            className="flex-1 px-4 py-2 border border-pastel-blue/30 rounded-lg focus:ring-2 focus:ring-pastel-blue focus:border-transparent"
          >
            <option value="">Select a data source</option>
            {sources.map((source) => (
              <option key={source.id} value={source.id}>
                {source.name} ({source.type})
              </option>
            ))}
          </select>
          <button
            onClick={handleCheckQuality}
            disabled={!selectedDataSourceId || checkLoading}
            className="px-6 py-2 bg-pastel-blue text-white rounded-lg hover:bg-pastel-blue/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {checkLoading ? 'Checking...' : 'Check Quality'}
          </button>
        </div>
        {checkError && (
          <div className="mt-2 text-sm text-red-600">{checkError}</div>
        )}
      </div>

      {selectedDataSourceId && (
        <>
          {/* Tabs */}
          <div className="flex gap-2 border-b border-pastel-blue/30">
            <button
              onClick={() => setActiveTab('quality')}
              className={`px-4 py-2 font-semibold transition-colors ${
                activeTab === 'quality'
                  ? 'text-pastel-blue border-b-2 border-pastel-blue'
                  : 'text-gray-soft-600 hover:text-pastel-blue'
              }`}
            >
              Quality Reports
            </button>
            <button
              onClick={() => setActiveTab('profile')}
              className={`px-4 py-2 font-semibold transition-colors ${
                activeTab === 'profile'
                  ? 'text-pastel-blue border-b-2 border-pastel-blue'
                  : 'text-gray-soft-600 hover:text-pastel-blue'
              }`}
            >
              Data Profile
            </button>
            <button
              onClick={() => setActiveTab('lineage')}
              className={`px-4 py-2 font-semibold transition-colors ${
                activeTab === 'lineage'
                  ? 'text-pastel-blue border-b-2 border-pastel-blue'
                  : 'text-gray-soft-600 hover:text-pastel-blue'
              }`}
            >
              Data Lineage
            </button>
          </div>

          {/* Quality Reports Tab */}
          {activeTab === 'quality' && (
            <div className="space-y-6">
              {reportsLoading ? (
                <div className="text-center py-8 text-gray-500">Loading...</div>
              ) : reports.length > 0 ? (
                <>
                  {/* Latest Report Summary */}
                  <div className="glass-strong p-6 rounded-xl border border-pastel-blue/30">
                    <h2 className="text-xl font-semibold text-gray-soft-700 mb-4">Latest Quality Report</h2>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className={`px-4 py-2 rounded-full text-sm font-semibold border ${getStatusColor(reports[0].status)}`}>
                          {reports[0].status?.toUpperCase() || 'UNKNOWN'}
                        </span>
                        <div className="text-sm text-gray-500">
                          {new Date(reports[0].created_at).toLocaleString()}
                        </div>
                      </div>
                      {reports[0].overall_score !== null && (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-gray-soft-700">Overall Score</span>
                            <span className="text-lg font-bold text-pastel-blue">
                              {(reports[0].overall_score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-3">
                            <div
                              className="bg-pastel-blue h-3 rounded-full transition-all"
                              style={{ width: `${reports[0].overall_score * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                      {qualityChartData.length > 0 && (
                        <div>
                          <h3 className="text-lg font-semibold text-gray-soft-700 mb-4">Quality Metrics</h3>
                          <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={qualityChartData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#9DD5B8" strokeOpacity={0.2} />
                              <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                              <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="score" fill="#8FC4D4" radius={[4, 4, 0, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                      {reports[0].freshness_metrics && (
                        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                          <div className="text-sm">
                            <strong>Data Freshness:</strong>{' '}
                            {reports[0].freshness_metrics.is_fresh ? (
                              <span className="text-green-600">Fresh</span>
                            ) : (
                              <span className="text-red-600">Stale</span>
                            )}
                            {reports[0].freshness_metrics.hours_since_update !== null && (
                              <span className="ml-2 text-gray-600">
                                ({reports[0].freshness_metrics.hours_since_update.toFixed(1)} hours ago)
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Quality History */}
                  <div className="glass-strong p-6 rounded-xl border border-pastel-blue/30">
                    <h2 className="text-xl font-semibold text-gray-soft-700 mb-4">Quality History</h2>
                    <div className="space-y-3">
                      {reports.map((report) => (
                        <div
                          key={report.id}
                          className="p-4 rounded-lg border border-pastel-blue/30 bg-white/50 backdrop-blur-sm hover:shadow-md transition-shadow"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                              <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getStatusColor(report.status)}`}>
                                {report.status?.toUpperCase() || 'UNKNOWN'}
                              </span>
                              {report.overall_score !== null && (
                                <span className="text-sm text-gray-600">
                                  Score: {(report.overall_score * 100).toFixed(1)}%
                                </span>
                              )}
                              {report.sample_size && (
                                <span className="text-sm text-gray-500">
                                  {report.sample_size.toLocaleString()} samples
                                </span>
                              )}
                            </div>
                            <div className="text-sm text-gray-500">
                              {new Date(report.created_at).toLocaleString()}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No quality reports available. Click "Check Quality" to generate a report.
                </div>
              )}
            </div>
          )}

          {/* Data Profile Tab */}
          {activeTab === 'profile' && (
            <div className="glass-strong p-6 rounded-xl border border-pastel-blue/30">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-soft-700">Data Profile</h2>
                <button
                  onClick={() => refetchProfile()}
                  disabled={!selectedDataSourceId || profileLoading}
                  className="px-4 py-2 bg-pastel-blue text-white rounded-lg hover:bg-pastel-blue/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
                >
                  {profileLoading ? 'Generating...' : 'Generate Profile'}
                </button>
              </div>
              {profileError && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
                  {profileError}
                </div>
              )}
              {profileLoading ? (
                <div className="text-center py-8 text-gray-500">Generating profile...</div>
              ) : profile ? (
                <div className="space-y-6">
                  {/* Overview */}
                  {profile.overview && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-soft-700 mb-3">Overview</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 bg-white/50 rounded-lg border border-pastel-blue/30">
                          <div className="text-sm text-gray-500">Total Rows</div>
                          <div className="text-2xl font-bold text-pastel-blue">
                            {profile.overview.total_rows?.toLocaleString() || 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 bg-white/50 rounded-lg border border-pastel-blue/30">
                          <div className="text-sm text-gray-500">Total Columns</div>
                          <div className="text-2xl font-bold text-pastel-blue">
                            {profile.overview.total_columns || 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 bg-white/50 rounded-lg border border-pastel-blue/30">
                          <div className="text-sm text-gray-500">Duplicate Rows</div>
                          <div className="text-2xl font-bold text-pastel-blue">
                            {profile.overview.duplicate_rows?.toLocaleString() || 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 bg-white/50 rounded-lg border border-pastel-blue/30">
                          <div className="text-sm text-gray-500">Memory Usage</div>
                          <div className="text-lg font-bold text-pastel-blue">
                            {(profile.overview.memory_usage_bytes / 1024 / 1024).toFixed(2)} MB
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Column Profiles */}
                  {profile.columns && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-soft-700 mb-3">Column Profiles</h3>
                      <div className="space-y-4">
                        {Object.entries(profile.columns).map(([columnName, columnData]: [string, any]) => (
                          <div
                            key={columnName}
                            className="p-4 bg-white/50 rounded-lg border border-pastel-blue/30"
                          >
                            <h4 className="font-semibold text-gray-soft-700 mb-2">{columnName}</h4>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                              <div>
                                <span className="text-gray-500">Type:</span>{' '}
                                <span className="font-medium">{columnData.dtype}</span>
                              </div>
                              <div>
                                <span className="text-gray-500">Nulls:</span>{' '}
                                <span className="font-medium">
                                  {columnData.null_count} ({columnData.null_percentage?.toFixed(1)}%)
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500">Unique:</span>{' '}
                                <span className="font-medium">
                                  {columnData.unique_count} ({columnData.unique_percentage?.toFixed(1)}%)
                                </span>
                              </div>
                              {columnData.numeric && (
                                <>
                                  <div>
                                    <span className="text-gray-500">Mean:</span>{' '}
                                    <span className="font-medium">{columnData.numeric.mean?.toFixed(2) || 'N/A'}</span>
                                  </div>
                                  <div>
                                    <span className="text-gray-500">Std:</span>{' '}
                                    <span className="font-medium">{columnData.numeric.std?.toFixed(2) || 'N/A'}</span>
                                  </div>
                                  <div>
                                    <span className="text-gray-500">Min:</span>{' '}
                                    <span className="font-medium">{columnData.numeric.min?.toFixed(2) || 'N/A'}</span>
                                  </div>
                                  <div>
                                    <span className="text-gray-500">Max:</span>{' '}
                                    <span className="font-medium">{columnData.numeric.max?.toFixed(2) || 'N/A'}</span>
                                  </div>
                                </>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p className="mb-4">No profile data available</p>
                  <button
                    onClick={() => refetchProfile()}
                    disabled={!selectedDataSourceId || profileLoading}
                    className="px-4 py-2 bg-pastel-blue text-white rounded-lg hover:bg-pastel-blue/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {profileLoading ? 'Generating...' : 'Generate Profile'}
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Data Lineage Tab */}
          {activeTab === 'lineage' && (
            <div className="glass-strong p-6 rounded-xl border border-pastel-blue/30">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-soft-700">Data Lineage</h2>
                <button
                  onClick={() => refetchLineage()}
                  disabled={!selectedDataSourceId || lineageLoading}
                  className="px-4 py-2 bg-pastel-blue text-white rounded-lg hover:bg-pastel-blue/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
                >
                  {lineageLoading ? 'Refreshing...' : 'Refresh'}
                </button>
              </div>
              {lineageError && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
                  {lineageError}
                </div>
              )}
              {lineageLoading ? (
                <div className="text-center py-8 text-gray-500">Loading lineage...</div>
              ) : lineage.length > 0 ? (
                <div className="space-y-3">
                  {lineage.map((item) => (
                    <div
                      key={item.id}
                      className="p-4 rounded-lg border border-pastel-blue/30 bg-white/50 backdrop-blur-sm"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-semibold text-gray-soft-700">
                            {item.target_type} #{item.target_id}
                          </div>
                          {item.description && (
                            <div className="text-sm text-gray-600 mt-1">{item.description}</div>
                          )}
                          {item.transformation && (
                            <div className="text-xs text-gray-500 mt-1">
                              Transformation: {JSON.stringify(item.transformation)}
                            </div>
                          )}
                        </div>
                        <div className="text-sm text-gray-500">
                          {new Date(item.created_at).toLocaleString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p className="mb-2">No lineage data available</p>
                  <p className="text-sm text-gray-400 mb-4">
                    Lineage records are created automatically when you train models using this data source.
                    Train a model to see lineage information here.
                  </p>
                  <button
                    onClick={() => refetchLineage()}
                    disabled={!selectedDataSourceId || lineageLoading}
                    className="px-4 py-2 bg-pastel-blue text-white rounded-lg hover:bg-pastel-blue/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {lineageLoading ? 'Refreshing...' : 'Refresh'}
                  </button>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

