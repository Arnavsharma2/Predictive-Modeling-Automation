'use client'

import { useState } from 'react'
import { useDataSources } from '@/hooks/useData'
import { useModels } from '@/hooks/useModels'
import { useTrainingJobs } from '@/hooks/useModels'
import { usePolling } from '@/hooks/usePolling'
import Link from 'next/link'

export default function Dashboard() {
  const [autoRefresh, setAutoRefresh] = useState(true)
  const { sources, loading: sourcesLoading, refetch: refetchSources } = useDataSources()
  const { models, loading: modelsLoading, refetch: refetchModels } = useModels()
  const { jobs, loading: jobsLoading, refetch: refetchJobs } = useTrainingJobs({ limit: 100 })

  // Auto-refresh every 10 seconds
  usePolling({
    enabled: autoRefresh,
    interval: 10000,
    onPoll: async () => {
      await Promise.all([refetchSources(), refetchModels(), refetchJobs()])
    }
  })
  
  const activeJobs = jobs.filter((job: any) => {
    if (!job || !job.status) return false
    const status = String(job.status).toLowerCase()
    return status === 'running' || status === 'pending'
  }).length

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-10">
        <h1 className="text-4xl font-bold text-gray-soft-700 mb-2">Dashboard</h1>
        <p className="text-gray-soft-600 font-medium">Monitor your analytics platform</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        <Link href="/dashboard/ingestion" className="group">
          <div className="glass rounded-2xl p-6 hover-glow handcrafted-outline cursor-pointer transition-all duration-300 border-pastel-blue/40">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-xs font-semibold text-gray-soft-500 uppercase tracking-wider">Data Sources</h3>
              <div className="w-11 h-11 bg-pastel-blue rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                <svg className="w-5 h-5 text-gray-soft-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
              </div>
            </div>
            <p className="text-4xl font-bold text-gray-soft-700 mb-2">
              {sourcesLoading ? (
                <span className="inline-block w-12 h-10 bg-gray-soft-200/50 rounded animate-pulse"></span>
              ) : (
                sources.length
              )}
            </p>
            <p className="text-xs text-gray-soft-500 font-medium">Active sources</p>
          </div>
        </Link>
        <Link href="/dashboard/models" className="group">
          <div className="glass rounded-2xl p-6 hover-glow handcrafted-outline cursor-pointer transition-all duration-300 border-pastel-mint/40">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-xs font-semibold text-gray-soft-500 uppercase tracking-wider">ML Models</h3>
              <div className="w-11 h-11 bg-pastel-mint rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                <svg className="w-5 h-5 text-gray-soft-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
            <p className="text-4xl font-bold text-gray-soft-700 mb-2">
              {modelsLoading ? (
                <span className="inline-block w-12 h-10 bg-gray-soft-200/50 rounded animate-pulse"></span>
              ) : (
                models.length
              )}
            </p>
            <p className="text-xs text-gray-soft-500 font-medium">Total models</p>
          </div>
        </Link>
        <Link href="/dashboard/predictions" className="group">
          <div className="glass rounded-2xl p-6 hover-glow handcrafted-outline cursor-pointer transition-all duration-300 border-pastel-green/40">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-xs font-semibold text-gray-soft-500 uppercase tracking-wider">Trained Models</h3>
              <div className="w-11 h-11 bg-pastel-green rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                <svg className="w-5 h-5 text-gray-soft-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
            <p className="text-4xl font-bold text-gray-soft-700 mb-2">
              {modelsLoading ? (
                <span className="inline-block w-12 h-10 bg-gray-soft-200/50 rounded animate-pulse"></span>
              ) : (
                models.filter((m: any) => m.status === 'trained' || m.status === 'deployed').length
              )}
            </p>
            <p className="text-xs text-gray-soft-500 font-medium">Ready for predictions</p>
          </div>
        </Link>
        <div className="glass rounded-2xl p-6 border-pastel-powder/40">
          <div className="flex items-center justify-between mb-5">
            <h3 className="text-xs font-semibold text-gray-soft-500 uppercase tracking-wider">Active Jobs</h3>
            <div className="w-11 h-11 bg-pastel-powder rounded-xl flex items-center justify-center shadow-sm">
              <svg className="w-5 h-5 text-gray-soft-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </div>
          </div>
          <p className="text-4xl font-bold text-gray-soft-700 mb-2">
            {jobsLoading ? (
              <span className="inline-block w-12 h-10 bg-gray-soft-200/50 rounded animate-pulse"></span>
            ) : (
              activeJobs
            )}
          </p>
          <p className="text-xs text-gray-soft-500 font-medium">Running now</p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass rounded-2xl p-6 border-pastel-blue/40">
          <h2 className="text-xl font-bold text-gray-soft-700 mb-6 flex items-center gap-3">
            <span className="w-1 h-6 bg-pastel-blue rounded-full"></span>
            Recent Data Sources
          </h2>
          {sourcesLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-6 h-6 border-2 border-pastel-blue/30 border-t-pastel-blue rounded-full animate-spin"></div>
            </div>
          ) : sources.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-soft-500 mb-3 font-medium">No data sources yet.</p>
              <Link href="/dashboard/ingestion" className="inline-flex items-center gap-2 text-pastel-blue hover:text-primary-600 font-semibold transition-colors">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Upload data
              </Link>
            </div>
          ) : (
            <ul className="space-y-2">
              {sources.slice(0, 5).map((source: any) => (
                <li key={source.id} className="flex justify-between items-center py-3 px-3 rounded-lg border-b border-pastel-blue/20 last:border-0 group hover:bg-pastel-blue/10 transition-colors">
                  <span className="text-gray-soft-700 font-medium group-hover:text-pastel-blue transition-colors">{source.name}</span>
                  <span className={`px-3 py-1 text-xs font-semibold rounded-full ${
                    source.status === 'active' ? 'bg-accent-50 text-accent-500 border border-accent-100' :
                    source.status === 'processing' ? 'bg-pastel-mint/50 text-primary-600 border border-pastel-mint' :
                    'bg-gray-soft-100 text-gray-soft-600 border border-gray-soft-200'
                  }`}>
                    {source.status}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        
        <div className="glass rounded-2xl p-6 border-pastel-mint/40">
          <h2 className="text-xl font-bold text-gray-soft-700 mb-6 flex items-center gap-3">
            <span className="w-1 h-6 bg-pastel-mint rounded-full"></span>
            Recent Models
          </h2>
          {modelsLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-6 h-6 border-2 border-pastel-mint/30 border-t-pastel-mint rounded-full animate-spin"></div>
            </div>
          ) : models.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-soft-500 mb-3 font-medium">No models yet.</p>
              <Link href="/dashboard/models" className="inline-flex items-center gap-2 text-pastel-mint hover:text-primary-600 font-semibold transition-colors">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Train a model
              </Link>
            </div>
          ) : (
            <ul className="space-y-2">
              {models.slice(0, 5).map((model: any) => (
                <li key={model.id} className="flex justify-between items-center py-3 px-3 rounded-lg border-b border-pastel-mint/20 last:border-0 group hover:bg-pastel-mint/10 transition-colors">
                  <div>
                    <span className="text-gray-soft-700 font-medium group-hover:text-pastel-mint transition-colors">{model.name}</span>
                    <span className="ml-2 text-xs text-gray-soft-500">({model.type})</span>
                  </div>
                  <span className={`px-3 py-1 text-xs font-semibold rounded-full ${
                    model.status === 'trained' || model.status === 'deployed' ? 'bg-accent-50 text-accent-500 border border-accent-100' :
                    model.status === 'training' ? 'bg-pastel-mint/50 text-primary-600 border border-pastel-mint' :
                    'bg-gray-soft-100 text-gray-soft-600 border border-gray-soft-200'
                  }`}>
                    {model.status}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}

