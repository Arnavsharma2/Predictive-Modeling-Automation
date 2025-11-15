'use client'

import { useState } from 'react'
import PredictionForm from '@/components/forms/PredictionForm'

export default function PredictionsPage() {
  const [activeTab, setActiveTab] = useState<'regression' | 'classification' | 'anomaly'>('regression')

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold text-gray-soft-700 mb-2">Predictions</h1>
      <p className="text-gray-soft-600 mb-8 font-medium">Make predictions with your trained models</p>
      
      <div className="mb-6">
        <div className="border-b border-pastel-green/30">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('regression')}
              className={`py-4 px-1 border-b-2 font-semibold text-sm ${
                activeTab === 'regression'
                  ? 'border-pastel-green text-pastel-green'
                  : 'border-transparent text-gray-soft-500 hover:text-gray-soft-700 hover:border-pastel-green/30'
              }`}
            >
              Regression
            </button>
            <button
              onClick={() => setActiveTab('classification')}
              className={`py-4 px-1 border-b-2 font-semibold text-sm ${
                activeTab === 'classification'
                  ? 'border-pastel-green text-pastel-green'
                  : 'border-transparent text-gray-soft-500 hover:text-gray-soft-700 hover:border-pastel-green/30'
              }`}
            >
              Classification
            </button>
            <button
              onClick={() => setActiveTab('anomaly')}
              className={`py-4 px-1 border-b-2 font-semibold text-sm ${
                activeTab === 'anomaly'
                  ? 'border-pastel-green text-pastel-green'
                  : 'border-transparent text-gray-soft-500 hover:text-gray-soft-700 hover:border-pastel-green/30'
              }`}
            >
              Anomaly Detection
            </button>
          </nav>
        </div>
      </div>

      <div className="glass rounded-2xl border-pastel-green/40 p-8">
        <h2 className="text-2xl font-bold text-gray-soft-700 mb-6 flex items-center gap-2">
          <span className="w-1 h-6 bg-pastel-green rounded-full"></span>
          {activeTab === 'regression' && 'Regression Prediction'}
          {activeTab === 'classification' && 'Classification'}
          {activeTab === 'anomaly' && 'Anomaly Detection'}
        </h2>
        <PredictionForm 
          modelType={
            activeTab === 'regression' ? 'regression' :
            activeTab === 'classification' ? 'classification' :
            'anomaly_detection'
          }
        />
      </div>
    </div>
  )
}

