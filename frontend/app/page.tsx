'use client'

import Link from 'next/link'
import FeatureImportanceChart from '@/components/charts/FeatureImportanceChart'
import ShapWaterfallChart from '@/components/charts/ShapWaterfallChart'
import ArchitectureDiagram from '@/components/ArchitectureDiagram'
import { useAuth } from '@/contexts/AuthContext'

export default function Home() {
  const { isAuthenticated } = useAuth()
  
  const getStartedHref = isAuthenticated ? '/dashboard' : '/auth/register'
  const learnMoreHref = isAuthenticated ? '/dashboard' : '/auth/register'
  // Sample SHAP data for demonstration
  const sampleShapData = {
    baseValue: 0.5,
    prediction: 0.73,
    contributions: [
      { feature: 'House Size (sqft)', shap_value: 0.15 },
      { feature: 'Location Score', shap_value: 0.12 },
      { feature: 'Age of Property', shap_value: -0.08 },
      { feature: 'Number of Bedrooms', shap_value: 0.06 },
      { feature: 'Neighborhood Rating', shap_value: 0.05 },
      { feature: 'Distance to School', shap_value: -0.03 },
      { feature: 'Property Condition', shap_value: 0.04 },
      { feature: 'Year Built', shap_value: 0.02 },
    ]
  }

  // Sample feature importance data
  const sampleFeatureImportance = {
    'House Size (sqft)': 0.245,
    'Location Score': 0.198,
    'Age of Property': -0.156,
    'Number of Bedrooms': 0.134,
    'Neighborhood Rating': 0.112,
    'Distance to School': -0.089,
    'Property Condition': 0.078,
    'Year Built': 0.065,
    'Garage Size': 0.054,
    'Lot Size': 0.043,
  }

  return (
    <>
      {/* Animated pastel background elements - Fixed to cover entire page - NO GRADIENTS */}
      <div className="fixed inset-0 pointer-events-none z-0" style={{ willChange: 'transform' }}>
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-pastel-blue/30 rounded-full blur-3xl float"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pastel-mint/25 rounded-full blur-3xl float" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-pastel-green/20 rounded-full blur-3xl float" style={{ animationDelay: '4s' }}></div>
        <div className="absolute top-3/4 right-1/3 w-80 h-80 bg-pastel-powder/25 rounded-full blur-3xl float" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/6 right-1/6 w-72 h-72 bg-pastel-blue/20 rounded-full blur-3xl float" style={{ animationDelay: '3s' }}></div>
        <div className="absolute bottom-1/3 left-1/3 w-80 h-80 bg-pastel-mint/20 rounded-full blur-3xl float" style={{ animationDelay: '5s' }}></div>
        
        {/* Foggy cloud edges */}
        <div className="foggy-edges"></div>
        
        {/* Paolo Ceric-inspired animated geometric pattern */}
        <div className="absolute top-1/2 left-1/2 geometric-spiral" style={{ width: '600px', height: '600px', transform: 'translate(-50%, -50%)' }}>
          <div className="geometric-pattern"></div>
        </div>
      </div>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl lg:text-7xl font-bold mb-4 leading-tight text-reveal" style={{ '--reveal-delay': '0.2s' } as React.CSSProperties}>
              <span className="block text-gray-soft-700">Automated Predictive</span>
              <span className="block gradient-text">Modeling Pipeline</span>
            </h1>
            
            <p className="text-lg lg:text-xl text-gray-soft-600 mb-6 max-w-3xl mx-auto text-reveal leading-relaxed" style={{ '--reveal-delay': '0.4s' } as React.CSSProperties}>
              An platform for building, training, and deploying machine learning models. Upload your data, train models, monitor performance, and generate predictions with automated preprocessing and feature engineering.
            </p>
            
            <div className="flex flex-col sm:flex-row justify-center gap-6 mb-8 text-reveal" style={{ '--reveal-delay': '0.6s' } as React.CSSProperties}>
              <Link
                href={getStartedHref}
                className="btn-primary hover-glow handcrafted-outline inline-flex items-center justify-center group"
              >
                <span>Get Started</span>
                <svg className="ml-2 w-5 h-5 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Link>
              <Link
                href={learnMoreHref}
                className="btn-secondary hover-glow handcrafted-outline inline-flex items-center justify-center"
              >
                Learn More
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Analytics & Explainability Section */}
      <section className="relative py-12 lg:py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center mb-8">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-gray-soft-700 mb-3">
              Advanced Model Explainability
            </h2>
            <p className="text-lg text-gray-soft-600 max-w-3xl mx-auto">
              Understand your model predictions with SHAP and LIME explanations. See exactly which features drive your predictions and why.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
            {/* SHAP Waterfall Chart */}
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-4 shadow-lg border border-gray-200/50">
              <ShapWaterfallChart
                baseValue={sampleShapData.baseValue}
                prediction={sampleShapData.prediction}
                contributions={sampleShapData.contributions}
                maxDisplay={8}
                height={400}
              />
              <p className="text-sm text-gray-soft-600 mt-3 text-center">
                SHAP waterfall visualization showing how each feature contributes to the final prediction
              </p>
            </div>

            {/* Feature Importance Chart */}
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-4 shadow-lg border border-gray-200/50">
              <FeatureImportanceChart
                importance={sampleFeatureImportance}
                title="Feature Importance Analysis"
                maxFeatures={10}
                height={400}
              />
              <p className="text-sm text-gray-soft-600 mt-3 text-center">
                Global feature importance showing which features have the most impact on model predictions
              </p>
            </div>
          </div>

          <div className="mt-8 text-center">
            <Link
              href={isAuthenticated ? '/dashboard/explainability' : '/auth/register'}
              className="btn-secondary hover-glow handcrafted-outline inline-flex items-center justify-center group"
            >
              <span>Explore Explainability</span>
              <svg className="ml-2 w-5 h-5 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
          </div>
        </div>
      </section>

      {/* Architecture Section */}
      <section className="relative py-12 lg:py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center mb-8">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-gray-soft-700 mb-3">
              System Architecture
            </h2>
            <p className="text-lg text-gray-soft-600 max-w-3xl mx-auto">
              Built with modern, scalable technologies. Our architecture ensures high performance, reliability, and seamless integration.
            </p>
          </div>

          <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 lg:p-8 shadow-lg border border-gray-200/50">
            <ArchitectureDiagram />
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white/60 backdrop-blur-sm rounded-lg p-5 border border-gray-200/50">
              <h3 className="text-xl font-semibold text-gray-soft-700 mb-2">Scalable Backend</h3>
              <p className="text-gray-soft-600">
                FastAPI-powered REST API with async operations, Redis caching, and cloud storage integration for optimal performance.
              </p>
            </div>
            <div className="bg-white/60 backdrop-blur-sm rounded-lg p-5 border border-gray-200/50">
              <h3 className="text-xl font-semibold text-gray-soft-700 mb-2">Modern Frontend</h3>
              <p className="text-gray-soft-600">
                Next.js 14 with React 18, TypeScript, and Tailwind CSS for a responsive, interactive user experience.
              </p>
            </div>
            <div className="bg-white/60 backdrop-blur-sm rounded-lg p-5 border border-gray-200/50">
              <h3 className="text-xl font-semibold text-gray-soft-700 mb-2">Data Pipeline</h3>
              <p className="text-gray-soft-600">
                Prefect orchestration with PostgreSQL + TimescaleDB for efficient time-series data management and ETL workflows.
              </p>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}

