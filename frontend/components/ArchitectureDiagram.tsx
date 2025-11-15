'use client'

import { useMemo } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  ConnectionMode,
  MarkerType,
  Position,
} from 'reactflow'
import 'reactflow/dist/style.css'

const nodeTypes = {}

export default function ArchitectureDiagram() {
  const nodes: Node[] = useMemo(() => [
    // Frontend Layer
    {
      id: 'frontend',
      type: 'default',
      position: { x: 250, y: 0 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-lg font-bold text-gray-soft-700 mb-1">Frontend</div>
            <div className="text-sm text-gray-soft-600">Next.js 14</div>
            <div className="text-xs text-gray-soft-500 mt-1">React 18 • TypeScript</div>
            <div className="text-xs text-gray-soft-500">Port: 3000</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(168, 213, 226, 0.2)',
        border: '2px solid #A8D5E2',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '200px',
        zIndex: 10,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    },
    {
      id: 'websocket',
      type: 'default',
      position: { x: 600, y: 0 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-base font-bold text-gray-soft-700 mb-1">WebSocket</div>
            <div className="text-xs text-gray-soft-600">Real-time Updates</div>
            <div className="text-xs text-gray-soft-500 mt-1">Training & Batch Jobs</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(168, 85, 247, 0.2)',
        border: '2px solid #A855F7',
        borderRadius: '12px',
        padding: '16px',
        minWidth: '160px',
        zIndex: 10,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    },

    // Backend Layer
    {
      id: 'backend',
      type: 'default',
      position: { x: 250, y: 250 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-lg font-bold text-gray-soft-700 mb-1">Backend API</div>
            <div className="text-sm text-gray-soft-600">FastAPI</div>
            <div className="text-xs text-gray-soft-500 mt-1">Async • SQLAlchemy 2.0</div>
            <div className="text-xs text-gray-soft-500">Port: 8000</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(200, 230, 201, 0.2)',
        border: '2px solid #C8E6C9',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '240px',
        zIndex: 10,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    },

    // Backend Sub-components - Better spacing
    {
      id: 'api-endpoints',
      type: 'default',
      position: { x: 0, y: 500 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-sm font-semibold text-gray-soft-700">API Endpoints</div>
            <div className="text-xs text-gray-soft-600 mt-1">REST Routes</div>
            <div className="text-xs text-gray-soft-500 mt-1">v1/*</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(200, 230, 201, 0.15)',
        border: '1px solid #81C784',
        borderRadius: '8px',
        padding: '12px',
        minWidth: '140px',
        zIndex: 10,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    },
    {
      id: 'ml-models',
      type: 'default',
      position: { x: 200, y: 500 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-sm font-semibold text-gray-soft-700">ML Models</div>
            <div className="text-xs text-gray-soft-600 mt-1">XGBoost • CatBoost</div>
            <div className="text-xs text-gray-soft-500 mt-1">LightGBM • Random Forest</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(200, 230, 201, 0.15)',
        border: '1px solid #81C784',
        borderRadius: '8px',
        padding: '12px',
        minWidth: '140px',
        zIndex: 10,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    },
    {
      id: 'auth-service',
      type: 'default',
      position: { x: 400, y: 500 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-sm font-semibold text-gray-soft-700">Auth Service</div>
            <div className="text-xs text-gray-soft-600 mt-1">JWT • OAuth2</div>
            <div className="text-xs text-gray-soft-500 mt-1">API Keys</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(200, 230, 201, 0.15)',
        border: '1px solid #81C784',
        borderRadius: '8px',
        padding: '12px',
        minWidth: '140px',
        zIndex: 10,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    },
    {
      id: 'etl-pipeline',
      type: 'default',
      position: { x: 700, y: 500 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-sm font-semibold text-gray-soft-700">ETL Pipeline</div>
            <div className="text-xs text-gray-soft-600 mt-1">Data Processing</div>
            <div className="text-xs text-gray-soft-500 mt-1">Feature Engineering</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(200, 230, 201, 0.15)',
        border: '1px solid #81C784',
        borderRadius: '8px',
        padding: '12px',
        minWidth: '140px',
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    },
    {
      id: 'storage-client',
      type: 'default',
      position: { x: 900, y: 500 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-sm font-semibold text-gray-soft-700">Storage Client</div>
            <div className="text-xs text-gray-soft-600 mt-1">S3 • Azure Blob</div>
            <div className="text-xs text-gray-soft-500 mt-1">Local Storage</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(200, 230, 201, 0.15)',
        border: '1px solid #81C784',
        borderRadius: '8px',
        padding: '12px',
        minWidth: '140px',
        zIndex: 10,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    },

    // Services Layer - Better spacing
    {
      id: 'postgres',
      type: 'default',
      position: { x: 0, y: 720 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-base font-bold text-gray-soft-700 mb-1">PostgreSQL</div>
            <div className="text-xs text-gray-soft-600">+ TimescaleDB</div>
            <div className="text-xs text-gray-soft-500 mt-1">Time-series Data</div>
            <div className="text-xs text-gray-soft-500">Port: 5432</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(181, 229, 207, 0.2)',
        border: '2px solid #B5E5CF',
        borderRadius: '12px',
        padding: '16px',
        minWidth: '180px',
        zIndex: 10,
      },
      sourcePosition: Position.Top,
      targetPosition: Position.Bottom,
    },
    {
      id: 'redis',
      type: 'default',
      position: { x: 220, y: 720 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-base font-bold text-gray-soft-700 mb-1">Redis</div>
            <div className="text-xs text-gray-soft-600">Cache & Pub/Sub</div>
            <div className="text-xs text-gray-soft-500 mt-1">Task Queue</div>
            <div className="text-xs text-gray-soft-500">Port: 6379</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(176, 224, 230, 0.2)',
        border: '2px solid #B0E0E6',
        borderRadius: '12px',
        padding: '16px',
        minWidth: '180px',
        zIndex: 10,
      },
      sourcePosition: Position.Top,
      targetPosition: Position.Bottom,
    },
    {
      id: 'mlflow',
      type: 'default',
      position: { x: 430, y: 720 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-base font-bold text-gray-soft-700 mb-1">MLflow</div>
            <div className="text-xs text-gray-soft-600">Model Registry</div>
            <div className="text-xs text-gray-soft-500 mt-1">Experiment Tracking</div>
            <div className="text-xs text-gray-soft-500">Port: 5000</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(147, 197, 253, 0.2)',
        border: '2px solid #93C5FD',
        borderRadius: '12px',
        padding: '16px',
        minWidth: '180px',
        zIndex: 10,
      },
      sourcePosition: Position.Top,
      targetPosition: Position.Bottom,
    },
    {
      id: 'prefect',
      type: 'default',
      position: { x: 800, y: 720 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-base font-bold text-gray-soft-700 mb-1">Prefect</div>
            <div className="text-xs text-gray-soft-600">Orchestrator</div>
            <div className="text-xs text-gray-soft-500 mt-1">Workflow Engine</div>
            <div className="text-xs text-gray-soft-500">Port: 4200</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(196, 181, 253, 0.2)',
        border: '2px solid #C4B5FD',
        borderRadius: '12px',
        padding: '16px',
        minWidth: '180px',
        zIndex: 10,
      },
      sourcePosition: Position.Top,
      targetPosition: Position.Bottom,
    },
    {
      id: 'celery',
      type: 'default',
      position: { x: 1020, y: 720 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-base font-bold text-gray-soft-700 mb-1">Celery Workers</div>
            <div className="text-xs text-gray-soft-600">Background Tasks</div>
            <div className="text-xs text-gray-soft-500 mt-1">Model Training</div>
            <div className="text-xs text-gray-soft-500">Async Processing</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(251, 191, 36, 0.2)',
        border: '2px solid #FBBF24',
        borderRadius: '12px',
        padding: '16px',
        minWidth: '180px',
        zIndex: 10,
      },
      sourcePosition: Position.Top,
      targetPosition: Position.Bottom,
    },

    // Storage Layer
    {
      id: 'storage',
      type: 'default',
      position: { x: 400, y: 920 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-lg font-bold text-gray-soft-700 mb-1">Cloud Storage</div>
            <div className="text-sm text-gray-soft-600">S3 / Azure Blob</div>
            <div className="text-xs text-gray-soft-500 mt-1">Model Artifacts</div>
            <div className="text-xs text-gray-soft-500">Data Files</div>
          </div>
        ),
      },
      style: {
        background: 'rgba(254, 215, 170, 0.2)',
        border: '2px solid #FED7AA',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '220px',
        zIndex: 10,
      },
      sourcePosition: Position.Top,
      targetPosition: Position.Bottom,
    },
  ], [])

  const edges: Edge[] = useMemo(() => [
    // Frontend to Backend
    {
      id: 'frontend-backend',
      source: 'frontend',
      target: 'backend',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#3B82F6', strokeWidth: 2.5 },
      label: 'HTTP/REST API',
      labelStyle: { fill: '#3B82F6', fontWeight: 600, fontSize: '11px' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#3B82F6' },
    },
    // Frontend to WebSocket
    {
      id: 'frontend-websocket',
      source: 'frontend',
      target: 'websocket',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#A855F7', strokeWidth: 2, strokeDasharray: '5 3' },
      label: 'WebSocket',
      labelStyle: { fill: '#A855F7', fontWeight: 600, fontSize: '10px' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#A855F7' },
    },
    // WebSocket to Backend
    {
      id: 'websocket-backend',
      source: 'websocket',
      target: 'backend',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#A855F7', strokeWidth: 2, strokeDasharray: '5 3' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#A855F7' },
    },
    // Backend to sub-components
    {
      id: 'backend-api',
      source: 'backend',
      target: 'api-endpoints',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#94A3B8', strokeWidth: 1.5, strokeDasharray: '3 2' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94A3B8' },
    },
    {
      id: 'backend-ml',
      source: 'backend',
      target: 'ml-models',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#94A3B8', strokeWidth: 1.5, strokeDasharray: '3 2' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94A3B8' },
    },
    {
      id: 'backend-auth',
      source: 'backend',
      target: 'auth-service',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#94A3B8', strokeWidth: 1.5, strokeDasharray: '3 2' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94A3B8' },
    },
    {
      id: 'backend-etl',
      source: 'backend',
      target: 'etl-pipeline',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#94A3B8', strokeWidth: 1.5, strokeDasharray: '3 2' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94A3B8' },
    },
    {
      id: 'backend-storage-client',
      source: 'backend',
      target: 'storage-client',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#94A3B8', strokeWidth: 1.5, strokeDasharray: '3 2' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94A3B8' },
    },
    // Backend to services
    {
      id: 'backend-postgres',
      source: 'backend',
      target: 'postgres',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748B', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#64748B' },
    },
    {
      id: 'backend-redis',
      source: 'backend',
      target: 'redis',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748B', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#64748B' },
    },
    {
      id: 'backend-mlflow',
      source: 'backend',
      target: 'mlflow',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748B', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#64748B' },
    },
    {
      id: 'backend-prefect',
      source: 'backend',
      target: 'prefect',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748B', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#64748B' },
    },
    {
      id: 'backend-storage',
      source: 'backend',
      target: 'storage',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748B', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#64748B' },
    },
    // Celery connections
    {
      id: 'celery-redis',
      source: 'celery',
      target: 'redis',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#F59E0B', strokeWidth: 2, strokeDasharray: '4 3' },
      label: 'Pub/Sub',
      labelStyle: { fill: '#F59E0B', fontWeight: 600, fontSize: '10px' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#F59E0B' },
    },
    {
      id: 'celery-postgres',
      source: 'celery',
      target: 'postgres',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748B', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#64748B' },
    },
    {
      id: 'celery-storage',
      source: 'celery',
      target: 'storage',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748B', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#64748B' },
    },
    {
      id: 'celery-mlflow',
      source: 'celery',
      target: 'mlflow',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#64748B', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#64748B' },
    },
    // WebSocket to Redis
    {
      id: 'websocket-redis',
      source: 'websocket',
      target: 'redis',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#A855F7', strokeWidth: 2, strokeDasharray: '4 3' },
      label: 'Pub/Sub',
      labelStyle: { fill: '#A855F7', fontWeight: 600, fontSize: '10px' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#A855F7' },
    },
    // Prefect to PostgreSQL
    {
      id: 'prefect-postgres',
      source: 'prefect',
      target: 'postgres',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#94A3B8', strokeWidth: 1.5, strokeDasharray: '3 2' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94A3B8' },
    },
    // Storage Client to Storage
    {
      id: 'storage-client-storage',
      source: 'storage-client',
      target: 'storage',
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#94A3B8', strokeWidth: 1.5, strokeDasharray: '3 2' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94A3B8' },
    },
  ], [])

  return (
    <div className="w-full py-8">
      <div className="w-full max-w-7xl mx-auto" style={{ height: '1000px' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          connectionMode={ConnectionMode.Loose}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          minZoom={0.25}
          maxZoom={1.5}
          elevateNodesOnSelect={true}
          elevateEdgesOnSelect={false}
        >
          <Background color="#e5e7eb" gap={16} />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              if (node.id === 'frontend') return '#A8D5E2'
              if (node.id === 'backend') return '#C8E6C9'
              if (node.id === 'postgres') return '#B5E5CF'
              if (node.id === 'redis') return '#B0E0E6'
              if (node.id === 'prefect') return '#C4B5FD'
              if (node.id === 'celery') return '#FBBF24'
              if (node.id === 'storage') return '#FED7AA'
              if (node.id === 'websocket') return '#A855F7'
              if (node.id === 'mlflow') return '#93C5FD'
              return '#94A3B8'
            }}
            maskColor="rgba(0, 0, 0, 0.1)"
          />
        </ReactFlow>
      </div>

      {/* Legend */}
      <div className="mt-8 space-y-4">
        <div className="text-center text-sm font-semibold text-gray-soft-700 mb-4">Component Types</div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-9 gap-4 max-w-7xl mx-auto">
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-pastel-blue/20 border-2 border-pastel-blue rounded"></div>
            <span className="text-sm text-gray-soft-600">Frontend</span>
          </div>
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-pastel-green/20 border-2 border-pastel-green rounded"></div>
            <span className="text-sm text-gray-soft-600">Backend</span>
          </div>
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-pastel-mint/20 border-2 border-pastel-mint rounded"></div>
            <span className="text-sm text-gray-soft-600">Database</span>
          </div>
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-pastel-powder/20 border-2 border-pastel-powder rounded"></div>
            <span className="text-sm text-gray-soft-600">Cache</span>
          </div>
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-blue-200/30 border-2 border-blue-300 rounded"></div>
            <span className="text-sm text-gray-soft-600">MLflow</span>
          </div>
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-purple-200/30 border-2 border-purple-300 rounded"></div>
            <span className="text-sm text-gray-soft-600">Orchestration</span>
          </div>
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-orange-200/30 border-2 border-orange-300 rounded"></div>
            <span className="text-sm text-gray-soft-600">Storage</span>
          </div>
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-amber-200/30 border-2 border-amber-400 rounded"></div>
            <span className="text-sm text-gray-soft-600">Workers</span>
          </div>
          <div className="flex items-center gap-2 justify-center">
            <div className="w-4 h-4 bg-purple-200/30 border-2 border-purple-400 rounded"></div>
            <span className="text-sm text-gray-soft-600">WebSocket</span>
          </div>
        </div>
        <div className="mt-6 flex flex-wrap justify-center gap-6 text-xs text-gray-soft-600">
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-blue-500"></div>
            <span>HTTP/REST</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 border-dashed border-purple-500" style={{ borderTopWidth: '2px' }}></div>
            <span>WebSocket</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 border-dashed border-amber-500" style={{ borderTopWidth: '2px' }}></div>
            <span>Pub/Sub</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 border-dashed border-gray-400" style={{ borderTopWidth: '2px' }}></div>
            <span>Data Flow</span>
          </div>
        </div>
      </div>
    </div>
  )
}
