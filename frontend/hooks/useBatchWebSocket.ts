/**
 * React hook for WebSocket connection to batch prediction job updates.
 * Provides real-time progress updates without polling.
 */
import { useEffect, useRef, useState, useCallback } from 'react';

interface BatchJobUpdate {
  type: string;
  job_id: number;
  status: string;
  progress: number | null;
  total_records: number | null;
  processed_records: number | null;
  failed_records: number | null;
  updated_at: string;
}

interface UseBatchWebSocketOptions {
  jobId: number;
  token: string;
  enabled?: boolean;
  onUpdate?: (update: BatchJobUpdate) => void;
  onError?: (error: Error) => void;
  onClose?: () => void;
}

interface UseBatchWebSocketReturn {
  isConnected: boolean;
  lastUpdate: BatchJobUpdate | null;
  error: Error | null;
  reconnect: () => void;
}

const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

/**
 * Hook to establish WebSocket connection for real-time batch job updates.
 */
export function useBatchWebSocket({
  jobId,
  token,
  enabled = true,
  onUpdate,
  onError,
  onClose,
}: UseBatchWebSocketOptions): UseBatchWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<BatchJobUpdate | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (!enabled || !token) {
      return;
    }

    try {
      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
      }

      // Connect to dashboard WebSocket and subscribe to batch job
      const wsUrl = `${WS_BASE_URL}/api/v1/ws/dashboard?token=${encodeURIComponent(token)}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`WebSocket connected for batch job ${jobId}`);
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;
        
        // Subscribe to batch job updates
        ws.send(JSON.stringify({
          type: 'subscribe',
          subscription_key: `batch_${jobId}`
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle different message types
          if (data.type === 'connected') {
            console.log('WebSocket connection confirmed');
            // Subscribe to batch job updates
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                type: 'subscribe',
                subscription_key: `batch_${jobId}`
              }));
            }
            return;
          }

          if (data.type === 'subscribed') {
            console.log(`Subscribed to ${data.subscription_key}`);
            return;
          }

          if (data.type === 'batch_job_update' && data.job_id === jobId) {
            const update: BatchJobUpdate = data;
            setLastUpdate(update);
            onUpdate?.(update);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
          const parseError = new Error('Failed to parse WebSocket message');
          setError(parseError);
          onError?.(parseError);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        const wsError = new Error('WebSocket connection error');
        setError(wsError);
        onError?.(wsError);
      };

      ws.onclose = (event) => {
        console.log(`WebSocket closed for batch job ${jobId}:`, event.code, event.reason);
        setIsConnected(false);
        wsRef.current = null;
        onClose?.();

        // Attempt to reconnect if not intentionally closed
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
          console.log(`Attempting to reconnect in ${delay}ms...`);

          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        }
      };

      wsRef.current = ws;

      // Send periodic ping to keep connection alive
      const pingInterval = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send('ping');
        }
      }, 30000); // Ping every 30 seconds

      // Clean up ping interval when connection closes
      ws.addEventListener('close', () => clearInterval(pingInterval));

    } catch (err) {
      console.error('Error creating WebSocket:', err);
      const connError = err instanceof Error ? err : new Error('Failed to create WebSocket');
      setError(connError);
      onError?.(connError);
    }
  }, [enabled, token, jobId, onUpdate, onError, onClose]);

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    connect();
  }, [connect]);

  useEffect(() => {
    if (enabled && token) {
      connect();
    }

    return () => {
      // Clean up on unmount
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
      }
    };
  }, [enabled, token, connect]);

  return {
    isConnected,
    lastUpdate,
    error,
    reconnect,
  };
}

