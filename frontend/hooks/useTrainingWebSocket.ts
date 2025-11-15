/**
 * React hook for WebSocket connection to training job updates.
 * Provides real-time progress updates without polling.
 */
import { useEffect, useRef, useState, useCallback } from 'react';

interface TrainingUpdate {
  type: string;
  job_id: number;
  status: string;
  progress: number | null;
  current_epoch: number | null;
  total_epochs: number | null;
  metrics: Record<string, any> | null;
  error_message: string | null;
  updated_at: string;
}

interface UseTrainingWebSocketOptions {
  jobId: number;
  token: string;
  enabled?: boolean;
  onUpdate?: (update: TrainingUpdate) => void;
  onError?: (error: Error) => void;
  onClose?: () => void;
}

interface UseTrainingWebSocketReturn {
  isConnected: boolean;
  lastUpdate: TrainingUpdate | null;
  error: Error | null;
  reconnect: () => void;
}

const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

/**
 * Hook to establish WebSocket connection for real-time training updates.
 *
 * @param options - Configuration options
 * @returns WebSocket connection state and controls
 *
 * @example
 * ```tsx
 * const { isConnected, lastUpdate } = useTrainingWebSocket({
 *   jobId: 123,
 *   token: accessToken,
 *   enabled: true,
 *   onUpdate: (update) => {
 *     console.log('Progress:', update.progress);
 *   }
 * });
 * ```
 */
export function useTrainingWebSocket({
  jobId,
  token,
  enabled = true,
  onUpdate,
  onError,
  onClose,
}: UseTrainingWebSocketOptions): UseTrainingWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<TrainingUpdate | null>(null);
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

      const wsUrl = `${WS_BASE_URL}/api/v1/ws/training/${jobId}?token=${encodeURIComponent(token)}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`WebSocket connected for job ${jobId}`);
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle different message types
          if (data.type === 'connected') {
            console.log('WebSocket connection confirmed');
            return;
          }

          if (data.type === 'progress_update') {
            const update: TrainingUpdate = data;
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
        console.log(`WebSocket closed for job ${jobId}:`, event.code, event.reason);
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
