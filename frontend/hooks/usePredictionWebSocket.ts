/**
 * React hook for WebSocket connection to live prediction streaming.
 * Provides real-time prediction updates for a specific model.
 */
import { useEffect, useRef, useState, useCallback } from 'react';

interface PredictionUpdate {
  type: string;
  model_id: number;
  prediction: number | string | any;
  features: Record<string, any>;
  confidence?: Record<string, number>;
  timestamp: string;
}

interface UsePredictionWebSocketOptions {
  modelId: number;
  token: string;
  enabled?: boolean;
  onPrediction?: (update: PredictionUpdate) => void;
  onError?: (error: Error) => void;
  onClose?: () => void;
}

interface UsePredictionWebSocketReturn {
  isConnected: boolean;
  lastPrediction: PredictionUpdate | null;
  error: Error | null;
  reconnect: () => void;
}

const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

/**
 * Hook to establish WebSocket connection for real-time prediction streaming.
 */
export function usePredictionWebSocket({
  modelId,
  token,
  enabled = true,
  onPrediction,
  onError,
  onClose,
}: UsePredictionWebSocketOptions): UsePredictionWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastPrediction, setLastPrediction] = useState<PredictionUpdate | null>(null);
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

      const wsUrl = `${WS_BASE_URL}/api/v1/ws/predictions/${modelId}?token=${encodeURIComponent(token)}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`WebSocket connected for model ${modelId} predictions`);
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle different message types
          if (data.type === 'connected') {
            console.log('WebSocket connection confirmed for predictions');
            return;
          }

          if (data.type === 'prediction') {
            const update: PredictionUpdate = data;
            setLastPrediction(update);
            onPrediction?.(update);
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
        console.log(`WebSocket closed for model ${modelId}:`, event.code, event.reason);
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
  }, [enabled, token, modelId, onPrediction, onError, onClose]);

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
    lastPrediction,
    error,
    reconnect,
  };
}

