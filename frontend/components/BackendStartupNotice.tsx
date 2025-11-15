'use client';

import { useState, useEffect } from 'react';

interface BackendStartupNoticeProps {
  className?: string;
  autoDismiss?: boolean;
  dismissAfter?: number; // milliseconds
}

export default function BackendStartupNotice({ 
  className = '', 
  autoDismiss = false,
  dismissAfter = 45000 // 45 seconds
}: BackendStartupNoticeProps) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (autoDismiss && isVisible) {
      const timer = setTimeout(() => {
        setIsVisible(false);
      }, dismissAfter);
      return () => clearTimeout(timer);
    }
  }, [autoDismiss, dismissAfter, isVisible]);

  if (!isVisible) return null;

  return (
    <div className={`fixed top-4 right-4 z-50 max-w-md rounded-lg bg-red-50 border-2 border-red-300 p-5 shadow-xl ${className}`}>
      <div className="flex items-start">
        <div className="flex-shrink-0">
          <svg
            className="h-6 w-6 text-red-600"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        </div>
        <div className="ml-3 flex-1">
          <h3 className="text-base font-semibold text-red-800 mb-2">
            Limited Functionality Notice
          </h3>
          <div className="mt-2 text-sm text-red-700 space-y-2">
            <p className="font-medium">
              This website implementation has limited functionality.
            </p>
            <p>
              Full features require monthly hosting fees for background services (Redis, Celery workers, MLflow, etc.) that are not currently deployed. 
              Basic authentication and API endpoints may work, but advanced features like model training, real-time updates, and experiment tracking are unavailable.
            </p>
          </div>
        </div>
        <div className="ml-4 flex-shrink-0">
          <button
            type="button"
            className="inline-flex text-red-600 hover:text-red-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 rounded-md"
            onClick={() => setIsVisible(false)}
          >
            <span className="sr-only">Dismiss</span>
            <svg
              className="h-5 w-5"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

