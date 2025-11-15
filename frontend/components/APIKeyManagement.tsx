'use client';

import { useState, useEffect } from 'react';
import { getAPIKeys, createAPIKey, deleteAPIKey, type APIKey, type APIKeyResponse } from '@/lib/auth';

export default function APIKeyManagement() {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [createdKey, setCreatedKey] = useState<APIKeyResponse | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    loadAPIKeys();
  }, []);

  const loadAPIKeys = async () => {
    try {
      const keys = await getAPIKeys();
      setApiKeys(keys);
    } catch (err: any) {
      setError('Failed to load API keys');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKeyName.trim()) return;

    setCreating(true);
    setError('');

    try {
      const newKey = await createAPIKey(newKeyName);
      setCreatedKey(newKey);
      setNewKeyName('');
      await loadAPIKeys();
    } catch (err: any) {
      setError(err.message || 'Failed to create API key');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      return;
    }

    try {
      await deleteAPIKey(id);
      await loadAPIKeys();
    } catch (err: any) {
      setError('Failed to delete API key');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (loading) {
    return <div className="text-center py-4">Loading API keys...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">API Keys</h3>
        <p className="mt-1 text-sm text-gray-500">
          Create API keys for programmatic access to the platform.
        </p>
      </div>

      {/* Created key modal */}
      {createdKey && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-yellow-400"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3 flex-1">
              <h3 className="text-sm font-medium text-yellow-800">
                Save your API key
              </h3>
              <div className="mt-2 text-sm text-yellow-700">
                <p>
                  Make sure to copy your API key now. You won't be able to see
                  it again!
                </p>
                <div className="mt-3 flex items-center space-x-2">
                  <code className="flex-1 bg-white px-3 py-2 rounded border border-yellow-300 text-xs break-all">
                    {createdKey.api_key}
                  </code>
                  <button
                    onClick={() => copyToClipboard(createdKey.api_key)}
                    className="px-3 py-2 bg-yellow-100 text-yellow-800 rounded hover:bg-yellow-200 text-xs font-medium"
                  >
                    Copy
                  </button>
                </div>
              </div>
              <div className="mt-4">
                <button
                  onClick={() => setCreatedKey(null)}
                  className="text-sm font-medium text-yellow-800 hover:text-yellow-900"
                >
                  Got it, dismiss
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      {/* Create new key form */}
      <form onSubmit={handleCreate} className="flex gap-2">
        <input
          type="text"
          value={newKeyName}
          onChange={(e) => setNewKeyName(e.target.value)}
          placeholder="Enter API key name (e.g., Production API)"
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          disabled={creating}
        />
        <button
          type="submit"
          disabled={creating || !newKeyName.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {creating ? 'Creating...' : 'Create Key'}
        </button>
      </form>

      {/* API keys list */}
      {apiKeys.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <p>No API keys yet. Create one to get started.</p>
        </div>
      ) : (
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            {apiKeys.map((key) => (
              <li key={key.id} className="px-4 py-4 sm:px-6">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center">
                      <p className="text-sm font-medium text-gray-900">
                        {key.name}
                      </p>
                      {!key.is_active && (
                        <span className="ml-2 px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded">
                          Inactive
                        </span>
                      )}
                    </div>
                    <div className="mt-1 flex items-center space-x-4 text-sm text-gray-500">
                      <span>
                        <code className="bg-gray-100 px-2 py-1 rounded">
                          {key.key_prefix}...
                        </code>
                      </span>
                      <span>Created {new Date(key.created_at).toLocaleDateString()}</span>
                      {key.last_used_at && (
                        <span>
                          Last used {new Date(key.last_used_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(key.id)}
                    className="ml-4 px-3 py-1 text-sm text-red-600 hover:text-red-800 font-medium"
                  >
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
