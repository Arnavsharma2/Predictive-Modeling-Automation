import axios from 'axios'

// Normalize API base URL - remove trailing slashes to prevent double slashes
const getApiBaseUrl = () => {
  const url = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  return url.replace(/\/+$/, '') // Remove trailing slashes
}

const API_BASE_URL = getApiBaseUrl()
const STORAGE_KEY = 'auth_user'
const TOKEN_KEY = 'auth_token'
const REFRESH_TOKEN_KEY = 'auth_refresh_token'

// User type
export interface User {
  id: number
  username: string
  email: string
  full_name?: string
  role: 'viewer' | 'data_scientist' | 'admin'
  is_active?: boolean
  is_superuser?: boolean
  created_at: string
  updated_at?: string
}

// API Key types
export interface APIKey {
  id: number
  name: string
  key_prefix: string
  is_active?: boolean
  created_at: string
  last_used_at?: string
}

export interface APIKeyResponse {
  id: number
  name: string
  api_key: string
  key_prefix: string
  created_at: string
}

// Login/Register request types
interface LoginRequest {
  username: string
  password: string
}

interface RegisterRequest {
  email: string
  username: string
  password: string
  full_name?: string
  role?: User['role']
}

// Storage helpers
export function getStoredUser(): User | null {
  if (typeof window === 'undefined') return null
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored ? JSON.parse(stored) : null
  } catch {
    return null
  }
}

export function storeUser(user: User): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(user))
  } catch (error) {
    console.error('Failed to store user:', error)
  }
}

export function getAccessToken(): string | null {
  if (typeof window === 'undefined') return null
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

function storeToken(token: string): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(TOKEN_KEY, token)
  } catch (error) {
    console.error('Failed to store token:', error)
  }
}

function storeRefreshToken(refreshToken: string): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken)
  } catch (error) {
    console.error('Failed to store refresh token:', error)
  }
}

function clearAuth(): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.removeItem(STORAGE_KEY)
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(REFRESH_TOKEN_KEY)
  } catch (error) {
    console.error('Failed to clear auth:', error)
  }
}

// API functions
export async function login(credentials: LoginRequest): Promise<void> {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/v1/auth/login`, {
      username: credentials.username,
      password: credentials.password,
    })

    const { access_token, refresh_token, token_type } = response.data

    if (access_token) {
      storeToken(access_token)
    }
    if (refresh_token) {
      storeRefreshToken(refresh_token)
    }
  } catch (error: any) {
    const message = error.response?.data?.detail || 'Login failed'
    throw new Error(message)
  }
}

export async function register(data: RegisterRequest): Promise<User> {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/v1/auth/register`, {
      email: data.email,
      username: data.username,
      password: data.password,
      full_name: data.full_name,
      role: data.role || 'viewer',
    })

    const { access_token, refresh_token, user } = response.data

    if (access_token) {
      storeToken(access_token)
    }
    if (refresh_token) {
      storeRefreshToken(refresh_token)
    }

    return user
  } catch (error: any) {
    const message = error.response?.data?.detail || 'Registration failed'
    throw new Error(message)
  }
}

export function logout(): void {
  clearAuth()
}

export async function getCurrentUser(): Promise<User> {
  const token = getAccessToken()
  if (!token) {
    throw new Error('Not authenticated')
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/api/v1/auth/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    return response.data
  } catch (error: any) {
    clearAuth()
    const message = error.response?.data?.detail || 'Failed to get user'
    throw new Error(message)
  }
}

// API Key functions
export async function getAPIKeys(): Promise<APIKey[]> {
  const token = getAccessToken()
  if (!token) {
    throw new Error('Not authenticated')
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/api/v1/api-keys`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    // Backend returns an array directly, not an object with a 'keys' property
    if (Array.isArray(response.data)) {
      return response.data
    }
    // Fallback: check if it's an object with a 'keys' property (for backwards compatibility)
    if (response.data && typeof response.data === 'object' && Array.isArray(response.data.keys)) {
      return response.data.keys
    }
    return []
  } catch (error: any) {
    const message = error.response?.data?.detail || 'Failed to get API keys'
    throw new Error(message)
  }
}

export async function createAPIKey(name: string): Promise<APIKeyResponse> {
  const token = getAccessToken()
  if (!token) {
    throw new Error('Not authenticated')
  }

  try {
    const response = await axios.post(
      `${API_BASE_URL}/api/v1/api-keys`,
      { name },
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
    return response.data
  } catch (error: any) {
    const message = error.response?.data?.detail || 'Failed to create API key'
    throw new Error(message)
  }
}

export async function deleteAPIKey(id: number): Promise<void> {
  const token = getAccessToken()
  if (!token) {
    throw new Error('Not authenticated')
  }

  try {
    await axios.delete(`${API_BASE_URL}/api/v1/api-keys/${id}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
  } catch (error: any) {
    const message = error.response?.data?.detail || 'Failed to delete API key'
    throw new Error(message)
  }
}

