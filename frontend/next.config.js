const path = require('path')

// Normalize API URL - remove trailing slashes
const normalizeUrl = (url) => {
  return url ? url.replace(/\/+$/, '') : url;
};

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['reactflow', '@reactflow/core', '@reactflow/controls', '@reactflow/background', '@reactflow/minimap'],
  env: {
    NEXT_PUBLIC_API_URL: normalizeUrl(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'),
    NEXT_PUBLIC_WS_URL: (() => {
      if (process.env.NEXT_PUBLIC_WS_URL) {
        return normalizeUrl(process.env.NEXT_PUBLIC_WS_URL);
      }
      const apiUrl = normalizeUrl(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');
      return apiUrl.replace(/^http/, 'ws');
    })(),
  },
  // Webpack alias configuration for path resolution (required for Vercel builds)
  webpack: (config) => {
    // Set up alias - ensure @ points to the frontend directory
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': path.resolve(__dirname),
    };
    
    // Ensure reactflow is properly resolved
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
    };

    return config;
  },
  async rewrites() {
    // Only rewrite in development, in production use direct API calls
    if (process.env.NODE_ENV === 'development') {
      const apiUrl = normalizeUrl(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');
      return [
        {
          source: '/api/:path*',
          destination: `${apiUrl}/api/:path*`,
        },
      ];
    }
    return [];
  },
}

module.exports = nextConfig

