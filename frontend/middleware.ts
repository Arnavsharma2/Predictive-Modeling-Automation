import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * Middleware for route protection.
 *
 * Note: Since we use localStorage for token storage (client-side only),
 * we cannot check authentication status in middleware (which runs server-side).
 * Route protection is handled client-side in individual page components using
 * the useAuth hook and useEffect redirects.
 *
 * This middleware is kept for future cookie-based auth if needed.
 */

export function middleware(request: NextRequest) {
  // Pass through all requests - auth is handled client-side
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
};
