"""Simple rate limiting ASGI middleware for FastAPI.

Uses a sliding window counter per client IP. No external dependencies required.
Status polling and static file endpoints are exempt from rate limiting.
"""

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Paths that are exempt from rate limiting (status polling + static assets)
EXEMPT_PREFIXES = (
    "/api/status/",
    "/api/batch-status/",
    "/api/videos/",          # Covers /api/videos/{id}/polish-status too
    "/css/",
    "/js/",
    "/index.html",
    "/favicon",
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter.

    Args:
        app: The ASGI application.
        calls_per_minute: Maximum number of requests per IP per minute.
            Set to 0 to disable rate limiting entirely.
    """

    def __init__(self, app, calls_per_minute=30):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.window = 60  # seconds
        self._requests = defaultdict(list)  # ip -> [timestamps]

    async def dispatch(self, request, call_next):
        # Disabled
        if self.calls_per_minute <= 0:
            return await call_next(request)

        path = request.url.path

        # Skip root page and exempt paths
        if path == "/" or any(path.startswith(p) for p in EXEMPT_PREFIXES):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Sliding window: remove timestamps older than the window
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if now - t < self.window]

        if len(self._requests[client_ip]) >= self.calls_per_minute:
            retry_after = int(self.window - (now - self._requests[client_ip][0])) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": str(retry_after)},
            )

        self._requests[client_ip].append(now)

        # Periodic cleanup: remove IPs with no recent activity (every ~100 requests)
        if len(self._requests) > 100:
            stale_ips = [
                ip for ip, ts in self._requests.items()
                if not ts or (now - ts[-1]) > self.window * 2
            ]
            for ip in stale_ips:
                del self._requests[ip]

        return await call_next(request)
