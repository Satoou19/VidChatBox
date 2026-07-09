"""VidChatBox — FastAPI application entry point.

This file is intentionally minimal: it sets up the app, middleware,
mounts route modules, and serves static files.
All business logic lives in backend/routes/ and backend/services/.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from backend.pipeline.project_manager import create_project
from backend.rate_limiter import RateLimitMiddleware
from backend.routes import projects, ingest, export, chat

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

load_dotenv(override=True)

app = FastAPI(title="VidChatBox API", version="1.1.0")

# ------------------------------------------------------------------
# Middleware — Fix #5 (CORS) + Fix #6 (Rate Limit)
# ------------------------------------------------------------------

# CORS: configurable via ALLOWED_ORIGINS env var
_origins_env = os.getenv("ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if _allowed_origins else ["*"],
    allow_credentials=bool(_allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: configurable via RATE_LIMIT_PER_MINUTE env var
_rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
app.add_middleware(RateLimitMiddleware, calls_per_minute=_rate_limit)

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

app.include_router(projects.router)
app.include_router(ingest.router)
app.include_router(export.router)
app.include_router(chat.router)

# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------

create_project("default")

# ------------------------------------------------------------------
# Static files (must be last — catch-all mount)
# ------------------------------------------------------------------

os.makedirs("./frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
