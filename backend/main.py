"""VidChatBox — FastAPI application entry point.

This file is intentionally minimal: it sets up the app, middleware,
mounts route modules, and serves static files.
All business logic lives in backend/routes/ and backend/services/.
"""

import os
import sys
from dotenv import load_dotenv

# ------------------------------------------------------------------
# PyInstaller Helper for Desktop Mode
# ------------------------------------------------------------------
def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Resolve to root workspace directory (parent of backend/)
        base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_path, relative_path)

if getattr(sys, 'frozen', False):
    # Change current working directory to the directory of the executable
    os.chdir(os.path.dirname(sys.executable))

# Load environment variables from the CWD (now set to exe directory if frozen)
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

frontend_dir = get_resource_path("frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
