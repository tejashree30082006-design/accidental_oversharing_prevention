"""Accidental Oversharing Detector - Backend API.

Main entry point for the FastAPI backend application.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import database initializer
from backend.app.core.database import init_db

# Import API routers
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.scan import router as scan_router
from backend.app.api.v1.protect import router as protect_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    # Initialize SQLite database and create tables if they do not exist
    init_db()
    yield


# Initialize the FastAPI application
app = FastAPI(
    title="Accidental Oversharing Detector API",
    description=(
        "Backend API for detecting, scoring, and protecting sensitive "
        "personal information (PII, EXIF/GPS, QR/barcodes) in user photos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
# This allows the Next.js frontend (Member 1) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local hackathon development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 endpoints
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(scan_router, prefix="/api/v1", tags=["Scan"])
app.include_router(protect_router, prefix="/api/v1", tags=["Protect"])


@app.get("/", summary="Root Status & Navigation")
def root() -> dict[str, str]:
    """Root endpoint providing links to API documentation and health status."""
    return {
        "message": "Welcome to the Accidental Oversharing Detector API",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
    }
