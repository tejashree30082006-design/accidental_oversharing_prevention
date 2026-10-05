"""Unit tests for the Health Check API."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

# Initialize test client
client = TestClient(app)


def test_health_check_returns_ok():
    """Verify that GET /api/v1/health returns status 200 and {'status': 'ok'}."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint_returns_welcome():
    """Verify that GET / returns status 200 with navigation information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "health_check" in data
    assert data["health_check"] == "/api/v1/health"
