"""Health Check Endpoint.

This endpoint is used by clients, monitoring tools, and load balancers to
verify that the FastAPI backend service is operational.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Service Health Check")
def get_health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict: A JSON response containing `{"status": "ok"}`
    """
    return {"status": "ok"}
