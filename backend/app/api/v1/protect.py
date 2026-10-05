"""Image Protection API Endpoints.

Handles remediation and protection of analyzed photos:
- POST /api/v1/protect: Creates a safe version with blurred sensitive regions and stripped EXIF.
- GET /api/v1/protect/download/{session_id}: Downloads the safe protected image.
"""

import io
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy.orm import Session

from backend.app.core.config import BASE_DIR
from backend.app.core.database import get_db
from backend.app.models.scan_session import ScanSession
from backend.app.schemas.finding import Finding, BoundingBox
from backend.app.schemas.scan import ProtectResponse
from backend.app.services.protection_service import (
    protect_image,
    save_protected_image,
    PROCESSED_IMAGE_DIR,
)

router = APIRouter()

UPLOAD_TEMP_DIR = BASE_DIR / "temp_uploads"


@router.post("/protect", response_model=ProtectResponse)
async def protect_photo(
    session_id: str = Form(..., description="ID of the scan session to protect"),
    file: Optional[UploadFile] = File(None, description="Optional image upload if not cached"),
    db: Session = Depends(get_db),
) -> ProtectResponse:
    """Create a protected version of the photo for a given scan session.

    Actions executed:
    1. Blurs sensitive text regions (phone numbers, addresses, room numbers).
    2. Blurs QR codes and barcodes.
    3. Redacts high-severity numbers (payment cards, passports).
    4. Strips all EXIF metadata and GPS coordinates.
    5. Saves safe image and marks session as protected.
    """
    db_session = (
        db.query(ScanSession)
        .filter(ScanSession.session_id == session_id)
        .first()
    )

    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan session '{session_id}' not found.",
        )

    # Reconstruct findings from database
    findings: list[Finding] = []
    for df in db_session.findings:
        bbox = None
        if df.x is not None and df.y is not None and df.width is not None and df.height is not None:
            bbox = BoundingBox(x=df.x, y=df.y, width=df.width, height=df.height)

        findings.append(
            Finding(
                id=df.finding_id or f"finding-{df.id}",
                category=df.category,
                severity=df.severity,
                confidence=df.confidence,
                description=df.description,
                source=df.source,
                bbox=bbox,
                recommended_action=df.recommended_action,
            )
        )

    # Obtain the image to protect: either from the uploaded file or the temporary cache
    pil_image: Optional[Image.Image] = None

    if file is not None and file.filename:
        file_bytes = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(file_bytes))
            pil_image.load()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded image is invalid: {exc}",
            )
    else:
        temp_path = UPLOAD_TEMP_DIR / f"{session_id}.jpg"
        if temp_path.exists():
            try:
                pil_image = Image.open(temp_path)
                pil_image.load()
            except Exception:
                pass

    if pil_image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Original image is no longer in temporary cache. "
                "Please provide the image file in the 'file' form field."
            ),
        )

    # Apply protection: blur, redaction, and metadata stripping
    protected_pil, actions = protect_image(pil_image, findings)

    # Save protected image
    saved_path = save_protected_image(protected_pil, session_id)

    # Update database record
    db_session.protected = True
    db.commit()

    download_url = f"/api/v1/protect/download/{session_id}"

    return ProtectResponse(
        session_id=session_id,
        original_risk_score=db_session.risk_score,
        original_risk_level=db_session.risk_level,
        protected_risk_score=0,
        protected_risk_level="LOW",
        actions_applied=actions,
        protected_image_path=str(saved_path),
        download_url=download_url,
        message="Image successfully protected and sanitized of all privacy risks.",
    )


@router.get("/protect/download/{session_id}")
def download_protected_image(session_id: str) -> FileResponse:
    """Download the protected, safe version of an image.

    Args:
        session_id: Session identifier.

    Returns:
        FileResponse: Clean image download attachment.
    """
    protected_file = PROCESSED_IMAGE_DIR / f"protected_{session_id}.jpg"
    if not protected_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Protected image for session '{session_id}' not found.",
        )

    return FileResponse(
        path=protected_file,
        media_type="image/jpeg",
        filename=f"safe_{session_id}.jpg",
    )
