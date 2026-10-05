"""Scan API Endpoints.

Handles image upload, privacy risk analysis, scoring, and report retrieval:
- POST /api/v1/scan: Upload and scan an image for privacy risks.
- GET /api/v1/scan/{scan_id}: Retrieve privacy report by scan ID.
"""

import io
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from PIL import Image
from sqlalchemy.orm import Session

from backend.app.core.config import BASE_DIR
from backend.app.core.database import get_db
from backend.app.models.scan_session import ScanSession
from backend.app.models.finding import Finding as DBFinding
from backend.app.schemas.finding import Finding, BoundingBox
from backend.app.schemas.scan import ScanResponse, RiskLevel
from backend.app.services.metadata_detector import scan_metadata_and_codes
from backend.app.services.ocr_detector import scan_image_text
from backend.app.services.gemma_reasoning import gemma_reasoner
from backend.app.services.risk_engine import deduplicate_findings, calculate_risk_score

router = APIRouter()

# Temporary upload cache directory (never permanent storage)
UPLOAD_TEMP_DIR = BASE_DIR / "temp_uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/scan", response_model=ScanResponse, status_code=status.HTTP_201_CREATED)
async def scan_image(
    file: UploadFile = File(..., description="Image file to analyze for privacy risks"),
    db: Session = Depends(get_db),
) -> ScanResponse:
    """Upload a photo to detect sensitive information, hidden metadata, and calculate risk score.

    Workflow:
    1. Read and validate uploaded image.
    2. Scan for EXIF GPS metadata, QR codes, and barcodes (Member 3).
    3. Scan for visible PII via OCR (Member 2).
    4. Refine ambiguous findings using Gemma 4 contextual reasoning.
    5. Deduplicate findings to prevent double-counting.
    6. Compute risk score (0-100) and risk level (LOW, MEDIUM, HIGH).
    7. Persist session metadata and findings to SQLite.
    8. Return structured privacy report.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image (JPEG, PNG, WEBP, etc.)",
        )

    file_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        pil_image.load()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image format: {exc}",
        )

    session_id = f"scan-{uuid.uuid4().hex[:12]}"

    # Save temporary image copy for the subsequent /protect step
    temp_path = UPLOAD_TEMP_DIR / f"{session_id}.jpg"
    try:
        # Save a working copy in temp_uploads
        pil_image.convert("RGB").save(temp_path, format="JPEG")
    except Exception:
        pass

    # 1. Member 3: EXIF metadata, GPS, QR codes, barcodes
    metadata_findings = scan_metadata_and_codes(pil_image)

    # 2. Member 2: OCR text and PII detection
    ocr_findings = scan_image_text(pil_image)

    # 3. Member 4 / Gemma 4: Contextual reasoning refinement
    raw_findings = metadata_findings + ocr_findings
    refined_findings = gemma_reasoner.refine_findings(raw_findings)

    # 4. Member 4: Deduplicate findings (avoid double counting)
    unique_findings = deduplicate_findings(refined_findings)

    # 5. Member 4: Compute privacy risk score
    risk_score, risk_level, summary = calculate_risk_score(unique_findings)

    # 6. Member 4: Save scan session to SQLite
    db_session = ScanSession(
        session_id=session_id,
        risk_score=risk_score,
        risk_level=risk_level.value,
        total_findings=len(unique_findings),
        protected=False,
    )
    db.add(db_session)

    # 7. Persist individual findings (only metadata, never raw secrets)
    for f in unique_findings:
        bx = f.bbox.x if f.bbox else None
        by = f.bbox.y if f.bbox else None
        bw = f.bbox.width if f.bbox else None
        bh = f.bbox.height if f.bbox else None

        db_finding = DBFinding(
            finding_id=f.id,
            session_id=session_id,
            category=f.category.value,
            severity=f.severity.value,
            confidence=f.confidence,
            description=f.description,
            source=f.source.value,
            x=bx,
            y=by,
            width=bw,
            height=bh,
            recommended_action=f.recommended_action.value,
        )
        db.add(db_finding)

    db.commit()
    db.refresh(db_session)

    return ScanResponse(
        session_id=db_session.session_id,
        created_at=db_session.created_at,
        risk_score=db_session.risk_score,
        risk_level=RiskLevel(db_session.risk_level),
        total_findings=db_session.total_findings,
        findings=unique_findings,
        protected=db_session.protected,
        summary=summary,
    )


@router.get("/scan/{scan_id}", response_model=ScanResponse)
def get_scan_results(
    scan_id: str,
    db: Session = Depends(get_db),
) -> ScanResponse:
    """Retrieve existing scan results and privacy report from SQLite.

    Args:
        scan_id: The session ID of the scan.
        db: Database session.

    Returns:
        ScanResponse: The stored scan report and findings.
    """
    db_session = (
        db.query(ScanSession)
        .filter(ScanSession.session_id == scan_id)
        .first()
    )

    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan session with ID '{scan_id}' not found.",
        )

    # Reconstruct Pydantic findings from database models
    schema_findings: list[Finding] = []
    for df in db_session.findings:
        bbox = None
        if df.x is not None and df.y is not None and df.width is not None and df.height is not None:
            bbox = BoundingBox(x=df.x, y=df.y, width=df.width, height=df.height)

        schema_findings.append(
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

    _, level, summary = calculate_risk_score(schema_findings)

    return ScanResponse(
        session_id=db_session.session_id,
        created_at=db_session.created_at,
        risk_score=db_session.risk_score,
        risk_level=RiskLevel(db_session.risk_level),
        total_findings=db_session.total_findings,
        findings=schema_findings,
        protected=db_session.protected,
        summary=summary,
    )
