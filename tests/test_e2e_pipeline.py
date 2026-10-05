"""End-to-End Pipeline Integration Tests.

Validates the full privacy detection and protection lifecycle:
IMAGE -> SCAN -> IDENTIFY -> WARN -> PROTECT -> DOWNLOAD SAFE IMAGE

Tested scenarios:
1. Clean image
2. Phone number / Email detection
3. QR code detection
4. Barcode detection
5. GPS EXIF metadata detection
6. Multiple risks image (QR + GPS)
7. Gemma 4 contextual reasoning
8. Protection & Safe Image download without GPS metadata
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.core.config import BASE_DIR
from backend.app.core.database import init_db
from backend.app.services.ocr_detector import detect_pii_in_text
from backend.app.services.gemma_reasoning import gemma_reasoner
from backend.app.schemas.finding import Finding, FindingCategory, FindingSeverity, FindingSource, RecommendedAction

client = TestClient(app)
SAMPLE_DIR = BASE_DIR / "sample_images"


@pytest.fixture(scope="module", autouse=True)
def setup_test_suite():
    """Ensure database tables and sample images are prepared."""
    init_db()


# -------------------------------------------------------------
# Scenario 1: Clean Image
# -------------------------------------------------------------
def test_e2e_clean_image():
    """Verify clean image produces score 0 and LOW risk."""
    clean_path = SAMPLE_DIR / "clean_sample.jpg"
    with open(clean_path, "rb") as f:
        response = client.post(
            "/api/v1/scan",
            files={"file": ("clean_sample.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["risk_score"] == 0
    assert data["risk_level"] == "LOW"
    assert data["total_findings"] == 0
    assert len(data["findings"]) == 0


# -------------------------------------------------------------
# Scenario 2: GPS Metadata Detection & Privacy Adherence
# -------------------------------------------------------------
def test_e2e_gps_metadata():
    """Verify GPS EXIF detection without leaking raw coordinates."""
    gps_path = SAMPLE_DIR / "gps_sample.jpg"
    with open(gps_path, "rb") as f:
        response = client.post(
            "/api/v1/scan",
            files={"file": ("gps_sample.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["risk_score"] == 35
    assert data["risk_level"] == "MEDIUM"
    assert data["total_findings"] == 1

    gps_finding = data["findings"][0]
    assert gps_finding["category"] == "GPS_LOCATION"
    assert gps_finding["source"] == "EXIF"
    assert gps_finding["recommended_action"] == "REMOVE_METADATA"
    # Ensure raw coordinates are not in description
    assert "37.77" not in gps_finding["description"]


# -------------------------------------------------------------
# Scenario 3: QR Code Detection
# -------------------------------------------------------------
def test_e2e_qr_code():
    """Verify QR code detection and bounding box extraction."""
    qr_path = SAMPLE_DIR / "qr_sample.jpg"
    with open(qr_path, "rb") as f:
        response = client.post(
            "/api/v1/scan",
            files={"file": ("qr_sample.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["risk_score"] >= 25
    assert data["total_findings"] >= 1

    qr_finding = next(f for f in data["findings"] if f["category"] == "QR_CODE")
    assert qr_finding["source"] == "QR"
    assert qr_finding["bbox"] is not None
    assert qr_finding["bbox"]["width"] > 0
    assert qr_finding["recommended_action"] == "BLUR"


# -------------------------------------------------------------
# Scenario 4: Multiple Risks (QR + GPS) -> HIGH Risk Level
# -------------------------------------------------------------
def test_e2e_multi_risk_and_protection_workflow():
    """Verify multi-risk image scoring, retrieval, protection, and safe download."""
    multi_path = SAMPLE_DIR / "multi_risk_sample.jpg"

    # Step 1: Scan
    with open(multi_path, "rb") as f:
        scan_res = client.post(
            "/api/v1/scan",
            files={"file": ("multi_risk_sample.jpg", f, "image/jpeg")},
        )

    assert scan_res.status_code == 201
    scan_data = scan_res.json()
    session_id = scan_data["session_id"]
    # 35 (GPS) + 25 (QR) = 60 -> HIGH
    assert scan_data["risk_score"] >= 60
    assert scan_data["risk_level"] == "HIGH"
    assert scan_data["total_findings"] >= 2

    # Step 2: Retrieve Scan
    get_res = client.get(f"/api/v1/scan/{session_id}")
    assert get_res.status_code == 200
    assert get_res.json()["session_id"] == session_id

    # Step 3: Protect
    protect_res = client.post(
        "/api/v1/protect",
        data={"session_id": session_id},
    )
    assert protect_res.status_code == 200
    protect_data = protect_res.json()
    assert protect_data["protected_risk_score"] == 0
    assert protect_data["protected_risk_level"] == "LOW"
    assert len(protect_data["actions_applied"]) >= 2
    assert "Stripped all EXIF metadata and GPS coordinates" in " ".join(protect_data["actions_applied"])

    # Step 4: Download Safe Image
    download_res = client.get(protect_data["download_url"])
    assert download_res.status_code == 200
    assert download_res.headers["content-type"] == "image/jpeg"

    # Inspect downloaded bytes: must contain zero EXIF metadata
    downloaded_img = Image.open(io.BytesIO(download_res.content))
    assert len(downloaded_img.getexif()) == 0


# -------------------------------------------------------------
# Scenario 5: PII Regex & Text Detection
# -------------------------------------------------------------
def test_pii_detection_logic():
    """Verify text patterns for phone numbers, emails, cards, and passports."""
    sample_text = (
        "Customer: Jane Doe, Email: jane.doe@example.org, "
        "Phone: +1 (555) 723-9912, Card: 4111 2222 3333 4444, "
        "Passport: USA98765432, Ref: PNR#ABC123"
    )

    findings = detect_pii_in_text(sample_text)
    categories = [f.category for f in findings]

    assert FindingCategory.EMAIL in categories
    assert FindingCategory.PHONE_NUMBER in categories
    assert FindingCategory.CARD_NUMBER in categories
    assert FindingCategory.PASSPORT_NUMBER in categories
    assert FindingCategory.BOOKING_REFERENCE in categories


# -------------------------------------------------------------
# Scenario 6: Gemma 4 Contextual Reasoning
# -------------------------------------------------------------
def test_gemma_contextual_reasoning():
    """Verify Gemma contextual reasoning upgrades room numbers and refines seats."""
    # Finding candidate with ambiguous room number
    room_finding = Finding(
        id="f-room",
        category=FindingCategory.OTHER,
        severity=FindingSeverity.LOW,
        confidence=0.7,
        description="Ambiguous number 1208",
        source=FindingSource.OCR,
        recommended_action=RecommendedAction.BLUR,
    )

    context = "KEYCARD ACCESS - HOTEL LUXE - ROOM NO: 1208"
    refined = gemma_reasoner.reason_about_finding(room_finding, context)

    assert refined is not None
    assert refined.category == FindingCategory.ROOM_NUMBER
    assert refined.severity == FindingSeverity.MEDIUM
    assert refined.source == FindingSource.GEMMA
    assert "room number" in refined.description.lower()
