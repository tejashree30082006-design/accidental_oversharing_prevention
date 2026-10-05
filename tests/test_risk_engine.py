"""Unit tests for the Privacy Risk Scoring Engine."""

import pytest
from backend.app.schemas.finding import (
    Finding,
    FindingCategory,
    FindingSeverity,
    FindingSource,
    RecommendedAction,
    BoundingBox,
)
from backend.app.schemas.scan import RiskLevel
from backend.app.services.risk_engine import (
    calculate_risk_score,
    deduplicate_findings,
    CATEGORY_WEIGHTS,
)


def test_clean_image_score():
    """Verify clean image with no findings produces score 0 and LOW risk."""
    score, level, summary = calculate_risk_score([])
    assert score == 0
    assert level == RiskLevel.LOW
    assert "No sensitive personal information" in summary


def test_single_gps_finding_score():
    """GPS location has weight 35, which yields MEDIUM risk (30-59)."""
    gps_finding = Finding(
        id="f-gps",
        category=FindingCategory.GPS_LOCATION,
        severity=FindingSeverity.HIGH,
        confidence=1.0,
        description="GPS coordinates detected in EXIF",
        source=FindingSource.EXIF,
        recommended_action=RecommendedAction.REMOVE_METADATA,
    )
    score, level, _ = calculate_risk_score([gps_finding])
    assert score == 35
    assert level == RiskLevel.MEDIUM


def test_multiple_findings_high_risk_and_cap():
    """Multiple findings sum up and cap at 100 with HIGH risk."""
    findings = [
        Finding(
            id="f-gps",
            category=FindingCategory.GPS_LOCATION,
            severity=FindingSeverity.HIGH,
            confidence=1.0,
            description="GPS coordinates",
            source=FindingSource.EXIF,
            recommended_action=RecommendedAction.REMOVE_METADATA,
        ),
        Finding(
            id="f-card",
            category=FindingCategory.CARD_NUMBER,
            severity=FindingSeverity.HIGH,
            confidence=0.98,
            description="Credit card number",
            source=FindingSource.OCR,
            bbox=BoundingBox(x=10, y=10, width=100, height=20),
            recommended_action=RecommendedAction.REDACT,
        ),
        Finding(
            id="f-pass",
            category=FindingCategory.PASSPORT_NUMBER,
            severity=FindingSeverity.HIGH,
            confidence=0.95,
            description="Passport number",
            source=FindingSource.OCR,
            bbox=BoundingBox(x=10, y=50, width=120, height=25),
            recommended_action=RecommendedAction.REDACT,
        ),
        Finding(
            id="f-qr",
            category=FindingCategory.QR_CODE,
            severity=FindingSeverity.HIGH,
            confidence=0.99,
            description="QR code",
            source=FindingSource.QR,
            bbox=BoundingBox(x=200, y=200, width=80, height=80),
            recommended_action=RecommendedAction.BLUR,
        ),
    ]
    # 35 + 30 + 30 + 25 = 120 -> capped at 100
    score, level, _ = calculate_risk_score(findings)
    assert score == 100
    assert level == RiskLevel.HIGH


def test_deduplication_prevents_double_counting():
    """Ensure duplicate findings (e.g. OCR phone number + Gemma confirmation) count once."""
    ocr_phone = Finding(
        id="f-ocr-phone",
        category=FindingCategory.PHONE_NUMBER,
        severity=FindingSeverity.MEDIUM,
        confidence=0.88,
        description="Phone number detected",
        source=FindingSource.OCR,
        bbox=BoundingBox(x=50, y=100, width=150, height=30),
        recommended_action=RecommendedAction.BLUR,
    )
    gemma_phone = Finding(
        id="f-gemma-phone",
        category=FindingCategory.PHONE_NUMBER,
        severity=FindingSeverity.HIGH,
        confidence=0.97,
        description="Confirmed mobile number with country code",
        source=FindingSource.GEMMA,
        bbox=BoundingBox(x=52, y=101, width=148, height=29),  # Heavily overlapping bbox
        recommended_action=RecommendedAction.BLUR,
    )

    deduped = deduplicate_findings([ocr_phone, gemma_phone])
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.97
    assert deduped[0].severity == FindingSeverity.HIGH

    score, level, _ = calculate_risk_score(deduped)
    assert score == 15  # PHONE_NUMBER weight is 15, not 30
    assert level == RiskLevel.LOW
