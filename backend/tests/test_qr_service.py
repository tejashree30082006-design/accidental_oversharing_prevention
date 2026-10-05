"""
Tests for qr_service.py

Covers:
  3. Image WITH a QR code  → QR_CODE finding returned
  4. Image WITHOUT QR      → empty findings list
  5. Bounding box format   → [x, y, w, h] with 4 floats (when present)
  6. Finding schema        → category / severity / source / action correct
  7. Robustness            → corrupt / empty bytes → no exception
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from backend.app.models.finding import FindingCategory, RecommendedAction, Severity, Source
from backend.app.services.qr_service import detect_qr_codes
from backend.tests.fixtures.image_factory import (
    make_image_with_qr,
    make_image_without_qr,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def img_with_qr() -> bytes:
    return make_image_with_qr("https://example.com/test-qr")


@pytest.fixture(scope="module")
def img_without_qr() -> bytes:
    return make_image_without_qr()


# ── Test 3: Image WITH QR code ───────────────────────────────────────────────

class TestQrDetectionPositive:
    """Image that contains a real QR code."""

    def test_at_least_one_finding_returned(self, img_with_qr):
        findings = detect_qr_codes(img_with_qr)
        assert len(findings) >= 1, (
            "Expected at least one QR finding for a QR-code image."
        )

    def test_finding_category_is_qr_code(self, img_with_qr):
        findings = detect_qr_codes(img_with_qr)
        assert any(f.category == FindingCategory.QR_CODE for f in findings), (
            "QR_CODE category not found in findings."
        )

    def test_finding_severity_is_medium(self, img_with_qr):
        findings = detect_qr_codes(img_with_qr)
        qr_findings = [f for f in findings if f.category == FindingCategory.QR_CODE]
        assert qr_findings, "No QR_CODE finding."
        assert qr_findings[0].severity == Severity.MEDIUM

    def test_finding_source_is_qr_detector(self, img_with_qr):
        findings = detect_qr_codes(img_with_qr)
        qr_findings = [f for f in findings if f.category == FindingCategory.QR_CODE]
        assert qr_findings[0].source == Source.QR_DETECTOR

    def test_finding_recommended_action_is_blur(self, img_with_qr):
        findings = detect_qr_codes(img_with_qr)
        qr_findings = [f for f in findings if f.category == FindingCategory.QR_CODE]
        assert qr_findings[0].recommended_action == RecommendedAction.BLUR

    def test_finding_confidence_is_between_0_and_1(self, img_with_qr):
        findings = detect_qr_codes(img_with_qr)
        for f in findings:
            assert 0.0 <= f.confidence <= 1.0, (
                f"confidence {f.confidence} is out of range."
            )

    def test_bounding_box_format_when_present(self, img_with_qr):
        """If a bbox is provided it must be [x, y, w, h] — 4 numeric values."""
        findings = detect_qr_codes(img_with_qr)
        for f in findings:
            if f.bbox is not None:
                assert isinstance(f.bbox, list), "bbox should be a list."
                assert len(f.bbox) == 4, "bbox should have exactly 4 elements [x,y,w,h]."
                for val in f.bbox:
                    assert isinstance(val, (int, float)), (
                        f"bbox value {val!r} is not numeric."
                    )
                # Width and height must be positive
                assert f.bbox[2] > 0, "bbox width must be > 0."
                assert f.bbox[3] > 0, "bbox height must be > 0."

    def test_finding_has_non_empty_description(self, img_with_qr):
        findings = detect_qr_codes(img_with_qr)
        for f in findings:
            assert f.description.strip(), "Finding description should not be empty."

    def test_finding_id_is_unique_per_call(self, img_with_qr):
        findings_a = detect_qr_codes(img_with_qr)
        findings_b = detect_qr_codes(img_with_qr)
        ids_a = {f.id for f in findings_a}
        ids_b = {f.id for f in findings_b}
        assert ids_a.isdisjoint(ids_b), "Finding IDs should be unique per invocation."

    def test_readable_qr_has_high_confidence(self, img_with_qr):
        """A readable QR should report confidence == 1.0."""
        findings = detect_qr_codes(img_with_qr)
        readable = [f for f in findings if f.confidence == 1.0]
        # At least one readable QR finding expected for a clean QR image.
        assert readable, (
            "Expected at least one finding with confidence=1.0 for a clean QR image."
        )


# ── Test 4: Image WITHOUT QR code ────────────────────────────────────────────

class TestQrDetectionNegative:
    """Solid-colour image with no QR code."""

    def test_no_findings_returned(self, img_without_qr):
        findings = detect_qr_codes(img_without_qr)
        assert findings == [], (
            f"Expected empty findings for non-QR image, got: {findings}"
        )

    def test_returns_list_type(self, img_without_qr):
        findings = detect_qr_codes(img_without_qr)
        assert isinstance(findings, list)


# ── Robustness tests ─────────────────────────────────────────────────────────

class TestQrRobustness:
    """Edge-case handling — service must never raise on bad input."""

    def test_empty_bytes_returns_empty_list(self):
        findings = detect_qr_codes(b"")
        assert isinstance(findings, list)
        assert findings == []

    def test_corrupt_bytes_returns_empty_list(self):
        findings = detect_qr_codes(b"\x89PNG\r\ngarbage-here")
        assert isinstance(findings, list)

    def test_plain_text_bytes_returns_empty_list(self):
        findings = detect_qr_codes(b"This is not an image.")
        assert isinstance(findings, list)
        assert findings == []
