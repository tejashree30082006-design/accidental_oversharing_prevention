"""
Tests for barcode_service.py

Covers:
  5. Image WITH a barcode   → BARCODE finding returned
  6. Image WITHOUT barcode  → empty findings list
  7. Bounding box format    → [x, y, w, h] with 4 floats
  8. Finding schema         → category / severity / source / action correct
  9. pyzbar unavailable     → graceful empty list (import-guard tested)
  10. Robustness            → corrupt / empty bytes → no exception
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from backend.app.models.finding import FindingCategory, RecommendedAction, Severity, Source
from backend.app.services.barcode_service import detect_barcodes, _PYZBAR_AVAILABLE
from backend.tests.fixtures.image_factory import (
    make_image_with_barcode,
    make_image_without_barcode,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def img_with_barcode() -> bytes:
    return make_image_with_barcode("1234567890")


@pytest.fixture(scope="module")
def img_without_barcode() -> bytes:
    return make_image_without_barcode()


# ── Skip marker when pyzbar is not available ─────────────────────────────────
requires_pyzbar = pytest.mark.skipif(
    not _PYZBAR_AVAILABLE,
    reason="pyzbar / libzbar not available on this platform.",
)


# ── Test 5: Image WITH barcode ───────────────────────────────────────────────

@requires_pyzbar
class TestBarcodeDetectionPositive:
    """Image that contains a Code 128 barcode."""

    def test_at_least_one_finding_returned(self, img_with_barcode):
        findings = detect_barcodes(img_with_barcode)
        assert len(findings) >= 1, (
            "Expected at least one barcode finding."
        )

    def test_finding_category_is_barcode_or_qr(self, img_with_barcode):
        """Category must be BARCODE (or QR_CODE for QR-type barcodes)."""
        findings = detect_barcodes(img_with_barcode)
        valid_cats = {FindingCategory.BARCODE, FindingCategory.QR_CODE}
        for f in findings:
            assert f.category in valid_cats, (
                f"Unexpected category {f.category!r}."
            )

    def test_primary_finding_category_is_barcode(self, img_with_barcode):
        """A Code 128 barcode should produce a BARCODE category finding."""
        findings = detect_barcodes(img_with_barcode)
        barcode_findings = [f for f in findings if f.category == FindingCategory.BARCODE]
        assert barcode_findings, "Expected at least one BARCODE-category finding."

    def test_finding_severity_is_medium(self, img_with_barcode):
        findings = detect_barcodes(img_with_barcode)
        for f in findings:
            assert f.severity == Severity.MEDIUM, (
                f"Expected MEDIUM severity, got {f.severity!r}."
            )

    def test_finding_source_is_barcode_detector(self, img_with_barcode):
        findings = detect_barcodes(img_with_barcode)
        for f in findings:
            assert f.source == Source.BARCODE_DETECTOR

    def test_finding_recommended_action_is_blur(self, img_with_barcode):
        findings = detect_barcodes(img_with_barcode)
        for f in findings:
            assert f.recommended_action == RecommendedAction.BLUR

    def test_finding_confidence_is_one(self, img_with_barcode):
        findings = detect_barcodes(img_with_barcode)
        for f in findings:
            assert f.confidence == 1.0, (
                f"pyzbar detections should have confidence 1.0, got {f.confidence}."
            )

    def test_bounding_box_format(self, img_with_barcode):
        """bbox must be [x, y, w, h] — 4 positive numeric values."""
        findings = detect_barcodes(img_with_barcode)
        for f in findings:
            assert f.bbox is not None, "Barcode finding should include a bounding box."
            assert isinstance(f.bbox, list), "bbox should be a list."
            assert len(f.bbox) == 4, "bbox must have exactly 4 elements [x, y, w, h]."
            for val in f.bbox:
                assert isinstance(val, (int, float)), f"bbox value {val!r} not numeric."
            # width and height must be positive
            assert f.bbox[2] > 0, "bbox width (index 2) must be > 0."
            assert f.bbox[3] > 0, "bbox height (index 3) must be > 0."

    def test_finding_id_is_unique_per_call(self, img_with_barcode):
        findings_a = detect_barcodes(img_with_barcode)
        findings_b = detect_barcodes(img_with_barcode)
        ids_a = {f.id for f in findings_a}
        ids_b = {f.id for f in findings_b}
        assert ids_a.isdisjoint(ids_b), "Finding IDs must be unique per invocation."

    def test_finding_has_non_empty_description(self, img_with_barcode):
        findings = detect_barcodes(img_with_barcode)
        for f in findings:
            assert f.description.strip(), "Finding description should not be empty."


# ── Test 6: Image WITHOUT barcode ────────────────────────────────────────────

class TestBarcodeDetectionNegative:
    """Solid-colour image containing no barcode."""

    def test_no_findings_for_plain_image(self, img_without_barcode):
        findings = detect_barcodes(img_without_barcode)
        assert findings == [], (
            f"Expected empty findings for barcode-free image, got: {findings}"
        )

    def test_returns_list_type(self, img_without_barcode):
        findings = detect_barcodes(img_without_barcode)
        assert isinstance(findings, list)


# ── pyzbar unavailability handling ───────────────────────────────────────────

class TestBarcodeGracefulDegradation:
    """Service must return an empty list if pyzbar is unavailable."""

    def test_returns_list_when_pyzbar_unavailable(self, monkeypatch, img_with_barcode):
        """Simulate a missing pyzbar by patching the availability flag."""
        import backend.app.services.barcode_service as svc
        original = svc._PYZBAR_AVAILABLE
        try:
            svc._PYZBAR_AVAILABLE = False
            findings = detect_barcodes(img_with_barcode)
            assert isinstance(findings, list)
            assert findings == [], (
                "Should return empty list when pyzbar is marked unavailable."
            )
        finally:
            svc._PYZBAR_AVAILABLE = original


# ── Robustness ───────────────────────────────────────────────────────────────

class TestBarcodeRobustness:
    """Edge-case handling — service must never raise on bad input."""

    def test_empty_bytes_returns_empty_list(self):
        findings = detect_barcodes(b"")
        assert isinstance(findings, list)
        assert findings == []

    def test_corrupt_bytes_returns_empty_list(self):
        findings = detect_barcodes(b"\xff\xd8garbage")
        assert isinstance(findings, list)

    def test_plain_text_bytes_returns_empty_list(self):
        findings = detect_barcodes(b"This is not an image.")
        assert isinstance(findings, list)
        assert findings == []
