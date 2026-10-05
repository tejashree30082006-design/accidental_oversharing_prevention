"""
Tests for metadata_service.py

Covers:
  1. Image WITH GPS EXIF        → GPS_LOCATION finding (HIGH severity)
  2. Image WITHOUT EXIF         → empty findings list
  3. Additional EXIF fields     → EXIF_METADATA findings (MEDIUM severity)
  4. GPS masking                → coordinates rounded to ≤ 2 d.p.
  5. Invalid / corrupt bytes    → graceful empty list (no exception)
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on the Python path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from backend.app.models.finding import FindingCategory, RecommendedAction, Severity, Source
from backend.app.services.metadata_service import analyse_metadata
from backend.tests.fixtures.image_factory import (
    make_image_with_gps_exif,
    make_image_without_exif,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def img_with_gps() -> bytes:
    return make_image_with_gps_exif()


@pytest.fixture(scope="module")
def img_without_exif() -> bytes:
    return make_image_without_exif()


# ── Test 1: Image WITH EXIF containing GPS ───────────────────────────────────

class TestMetadataWithGps:
    """Image that contains GPS EXIF data."""

    def test_at_least_one_finding_returned(self, img_with_gps):
        findings = analyse_metadata(img_with_gps)
        assert len(findings) >= 1, "Expected at least one finding for GPS image."

    def test_gps_finding_present(self, img_with_gps):
        findings = analyse_metadata(img_with_gps)
        categories = [f.category for f in findings]
        assert FindingCategory.GPS_LOCATION in categories, (
            "GPS_LOCATION finding should be present."
        )

    def test_gps_finding_severity_is_high(self, img_with_gps):
        findings = analyse_metadata(img_with_gps)
        gps_findings = [f for f in findings if f.category == FindingCategory.GPS_LOCATION]
        assert gps_findings, "No GPS_LOCATION finding found."
        assert gps_findings[0].severity == Severity.HIGH

    def test_gps_finding_confidence_is_one(self, img_with_gps):
        findings = analyse_metadata(img_with_gps)
        gps_findings = [f for f in findings if f.category == FindingCategory.GPS_LOCATION]
        assert gps_findings[0].confidence == 1.0

    def test_gps_finding_source_is_exif(self, img_with_gps):
        findings = analyse_metadata(img_with_gps)
        gps_findings = [f for f in findings if f.category == FindingCategory.GPS_LOCATION]
        assert gps_findings[0].source == Source.EXIF

    def test_gps_finding_recommended_action(self, img_with_gps):
        findings = analyse_metadata(img_with_gps)
        gps_findings = [f for f in findings if f.category == FindingCategory.GPS_LOCATION]
        assert gps_findings[0].recommended_action == RecommendedAction.REMOVE_METADATA

    def test_gps_finding_has_no_bbox(self, img_with_gps):
        findings = analyse_metadata(img_with_gps)
        gps_findings = [f for f in findings if f.category == FindingCategory.GPS_LOCATION]
        assert gps_findings[0].bbox is None, "GPS finding should have no bounding box."

    def test_gps_coordinates_are_masked_in_description(self, img_with_gps):
        """
        GPS coordinates exposed in the description must be masked to at
        most 2 decimal places (≈ 1 km resolution).
        """
        findings = analyse_metadata(img_with_gps)
        gps_findings = [f for f in findings if f.category == FindingCategory.GPS_LOCATION]
        if not gps_findings:
            pytest.skip("No GPS finding to check.")
        desc = gps_findings[0].description
        # Extract any float-looking tokens from the description
        import re
        floats_in_desc = re.findall(r"-?\d+\.\d+", desc)
        for f in floats_in_desc:
            decimal_places = len(f.split(".")[1])
            assert decimal_places <= 2, (
                f"GPS coordinate {f!r} in description has more than 2 decimal "
                "places — potential privacy leak."
            )

    def test_gps_finding_has_unique_id(self, img_with_gps):
        findings_a = analyse_metadata(img_with_gps)
        findings_b = analyse_metadata(img_with_gps)
        ids_a = {f.id for f in findings_a}
        ids_b = {f.id for f in findings_b}
        # Each call should produce fresh UUIDs
        assert ids_a.isdisjoint(ids_b), "Finding IDs should be unique per invocation."

    def test_additional_exif_fields_detected(self, img_with_gps):
        """DateTimeOriginal / Make / Model etc. should surface as EXIF_METADATA findings."""
        findings = analyse_metadata(img_with_gps)
        exif_findings = [
            f for f in findings if f.category == FindingCategory.EXIF_METADATA
        ]
        assert len(exif_findings) >= 1, (
            "Expected EXIF_METADATA findings for camera make/model/software fields."
        )

    def test_exif_metadata_findings_severity_is_medium(self, img_with_gps):
        findings = analyse_metadata(img_with_gps)
        for f in findings:
            if f.category == FindingCategory.EXIF_METADATA:
                assert f.severity == Severity.MEDIUM


# ── Test 2: Image WITHOUT EXIF ───────────────────────────────────────────────

class TestMetadataWithoutExif:
    """Plain image with no EXIF metadata at all."""

    def test_no_findings_returned(self, img_without_exif):
        findings = analyse_metadata(img_without_exif)
        assert findings == [], (
            f"Expected empty findings list for no-EXIF image, got: {findings}"
        )

    def test_returns_list_type(self, img_without_exif):
        findings = analyse_metadata(img_without_exif)
        assert isinstance(findings, list)


# ── Test 3: Robustness ───────────────────────────────────────────────────────

class TestMetadataRobustness:
    """Edge-case handling — service must never raise on bad input."""

    def test_empty_bytes_returns_empty_list(self):
        findings = analyse_metadata(b"")
        assert isinstance(findings, list)
        assert findings == []

    def test_corrupt_bytes_returns_empty_list(self):
        findings = analyse_metadata(b"\xff\xfe\x00\x01garbage")
        assert isinstance(findings, list)
        assert findings == []

    def test_non_image_text_returns_empty_list(self):
        findings = analyse_metadata(b"Hello, world!")
        assert isinstance(findings, list)
        assert findings == []
