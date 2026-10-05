"""
Metadata detection service.

Uses Pillow (PIL) to extract and analyse EXIF metadata from images.
Detects privacy-sensitive fields including GPS coordinates, timestamps,
camera information, software, author, and copyright.

PRIVACY NOTE:
  - Raw GPS coordinates are NEVER written to logs.
  - GPS coordinates returned to the frontend are masked (rounded to 2
    decimal places, ≈ 1 km accuracy) to prevent precise tracking.
  - Uploaded images are read from in-memory bytes and are not persisted
    to disk by this service.
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from backend.app.models.finding import (
    BoundingBox,
    Finding,
    FindingCategory,
    RecommendedAction,
    Severity,
    Source,
)

logger = logging.getLogger(__name__)

# EXIF tag IDs for fields we care about
_TAG_ID_GPS_INFO = 34853  # GPSInfo IFD pointer


def _tag_name(tag_id: int) -> str:
    """Return the human-readable name for an EXIF tag ID."""
    return TAGS.get(tag_id, str(tag_id))


def _extract_exif(image: Image.Image) -> dict:
    """
    Return raw EXIF data from a PIL Image as a flat tag-name → value dict.

    Returns an empty dict if the image has no EXIF.
    """
    try:
        exif_data = image._getexif()  # type: ignore[attr-defined]
        if exif_data is None:
            return {}
        return {_tag_name(k): v for k, v in exif_data.items()}
    except (AttributeError, Exception):
        return {}


def _has_gps(exif: dict) -> bool:
    """Return True if the EXIF dict contains a non-empty GPSInfo IFD."""
    gps_info = exif.get("GPSInfo")
    return bool(gps_info)


def _mask_gps_for_frontend(gps_info: dict) -> dict:
    """
    Return a safe, masked representation of GPS data for the frontend.

    - Latitude and longitude are rounded to 2 decimal places
      (~1.1 km precision) so the precise location is never exposed.
    - The raw DMS tuples are replaced with approximate decimal degrees.
    - Altitude is rounded to the nearest 100 m.
    """
    def _dms_to_decimal(dms, ref: str) -> Optional[float]:
        """Convert DMS rational tuple to signed decimal degrees."""
        try:
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
            decimal = degrees + minutes / 60 + seconds / 3600
            if ref in ("S", "W"):
                decimal = -decimal
            return round(decimal, 2)  # mask to ~1 km
        except Exception:
            return None

    safe: dict = {}
    tag_names = {v: k for k, v in GPSTAGS.items()}

    lat_raw = gps_info.get(2)   # GPSLatitude
    lat_ref = gps_info.get(1)   # GPSLatitudeRef
    lon_raw = gps_info.get(4)   # GPSLongitude
    lon_ref = gps_info.get(3)   # GPSLongitudeRef
    alt_raw = gps_info.get(6)   # GPSAltitude

    if lat_raw and lat_ref:
        safe["latitude_approx"] = _dms_to_decimal(lat_raw, lat_ref)
    if lon_raw and lon_ref:
        safe["longitude_approx"] = _dms_to_decimal(lon_raw, lon_ref)
    if alt_raw is not None:
        try:
            altitude = float(alt_raw)
            safe["altitude_approx_m"] = round(altitude / 100) * 100
        except Exception:
            pass
    safe["note"] = "Coordinates masked to ~1 km accuracy for privacy."
    return safe


def _build_gps_finding(gps_info: dict) -> Finding:
    """Build the HIGH-severity GPS_LOCATION Finding."""
    # Log the PRESENCE of GPS data, never the coordinates themselves.
    logger.info("GPS metadata detected in uploaded image.")

    masked = _mask_gps_for_frontend(gps_info)
    description = (
        "The image contains GPS location metadata. "
        f"Approximate location: lat≈{masked.get('latitude_approx')}, "
        f"lon≈{masked.get('longitude_approx')}."
    )

    return Finding(
        category=FindingCategory.GPS_LOCATION,
        severity=Severity.HIGH,
        confidence=1.0,
        description=description,
        source=Source.EXIF,
        bbox=None,
        recommended_action=RecommendedAction.REMOVE_METADATA,
    )


def _build_metadata_finding(field: str, value: str) -> Finding:
    """Build a MEDIUM-severity EXIF_METADATA Finding for non-GPS fields."""
    return Finding(
        category=FindingCategory.EXIF_METADATA,
        severity=Severity.MEDIUM,
        confidence=1.0,
        description=f"EXIF field '{field}' is present: {value!r}",
        source=Source.EXIF,
        bbox=None,
        recommended_action=RecommendedAction.REMOVE_METADATA,
    )


# Fields we report on (besides GPS which gets its own special handling)
_FIELDS_OF_INTEREST = {
    "DateTimeOriginal",
    "DateTime",
    "Make",        # Camera Make
    "Model",       # Camera Model
    "Software",
    "Artist",      # Author
    "Copyright",
}


def analyse_metadata(image_bytes: bytes) -> List[Finding]:
    """
    Analyse image bytes for privacy-sensitive EXIF metadata.

    This is the primary public API of this service.

    Args:
        image_bytes: Raw image data (JPEG, PNG, TIFF, etc.).

    Returns:
        A list of :class:`~backend.app.models.finding.Finding` objects,
        one per detected privacy concern.  Returns an empty list when no
        EXIF metadata is found.

    Notes:
        - Images are processed in-memory; no files are written to disk.
        - GPS coordinates are never written to log output.
        - GPS values returned inside Finding descriptions are masked to
          approximately 1 km resolution.
    """
    findings: List[Finding] = []

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        logger.warning("Could not open image for metadata analysis: %s", exc)
        return findings

    exif = _extract_exif(image)
    if not exif:
        logger.debug("No EXIF metadata found in image.")
        return findings

    logger.debug("EXIF data found; checking for sensitive fields.")

    # 1. GPS — highest priority
    if _has_gps(exif):
        gps_info = None
        try:
            raw_exif = image._getexif()  # type: ignore[attr-defined]
            if raw_exif:
                gps_info = raw_exif.get(_TAG_ID_GPS_INFO)
        except Exception:
            pass
        findings.append(_build_gps_finding(gps_info or {}))

    # 2. Other sensitive fields
    for field in _FIELDS_OF_INTEREST:
        if field in exif:
            value = exif[field]
            findings.append(_build_metadata_finding(field, str(value)))

    return findings
