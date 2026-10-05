"""
Barcode detection service.

Uses pyzbar to detect and locate common 1-D and 2-D barcodes in images.
Supported symbologies include Code 128, EAN-13, EAN-8, UPC-A, UPC-E,
QR Code, Data Matrix, PDF417, Aztec, and more.

PRIVACY NOTE:
  - Decoded barcode content is NEVER written to application logs.
  - Images are processed from in-memory bytes only; no files are written
    to disk by this service.

PLATFORM NOTE:
  - pyzbar requires the zbar shared library.
    • Windows: bundled DLLs are included in the pyzbar wheel.
    • Linux:   install with `apt-get install libzbar0`
    • macOS:   install with `brew install zbar`
  - If pyzbar is unavailable, the service gracefully returns an empty
    list rather than raising an error, and logs a warning.
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional

import numpy as np
from PIL import Image

from backend.app.models.finding import (
    Finding,
    FindingCategory,
    RecommendedAction,
    Severity,
    Source,
)

logger = logging.getLogger(__name__)

# ── pyzbar import (graceful fallback) ──────────────────────────────────────
_PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar as _pyzbar  # type: ignore
    _PYZBAR_AVAILABLE = True
except Exception as _pyzbar_import_err:
    logger.warning(
        "pyzbar is not available (%s). Barcode detection will be disabled. "
        "Install the zbar shared library for your platform.",
        _pyzbar_import_err,
    )


# ── QR types that pyzbar can also detect ───────────────────────────────────
# We route pyzbar's 'QRCODE' type to FindingCategory.QR_CODE so that QR
# detections from pyzbar are also surfaced with the correct category.
_PYZBAR_QR_TYPES = {"QRCODE"}


def _pil_to_pyzbar_image(image_bytes: bytes) -> Optional[Image.Image]:
    """
    Open image bytes as a PIL Image in a mode that pyzbar accepts (L or RGB).

    Returns None if the image cannot be opened.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # pyzbar works best with grayscale or RGB; convert RGBA/P/etc.
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        return img
    except Exception as exc:
        logger.warning("Could not open image for barcode detection: %s", exc)
        return None


def _pyzbar_rect_to_bbox(rect) -> List[float]:
    """
    Convert a pyzbar Rect namedtuple to [x, y, width, height].

    pyzbar.Rect has fields: left, top, width, height.
    """
    return [float(rect.left), float(rect.top), float(rect.width), float(rect.height)]


def _build_barcode_finding(
    symbology: str,
    bbox: Optional[List[float]],
) -> Finding:
    """
    Construct a standardised Finding for a detected barcode.

    Args:
        symbology: The barcode type string reported by pyzbar
                   (e.g. "CODE128", "EAN13", "QRCODE").
        bbox:      Bounding box [x, y, w, h] in pixels, or None.

    Returns:
        A :class:`~backend.app.models.finding.Finding` instance.
    """
    is_qr = symbology.upper() in _PYZBAR_QR_TYPES

    if is_qr:
        category = FindingCategory.QR_CODE
        description = (
            f"A QR code ({symbology}) was detected via barcode scanner. "
            "The embedded content may expose sensitive information if shared."
        )
    else:
        category = FindingCategory.BARCODE
        description = (
            f"A {symbology} barcode was detected in the image. "
            "The embedded data may expose product, identity, or tracking "
            "information if shared."
        )

    return Finding(
        category=category,
        severity=Severity.MEDIUM,
        confidence=1.0,
        description=description,
        source=Source.BARCODE_DETECTOR,
        bbox=bbox,
        recommended_action=RecommendedAction.BLUR,
    )


def detect_barcodes(image_bytes: bytes) -> List[Finding]:
    """
    Detect barcodes in image bytes using pyzbar.

    This is the primary public API of this service.

    Args:
        image_bytes: Raw image data (JPEG, PNG, etc.).

    Returns:
        A list of :class:`~backend.app.models.finding.Finding` objects,
        one per detected barcode symbol.  Returns an empty list when no
        barcode is found or when pyzbar is unavailable.

    Notes:
        - Images are processed in-memory; no files are written to disk.
        - Decoded barcode content is never written to log output.
        - If pyzbar cannot be imported, a warning is logged and an empty
          list is returned so the overall scan pipeline is not disrupted.
    """
    if not _PYZBAR_AVAILABLE:
        logger.warning("Barcode detection skipped: pyzbar not available.")
        return []

    findings: List[Finding] = []

    pil_image = _pil_to_pyzbar_image(image_bytes)
    if pil_image is None:
        return findings

    try:
        decoded_objects = _pyzbar.decode(pil_image)
    except Exception as exc:
        logger.warning("pyzbar decode raised an exception: %s", exc)
        return findings

    for obj in decoded_objects:
        symbology = obj.type  # e.g. "CODE128", "EAN13", "QRCODE"
        bbox = _pyzbar_rect_to_bbox(obj.rect)

        # Log the presence and type of barcode, never the decoded data.
        logger.info("Barcode detected: symbology=%s. Content suppressed from logs.", symbology)

        findings.append(_build_barcode_finding(symbology=symbology, bbox=bbox))

    return findings
