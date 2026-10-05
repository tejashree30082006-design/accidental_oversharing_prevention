"""
QR code detection service.

Uses OpenCV's built-in QRCodeDetector to locate and decode QR codes
inside uploaded images.

PRIVACY NOTE:
  - Decoded QR content (e.g. URLs, contact data) is NEVER written to
    application logs.
  - Images are processed from in-memory bytes only; no files are written
    to disk by this service.
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional, Tuple

import cv2
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


def _pil_to_cv2_gray(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convert raw image bytes to an OpenCV grayscale ndarray.

    Returns None if the conversion fails.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb_array = np.array(pil_image, dtype=np.uint8)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    except Exception as exc:
        logger.warning("Could not convert image to grayscale for QR detection: %s", exc)
        return None


def _points_to_bbox(points: np.ndarray) -> List[float]:
    """
    Convert the 4-corner polygon returned by OpenCV QR detector to an
    [x, y, width, height] bounding box.

    Args:
        points: Array of shape (4, 2) with (x, y) corner coordinates.

    Returns:
        [x_min, y_min, width, height] as floats.
    """
    pts = points.reshape(-1, 2).astype(float)
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _build_qr_finding(
    is_readable: bool,
    bbox: Optional[List[float]],
) -> Finding:
    """
    Construct a standardised QR_CODE Finding.

    Args:
        is_readable: Whether the QR code content was successfully decoded.
        bbox:        Bounding box [x, y, w, h], or None if unavailable.

    Returns:
        A :class:`~backend.app.models.finding.Finding` instance.
    """
    if is_readable:
        description = (
            "A readable QR code was detected in the image. "
            "The embedded content may expose sensitive information if shared."
        )
        confidence = 1.0
    else:
        description = (
            "A QR code pattern was detected in the image but could not be "
            "fully decoded. It may still expose information when scanned with "
            "a dedicated reader."
        )
        confidence = 0.75

    return Finding(
        category=FindingCategory.QR_CODE,
        severity=Severity.MEDIUM,
        confidence=confidence,
        description=description,
        source=Source.QR_DETECTOR,
        bbox=bbox,
        recommended_action=RecommendedAction.BLUR,
    )


def detect_qr_codes(image_bytes: bytes) -> List[Finding]:
    """
    Detect QR codes in image bytes using OpenCV QRCodeDetector.

    This is the primary public API of this service.

    Args:
        image_bytes: Raw image data (JPEG, PNG, etc.).

    Returns:
        A list of :class:`~backend.app.models.finding.Finding` objects.
        Returns an empty list when no QR code is found.

    Notes:
        - Images are processed in-memory; no files are written to disk.
        - Decoded QR content is never written to log output.
        - When a QR code is found but decoding fails, a lower-confidence
          finding is still returned so the user is warned.
    """
    findings: List[Finding] = []

    gray = _pil_to_cv2_gray(image_bytes)
    if gray is None:
        return findings

    detector = cv2.QRCodeDetector()

    # detectAndDecode returns (decoded_text, points, straight_qrcode)
    decoded_text, points, _ = detector.detectAndDecode(gray)

    if points is None:
        # Try the wechat detector as a fallback for harder cases
        try:
            wechat = cv2.wechat_qrcode_WeChatQRCode()
            texts, _ = wechat.detectAndDecode(gray)
            if texts:
                # Fallback decoded something; treat as readable but no bbox
                logger.info("QR code detected (WeChatQRCode fallback), readable.")
                findings.append(_build_qr_finding(is_readable=True, bbox=None))
        except Exception:
            pass  # WeChatQRCode not available; skip
        return findings

    # A QR code was located
    bbox = _points_to_bbox(points)
    is_readable = bool(decoded_text)

    if is_readable:
        # Log ONLY that a readable QR was found; never log the content.
        logger.info("QR code detected and readable. Content suppressed from logs.")
    else:
        logger.info("QR code pattern detected but content not decoded.")

    findings.append(_build_qr_finding(is_readable=is_readable, bbox=bbox))
    return findings
