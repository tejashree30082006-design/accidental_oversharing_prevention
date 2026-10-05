"""Member 3 Module: EXIF, GPS, QR Code, and Barcode Detection Service.

Detects hidden and visual privacy risks:
1. EXIF Metadata & GPS Coordinates (using Pillow)
2. QR Codes with bounding boxes (using OpenCV QRCodeDetector)
3. Barcodes with bounding boxes (using pyzbar / OpenCV)

Strictly obeys privacy requirements: Raw GPS coordinates and decoded QR payloads
are never persisted or placed in finding descriptions.
"""

import uuid
from typing import List
import cv2
import numpy as np
from PIL import Image, ExifTags

from backend.app.schemas.finding import (
    Finding,
    FindingCategory,
    FindingSeverity,
    FindingSource,
    RecommendedAction,
    BoundingBox,
)


def detect_exif_gps(image: Image.Image) -> List[Finding]:
    """Scan an image for embedded EXIF metadata, specifically GPS location coordinates.

    Privacy compliance: The exact raw coordinates are NOT stored in the finding.

    Args:
        image: PIL Image object.

    Returns:
        List[Finding]: EXIF / GPS findings if present.
    """
    findings: List[Finding] = []

    try:
        exif = image.getexif()
        if not exif:
            return findings

        # Tag 34853 corresponds to GPSInfo in standard EXIF specifications
        has_gps = False
        if 34853 in exif:
            has_gps = True

        # Also check IFD dictionary if available
        if not has_gps:
            for ifd_id in (ExifTags.IFD.GPSInfo,):
                try:
                    ifd = exif.get_ifd(ifd_id)
                    if ifd:
                        has_gps = True
                        break
                except Exception:
                    pass

        if has_gps:
            findings.append(
                Finding(
                    id=f"finding-gps-{uuid.uuid4().hex[:6]}",
                    category=FindingCategory.GPS_LOCATION,
                    severity=FindingSeverity.HIGH,
                    confidence=1.0,
                    description=(
                        "Embedded GPS location coordinates were detected in EXIF metadata. "
                        "Sharing this image publicly exposes the exact geographical location "
                        "where the photo was taken."
                    ),
                    source=FindingSource.EXIF,
                    bbox=None,  # Metadata is file-level, not a visual bounding box
                    recommended_action=RecommendedAction.REMOVE_METADATA,
                )
            )

    except Exception:
        # Gracefully handle corrupted or non-standard EXIF headers
        pass

    return findings


def detect_qr_codes(image: Image.Image) -> List[Finding]:
    """Detect visible QR codes and determine their spatial bounding boxes using OpenCV.

    Privacy compliance: Decoded QR payloads are NOT stored in the finding.

    Args:
        image: PIL Image object.

    Returns:
        List[Finding]: Detected QR codes with bounding boxes.
    """
    findings: List[Finding] = []

    try:
        # Convert PIL Image to OpenCV format
        rgb_arr = np.array(image.convert("RGB"))
        cv_img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        detector = cv2.QRCodeDetector()
        # detectAndDecodeMulti detects all QR codes in the image
        has_qr, decoded_info, points, _ = detector.detectAndDecodeMulti(cv_img)

        if has_qr and points is not None:
            for pts in points:
                # pts is an array of 4 corner points [[x, y], [x, y], [x, y], [x, y]]
                pts = pts.reshape(-1, 2)
                x_min = float(np.min(pts[:, 0]))
                y_min = float(np.min(pts[:, 1]))
                x_max = float(np.max(pts[:, 0]))
                y_max = float(np.max(pts[:, 1]))

                w = max(10.0, x_max - x_min)
                h = max(10.0, y_max - y_min)

                findings.append(
                    Finding(
                        id=f"finding-qr-{uuid.uuid4().hex[:6]}",
                        category=FindingCategory.QR_CODE,
                        severity=FindingSeverity.HIGH,
                        confidence=0.98,
                        description=(
                            "A QR code was detected in the image. QR codes frequently encode "
                            "sensitive tokens, Wi-Fi credentials, boarding passes, or private links."
                        ),
                        source=FindingSource.QR,
                        bbox=BoundingBox(
                            x=round(x_min, 1),
                            y=round(y_min, 1),
                            width=round(w, 1),
                            height=round(h, 1),
                        ),
                        recommended_action=RecommendedAction.BLUR,
                    )
                )

    except Exception:
        pass

    return findings


def detect_barcodes(image: Image.Image) -> List[Finding]:
    """Detect visible barcodes using pyzbar.

    Args:
        image: PIL Image object.

    Returns:
        List[Finding]: Detected barcodes with bounding boxes.
    """
    findings: List[Finding] = []

    try:
        from pyzbar.pyzbar import decode, ZBarSymbol

        # Limit to 1D barcode formats to differentiate from QR codes
        barcode_symbols = [
            ZBarSymbol.CODE128,
            ZBarSymbol.CODE39,
            ZBarSymbol.EAN13,
            ZBarSymbol.EAN8,
            ZBarSymbol.UPCA,
            ZBarSymbol.UPCE,
            ZBarSymbol.I25,
        ]

        decoded_objects = decode(image, symbols=barcode_symbols)

        for obj in decoded_objects:
            rect = obj.rect
            findings.append(
                Finding(
                    id=f"finding-barcode-{uuid.uuid4().hex[:6]}",
                    category=FindingCategory.BARCODE,
                    severity=FindingSeverity.MEDIUM,
                    confidence=0.95,
                    description=(
                        "A 1D barcode was detected. Barcodes on shipping labels, tickets, "
                        "and identification cards can reveal account numbers and tracking details."
                    ),
                    source=FindingSource.BARCODE,
                    bbox=BoundingBox(
                        x=float(rect.left),
                        y=float(rect.top),
                        width=float(rect.width),
                        height=float(rect.height),
                    ),
                    recommended_action=RecommendedAction.BLUR,
                )
            )

    except Exception:
        # pyzbar or underlying zbar DLLs might not be present on some operating systems
        pass

    return findings


def scan_metadata_and_codes(image: Image.Image) -> List[Finding]:
    """Unified detector executing all Member 3 detections (EXIF, QR, Barcodes).

    Args:
        image: PIL Image object.

    Returns:
        List[Finding]: Aggregate list of findings from Member 3's suite.
    """
    findings: List[Finding] = []
    findings.extend(detect_exif_gps(image))
    findings.extend(detect_qr_codes(image))
    findings.extend(detect_barcodes(image))
    return findings
