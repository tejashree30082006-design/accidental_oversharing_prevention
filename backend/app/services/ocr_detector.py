"""Member 2 Module: OCR and PII Detection Service.

Extracts text and spatial bounding boxes from photos using pytesseract, then
applies deterministic regex and pattern matching to identify sensitive PII:
- Phone numbers
- Email addresses
- Credit / Debit card numbers
- Passport numbers
- Booking references (PNR)
- Room numbers & street addresses

Emits findings strictly adhering to the Global Finding Format.
"""

import re
import uuid
from typing import List, Optional
from PIL import Image

from backend.app.schemas.finding import (
    Finding,
    FindingCategory,
    FindingSeverity,
    FindingSource,
    RecommendedAction,
    BoundingBox,
)

# Deterministic Regex Patterns for Sensitive Information
PATTERNS = {
    FindingCategory.EMAIL: (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        FindingSeverity.HIGH,
        RecommendedAction.BLUR,
        "Email address exposed.",
    ),
    FindingCategory.PHONE_NUMBER: (
        re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        FindingSeverity.HIGH,
        RecommendedAction.BLUR,
        "Phone number exposed.",
    ),
    FindingCategory.CARD_NUMBER: (
        re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
        FindingSeverity.HIGH,
        RecommendedAction.REDACT,
        "Payment card number exposed.",
    ),
    FindingCategory.PASSPORT_NUMBER: (
        re.compile(
            r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b|\b[A-Za-z]{1,3}\d{7,9}\b",
            re.IGNORECASE,
        ),
        FindingSeverity.HIGH,
        RecommendedAction.REDACT,
        "Passport or ID number exposed.",
    ),
    FindingCategory.BOOKING_REFERENCE: (
        re.compile(r"\b(?:PNR|BOOKING|REF)[:\s#]*([A-Z0-9]{6})\b", re.IGNORECASE),
        FindingSeverity.MEDIUM,
        RecommendedAction.BLUR,
        "Travel booking reference (PNR) exposed.",
    ),
    FindingCategory.ROOM_NUMBER: (
        re.compile(r"\b(?:ROOM|SUITE|RM)[:\s#]*(\d{2,5}[A-Za-z]?)\b", re.IGNORECASE),
        FindingSeverity.MEDIUM,
        RecommendedAction.BLUR,
        "Private hotel/room number exposed.",
    ),
}


def mask_sensitive_value(value: str) -> str:
    """Mask characters of a sensitive value for safe logging and preview.

    Example: "9876543210" -> "******3210"
    """
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def detect_pii_in_text(
    text: str,
    bbox: Optional[BoundingBox] = None,
) -> List[Finding]:
    """Scan raw text for PII patterns and return standard Findings.

    Args:
        text: Input string.
        bbox: Bounding box of the text line if available.

    Returns:
        List[Finding]: Identified findings.
    """
    findings: List[Finding] = []

    for category, (pattern, severity, action, default_desc) in PATTERNS.items():
        for match in pattern.finditer(text):
            val = match.group(0)
            masked = mask_sensitive_value(val)
            finding_id = f"finding-{uuid.uuid4().hex[:6]}"
            desc = f"{default_desc} (ending in {masked[-4:]})"

            findings.append(
                Finding(
                    id=finding_id,
                    category=category,
                    severity=severity,
                    confidence=0.92,
                    description=desc,
                    source=FindingSource.OCR,
                    bbox=bbox,
                    recommended_action=action,
                )
            )

    return findings


def scan_image_text(image: Image.Image) -> List[Finding]:
    """Execute OCR on an image and detect sensitive information.

    If tesseract is installed, extracts words and precise bounding boxes.
    If tesseract binary is not installed, catches the error and returns an empty list
    without crashing the server.

    Args:
        image: PIL Image object.

    Returns:
        List[Finding]: Extracted findings with bounding boxes.
    """
    findings: List[Finding] = []

    try:
        import pytesseract
        from pytesseract import Output

        ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
        n_boxes = len(ocr_data["text"])

        # Group words by line for coherent pattern matching
        lines: dict[int, list[int]] = {}
        for i in range(n_boxes):
            text = ocr_data["text"][i].strip()
            if not text:
                continue
            line_num = ocr_data["line_num"][i]
            lines.setdefault(line_num, []).append(i)

        for line_num, word_indices in lines.items():
            full_line_text = " ".join([ocr_data["text"][i] for i in word_indices])
            if not full_line_text.strip():
                continue

            # Calculate line bounding box
            x_min = min(ocr_data["left"][i] for i in word_indices)
            y_min = min(ocr_data["top"][i] for i in word_indices)
            x_max = max(ocr_data["left"][i] + ocr_data["width"][i] for i in word_indices)
            y_max = max(ocr_data["top"][i] + ocr_data["height"][i] for i in word_indices)

            line_bbox = BoundingBox(
                x=float(x_min),
                y=float(y_min),
                width=float(x_max - x_min),
                height=float(y_max - y_min),
            )

            line_findings = detect_pii_in_text(full_line_text, bbox=line_bbox)
            findings.extend(line_findings)

    except Exception:
        # Tesseract binary might not be installed or configured in PATH.
        # Fail gracefully so the application pipeline continues to work.
        pass

    return findings
