"""
Shared Finding model for all privacy/security detection services.

A Finding represents a single detected privacy or security concern
within an image, such as GPS metadata, a QR code, or a barcode.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class FindingCategory(str, Enum):
    """Categories of privacy/security findings."""
    GPS_LOCATION = "GPS_LOCATION"
    EXIF_METADATA = "EXIF_METADATA"
    QR_CODE = "QR_CODE"
    BARCODE = "BARCODE"


class Severity(str, Enum):
    """Severity level of the finding."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class RecommendedAction(str, Enum):
    """Recommended action to take for the finding."""
    REMOVE_METADATA = "REMOVE_METADATA"
    BLUR = "BLUR"
    REDACT = "REDACT"
    REVIEW = "REVIEW"


class Source(str, Enum):
    """Source of the detection."""
    EXIF = "EXIF"
    QR_DETECTOR = "QR_DETECTOR"
    BARCODE_DETECTOR = "BARCODE_DETECTOR"


# BoundingBox: [x, y, width, height] in pixels, top-left origin.
BoundingBox = Optional[List[float]]


class Finding(BaseModel):
    """
    Standardised finding returned by all detection services.

    Attributes:
        id:                 Unique identifier for this finding.
        category:           Classification of the finding.
        severity:           How critical the finding is.
        confidence:         Detector confidence, 0.0 – 1.0.
        description:        Human-readable explanation.
        source:             Which detector produced this finding.
        bbox:               [x, y, w, h] in pixels, or null.
        recommended_action: What the user should do to remediate.
    """
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: FindingCategory
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    source: Source
    bbox: BoundingBox = None
    recommended_action: RecommendedAction
