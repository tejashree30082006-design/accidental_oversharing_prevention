"""Finding Pydantic Schemas.

Conforms to the project's Global Finding Format.
All detection sources (OCR, QR, Barcode, EXIF, Gemma) emit findings in this schema.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FindingCategory(str, Enum):
    """Categorization of detected privacy risks."""

    GPS_LOCATION = "GPS_LOCATION"
    CARD_NUMBER = "CARD_NUMBER"
    PASSPORT_NUMBER = "PASSPORT_NUMBER"
    QR_CODE = "QR_CODE"
    BOOKING_REFERENCE = "BOOKING_REFERENCE"
    BARCODE = "BARCODE"
    ADDRESS = "ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    ROOM_NUMBER = "ROOM_NUMBER"
    EMAIL = "EMAIL"
    OTHER = "OTHER"


class FindingSeverity(str, Enum):
    """Severity levels for detected privacy risks."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FindingSource(str, Enum):
    """Detection source or detector module."""

    OCR = "OCR"
    QR = "QR"
    BARCODE = "BARCODE"
    EXIF = "EXIF"
    GEMMA = "GEMMA"


class RecommendedAction(str, Enum):
    """Remediation actions applied to protect sensitive regions."""

    BLUR = "BLUR"
    REDACT = "REDACT"
    REMOVE_METADATA = "REMOVE_METADATA"
    REVIEW = "REVIEW"


class BoundingBox(BaseModel):
    """Bounding box coordinates for visual regions in pixels."""

    x: float = Field(..., description="Top-left X coordinate")
    y: float = Field(..., description="Top-left Y coordinate")
    width: float = Field(..., description="Width of the bounding region")
    height: float = Field(..., description="Height of the bounding region")


class FindingBase(BaseModel):
    """Core finding data adhering to the Global Finding Format."""

    id: str = Field(..., description="Unique finding identifier, e.g. finding-001")
    category: FindingCategory = Field(..., description="Finding category type")
    severity: FindingSeverity = Field(..., description="Risk severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence between 0.0 and 1.0")
    description: str = Field(..., description="Human-readable description of the risk")
    source: FindingSource = Field(..., description="Module that produced the detection")
    bbox: Optional[BoundingBox] = Field(None, description="Pixel coordinates, if applicable")
    recommended_action: RecommendedAction = Field(..., description="Action to remediate the risk")


from pydantic import BaseModel, Field, ConfigDict


class Finding(FindingBase):
    """Finding response model."""

    model_config = ConfigDict(from_attributes=True)
