"""
findings.py — Shared data models for OCR results and PII (Personally Identifiable
Information) findings.

This module defines the canonical data structures exchanged between:
  - ocr_service.py  (Member 2 — OCR layer)
  - pii_detector.py (Member 2 — detection layer)
  - risk_engine     (Member 4 — risk scoring layer)

Interface contract (for Member 4):
────────────────────────────────────────────────────────────────────────────────
  Finding
    .entity_type      : str   — PII category (see EntityType enum values)
    .text             : str   — The matched sensitive text (DO NOT log/store verbatim)
    .bbox             : BoundingBox — pixel coordinates of the text in the source image
    .confidence       : float — 0.0-1.0 combined OCR x detection confidence
    .context_snippet  : str   — a few surrounding words (without the sensitive value)
    .source_page      : int   — 0-indexed page number (0 for single-page images)

  BoundingBox
    .x, .y           : int   — top-left corner (pixels)
    .width, .height  : int   — dimensions (pixels)
    .as_dict()       : dict  — serialisable form
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class EntityType(str, Enum):
    """Canonical PII entity categories detected by pii_detector.py."""

    PHONE_NUMBER = "PHONE_NUMBER"
    EMAIL = "EMAIL"
    CARD_NUMBER = "CARD_NUMBER"
    PASSPORT_NUMBER = "PASSPORT_NUMBER"
    BOOKING_REFERENCE = "BOOKING_REFERENCE"
    ADDRESS = "ADDRESS"
    ROOM_NUMBER = "ROOM_NUMBER"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space bounding rectangle for a detected text region.

    Coordinates are relative to the top-left corner of the source image.
    For multi-page PDFs, coordinates are per-page.

    Attributes:
        x:      Left edge (pixels).
        y:      Top edge (pixels).
        width:  Width of the box (pixels).
        height: Height of the box (pixels).
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(
                f"BoundingBox dimensions must be non-negative, "
                f"got width={self.width}, height={self.height}"
            )

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_tesseract(cls, left: int, top: int, width: int, height: int) -> "BoundingBox":
        """Construct from Tesseract (left, top, width, height) convention."""
        return cls(x=left, y=top, width=width, height=height)

    @classmethod
    def union(cls, boxes: List["BoundingBox"]) -> "BoundingBox":
        """Return the smallest box that contains all given boxes."""
        if not boxes:
            raise ValueError("Cannot compute union of an empty list of BoundingBoxes")
        min_x = min(b.x for b in boxes)
        min_y = min(b.y for b in boxes)
        max_x = max(b.x + b.width for b in boxes)
        max_y = max(b.y + b.height for b in boxes)
        return cls(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)


@dataclass
class OCRWord:
    """Single word returned by the OCR engine (internal intermediate object).

    NOT exposed to Member 4 / risk engine.

    Attributes:
        text:       Raw text of the word as recognised by OCR.
        bbox:       Bounding box in source-image pixel coordinates.
        confidence: OCR confidence for this word in [0.0, 1.0].
        page:       0-indexed page number.
    """

    text: str
    bbox: BoundingBox
    confidence: float
    page: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"OCRWord confidence must be in [0.0, 1.0], got {self.confidence}"
            )


@dataclass
class Finding:
    """A single detected PII / sensitive-information candidate.

    PRIMARY OUTPUT of the Member 2 pipeline.
    PRIMARY INPUT  to Member 4 risk engine.

    Security rules:
      - ``text`` contains raw matched PII. NEVER log this field.
      - Use ``to_safe_dict()`` for any logging or audit-trail output.
      - ``context_snippet`` is safe to log (no sensitive values included).

    Attributes:
        entity_type:      Category of PII (EntityType enum value).
        text:             The sensitive matched text.  DO NOT LOG.
        bbox:             Location in the source image (pixel coordinates).
        confidence:       Combined OCR x pattern confidence in [0.0, 1.0].
        context_snippet:  Surrounding non-sensitive words for context.
        source_page:      0-indexed page/frame number.
        raw_pattern_name: Internal regex rule name (debug only).
        ocr_word_indices: Indices into the OCRWord list for this finding.
    """

    entity_type: EntityType
    text: str
    bbox: BoundingBox
    confidence: float
    context_snippet: str = ""
    source_page: int = 0
    raw_pattern_name: str = ""
    ocr_word_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Finding confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise for API / risk-engine consumption.

        WARNING: Includes raw ``text``. Strip before logging.
        """
        return {
            "entity_type": self.entity_type.value,
            "text": self.text,
            "bbox": self.bbox.as_dict(),
            "confidence": round(self.confidence, 4),
            "context_snippet": self.context_snippet,
            "source_page": self.source_page,
            "raw_pattern_name": self.raw_pattern_name,
        }

    def to_safe_dict(self) -> dict:
        """Like ``to_dict`` but with ``text`` replaced by ``[REDACTED]``.

        Use for logging, audit trails, and any displayable output.
        """
        d = self.to_dict()
        d["text"] = "[REDACTED]"
        return d
