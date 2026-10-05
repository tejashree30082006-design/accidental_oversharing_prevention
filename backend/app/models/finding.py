"""Finding SQLAlchemy Model.

Stores privacy risk findings identified during photo analysis. In compliance
with privacy principles, this model stores metadata, bounding boxes, and
recommended remediation actions without storing raw unmasked sensitive data.
"""

from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class Finding(Base):
    """Finding entity representing a specific detected privacy risk."""

    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    finding_id = Column(String(64), index=True, nullable=True)  # e.g., "finding-001"
    session_id = Column(
        String(64),
        ForeignKey("scan_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category = Column(String(64), nullable=False)  # e.g., PHONE_NUMBER, GPS_LOCATION
    severity = Column(String(16), nullable=False)  # LOW, MEDIUM, HIGH
    confidence = Column(Float, default=1.0, nullable=False)
    description = Column(String(255), nullable=False)
    source = Column(String(32), nullable=False)  # OCR, QR, BARCODE, EXIF, GEMMA

    # Bounding box coordinates (nullable for non-visual findings like EXIF)
    x = Column(Float, nullable=True)
    y = Column(Float, nullable=True)
    width = Column(Float, nullable=True)
    height = Column(Float, nullable=True)

    recommended_action = Column(String(32), nullable=False)  # BLUR, REDACT, REMOVE_METADATA, REVIEW

    # Relationship back to the parent ScanSession
    scan_session = relationship("ScanSession", back_populates="findings")

    def __repr__(self) -> str:
        return f"<Finding(id='{self.finding_id}', category='{self.category}', severity='{self.severity}')>"
