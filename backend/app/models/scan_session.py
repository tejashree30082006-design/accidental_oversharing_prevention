"""ScanSession SQLAlchemy Model.

Represents an analysis session for an uploaded photo, tracking risk score,
risk classification, and processing status.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class ScanSession(Base):
    """ScanSession entity representing an image privacy analysis run."""

    __tablename__ = "scan_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    risk_score = Column(Integer, default=0, nullable=False)
    risk_level = Column(String(16), default="LOW", nullable=False)  # LOW, MEDIUM, HIGH
    total_findings = Column(Integer, default=0, nullable=False)
    protected = Column(Boolean, default=False, nullable=False)

    # One-to-many relationship with findings detected in this session
    findings = relationship(
        "Finding",
        back_populates="scan_session",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    def __repr__(self) -> str:
        return f"<ScanSession(session_id='{self.session_id}', score={self.risk_score}, level='{self.risk_level}')>"
