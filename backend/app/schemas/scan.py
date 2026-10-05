"""Scan and Protection Pydantic Schemas.

Defines the API request and response data models for image scanning,
risk reporting, and image protection workflows.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from backend.app.schemas.finding import Finding


class RiskLevel(str, Enum):
    """Categorical risk classification based on calculated score."""

    LOW = "LOW"        # 0–29
    MEDIUM = "MEDIUM"  # 30–59
    HIGH = "HIGH"      # 60–100


class ScanResponse(BaseModel):
    """Response returned when an image is scanned or retrieved."""

    session_id: str = Field(..., description="Unique analysis session UUID")
    created_at: datetime = Field(..., description="Timestamp of the scan session")
    risk_score: int = Field(..., ge=0, le=100, description="Overall privacy risk score (0-100)")
    risk_level: RiskLevel = Field(..., description="Privacy risk classification (LOW, MEDIUM, HIGH)")
    total_findings: int = Field(..., description="Total count of unique privacy findings")
    findings: list[Finding] = Field(default_factory=list, description="List of detected privacy risks")
    protected: bool = Field(default=False, description="Whether this image session has been protected")
    summary: str = Field(..., description="High-level summary of privacy risks identified")

    model_config = ConfigDict(from_attributes=True)


class ProtectResponse(BaseModel):
    """Response returned after an image has been protected."""

    session_id: str = Field(..., description="Session identifier")
    original_risk_score: int = Field(..., description="Risk score before protection")
    original_risk_level: str = Field(..., description="Risk level before protection")
    protected_risk_score: int = Field(0, description="Risk score after protection (typically 0)")
    protected_risk_level: str = Field("LOW", description="Risk level after protection (LOW)")
    actions_applied: list[str] = Field(default_factory=list, description="List of protection operations applied")
    protected_image_path: Optional[str] = Field(None, description="Local path or key to protected image")
    download_url: str = Field(..., description="URL to download the protected safe image")
    message: str = Field(..., description="Status description of protection process")
