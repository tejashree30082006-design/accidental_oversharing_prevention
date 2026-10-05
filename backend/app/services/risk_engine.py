"""Privacy Risk Scoring and Deduplication Engine.

Calculates privacy risk scores (0-100) based on weighted risk categories,
deduplicates overlapping findings to prevent double-counting, and assigns
categorical risk levels (LOW, MEDIUM, HIGH).
"""

from typing import List, Tuple
from backend.app.schemas.finding import (
    Finding,
    FindingCategory,
    FindingSeverity,
    BoundingBox,
)
from backend.app.schemas.scan import RiskLevel

# Starting weights for unique finding categories
CATEGORY_WEIGHTS: dict[FindingCategory, int] = {
    FindingCategory.GPS_LOCATION: 35,
    FindingCategory.CARD_NUMBER: 30,
    FindingCategory.PASSPORT_NUMBER: 30,
    FindingCategory.QR_CODE: 25,
    FindingCategory.BOOKING_REFERENCE: 20,
    FindingCategory.BARCODE: 20,
    FindingCategory.ADDRESS: 15,
    FindingCategory.PHONE_NUMBER: 15,
    FindingCategory.ROOM_NUMBER: 15,
    FindingCategory.EMAIL: 10,
    FindingCategory.OTHER: 10,
}


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    x_a = max(box_a.x, box_b.x)
    y_a = max(box_a.y, box_b.y)
    x_b = min(box_a.x + box_a.width, box_b.x + box_b.width)
    y_b = min(box_a.y + box_a.height, box_b.y + box_b.height)

    inter_width = max(0.0, x_b - x_a)
    inter_height = max(0.0, y_b - y_a)
    inter_area = inter_width * inter_height

    area_a = box_a.width * box_a.height
    area_b = box_b.width * box_b.height
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def are_findings_duplicate(f1: Finding, f2: Finding) -> bool:
    """Determine whether two findings represent the same physical sensitive item.

    Rules:
    - Same category and overlapping bounding boxes (IoU > 0.3)
    - If neither has a bounding box (e.g. EXIF metadata), check if they share the same category.
    """
    if f1.category != f2.category:
        return False

    # If neither has a bounding box (e.g. both are GPS EXIF findings)
    if f1.bbox is None and f2.bbox is None:
        return True

    # If only one has a bounding box, check if one was an AI refinement of the other
    if (f1.bbox is None) != (f2.bbox is None):
        return False

    # Both have bounding boxes: evaluate spatial overlap
    return compute_iou(f1.bbox, f2.bbox) > 0.3


def deduplicate_findings(findings: List[Finding]) -> List[Finding]:
    """Deduplicate findings to prevent double-counting risks.

    For example, if OCR finds a phone number and Gemma confirms that same phone number,
    merge them into a single finding taking the maximum confidence and higher severity.

    Args:
        findings: Raw list of findings from all detection sources.

    Returns:
        List[Finding]: Deduplicated list of findings.
    """
    if not findings:
        return []

    unique_findings: List[Finding] = []

    for finding in findings:
        matched_idx = -1
        for idx, existing in enumerate(unique_findings):
            if are_findings_duplicate(existing, finding):
                matched_idx = idx
                break

        if matched_idx == -1:
            unique_findings.append(finding)
        else:
            # Merge findings: take highest confidence and highest severity
            existing = unique_findings[matched_idx]
            merged_confidence = max(existing.confidence, finding.confidence)

            # Severity precedence: HIGH > MEDIUM > LOW
            severity_order = {FindingSeverity.LOW: 1, FindingSeverity.MEDIUM: 2, FindingSeverity.HIGH: 3}
            higher_severity = (
                finding.severity
                if severity_order.get(finding.severity, 1) > severity_order.get(existing.severity, 1)
                else existing.severity
            )

            # Combine or enrich description
            merged_desc = existing.description
            if finding.source.value not in existing.source.value and finding.source != existing.source:
                merged_desc = f"{existing.description} (Confirmed by {finding.source.value})"

            # Update the existing record with the merged values
            unique_findings[matched_idx] = Finding(
                id=existing.id,
                category=existing.category,
                severity=higher_severity,
                confidence=round(merged_confidence, 2),
                description=merged_desc,
                source=existing.source,
                bbox=existing.bbox or finding.bbox,
                recommended_action=existing.recommended_action,
            )

    return unique_findings


def calculate_risk_score(findings: List[Finding]) -> Tuple[int, RiskLevel, str]:
    """Calculate the cumulative privacy risk score and assign risk level.

    Args:
        findings: Deduplicated list of findings.

    Returns:
        Tuple[int, RiskLevel, str]:
            - Final risk score (0-100)
            - Risk classification level (LOW, MEDIUM, HIGH)
            - Human-readable summary explanation
    """
    if not findings:
        return 0, RiskLevel.LOW, "No sensitive personal information or risky metadata was detected."

    # Calculate sum of weights for each unique finding
    raw_score = sum(CATEGORY_WEIGHTS.get(f.category, 10) for f in findings)

    # Maximum score is strictly capped at 100
    final_score = min(100, raw_score)

    # Classify risk level based on thresholds
    # 0–29 = LOW, 30–59 = MEDIUM, 60–100 = HIGH
    if final_score >= 60:
        level = RiskLevel.HIGH
    elif final_score >= 30:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    # Build clear explanation of what is exposed
    categories_found = [f.category.value.replace("_", " ").title() for f in findings]
    category_summary = ", ".join(list(dict.fromkeys(categories_found)))

    summary = (
        f"Detected {len(findings)} sensitive item(s): {category_summary}. "
        f"This photo carries a {level.value} privacy risk (score: {final_score}/100) "
        "and could expose personal data if published without protection."
    )

    return final_score, level, summary
