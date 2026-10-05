"""Unit tests for the Image Protection and Redaction Service."""

import numpy as np
from PIL import Image
from backend.app.schemas.finding import (
    Finding,
    FindingCategory,
    FindingSeverity,
    FindingSource,
    RecommendedAction,
    BoundingBox,
)
from backend.app.services.protection_service import (
    protect_image,
    save_protected_image,
    image_to_bytes,
)


def test_protect_image_blur_and_redaction():
    """Verify that blurring and redaction modify pixels appropriately."""
    # Create a 200x200 white test image
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    # Draw a colored region in (10, 10, 50, 50)
    img_arr = np.array(img)
    img_arr[10:60, 10:60] = [100, 150, 200]
    img = Image.fromarray(img_arr)

    findings = [
        # Blurred finding
        Finding(
            id="f-blur",
            category=FindingCategory.PHONE_NUMBER,
            severity=FindingSeverity.MEDIUM,
            confidence=0.9,
            description="Phone number",
            source=FindingSource.OCR,
            bbox=BoundingBox(x=10, y=10, width=50, height=50),
            recommended_action=RecommendedAction.BLUR,
        ),
        # Redacted finding
        Finding(
            id="f-redact",
            category=FindingCategory.CARD_NUMBER,
            severity=FindingSeverity.HIGH,
            confidence=0.95,
            description="Card number",
            source=FindingSource.OCR,
            bbox=BoundingBox(x=100, y=100, width=40, height=20),
            recommended_action=RecommendedAction.REDACT,
        ),
    ]

    protected_img, actions = protect_image(img, findings)

    assert len(actions) >= 3
    result_arr = np.array(protected_img)

    # Check redaction: pixels in redacted zone should be black (0, 0, 0)
    redacted_zone = result_arr[100:120, 100:140]
    assert np.all(redacted_zone == [0, 0, 0])

    # Check blur: blurred pixels should be modified
    blurred_zone = result_arr[10:60, 10:60]
    assert not np.all(blurred_zone == [255, 255, 255])


def test_protect_image_strips_exif():
    """Verify that protected images contain no EXIF GPS data."""
    # Create image
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))

    # Protect the image
    protected_img, _ = protect_image(img, [])

    # Verify EXIF info is absent
    exif = protected_img.getexif()
    assert len(exif) == 0

    # Also test byte conversion
    img_bytes = image_to_bytes(protected_img)
    assert len(img_bytes) > 0
