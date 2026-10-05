"""Image Protection and Redaction Service.

Provides automatic privacy protection for images:
- Heavy Gaussian blurring / pixelation for sensitive text regions, QR codes, and barcodes.
- Redaction (solid masking) for high-sensitivity data (cards, passport numbers).
- Complete removal of EXIF metadata (including GPS coordinates).
"""

import io
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

from backend.app.schemas.finding import Finding, RecommendedAction
from backend.app.core.config import BASE_DIR

# Temporary storage for protected images available for download
PROCESSED_IMAGE_DIR = BASE_DIR / "temp_processed"
PROCESSED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def apply_blur(image_cv: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Apply strong Gaussian blur to a target bounding box in an OpenCV image.

    Args:
        image_cv: OpenCV image array (BGR).
        x, y, w, h: Coordinates and dimensions of the bounding box.

    Returns:
        np.ndarray: Modified image with the blurred region.
    """
    img_h, img_w = image_cv.shape[:2]

    # Clip coordinates to image boundaries
    x1 = max(0, min(x, img_w - 1))
    y1 = max(0, min(y, img_h - 1))
    x2 = max(0, min(x + w, img_w))
    y2 = max(0, min(y + h, img_h))

    if x2 <= x1 or y2 <= y1:
        return image_cv

    roi = image_cv[y1:y2, x1:x2]

    # Calculate kernel size proportional to region size (must be odd integers)
    kw = max(15, (roi.shape[1] // 3) * 2 + 1)
    kh = max(15, (roi.shape[0] // 3) * 2 + 1)
    if kw % 2 == 0:
        kw += 1
    if kh % 2 == 0:
        kh += 1

    blurred_roi = cv2.GaussianBlur(roi, (kw, kh), sigmaX=30, sigmaY=30)
    image_cv[y1:y2, x1:x2] = blurred_roi
    return image_cv


def apply_redaction(image_cv: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Apply solid black redaction rectangle over a bounding box.

    Args:
        image_cv: OpenCV image array (BGR).
        x, y, w, h: Coordinates and dimensions of the bounding box.

    Returns:
        np.ndarray: Modified image with blacked-out region.
    """
    img_h, img_w = image_cv.shape[:2]

    x1 = max(0, min(x, img_w - 1))
    y1 = max(0, min(y, img_h - 1))
    x2 = max(0, min(x + w, img_w))
    y2 = max(0, min(y + h, img_h))

    if x2 <= x1 or y2 <= y1:
        return image_cv

    # Draw solid black rectangle (0, 0, 0)
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return image_cv


def protect_image(
    image: Image.Image,
    findings: List[Finding],
) -> Tuple[Image.Image, List[str]]:
    """Protect an image by blurring/redacting sensitive regions and stripping EXIF metadata.

    Args:
        image: Original PIL image.
        findings: List of privacy findings to remediate.

    Returns:
        Tuple[Image.Image, List[str]]:
            - Clean, protected PIL image without EXIF metadata.
            - List of actions applied during protection.
    """
    # Convert PIL Image to RGB (ensuring no alpha channel issues during blur)
    rgb_image = image.convert("RGB")
    image_np = np.array(rgb_image)
    # Convert RGB to BGR for OpenCV processing
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    actions_applied: List[str] = []

    for finding in findings:
        if finding.bbox is not None:
            bx = int(round(finding.bbox.x))
            by = int(round(finding.bbox.y))
            bw = int(round(finding.bbox.width))
            bh = int(round(finding.bbox.height))

            if finding.recommended_action == RecommendedAction.REDACT:
                image_cv = apply_redaction(image_cv, bx, by, bw, bh)
                actions_applied.append(
                    f"Redacted {finding.category.value} at (x={bx}, y={by}, w={bw}, h={bh})"
                )
            else:
                # Default to BLUR for BLUR, REVIEW, and visual targets
                image_cv = apply_blur(image_cv, bx, by, bw, bh)
                actions_applied.append(
                    f"Blurred {finding.category.value} at (x={bx}, y={by}, w={bw}, h={bh})"
                )

    # Convert back to RGB PIL Image
    protected_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    protected_pil = Image.fromarray(protected_rgb)

    # Note: Creating a brand new PIL Image from pixel array naturally strips
    # all original EXIF metadata, GPS tags, camera details, and ICC profiles.
    actions_applied.append("Stripped all EXIF metadata and GPS coordinates from output image.")

    return protected_pil, actions_applied


def save_protected_image(image: Image.Image, session_id: str) -> Path:
    """Save the protected image to a temporary file on disk without EXIF metadata.

    Args:
        image: Protected PIL Image.
        session_id: Session identifier.

    Returns:
        Path: Path to the saved protected image file.
    """
    output_path = PROCESSED_IMAGE_DIR / f"protected_{session_id}.jpg"
    # Explicitly do NOT supply exif parameter when saving
    image.save(output_path, format="JPEG", quality=95)
    return output_path


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """Convert a PIL Image to raw bytes without EXIF metadata.

    Args:
        image: Protected PIL Image.
        format: Target image format (JPEG or PNG).

    Returns:
        bytes: Raw image file bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=95)
    buffer.seek(0)
    return buffer.getvalue()
