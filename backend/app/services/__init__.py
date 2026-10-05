"""Services Package for Privacy Analysis and Image Protection."""

from backend.app.services.risk_engine import (
    calculate_risk_score,
    deduplicate_findings,
    CATEGORY_WEIGHTS,
)
from backend.app.services.protection_service import (
    protect_image,
    save_protected_image,
    image_to_bytes,
)
from backend.app.services.ocr_detector import (
    scan_image_text,
    detect_pii_in_text,
)
from backend.app.services.metadata_detector import (
    detect_exif_gps,
    detect_qr_codes,
    detect_barcodes,
    scan_metadata_and_codes,
)
from backend.app.services.gemma_reasoning import (
    gemma_reasoner,
    GemmaContextualReasoner,
)

__all__ = [
    "calculate_risk_score",
    "deduplicate_findings",
    "CATEGORY_WEIGHTS",
    "protect_image",
    "save_protected_image",
    "image_to_bytes",
    "scan_image_text",
    "detect_pii_in_text",
    "detect_exif_gps",
    "detect_qr_codes",
    "detect_barcodes",
    "scan_metadata_and_codes",
    "gemma_reasoner",
    "GemmaContextualReasoner",
]
