"""
ocr_service.py — OCR extraction layer for hotel document / image analysis.

Responsibilities (Member 2):
  - Accept a PIL Image or raw image bytes.
  - Run Tesseract OCR via pytesseract.
  - Return a list of OCRWord objects with accurate bounding boxes and
    per-word confidence scores.
  - Pre-process the image to improve OCR accuracy (grayscale, denoise, threshold).
  - Support multi-page PDFs via page parameter.

NOT responsible for:
  - Identifying which words are PII (see pii_detector.py).
  - Risk scoring (see risk_engine — Member 4).
  - Storing or logging raw OCR text.

Interface for Member 4:
  run_ocr(image) -> List[OCRWord]
    Each OCRWord has: .text, .bbox (BoundingBox), .confidence, .page

Dependencies:
  pip install pytesseract pillow opencv-python-headless numpy
  Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

try:
    import pytesseract
    from PIL import Image
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

from app.models.findings import BoundingBox, OCRWord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum OCR confidence (0-100 Tesseract scale) to include a word.
# Words below this threshold are dropped to reduce noise.
MIN_WORD_CONFIDENCE: int = 30

# Tesseract config: OEM 3 = LSTM engine; PSM 6 = uniform block of text.
# Adjust PSM per document type if needed (e.g., PSM 11 for sparse text).
TESSERACT_CONFIG: str = "--oem 3 --psm 6"

# Image pre-processing: upscale factor for small images (improves OCR accuracy)
UPSCALE_FACTOR: float = 2.0
UPSCALE_MIN_DIMENSION: int = 600  # only upscale if shortest side < this value


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def _preprocess_image(image: "Image.Image") -> "np.ndarray":
    """Convert PIL Image to a binarised OpenCV array for better OCR accuracy.

    Pipeline:
      1. Convert to grayscale.
      2. Upscale if the image is small (< UPSCALE_MIN_DIMENSION on shortest side).
      3. Apply Gaussian blur to reduce noise.
      4. Otsu adaptive threshold for clean black/white output.

    Args:
        image: PIL Image (any mode).

    Returns:
        numpy ndarray (grayscale, thresholded) ready for pytesseract.
    """
    # Convert to RGB then numpy
    img_array = np.array(image.convert("RGB"))

    # Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Upscale small images
    h, w = gray.shape
    if min(h, w) < UPSCALE_MIN_DIMENSION:
        gray = cv2.resize(
            gray,
            (int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR)),
            interpolation=cv2.INTER_CUBIC,
        )

    # Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarise (Otsu threshold)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def _scale_bbox(bbox: BoundingBox, scale: float) -> BoundingBox:
    """Reverse the upscale transform on a bounding box."""
    if scale == 1.0:
        return bbox
    return BoundingBox(
        x=int(bbox.x / scale),
        y=int(bbox.y / scale),
        width=int(bbox.width / scale),
        height=int(bbox.height / scale),
    )


# ---------------------------------------------------------------------------
# Core OCR function
# ---------------------------------------------------------------------------

def run_ocr(
    image: Union["Image.Image", bytes, str, Path],
    page: int = 0,
    lang: str = "eng",
    min_confidence: int = MIN_WORD_CONFIDENCE,
    tesseract_config: str = TESSERACT_CONFIG,
) -> List[OCRWord]:
    """Run Tesseract OCR on an image and return per-word results.

    This is the primary entry point for the OCR pipeline.

    Args:
        image:            PIL Image, raw bytes, file path (str or Path), or
                          numpy ndarray. Bytes/path are auto-converted.
        page:             0-indexed page number (for multi-page document tracking).
        lang:             Tesseract language code(s), e.g. "eng", "eng+hin".
        min_confidence:   Minimum word-level confidence (0-100). Words below
                          this are discarded.
        tesseract_config: Raw Tesseract CLI config string.

    Returns:
        List of OCRWord objects, one per recognised word above min_confidence.
        Ordered left-to-right, top-to-bottom (Tesseract natural order).

    Raises:
        RuntimeError: If pytesseract or Tesseract binary is not available.
        ValueError:   If the image argument type is not supported.
    """
    if not _TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract or Pillow is not installed. "
            "Run: pip install pytesseract pillow"
        )

    pil_image = _load_image(image)
    processed = _preprocess_image(pil_image)

    # Determine scale factor applied during pre-processing
    orig_w, orig_h = pil_image.size
    proc_h, proc_w = processed.shape[:2]
    scale_x = proc_w / orig_w
    scale_y = proc_h / orig_h
    # We use uniform upscaling so scale_x == scale_y; average for safety
    scale = (scale_x + scale_y) / 2.0

    # Run Tesseract — get data with bounding boxes and confidence per word
    try:
        data = pytesseract.image_to_data(
            processed,
            lang=lang,
            config=tesseract_config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract binary not found. Install from: "
            "https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    words: List[OCRWord] = []
    num_tokens = len(data["text"])

    for i in range(num_tokens):
        raw_text: str = data["text"][i]
        conf_raw: int = data["conf"][i]

        # Tesseract returns -1 for block/line/paragraph level rows
        if conf_raw < 0:
            continue

        # Filter whitespace-only tokens
        cleaned = raw_text.strip()
        if not cleaned:
            continue

        # Apply minimum confidence filter
        if conf_raw < min_confidence:
            logger.debug(
                "Dropping low-confidence token (conf=%d, threshold=%d)",
                conf_raw,
                min_confidence,
            )
            continue

        # Build bounding box in *processed image* space
        proc_bbox = BoundingBox(
            x=data["left"][i],
            y=data["top"][i],
            width=data["width"][i],
            height=data["height"][i],
        )

        # Map back to original image coordinates
        orig_bbox = _scale_bbox(proc_bbox, scale)

        # Normalise confidence to [0.0, 1.0]
        confidence = conf_raw / 100.0

        words.append(
            OCRWord(
                text=cleaned,
                bbox=orig_bbox,
                confidence=confidence,
                page=page,
            )
        )

    logger.info(
        "OCR complete: page=%d, words_found=%d (min_conf=%d)",
        page,
        len(words),
        min_confidence,
    )
    return words


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_image(source: Union["Image.Image", bytes, str, Path, "np.ndarray"]) -> "Image.Image":
    """Normalise various image input types to a PIL Image."""
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, (str, Path)):
        return Image.open(Path(source))
    if isinstance(source, bytes):
        import io
        return Image.open(io.BytesIO(source))
    if isinstance(source, np.ndarray):
        # OpenCV arrays are BGR; PIL expects RGB
        if source.ndim == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        return Image.fromarray(source)
    raise ValueError(
        f"Unsupported image type: {type(source).__name__}. "
        "Pass PIL.Image, bytes, file path, or numpy ndarray."
    )


# ---------------------------------------------------------------------------
# Utility: build a plain text corpus from OCR words (for context extraction)
# ---------------------------------------------------------------------------

def words_to_plain_text(words: List[OCRWord]) -> str:
    """Join OCR words into a single spaced string.

    Used internally by pii_detector to run regex over a text corpus.
    Do NOT store or log the return value — it may contain PII.
    """
    return " ".join(w.text for w in words)


def get_word_at_char_offset(
    words: List[OCRWord], char_offset: int
) -> Optional[int]:
    """Return the index of the OCRWord that contains the given character offset.

    Used by pii_detector to map regex match positions back to OCR bounding boxes.

    Args:
        words:       Ordered list of OCRWord objects.
        char_offset: Character offset in the string produced by words_to_plain_text().

    Returns:
        Index into ``words``, or None if out of range.
    """
    pos = 0
    for idx, word in enumerate(words):
        end = pos + len(word.text)
        if pos <= char_offset < end:
            return idx
        pos = end + 1  # +1 for the space separator
    return None


def get_words_in_char_span(
    words: List[OCRWord], start: int, end: int
) -> List[int]:
    """Return indices of all OCRWords that overlap with [start, end) in the corpus.

    Used by pii_detector to collect all word boxes for a multi-word match.
    """
    result: List[int] = []
    pos = 0
    for idx, word in enumerate(words):
        word_end = pos + len(word.text)
        # Check overlap: word spans [pos, word_end)
        if pos < end and word_end > start:
            result.append(idx)
        pos = word_end + 1
    return result
