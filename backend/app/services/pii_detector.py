"""
pii_detector.py — Sensitive information candidate detector.

Responsibilities (Member 2):
  - Accept a list of OCRWord objects (output of ocr_service.run_ocr).
  - Reconstruct a text corpus from the words.
  - Run regex / rule-based patterns to locate PII candidates.
  - Use context clues (surrounding keywords) to avoid false positives.
    e.g. a 10-digit number next to "PHONE" or "MOB" is a phone; a bare
    10-digit number with no context is not automatically flagged.
  - Map each match back to the OCRWord bounding boxes.
  - Return a list of Finding objects (one per distinct PII candidate).

NOT responsible for:
  - Risk scoring (Member 4).
  - Storing or logging the matched text values.
  - Modifying or returning modified images.

Detected entity types:
  PHONE_NUMBER        — context-triggered or clearly formatted phone numbers
  EMAIL               — standard email addresses
  CARD_NUMBER         — credit / debit card numbers (Luhn-validated)
  PASSPORT_NUMBER     — standard passport formats (with or without keyword)
  BOOKING_REFERENCE   — PNR / booking / confirmation codes with keyword context
  ADDRESS             — multi-word postal addresses (keyword-triggered)
  ROOM_NUMBER         — room / suite numbers (keyword-triggered)

Design principles:
  - Do NOT flag every standalone number.  Context keywords are required for
    ambiguous patterns (phone, booking, room, address).
  - Unambiguous patterns (email, card with Luhn check) may fire without context.
  - Combined confidence = OCR confidence of matched words (averaged).
  - context_snippet is built from neighbours of the match, with the matched
    value replaced by [REDACTED] so it is safe to log.

Interface for Member 4:
  detect_pii(words: List[OCRWord]) -> List[Finding]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Pattern, Tuple

from app.models.findings import BoundingBox, EntityType, Finding, OCRWord
from app.services.ocr_service import get_words_in_char_span, words_to_plain_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Context window: number of characters to include on each side for snippet
CONTEXT_WINDOW_CHARS: int = 40

# Minimum combined OCR confidence to emit a Finding
MIN_FINDING_CONFIDENCE: float = 0.30


# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PIIPattern:
    """A single regex rule for detecting a PII entity type.

    Attributes:
        name:             Human-readable rule name (used in Finding.raw_pattern_name).
        entity_type:      The EntityType this rule detects.
        pattern:          Compiled regex. The FIRST capture group (group 1) must
                          contain the canonical sensitive value.
        context_keywords: Optional list of keyword regexes. If non-empty, the
                          match is only accepted if at least one keyword appears
                          within CONTEXT_RADIUS characters of the match.
        context_radius:   Character radius to search for context keywords.
        requires_luhn:    If True, the match is validated with the Luhn algorithm.
        base_confidence:  Pattern-level confidence multiplier (0.0-1.0). Lower
                          values indicate patterns that fire more broadly.
    """

    name: str
    entity_type: EntityType
    pattern: Pattern
    context_keywords: Tuple[str, ...] = ()
    context_radius: int = 80
    requires_luhn: bool = False
    base_confidence: float = 1.0


# ---------------------------------------------------------------------------
# Helper: Luhn algorithm for card validation
# ---------------------------------------------------------------------------

def _luhn_check(number_str: str) -> bool:
    """Return True if the digit string passes the Luhn checksum."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Pattern Registry
# ---------------------------------------------------------------------------

def _build_patterns() -> List[PIIPattern]:
    """Return the ordered list of PII detection patterns.

    Patterns are evaluated in order; the first matching pattern for a given
    span wins (no double-counting). Order matters for specificity.
    """
    return [
        # ------------------------------------------------------------------
        # 1. EMAIL — unambiguous structural pattern; no context required
        # ------------------------------------------------------------------
        PIIPattern(
            name="email_standard",
            entity_type=EntityType.EMAIL,
            pattern=re.compile(
                r"\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b",
                re.IGNORECASE,
            ),
            context_keywords=(),
            base_confidence=0.97,
        ),

        # ------------------------------------------------------------------
        # 2. CARD_NUMBER — 13-19 digits, optional spaces/dashes; Luhn check
        # ------------------------------------------------------------------
        PIIPattern(
            name="card_number_spaced",
            entity_type=EntityType.CARD_NUMBER,
            pattern=re.compile(
                r"\b((?:\d[ \-]?){13,19}\d)\b",
                re.IGNORECASE,
            ),
            requires_luhn=True,
            base_confidence=0.92,
        ),

        # ------------------------------------------------------------------
        # 3. PASSPORT_NUMBER
        #    Format: 1-2 letters + 6-7 digits (with optional context keyword)
        #    Context boosts confidence; structural match alone is accepted
        #    (since passports follow strict international standards).
        # ------------------------------------------------------------------
        PIIPattern(
            name="passport_with_keyword",
            entity_type=EntityType.PASSPORT_NUMBER,
            pattern=re.compile(
                r"(?:passport|pass\s*no\.?|passport\s*no\.?|p\.?n\.?o\.?)\s*[:\-]?\s*([A-Z]{1,2}\d{6,7})\b",
                re.IGNORECASE,
            ),
            base_confidence=0.95,
        ),
        PIIPattern(
            name="passport_standalone",
            entity_type=EntityType.PASSPORT_NUMBER,
            pattern=re.compile(
                r"\b([A-Z]{1,2}\d{7})\b",
                re.IGNORECASE,
            ),
            context_keywords=(
                r"passport", r"pass\s*no", r"travel\s*doc", r"visa",
            ),
            context_radius=100,
            base_confidence=0.75,
        ),

        # ------------------------------------------------------------------
        # 4. BOOKING_REFERENCE — PNR / booking codes (keyword required)
        #    Format: 5-8 alphanumeric characters, context MANDATORY to avoid
        #    treating every short alphanumeric as a booking code.
        # ------------------------------------------------------------------
        PIIPattern(
            name="booking_ref_with_keyword",
            entity_type=EntityType.BOOKING_REFERENCE,
            pattern=re.compile(
                r"(?:pnr|booking\s*(?:ref(?:erence)?|id|no\.?)|confirmation\s*(?:no\.?|code|id)|ref(?:erence)?\s*no\.?)\s*[:\-]?\s*([A-Z0-9]{5,8})\b",
                re.IGNORECASE,
            ),
            base_confidence=0.94,
        ),

        # ------------------------------------------------------------------
        # 5. ROOM_NUMBER — context-triggered only
        #    Format: 1-5 digit (or alphanumeric) room identifiers
        # ------------------------------------------------------------------
        PIIPattern(
            name="room_number_with_keyword",
            entity_type=EntityType.ROOM_NUMBER,
            pattern=re.compile(
                r"(?:room\s*(?:no\.?|number|#)?|suite\s*(?:no\.?|#)?|cabin\s*(?:no\.?|#)?|rm\.?\s*(?:no\.?|#)?)\s*[:\-]?\s*([A-Z]?\d{1,5}[A-Z]?)\b",
                re.IGNORECASE,
            ),
            base_confidence=0.92,
        ),

        # ------------------------------------------------------------------
        # 6. PHONE_NUMBER — context-triggered to avoid false positives
        #    Accepts international (+91 ...), local (10-digit), and formatted
        #    numbers, but ONLY when a phone-related keyword is nearby.
        # ------------------------------------------------------------------
        PIIPattern(
            name="phone_with_keyword",
            entity_type=EntityType.PHONE_NUMBER,
            pattern=re.compile(
                r"(?:(?:ph(?:one)?|mob(?:ile)?|tel(?:ephone)?|call|contact|fax|whatsapp)\s*(?:no\.?|number|#)?\s*[:\-]?\s*)"
                r"(\+?[\d][\d\s\-\(\)\.]{7,17}[\d])\b",
                re.IGNORECASE,
            ),
            base_confidence=0.91,
        ),
        PIIPattern(
            name="phone_international",
            entity_type=EntityType.PHONE_NUMBER,
            pattern=re.compile(
                r"(\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,5}[\s\-]?\d{4,6})\b",
            ),
            context_keywords=(
                r"ph(?:one)?", r"mob(?:ile)?", r"tel", r"contact", r"call",
            ),
            context_radius=80,
            base_confidence=0.85,
        ),

        # ------------------------------------------------------------------
        # 7. ADDRESS — multi-part, keyword-triggered
        #    Captures a sequence of words following an address label.
        #    This is intentionally broad; Member 4 applies risk scoring.
        # ------------------------------------------------------------------
        PIIPattern(
            name="address_with_keyword",
            entity_type=EntityType.ADDRESS,
            pattern=re.compile(
                r"(?:address|addr\.?|residence|flat\s*no\.?|door\s*no\.?|plot\s*no\.?|house\s*no\.?|street|road|avenue|lane|nagar|colony|district|state|pin\s*code|zip)\s*[:\-]?\s*"
                r"([A-Za-z0-9\s,\.\-/]{8,80})",
                re.IGNORECASE,
            ),
            base_confidence=0.80,
        ),
    ]


_PATTERNS: List[PIIPattern] = _build_patterns()


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _has_context_keyword(
    corpus: str,
    match_start: int,
    match_end: int,
    keywords: Tuple[str, ...],
    radius: int,
) -> bool:
    """Return True if any keyword pattern appears within ``radius`` chars of match."""
    window_start = max(0, match_start - radius)
    window_end = min(len(corpus), match_end + radius)
    window = corpus[window_start:window_end]
    for kw in keywords:
        if re.search(kw, window, re.IGNORECASE):
            return True
    return False


def _build_context_snippet(
    corpus: str,
    match_start: int,
    match_end: int,
    window: int = CONTEXT_WINDOW_CHARS,
) -> str:
    """Return surrounding text with the matched span replaced by [REDACTED].

    Safe to log.
    """
    left = corpus[max(0, match_start - window): match_start].strip()
    right = corpus[match_end: min(len(corpus), match_end + window)].strip()
    parts = []
    if left:
        parts.append(left)
    parts.append("[REDACTED]")
    if right:
        parts.append(right)
    return " ... ".join(parts)


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def detect_pii(words: List[OCRWord]) -> List[Finding]:
    """Detect PII candidates in a list of OCR words.

    This is the primary entry point for the PII detection pipeline.

    Algorithm:
      1. Join words into a text corpus (space-separated).
      2. For each PIIPattern, run regex over the corpus.
      3. Validate context keywords (if required).
      4. Validate Luhn checksum (for CARD_NUMBER).
      5. Map matched character spans back to OCRWord bounding boxes.
      6. Compute combined confidence (pattern × average word OCR confidence).
      7. Emit a Finding for each validated match.

    Args:
        words: Ordered list of OCRWord objects from ocr_service.run_ocr().

    Returns:
        List of Finding objects. May be empty if no PII is detected.
        Ordered by (source_page, bbox.y, bbox.x).

    Security:
        This function does NOT log matched text values. Only safe metadata
        (entity_type, confidence, context_snippet) is logged.
    """
    if not words:
        return []

    corpus = words_to_plain_text(words)
    findings: List[Finding] = []

    # Track matched spans to avoid duplicate Findings from overlapping patterns
    matched_spans: List[Tuple[int, int]] = []

    for pattern in _PATTERNS:
        for m in pattern.pattern.finditer(corpus):
            # The canonical value is always in group 1
            try:
                value = m.group(1)
                value_start = m.start(1)
                value_end = m.end(1)
            except IndexError:
                value = m.group(0)
                value_start = m.start(0)
                value_end = m.end(0)

            if not value or not value.strip():
                continue

            # De-duplicate: skip if this span overlaps a previously matched span
            if _overlaps_any(value_start, value_end, matched_spans):
                continue

            # Context keyword check (only when pattern requires it)
            if pattern.context_keywords:
                if not _has_context_keyword(
                    corpus,
                    m.start(0),
                    m.end(0),
                    pattern.context_keywords,
                    pattern.context_radius,
                ):
                    logger.debug(
                        "Pattern '%s': skipping match (no context keyword found)",
                        pattern.name,
                    )
                    continue

            # Luhn validation for card numbers
            if pattern.requires_luhn:
                digits_only = re.sub(r"\D", "", value)
                if not _luhn_check(digits_only):
                    logger.debug(
                        "Pattern '%s': Luhn check failed, skipping", pattern.name
                    )
                    continue

            # Map value span to OCR word indices
            word_indices = get_words_in_char_span(words, value_start, value_end)
            if not word_indices:
                # Fallback: no words found for span — use rough estimate
                logger.debug(
                    "Pattern '%s': no OCR words found for char span [%d, %d]",
                    pattern.name,
                    value_start,
                    value_end,
                )
                continue

            # Compute bounding box: union of all matching word boxes
            matched_boxes = [words[i].bbox for i in word_indices]
            combined_bbox = BoundingBox.union(matched_boxes)

            # Compute confidence: pattern_base × average OCR confidence of matched words
            avg_ocr_conf = sum(words[i].confidence for i in word_indices) / len(word_indices)
            confidence = min(1.0, pattern.base_confidence * avg_ocr_conf)

            if confidence < MIN_FINDING_CONFIDENCE:
                logger.debug(
                    "Pattern '%s': combined confidence %.3f below threshold %.3f, skipping",
                    pattern.name,
                    confidence,
                    MIN_FINDING_CONFIDENCE,
                )
                continue

            # Build safe context snippet (no sensitive value included)
            snippet = _build_context_snippet(corpus, value_start, value_end)

            # Source page from the first matched word
            source_page = words[word_indices[0]].page

            finding = Finding(
                entity_type=pattern.entity_type,
                text=value.strip(),          # DO NOT LOG THIS FIELD
                bbox=combined_bbox,
                confidence=round(confidence, 4),
                context_snippet=snippet,
                source_page=source_page,
                raw_pattern_name=pattern.name,
                ocr_word_indices=word_indices,
            )
            findings.append(finding)
            matched_spans.append((value_start, value_end))

            logger.info(
                "PII detected: type=%s pattern=%s confidence=%.3f page=%d",
                pattern.entity_type.value,
                pattern.name,
                confidence,
                source_page,
            )

    # Sort findings: page → top → left
    findings.sort(key=lambda f: (f.source_page, f.bbox.y, f.bbox.x))
    return findings


# ---------------------------------------------------------------------------
# Convenience: run full pipeline (OCR + PII) on an image
# ---------------------------------------------------------------------------

def detect_pii_in_image(
    image,
    page: int = 0,
    lang: str = "eng",
) -> List[Finding]:
    """Run OCR + PII detection on a single image.

    Convenience wrapper. Equivalent to:
        words = ocr_service.run_ocr(image, page=page, lang=lang)
        findings = detect_pii(words)

    Args:
        image:  PIL Image, bytes, file path, or numpy ndarray.
        page:   0-indexed page number.
        lang:   Tesseract language code.

    Returns:
        List of Finding objects sorted by position.
    """
    from app.services.ocr_service import run_ocr  # avoid circular at module level

    words = run_ocr(image, page=page, lang=lang)
    return detect_pii(words)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _overlaps_any(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    """Return True if [start, end) overlaps any span in the list."""
    for s, e in spans:
        if start < e and end > s:
            return True
    return False
