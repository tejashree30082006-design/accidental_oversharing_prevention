# Member 4 Interface Guide — OCR & PII Service

## How to Consume Member 2's Output

### Import
```python
from app.services.pii_detector import detect_pii, detect_pii_in_image
from app.services.ocr_service import run_ocr
from app.models.findings import Finding, EntityType, BoundingBox
```

### Full Pipeline (recommended)
```python
findings: list[Finding] = detect_pii_in_image(image_path_or_pil)
```

### Split Pipeline
```python
words = run_ocr(image, page=0, lang="eng")
findings = detect_pii(words)
```

---

## Finding Object Fields

| Field | Type | Description | Safe to Log? |
|-------|------|-------------|--------------|
| `entity_type` | `EntityType` | PII category (enum) | YES |
| `text` | `str` | Raw matched value | **NO — NEVER LOG** |
| `bbox` | `BoundingBox` | Pixel location in image | YES |
| `confidence` | `float` | Combined confidence [0.0-1.0] | YES |
| `context_snippet` | `str` | Surrounding words (REDACTED) | YES |
| `source_page` | `int` | 0-indexed page number | YES |
| `raw_pattern_name` | `str` | Debug: which rule matched | YES |

### Serialisation
```python
# For API response / risk engine (contains raw PII text):
finding.to_dict()

# For logging / audit (text replaced with [REDACTED]):
finding.to_safe_dict()
```

---

## EntityType Values
```python
EntityType.PHONE_NUMBER        # "PHONE_NUMBER"
EntityType.EMAIL               # "EMAIL"
EntityType.CARD_NUMBER         # "CARD_NUMBER"
EntityType.PASSPORT_NUMBER     # "PASSPORT_NUMBER"
EntityType.BOOKING_REFERENCE   # "BOOKING_REFERENCE"
EntityType.ADDRESS             # "ADDRESS"
EntityType.ROOM_NUMBER         # "ROOM_NUMBER"
```

---

## BoundingBox
```python
bbox.x        # int — left pixel
bbox.y        # int — top pixel
bbox.width    # int — width in pixels
bbox.height   # int — height in pixels
bbox.as_dict()  # {"x": ..., "y": ..., "width": ..., "height": ...}
```

---

## Confidence Scoring
- `confidence` = `pattern_base_confidence × avg(OCR word confidences)`
- Range: `0.0` to `1.0`
- Findings below `0.30` are suppressed before being returned.
- Recommended risk thresholds for Member 4:
  - `>= 0.80` → HIGH risk
  - `0.50 – 0.79` → MEDIUM risk
  - `0.30 – 0.49` → LOW risk

---

## Security Contract
- `Finding.text` contains raw PII. Never log, print, or store verbatim.
- Always use `Finding.to_safe_dict()` for any logging or audit trail.
- The OCR corpus string is also sensitive — never store or return it from APIs.
