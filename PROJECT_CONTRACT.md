# PROJECT_CONTRACT.md — Hotel Document PII Analysis System

## System Overview
A backend pipeline that accepts hotel document images, extracts text via OCR,
detects personally identifiable information (PII), and scores privacy risk.

## Member Responsibilities

| Member | Role | Files Owned |
|--------|------|-------------|
| Member 1 | API / Ingestion Layer | `backend/app/api/` |
| Member 2 | OCR & PII Detection | `backend/app/services/ocr_service.py`, `backend/app/services/pii_detector.py`, `backend/app/models/findings.py` |
| Member 3 | Image Pre-processing | `backend/app/services/image_processor.py` |
| Member 4 | Risk Engine | `backend/app/services/risk_engine.py` |
| Member 5 | Frontend / UI | `frontend/` |

## Member 2 Deliverables
- [x] `backend/app/models/findings.py` — shared data models
- [x] `backend/app/services/ocr_service.py` — Tesseract OCR wrapper
- [x] `backend/app/services/pii_detector.py` — regex/rule-based PII detection
- [x] `backend/tests/unit/test_ocr_service.py`
- [x] `backend/tests/unit/test_pii_detector.py`

## PII Categories Detected
- PHONE_NUMBER
- EMAIL
- CARD_NUMBER (Luhn-validated)
- PASSPORT_NUMBER
- BOOKING_REFERENCE
- ADDRESS
- ROOM_NUMBER

## Interface Constraints
- Do NOT log `Finding.text` directly — use `Finding.to_safe_dict()`.
- Do NOT store full OCR text corpus.
- `Finding.confidence` is in [0.0, 1.0].
- All bounding boxes are in pixel coordinates of the source image.
