# PROJECT CONTRACT: ACCIDENTAL OVERSHARING DETECTOR

## 1. Project Overview
The Accidental Oversharing Detector scans user-uploaded photos to identify sensitive information (PII, GPS, QR/barcodes) that might accidentally be exposed online, assigns a privacy risk score, and provides an automatic protection/redaction engine with safe image downloads.

## 2. Team Roles & Responsibilities
- **MEMBER 1:** Frontend / UI (Next.js, React, Tailwind CSS)
- **MEMBER 2:** OCR / Text / Sensitive Information Detection (Pytesseract, regex, PII detection)
- **MEMBER 3:** EXIF / GPS / QR / Barcode Detection (Pillow EXIF, OpenCV QRCodeDetector, pyzbar)
- **MEMBER 4 (Current):** Backend (FastAPI), SQLite database, Risk scoring, Image protection, Gemma 4 contextual reasoning, Service integration, E2E testing.

## 3. Global Finding Format
All detection modules (OCR, QR, Barcode, EXIF, Gemma) must emit findings conforming to this standard schema:

```json
{
  "id": "finding-001",
  "category": "PHONE_NUMBER",
  "severity": "HIGH",
  "confidence": 0.96,
  "description": "A phone number may be publicly exposed.",
  "source": "OCR",
  "bbox": {
    "x": 120,
    "y": 250,
    "width": 180,
    "height": 35
  },
  "recommended_action": "BLUR"
}
```

### Enumerations:
- **Source:** `OCR`, `QR`, `BARCODE`, `EXIF`, `GEMMA`
- **Severity:** `LOW`, `MEDIUM`, `HIGH`
- **Recommended Action:** `BLUR`, `REDACT`, `REMOVE_METADATA`, `REVIEW`

## 4. Risk Score Weights
Risk scores range from 0 to 100 based on unique findings (no double-counting):
- `GPS_LOCATION`: 35
- `CARD_NUMBER`: 30
- `PASSPORT_NUMBER`: 30
- `QR_CODE`: 25
- `BOOKING_REFERENCE`: 20
- `BARCODE`: 20
- `ADDRESS`: 15
- `PHONE_NUMBER`: 15
- `ROOM_NUMBER`: 15
- `EMAIL`: 10
- `OTHER`: 10

### Risk Levels:
- **0–29:** LOW
- **30–59:** MEDIUM
- **60–100:** HIGH

## 5. API Endpoints
- `GET /api/v1/health` - Check backend service health
- `POST /api/v1/scan` - Upload and scan an image for privacy risks
- `GET /api/v1/scan/{scan_id}` - Retrieve stored scan results
- `POST /api/v1/protect` - Create and download a protected version of the image

## 6. Privacy & Data Handling Rules
- Never permanently store original uploaded images, raw GPS coordinates, decoded QR payloads, or unmasked sensitive credentials.
- Temporary files must be cleaned up after processing.
- The database stores only analytical findings and scan session metadata.
