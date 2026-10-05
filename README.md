
# Accidental Oversharing Detector (`Summa`)

An AI-driven and computer-vision powered privacy protection tool that detects sensitive personal data (PII, GPS coordinates, QR codes, and barcodes) in photos before they are posted online.

## Workflow
1. **Upload** image.
2. **Scan** for visible text, QR/barcodes, and EXIF metadata.
3. **Reason** contextually using Gemma 4.
4. **Score** the privacy risk from 0–100.
5. **Protect** sensitive regions via blurring/redaction and EXIF stripping.
6. **Download** safe, privacy-compliant image.

## Project Structure
```text
Summa/
├── frontend/               # Next.js / React UI (Member 1)
├── backend/                # FastAPI backend service (Member 4)
│   └── app/
│       ├── api/            # API routers & endpoints (v1)
│       ├── core/           # Configuration, security, logging
│       ├── models/         # SQLAlchemy database models
│       ├── schemas/        # Pydantic data schemas
│       ├── services/       # Core business logic & integrations
│       └── main.py         # FastAPI application entrypoint
├── database/               # SQLite database storage & migrations
├── sample_images/          # Benchmark & test images
├── tests/                  # Automated backend and integration tests
├── docs/                   # Documentation and API references
├── PROJECT_CONTRACT.md     # Team contracts and shared data schemas
├── requirements.txt        # Python backend dependencies
└── README.md               # Project documentation
```

## Getting Started

### 1. Backend Setup
```bash
# Navigate to the Summa root directory
cd Summa

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend development server
uvicorn backend.app.main:app --reload --port 8000
```

### 2. Verify Health Endpoint
Open your browser or make a curl request:
```bash
curl http://127.0.0.1:8000/api/v1/health
```
Expected response:
```json
{"status": "ok"}
```
Interactive API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

