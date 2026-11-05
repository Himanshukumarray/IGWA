# IGWA OCR API

The OCR service of the IGWA police software backend. A client uploads a PDF or image, the backend validates it, sends it to Gemini Flash for OCR, keeps the result temporarily in memory and returns a `document_id`. The OCR text is then fetched with that ID.

## OCR flow

```text
POST /ocr/upload -> validate file -> Gemini Flash OCR -> store in memory -> document_id
GET  /ocr/{document_id} -> OCR result from memory
```

Uploaded files are processed in memory and never written to disk.

## Folder structure

```text
IGWA/
├── services/ocr/      routes.py, service.py, gemini_client.py, schemas.py, utils.py
├── config/            settings.py
├── main.py
├── requirements.txt
├── .env / .env.example
└── README.md
```

## Requirements

- Python 3.10+
- A Gemini API key from https://aistudio.google.com/apikey

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Gemini key configuration

Copy `.env.example` to `.env` and set your key:

```env
APP_NAME=IGWA
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash

MAX_FILE_SIZE_MB=20
```

## Run

```powershell
python -m uvicorn main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/ocr/upload` | Upload PDF/PNG/JPG/JPEG/WEBP, run OCR, get `document_id` |
| GET | `/ocr/{document_id}` | Fetch the stored OCR result |
| GET | `/health` | Liveness check (does not call Gemini) |

Errors: 400 (unsupported/empty file), 413 (too large), 404 (unknown document), 502 (Gemini failure).

## Example cURL

```bash
curl -X POST "http://localhost:8000/ocr/upload" -F "file=@sample.pdf"
curl "http://localhost:8000/ocr/3a6a76b5-0c20-4a57-975a-39d13bfbf53e"
```

Upload response:

```json
{
  "success": true,
  "message": "OCR processing completed successfully",
  "document_id": "3a6a76b5-0c20-4a57-975a-39d13bfbf53e",
  "status": "completed",
  "filename": "sample.pdf"
}
```

Get OCR response:

```json
{
  "success": true,
  "document_id": "3a6a76b5-0c20-4a57-975a-39d13bfbf53e",
  "status": "completed",
  "filename": "sample.pdf",
  "ocr_data": {
    "pages": [{ "page_number": 1, "text": "..." }],
    "full_text": "..."
  }
}
```

## Temporary memory storage

This version stores OCR results in application memory.

If the FastAPI server restarts, all OCR results are lost.

This is intentional for the current development phase.
