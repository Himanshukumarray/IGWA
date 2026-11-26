# IGWA OCR API

The OCR service of the IGWA police software backend. Upload a PDF or image and get the extracted text and
metadata. Results are kept temporarily in memory; uploaded files are never written to disk.

## Flow

```text
POST /python/ocr/upload
  -> validate file
  -> OCR the whole document: Gemini (returns text + metadata), Groq if Gemini fails
  -> store the result in memory and respond with a document_id / job_id
  -> background text-only metadata request, only if the OCR response carried no usable metadata

GET /python/ocr/{document_id}      -> stored OCR result
GET /python/ocr/{job_id}/metadata  -> metadata status / result
```

## Folder structure

```text
IGWA/
├── services/ocr/
│   ├── routes.py              upload endpoint
│   ├── service.py             OCR and metadata flow
│   ├── gemini_client.py       Gemini prompts, response validation, page recovery
│   ├── gemini_key_manager.py  multi-key failover
│   ├── groq_client.py         Groq fallback
│   ├── application.py         cuts the attached application out of the OCR pages
│   ├── schemas.py             request/response models
│   └── utils.py               upload validation
├── config/settings.py
├── tests/                     mocked tests (no network, no real keys)
├── main.py
├── requirements.txt
└── .env / .env.example
```

## Setup (Windows PowerShell)

Requires Python 3.10+ and a Gemini API key from https://aistudio.google.com/apikey.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then set your keys
python -m uvicorn main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

Tests: `python -m pytest`

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/python/ocr/upload` | Upload PDF/PNG/JPG/JPEG/WEBP, get text and metadata |
| GET | `/python/ocr/{document_id}` | Fetch the stored OCR result |
| GET | `/python/ocr/{job_id}/metadata` | Poll background metadata status/result (`job_id` == `document_id`) |
| GET | `/health` (also `/python/ocr/health`) | Liveness check and Gemini key counts |

Errors: 400 (unsupported/empty file), 413 (too large), 404 (unknown document), 502 (OCR failed on every provider).

```bash
curl -X POST "http://localhost:8000/python/ocr/upload" -F "file=@sample.pdf"
```

```json
{
  "success": true,
  "message": "OCR and metadata extraction completed successfully",
  "document_id": "3a6a76b5-0c20-4a57-975a-39d13bfbf53e",
  "job_id": "3a6a76b5-0c20-4a57-975a-39d13bfbf53e",
  "status": "completed",
  "filename": "sample.pdf",
  "ocr_model": "gemini-3.1-flash-lite",
  "ocr_provider": "gemini",
  "ocr_data": { "pages": [{ "page_number": 1, "text": "..." }], "full_text": "..." },
  "metadata": { "document_type": "FIR", "fir_number": "..." },
  "metadata_model": "gemini-3.1-flash-lite",
  "metadata_status": "completed",
  "metadata_error": null,
  "isFir": true
}
```

## OCR

- The whole PDF/image is sent in one request. Prompts forbid translation and transliteration: Hindi stays in
  Devanagari, English stays English, digits are copied exactly. `full_text` is joined from the pages in Python.
- Every page must be returned exactly once. If Gemini skips pages, only those pages are re-OCRed. If the output
  is truncated or unparsable, the PDF is OCRed page by page.
- If Gemini cannot serve the request, Groq OCRs the document page by page. Whether `GROQ_MODEL` accepts image
  input is not verified by this code: check Groq's model documentation.
- Review critical identifiers: text quality is whatever the provider returns.

## Metadata

Gemini normally returns the metadata with the OCR text, so the upload response has `metadata_status: "completed"`.
Otherwise (Groq did the OCR, or the metadata was missing or invalid) the status is `pending`: poll
`GET /python/ocr/{job_id}/metadata` every 2-3 s until it is `completed` or `failed`. `skipped` means no text was
found. Statuses: `pending`, `processing`, `completed`, `failed`, `skipped`.

- `metadata.application_summary` is a 3-5 sentence summary of the whole uploaded document (any type), in the
  document's main language, or `null`. It is written by the model, so review it before relying on it.
- `isFir` is `true` only when the document is confidently an official registered FIR.
- `metadata.application` is `{found, start_page, end_page, text}`. The model only reports where the attached
  application is; the text is cut from the OCR pages. `found=false` means there is none; `application=null` or
  `found=true, text=null` means it could not be determined.
- `fir_date` and `incident_date` are `YYYY-MM-DD` or `null`; unsupported `case_type` / `case_status` values become
  `null`. `fir_document` is the upload's `document_id` when `isFir` is true.

## Gemini keys

Configure `GEMINI_API_KEYS=k1,k2,...` or `GEMINI_API_KEY_1..10` or a single `GEMINI_API_KEY` (in that order of
precedence). Keys are read from `.env` only and never logged (logs use `Key-N`).

| Failure | Action |
| --- | --- |
| 401 / invalid key | key disabled until restart; next key |
| 402 billing, 403 permission | key cooldown; next key |
| 429 | cooldown (`Retry-After`, daily quota 1h); next key |
| 5xx, network, timeout | next key, up to `GEMINI_SHARED_FAILOVER_MAX_KEYS` keys, then Groq |
| 400, 413 | no rotation (same request fails everywhere); Groq |
| empty / truncated / invalid output | next key, up to `GEMINI_MAX_INVALID_RESPONSE_ATTEMPTS` |

Key health is kept in memory per process.

## Timeouts

| Setting | Default | Covers |
| --- | --- | --- |
| `GEMINI_REQUEST_TIMEOUT_SECONDS` | 150 | one Gemini attempt |
| `GEMINI_OVERALL_BUDGET_SECONDS` | 310 | Gemini failover chain |
| `OCR_OVERALL_DEADLINE_SECONDS` | 400 | OCR, Gemini + Groq |
| `METADATA_TIMEOUT_SECONDS` | 90 | each background metadata attempt (only when needed) |

A large or scanned PDF can take minutes, so set the client/proxy timeout accordingly.

## Temporary memory storage

Results and pending background tasks live in process memory: a restart or reload loses them, and multiple uvicorn
workers do not share them (run a single worker).
