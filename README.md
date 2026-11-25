# IGWA OCR API

The OCR service of the IGWA police software backend. A client uploads a PDF or image, the backend validates it, sends it to Gemini Flash for OCR, keeps the result temporarily in memory and returns a `document_id`. The OCR text is then fetched with that ID.

## OCR flow

```text
POST /python/ocr/upload -> validate file -> Gemini Flash OCR -> store in memory -> document_id
GET  /python/ocr/{document_id} -> OCR result from memory
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
| POST | `/python/ocr/upload` | Upload PDF/PNG/JPG/JPEG/WEBP, run OCR, get `document_id` |
| GET | `/python/ocr/{document_id}` | Fetch the stored OCR result |
| GET | `/python/ocr/{job_id}/metadata` | Poll background metadata status/result (`job_id` == `document_id`) |
| GET | `/health` | Liveness check (does not call Gemini) |

Errors: 400 (unsupported/empty file), 413 (too large), 404 (unknown document), 502 (Gemini failure).

## Example cURL

```bash
curl -X POST "http://localhost:8000/python/ocr/upload" -F "file=@sample.pdf"
curl "http://localhost:8000/python/ocr/3a6a76b5-0c20-4a57-975a-39d13bfbf53e"
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

## Metadata in the same request

Gemini returns the OCR text and the 15 metadata fields in ONE request, so `POST /python/ocr/upload` responds with
`metadata` filled and `metadata_status: "completed"`. No second pass runs. Only if the metadata is missing,
empty or invalid (or Groq did the OCR) does the old background extraction run: then `metadata` is `null`,
`metadata_status` is `pending` and you poll `GET /python/ocr/{job_id}/metadata`. Bad metadata never discards the OCR text.

## Asynchronous metadata fallback (API contract)

`POST /python/ocr/upload` now returns as soon as OCR finishes, with `job_id`, `metadata: null` and
`metadata_status: "pending"`. Metadata (Gemini, Groq fallback) runs in a FastAPI background
task; poll `GET /python/ocr/{job_id}/metadata` every 2-3 s until `metadata_status` is `completed`,
`failed` or `skipped` (empty OCR text), with a timeout and cancel on page leave.
Statuses: `pending`, `processing`, `completed`, `failed`, `skipped`.

Limitations: jobs live in process memory and BackgroundTasks is not a durable queue, so a
restart/reload loses results and any job still `pending`/`processing`; multiple uvicorn
workers do not share state (run a single worker, or move to a shared store + real queue).

## OCR engine: Gemini with key failover, Groq fallback

The uploaded PDF/image goes to Gemini as a whole document (credentials tried in order, see below); if every
Gemini credential fails or is unavailable, the same document goes to Groq. There is no local OCR model.
`ocr_provider` in the response is `gemini` or `groq`. Both prompts forbid translation and transliteration:
Hindi stays in Devanagari, English stays English, digits are copied exactly. Gemini's output is parsed and
validated, never post-edited. OCR text quality is whatever the provider returns (no local quality scoring),
so review critical identifiers. Encrypted/corrupt files are rejected by the provider path as an OCR failure (502).

### Gemini API key failover

`services/ocr/gemini_key_manager.py` picks keys per request, one at a time, for both OCR and background
metadata. With `GEMINI_KEY_SELECTION=random` (default) every request shuffles the currently healthy keys
(sampling without replacement, so a failed key is never retried in the same request) and walks that order
until a valid result arrives; `ordered` keeps the configured order. Configure `GEMINI_API_KEYS=k1,k2,...` (any
number) or `GEMINI_API_KEY_1..10` (precedence: list > numbered > legacy `GEMINI_API_KEY`; placeholders and
duplicates are ignored). Keys live only in `.env`: nothing is hardcoded and keys are never logged (logs use
`Key-N`). `reload_manager()` re-reads the settings and keeps the health of surviving keys.

| Failure | Action |
| --- | --- |
| 401, "API key not valid/expired/leaked" (the Gemini API reports an invalid key as HTTP 400) | key disabled until the process restarts / `check_keys` succeeds; next key immediately |
| 402 billing, 403 permission | key cooldown (default 1h); next key |
| 429 | cooldown = `Retry-After` (else 60s); daily quota = 1h; next key |
| 5xx, timeout, network | key benched for `GEMINI_SHARED_FAILURE_COOLDOWN_SECONDS`; next key, but after `GEMINI_SHARED_FAILOVER_MAX_KEYS` distinct keys fail this way the provider counts as down: stop, fast-fail for the cooldown, go to Groq |
| 400 (invalid PDF / bad request), 413 | **no rotation**, no key penalized (the same request fails on every key) |
| 404 model not found | confirmed on one other key, then stop (a wrong model name would fail everywhere) |
| empty / truncated / invalid-JSON / schema-invalid / error-only / bad-finish-reason response | next key (max `GEMINI_MAX_INVALID_RESPONSE_ATTEMPTS`), key not cooled down |
| response only skips pages | only the missing pages are re-OCRed (one sub-PDF call) and merged in page order |
| repeated truncation of a multi-page PDF | page-by-page OCR |
| all keys unavailable / Gemini down | Groq (existing fallback) |

SDK-internal retries are disabled (`attempts=1`) so failover is the only retry layer; there is no sleep between
keys (`GEMINI_FAILOVER_DELAY_SECONDS=0`). Failure *detection* is bounded by `GEMINI_REQUEST_TIMEOUT_SECONDS` per
attempt, the Gemini chain by `GEMINI_OVERALL_BUDGET_SECONDS`, Gemini + Groq by `OCR_OVERALL_DEADLINE_SECONDS`; the
*switch* itself takes well under a millisecond. Per-key health (counters, last error, cooldown expiry, average
latency, active requests) is available from `get_manager().snapshot()`; `/health` shows only
`gemini_keys: {configured, available}`. Every request logs `[OCR] request=<id> ...` lines (order, attempt,
`Key-N`, category, timing) and a final `Status: SUCCESS|FAILED` line, never document text.
Limits: cooldown/health state is per process (each uvicorn worker has its own view; no shared store is used) and
resets on restart; results are in memory, so a failed upload (502) is not stored: the client still has the file and
must re-upload. A hung connection is bounded by the per-attempt timeout, not a separate connect timeout.

### Parallel batch OCR (multi-page PDFs)

A PDF that needs more than one batch (more than `OCR_BATCH_SIZE` = 5 pages; smaller batches for scanned/heavy PDFs,
see `OCR_HEAVY_PAGE_BYTES`) is cut into page batches by `services/ocr/batch_ocr.py` (PyMuPDF `insert_pdf`: pages are
copied, not re-rendered) and OCRed concurrently on different Gemini keys. Documents of up to one batch, images and
`OCR_BATCH_PROCESSING_ENABLED=false` keep the single-request path above (which also returns the metadata in the same
call). The API, request and response format are unchanged: the frontend never sees batches, keys or retries.

* **Allocation.** `healthy keys -> active batches / standby`: 6->4/2, 5->3/2, 4->2/2, 3->2/1, 2->1/1, 1->1/0, 0->0/0
  (`OCR_MAX_PARALLEL_BATCHES`, `GEMINI_RESERVED_KEYS`; nothing is hardcoded to six keys or four batches). Keys are chosen
  randomly among the eligible ones (not disabled, not cooling down, not at their cap). Standby keys are simply the
  healthy keys the document's workers do not occupy: a failed batch takes one of them at once, and recovery work is
  queued ahead of fresh batches. When a batch finishes the next one starts immediately (no waves).
* **Strict limits** (process-local, see limits below): `GEMINI_MAX_CONCURRENT_PER_KEY` (default 1; a key at its cap is
  never handed out), `OCR_GLOBAL_MAX_CONCURRENT_BATCHES` (all documents together; 10 uploads x 4 batches never become 40
  requests), workers per document. Waiting batches are served first come first served, so one large document cannot
  starve a small one. Reservations are released after success, failure and cancellation.
* **Failure handling per batch** (nothing else restarts; validated pages are kept in a registry keyed by original page
  number and are never overwritten):

| Batch failure | Action |
| --- | --- |
| billing / auth / 403 | key excluded as before (cooldown / disabled); another key at once |
| 429 | cooldown from `Retry-After`; another key, **unless a second key is also rate limited**: then a shared project quota is assumed, the document pauses briefly, halves its concurrency and stops rotating (Groq / failure) |
| 5xx, network | key benched briefly; another key; no provider-wide circuit breaker from batch traffic |
| processing timeout (adaptive, see below), truncation, rejected input (400/413) | batch is split in halves (5 -> 2 + 3 ...) down to single pages (`OCR_BATCH_MIN_SIZE`); the key is **not** penalized |
| empty / invalid JSON / schema / error-only / bad page numbers | another key (max `GEMINI_MAX_INVALID_RESPONSE_ATTEMPTS`), then split |
| response skips pages | valid pages kept, only the missing pages are redone |
| no eligible key left, or Gemini recovery exhausted | **only that batch** goes to Groq (per-page images, SDK retries off, `OCR_GROQ_MAX_CONCURRENT`) |
| still incomplete, or the overall deadline passes | HTTP 502 as before; nothing is stored, no partial result is returned |

* **Adaptive timeout.** `OCR_BATCH_TIMEOUT_SECONDS` (90) is the budget of a full batch; it scales with pages and PDF
  size, tightens with latency measured earlier in the request, stays within `OCR_BATCH_TIMEOUT_MAX_SECONDS` and the
  remaining budget (`OCR_OVERALL_DEADLINE_SECONDS` minus `OCR_METADATA_RESERVE_SECONDS`). A timeout caused by the
  application's own remaining budget is nobody's fault and counts against no key.
* **Merge and metadata.** Pages are mapped from batch-local to original numbers, checked (all present, none duplicated),
  and `full_text` is built in Python (`build_full_text`); no LLM rewrites text. Batches do not extract metadata. After the
  merge one metadata extraction runs on the page-marked full text (`OCR_METADATA_AFTER_MERGE`), still inside the upload
  request like the single-call path, so `metadata_status` is normally `completed` in the upload response. If it does not
  finish in time it falls back to the existing background task (`pending`, poll `/metadata`).
* **Limits.** Progress lives in memory only: a restart or a terminal failure loses it and the client must re-upload
  (durable recovery would need stored PDFs, a job queue and a status endpoint: not built, needs approval). The caps are
  per process: with N uvicorn workers the real ceiling is N times higher, and keys of one Google project share one quota.
  Cancelling a request cancels the local calls; Gemini/Groq may still finish a request they already received. Very large
  documents are sent whole to the metadata model (no chunked aggregation yet). `GEMINI_OVERALL_BUDGET_SECONDS` and
  `GEMINI_SHARED_FAILOVER_MAX_KEYS` still govern the single-request path; batches use `OCR_OVERALL_DEADLINE_SECONDS` and
  `GEMINI_SHARED_FAILOVER_MAX_KEYS` as the number of distinct keys a batch tries after 5xx/network errors.
* **Benchmarks.** `python benchmarks/benchmark_ocr_batches.py` (mocked, no network) and
  `python benchmarks/live_ocr_check.py file.pdf` (real calls, run deliberately).

### isFir (FIR vs application)

Every response (`POST /python/ocr/upload`, `GET /python/ocr/{id}`, `GET /python/ocr/{job_id}/metadata`) has a
boolean `isFir` right after `metadata_error`. It comes from a `document_class` value
(`FIR` / `APPLICATION` / `OTHER` / `UNCLEAR`) that Gemini returns inside the same request as the OCR and metadata
(or from the text-only Gemini/Groq metadata fallback). `isFir` is `true` only for `FIR`. An application that
quotes an FIR number is `APPLICATION`; an official FIR form followed by attached letters is still `FIR`.
Missing/`UNCLEAR`/invalid classification gives `false` and a "classification unresolved" log line (a confirmed
non-FIR logs as "classified as non-FIR"). While metadata is still pending, `isFir` is `false` until the
background step completes. `document_class` itself is internal and never appears in the `metadata` object.

### metadata.application (attached application/complaint)

`metadata.application` is `{found, start_page, end_page, text}`. The model only reports *where* the application is
(`application_location`: page range, plus short verbatim `start_text`/`end_text` quotes when it starts/ends
mid-page) in the existing single Gemini request (or the existing background metadata request after Groq OCR, where
pages are marked `[[PAGE n]]`). `text` is cut from `ocr_data.pages` in Python (`services/ocr/application.py`):
nothing is regenerated and no extra model call is made.

- `found=false` (all other fields `null`): confirmed no application.
- `found=true`, `text=null`: located, but a mid-page boundary quote was not found verbatim, so no text is
  returned rather than risk returning FIR text.
- `application=null` (or `metadata=null` while pending/failed): unavailable, e.g. the model returned no/invalid
  location. This is never reported as "no application".

### Case-management metadata fields

`metadata` also carries `fir_date`, `incident_date` (both `YYYY-MM-DD` or `null`), `title`, `case_type`
(`CaseType` enum), `case_status` (`CASE_STATUS` enum) and `fir_document`. Unsupported enum values or unclear dates
are normalized to `null` instead of failing the metadata. `fir_document` is set by the server to the upload's
`document_id` only when `isFir` is true (never taken from model output). Existing `accused_names` and
`incident_date_time` remain the accused-persons and verbatim incident date/time fields.
