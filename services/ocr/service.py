"""Upload -> OCR (Gemini, then Groq) -> stored record; metadata in the same call or in the background."""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import BackgroundTasks, HTTPException, UploadFile
from pydantic import ValidationError

from config.settings import Settings, get_settings
from services.ocr import gemini_client, gemini_key_manager, groq_client, utils
from services.ocr.application import attach_application, build_marked_text
from services.ocr.gemini_client import GeminiMetadataError, GeminiOCRError
from services.ocr.groq_client import GroqMetadataError, GroqOCRError
from services.ocr.schemas import (
    MetadataStatus,
    MetadataStatusResponse,
    OCRData,
    OCRMetadata,
    OCRPage,
    OCRResultResponse,
    OCRStatus,
    OCRUploadResponse,
    metadata_from_dict,
    ocr_data_from_dict,
)

logger = logging.getLogger(__name__)

# Temporary in-memory storage: document_id (== job_id) -> stored OCR record.
# NOTE: process-local. Results AND pending background metadata tasks are lost on restart
# or reload, and multiple uvicorn workers do not share it. FastAPI BackgroundTasks is not
# a durable queue. Within one process it is race-free because every read-modify-write of
# a record happens without an await (single event-loop thread).
ocr_results: dict[str, OCRResultResponse] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


_OCR_FAILED = "OCR provider failed to process the document"
_REMOTE_ERRORS = (GeminiOCRError, GroqOCRError, ValidationError, ValueError, asyncio.TimeoutError)
_MAX_RETRY_AFTER_SECONDS = 10.0

# ---- OCR providers (Gemini -> Groq) ------------------------------------------------------


def _retry_delay(exc: BaseException, attempt: int) -> float | None:
    """Seconds to wait before retrying a Groq error, or None when retrying is pointless.

    Billing/auth/bad-request errors are never retried. Rate limits are retried only when the
    provider's Retry-After is short; 5xx/timeouts/connection errors use a small exponential backoff.
    """
    cause = exc.__cause__ or exc
    _status, category, _ = gemini_client.describe_provider_error(cause)
    if category == "rate_limit":
        headers = getattr(getattr(cause, "response", None), "headers", None)
        try:
            delay = float(headers.get("retry-after")) if headers is not None else None
        except (TypeError, ValueError):
            return None
        return delay if delay is not None and 0 <= delay <= _MAX_RETRY_AFTER_SECONDS else None
    if category in ("server_error", "timeout", "connection"):
        return min(1.0 * 2**attempt, 4.0)
    return None


def _error_category(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, (ValidationError, ValueError)):
        return "invalid_output"
    if getattr(exc, "category", None):
        return str(exc.category)
    return gemini_client.describe_provider_error(exc.__cause__ or exc)[1]


async def _call_gemini(payload: bytes, mime: str, timeout: float, s: Settings) -> dict:
    """One Gemini OCR request. Retries and key failover happen inside the key manager."""
    return await asyncio.wait_for(gemini_client.extract_text(payload, mime), timeout)


async def _call_groq(payload: bytes, mime: str, timeout: float, s: Settings) -> dict:
    """Groq OCR with bounded retries. Pages already obtained are kept in `done` and never requested
    again, so a retry after an HTTP 429 only sends the pages that are still missing. No retry starts
    when the remaining time could not finish it."""
    done: dict[int, str] = {}
    started = time.monotonic()
    retries = s.ocr_groq_max_retries
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(
                groq_client.extract_text(payload, mime, done=done),
                max(timeout - (time.monotonic() - started), 0.1))
        except GroqOCRError as exc:
            delay = _retry_delay(exc, attempt) if attempt < retries else None
            if delay is None or timeout - (time.monotonic() - started) < delay + 5.0:
                raise
            logger.warning("Groq OCR retry in %.1fs: category=%s", delay, _error_category(exc))
            await _sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)  # separate seam so tests can skip real backoff delays


def _verify_all_pages(data: OCRData, expected: int | None) -> None:
    """Reject a response that skipped, duplicated or invented pages (it falls back to Groq)."""
    if expected is not None and [p.page_number for p in data.pages] != list(range(1, expected + 1)):
        raise ValueError("provider did not return every page exactly once")


async def _run_ocr(
    payload: bytes, mime: str, document_id: str, s: Settings
) -> tuple[OCRData, str, str, OCRMetadata | None]:
    """Whole document to Gemini, then Groq. Returns (data, model, provider, metadata).

    Gemini returns the metadata in the same response; Groq OCR does not, so metadata is None there.
    """
    providers = []
    if s.ocr_enable_gemini_fallback:
        providers.append(("gemini", _call_gemini, s.gemini_model))
    if s.ocr_enable_groq_fallback:
        providers.append(("groq", _call_groq, s.groq_model))

    failures: list[str] = []
    deadline = time.monotonic() + s.ocr_overall_deadline_seconds  # Gemini + Groq share one budget
    expected = gemini_client.count_pages(payload, mime)  # None: unreadable PDF, page check skipped
    for name, call, model in providers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failures.append(f"{name}=deadline_exceeded")
            logger.warning("[OCR] request=%s overall deadline reached before %s", document_id, name)
            break
        if name == "groq" and failures:
            logger.warning("job=%s Groq fallback used (Gemini unavailable)", document_id)
        try:
            logger.info("%s OCR started: document_id=%s pages=%s", name, document_id, expected)
            call_started = time.perf_counter()
            raw = await call(payload, mime, remaining, s)
            call_seconds = time.perf_counter() - call_started
            timings: dict[str, float] = {}
            data = ocr_data_from_dict(raw, timings)
            _verify_all_pages(data, expected)
            metadata = metadata_from_dict(raw) if name == "gemini" else None
            logger.info(
                "job=%s OCR timing: provider=%s call=%.2fs parse_validate=%.3fs full_text_join=%.4fs "
                "pages=%d chars=%d metadata_in_same_call=%s",
                document_id, name, call_seconds, timings["parse"], timings["full_text"],
                len(data.pages), len(data.full_text), metadata is not None)
            return data, model, name, metadata
        except _REMOTE_ERRORS as exc:
            failures.append(f"{name}={_error_category(exc)}")
            logger.warning("%s OCR failed: document_id=%s reason=%s", name, document_id, failures[-1])
    logger.error("[OCR] request=%s Status: FAILED reasons=%s", document_id,
                 " ".join(failures) or "no_provider_enabled")
    raise HTTPException(status_code=502, detail=_OCR_FAILED)


# ---- metadata -----------------------------------------------------------------------------


def _is_fir(metadata: OCRMetadata | None, job_id: str) -> bool:
    """isFir from the model's classification. True only for a confident FIR.

    APPLICATION/OTHER are confirmed non-FIR (false). A missing or UNCLEAR classification also
    yields false, but is logged as an unresolved classification so it is never mistaken for a
    confirmed non-FIR.
    """
    label = metadata.document_class if metadata is not None else None
    if label == "FIR":
        logger.info("job=%s Document classified as FIR", job_id)
        return True
    if label in ("APPLICATION", "OTHER"):
        logger.info("job=%s Document classified as non-FIR (%s)", job_id, label)
        return False
    logger.warning("job=%s Document classification unresolved (%s): isFir defaults to false",
                   job_id, label or "missing")
    return False


def _finalize_metadata(metadata: OCRMetadata, pages: list[OCRPage], job_id: str) -> tuple[OCRMetadata, bool]:
    """Python-owned metadata: application text and fir_document (the uploaded document's id when the
    document is an FIR; any value the model sent is discarded). Returns (metadata, isFir)."""
    is_fir = _is_fir(metadata, job_id)
    metadata = attach_application(metadata, pages, job_id)
    return metadata.model_copy(update={"fir_document": job_id if is_fir else None}), is_fir


async def extract_metadata(
    job_id: str, ocr_text: str
) -> tuple[OCRMetadata | None, str | None]:
    """Gemini metadata extraction with Groq fallback. Returns (metadata, model).

    metadata is None only when both providers failed. Each client validates its output
    against OCRMetadata; invalid or truncated output raises and triggers the fallback.
    """
    settings = get_settings()
    timeout = settings.metadata_timeout_seconds
    try:
        logger.info("job=%s Gemini metadata extraction started", job_id)
        metadata = await asyncio.wait_for(gemini_client.extract_metadata(ocr_text), timeout)
        return metadata, settings.gemini_model
    except (GeminiMetadataError, ValidationError, asyncio.TimeoutError) as gemini_exc:
        gemini_reason = type(gemini_exc).__name__
        logger.warning("job=%s Gemini failed (%s); Groq fallback started", job_id, gemini_reason)
    try:
        metadata = await asyncio.wait_for(groq_client.extract_metadata(ocr_text), timeout)
        return metadata, settings.groq_model
    except (GroqMetadataError, ValidationError, asyncio.TimeoutError) as groq_exc:
        logger.error(
            "job=%s Metadata extraction failed gemini=%s groq=%s",
            job_id, gemini_reason, type(groq_exc).__name__,
        )
        return None, None


# ---- requests -----------------------------------------------------------------------------


async def process_ocr(file: UploadFile) -> OCRResultResponse:
    """Validate and OCR the upload and store the job record. Never calls metadata providers."""
    settings = get_settings()
    filename = utils.sanitize_filename(file.filename)
    ext = utils.validate_upload_metadata(filename, file.content_type)
    data = await utils.read_upload_limited(file, settings.max_file_size_bytes)
    mime_type = utils.validate_content_matches_extension(data, ext)

    document_id = str(uuid.uuid4())
    gemini_key_manager.request_id_var.set(document_id)  # correlates every [OCR] log line of this request
    started = time.perf_counter()
    logger.info("OCR upload received: document_id=%s filename=%s", document_id, filename)

    ocr_data, ocr_model, ocr_provider, metadata = await _run_ocr(data, mime_type, document_id, settings)

    now = _now()
    has_text = bool(ocr_data.full_text.strip())
    if has_text and metadata is not None:
        # Same-call location -> application text cut from pages[] in Python (no extra request).
        metadata, is_fir = _finalize_metadata(metadata, ocr_data.pages, document_id)
        metadata_state = dict(
            metadata=metadata, metadata_model=ocr_model, metadata_status=MetadataStatus.COMPLETED,
            isFir=is_fir,
        )
    else:  # no usable same-call metadata: extract in the background from the OCR text
        metadata_state = dict(
            metadata_status=MetadataStatus.PENDING if has_text else MetadataStatus.SKIPPED,
            metadata_error=None if has_text else "No OCR text available for metadata extraction",
        )
    record = OCRResultResponse(
        document_id=document_id,
        job_id=document_id,
        status=OCRStatus.COMPLETED,
        filename=filename,
        ocr_model=ocr_model,
        ocr_provider=ocr_provider,
        ocr_data=ocr_data,
        created_at=now,
        updated_at=now,
        **metadata_state,
    )
    ocr_results[document_id] = record
    logger.info("[OCR] request=%s Provider: %s Total processing time: %.2fs Status: SUCCESS",
                document_id, ocr_provider, time.perf_counter() - started)
    return record


async def process_upload(
    file: UploadFile, background_tasks: BackgroundTasks
) -> OCRUploadResponse:
    """Run OCR, store the job, schedule metadata extraction if needed and return immediately."""
    request_started = time.perf_counter()
    record = await process_ocr(file)
    job_id = record.document_id
    if record.metadata_status == MetadataStatus.COMPLETED:
        logger.info("job=%s Metadata returned with the OCR (single request)", job_id)
        message = "OCR and metadata extraction completed successfully"
    elif record.metadata_status == MetadataStatus.PENDING:
        # Only the OCR text is passed on: no UploadFile, file bytes or request state.
        background_tasks.add_task(
            process_metadata_background, job_id, record.ocr_data.full_text, record.ocr_data.pages
        )
        logger.info("job=%s Metadata scheduled", job_id)
        message = "OCR completed; metadata extraction is running in the background"
    else:
        logger.warning("job=%s Metadata skipped (empty OCR text)", job_id)
        message = "OCR completed; no text found so metadata extraction was skipped"
    logger.info("job=%s OCR API total response time=%.2fs", job_id, time.perf_counter() - request_started)
    return OCRUploadResponse(
        message=message,
        **record.model_dump(exclude={"success", "created_at", "updated_at"}),
    )


def _update_job(job_id: str, **changes: object) -> None:
    """Replace the stored record in one step (no await, so no interleaving)."""
    record = ocr_results.get(job_id)
    if record is not None:
        ocr_results[job_id] = record.model_copy(update={**changes, "updated_at": _now()})


async def process_metadata_background(
    job_id: str, ocr_text: str, pages: list[OCRPage] | None = None
) -> None:
    """Background task: extract metadata for a stored job and always reach a terminal status.

    With `pages`, the metadata request sees page markers, reports where the application is, and
    Python cuts the application text from those pages (still one metadata request).
    """
    started = time.monotonic()
    if pages:
        ocr_text = build_marked_text(pages)
    _update_job(job_id, metadata_status=MetadataStatus.PROCESSING)
    logger.info("job=%s Metadata processing started", job_id)
    try:
        metadata, model = await extract_metadata(job_id, ocr_text)
    except Exception as exc:  # task boundary: never leave the job stuck in PROCESSING
        logger.error("job=%s Metadata task crashed: %s", job_id, type(exc).__name__)
        metadata, model = None, None
    elapsed = time.monotonic() - started
    if metadata is None:
        _update_job(
            job_id,
            metadata_status=MetadataStatus.FAILED,
            metadata_error="Metadata extraction could not be completed.",
            isFir=False,
        )
        logger.error("job=%s Metadata failed duration=%.2fs", job_id, elapsed)
        logger.warning("job=%s Document classification failed (metadata unavailable): isFir defaults to false",
                       job_id)
    else:
        metadata, is_fir = _finalize_metadata(metadata, pages or [], job_id)
        _update_job(
            job_id,
            metadata=metadata,
            metadata_model=model,
            metadata_status=MetadataStatus.COMPLETED,
            metadata_error=None,
            isFir=is_fir,
        )
        logger.info("job=%s Metadata completed provider=%s duration=%.2fs", job_id, model, elapsed)


def get_metadata_status(job_id: str) -> MetadataStatusResponse:
    record = get_ocr_result(job_id)
    return MetadataStatusResponse(
        job_id=job_id,
        metadata_status=record.metadata_status,
        metadata=record.metadata,
        metadata_provider=record.metadata_model,
        metadata_error=record.metadata_error,
        isFir=record.isFir,
    )


def get_ocr_result(document_id: str) -> OCRResultResponse:
    logger.info("OCR result requested: document_id=%s", document_id)
    result = ocr_results.get(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="OCR document not found")
    return result
