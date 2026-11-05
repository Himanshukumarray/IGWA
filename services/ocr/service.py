import logging
import uuid

from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from config.settings import get_settings
from services.ocr import gemini_client, groq_client, utils
from services.ocr.gemini_client import GeminiOCRError
from services.ocr.gemini_client import GeminiMetadataError
from services.ocr.groq_client import GroqMetadataError, GroqOCRError
from services.ocr.schemas import (
    MetadataStatus,
    OCRMetadata,
    OCRResultResponse,
    OCRStatus,
    OCRUploadResponse,
    ocr_data_from_dict,
)

logger = logging.getLogger(__name__)

# Temporary in-memory storage: document_id -> stored OCR record.
# NOTE: OCR results will be lost when the FastAPI server restarts.
# This is intentional for the current development phase.
ocr_results: dict[str, OCRResultResponse] = {}

async def process_upload(file: UploadFile) -> OCRUploadResponse:
    settings = get_settings()
    filename = utils.sanitize_filename(file.filename)
    ext = utils.validate_upload_metadata(filename, file.content_type)
    data = await utils.read_upload_limited(file, settings.max_file_size_bytes)
    mime_type = utils.validate_content_matches_extension(data, ext)

    document_id = str(uuid.uuid4())
    logger.info("OCR upload received: document_id=%s filename=%s", document_id, filename)

    payload, payload_mime = utils.preprocess_document(data, mime_type)

    try:
        logger.info("Gemini OCR started: document_id=%s", document_id)
        raw = await gemini_client.extract_text(payload, payload_mime)
        ocr_data = ocr_data_from_dict(raw)
        ocr_model = settings.gemini_model
        logger.info("Gemini OCR completed: document_id=%s", document_id)
    except (GeminiOCRError, ValidationError) as gemini_exc:
        logger.warning(
            "Gemini OCR failed, falling back to Groq: document_id=%s reason=%s",
            document_id,
            type(gemini_exc).__name__,
        )
        try:
            raw = await groq_client.extract_text(payload, payload_mime)
            ocr_data = ocr_data_from_dict(raw)
            ocr_model = settings.groq_model
            logger.info("Groq OCR completed: document_id=%s", document_id)
        except (GroqOCRError, ValidationError) as groq_exc:
            logger.error(
                "OCR failed: document_id=%s gemini=%s groq=%s",
                document_id,
                type(gemini_exc).__name__,
                type(groq_exc).__name__,
            )
            raise HTTPException(
                status_code=502, detail="OCR provider failed to process the document"
            ) from groq_exc

    metadata: OCRMetadata | None = None
    metadata_model: str | None = None
    metadata_status = MetadataStatus.COMPLETED
    metadata_error: str | None = None
    if not ocr_data.full_text.strip():
        metadata_status = MetadataStatus.SKIPPED
        metadata_error = "No OCR text available for metadata extraction"
        logger.warning("Metadata skipped (empty OCR text): document_id=%s", document_id)
    else:
        try:
            logger.info("Gemini metadata extraction started: document_id=%s", document_id)
            metadata = await gemini_client.extract_metadata(ocr_data.full_text)
            metadata_model = settings.gemini_model
            logger.info("Gemini metadata extraction completed: document_id=%s", document_id)
        except (GeminiMetadataError, ValidationError) as gemini_meta_exc:
            logger.warning(
                "Gemini metadata failed, falling back to Groq: document_id=%s reason=%s",
                document_id,
                type(gemini_meta_exc).__name__,
            )
            try:
                metadata = await groq_client.extract_metadata(ocr_data.full_text)
                metadata_model = settings.groq_model
                logger.info("Groq metadata extraction completed: document_id=%s", document_id)
            except (GroqMetadataError, ValidationError) as groq_meta_exc:
                metadata_status = MetadataStatus.FAILED
                metadata_error = "Gemini and Groq metadata requests failed"
                logger.error(
                    "Metadata extraction failed: document_id=%s gemini=%s groq=%s",
                    document_id,
                    type(gemini_meta_exc).__name__,
                    type(groq_meta_exc).__name__,
                )

    ocr_results[document_id] = OCRResultResponse(
        document_id=document_id,
        status=OCRStatus.COMPLETED,
        filename=filename,
        ocr_model=ocr_model,
        ocr_data=ocr_data,
        metadata=metadata,
        metadata_model=metadata_model,
        metadata_status=metadata_status,
        metadata_error=metadata_error,
    )
    logger.info("OCR stored in memory: document_id=%s model=%s", document_id, ocr_model)

    return OCRUploadResponse(
        message=(
            "OCR and metadata extraction completed successfully"
            if metadata_status == MetadataStatus.COMPLETED
            else "OCR completed; metadata extraction did not complete"
        ),
        document_id=document_id,
        status=OCRStatus.COMPLETED,
        filename=filename,
        ocr_model=ocr_model,
        ocr_data=ocr_data,
        metadata=metadata,
        metadata_model=metadata_model,
        metadata_status=metadata_status,
        metadata_error=metadata_error,
    )

def get_ocr_result(document_id: str) -> OCRResultResponse:
    logger.info("OCR result requested: document_id=%s", document_id)
    result = ocr_results.get(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="OCR document not found")
    return result
