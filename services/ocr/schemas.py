from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

class OCRStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class OCRPage(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str = ""

class OCRData(BaseModel):
    pages: list[OCRPage] = Field(default_factory=list)
    full_text: str = ""

class MetadataStatus(str, Enum):
    PENDING = "pending"  # background task scheduled
    PROCESSING = "processing"  # background extraction running
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class OCRMetadata(BaseModel):
    """The 15 metadata fields extracted from OCR text of police documents."""

    document_type: str | None = None
    fir_number: str | None = None
    case_number: str | None = None
    document_date: str | None = None
    police_station: str | None = None
    district: str | None = None
    investigating_officer: str | None = None

    legal_sections: list[str] = Field(default_factory=list)

    complainant_name: str | None = None

    accused_names: list[str] = Field(default_factory=list)
    witness_names: list[str] = Field(default_factory=list)

    incident_date_time: str | None = None
    incident_location: str | None = None

    document_narrative: str | None = None

    investigation_actions_findings: list[str] = Field(default_factory=list)

class OCRUploadResponse(BaseModel):
    success: bool = True
    message: str = "OCR processing completed successfully"
    document_id: str
    job_id: str = Field(..., description="Same value as document_id; use it to poll metadata")
    status: OCRStatus
    filename: str
    ocr_model: str = Field(..., description="Model that produced the OCR text")
    ocr_provider: str | None = Field(
        default=None, description="gemini or groq"
    )
    ocr_data: OCRData | None = None
    metadata: OCRMetadata | None = None
    metadata_model: str | None = Field(
        default=None, description="Model that produced the metadata (may differ from ocr_model)"
    )
    metadata_status: MetadataStatus = MetadataStatus.COMPLETED
    metadata_error: str | None = Field(
        default=None, description="Set when metadata extraction failed or was skipped"
    )

class OCRResultResponse(BaseModel):
    """Stored OCR result as returned by GET /python/ocr/{document_id}."""

    success: bool = True
    document_id: str
    job_id: str = ""
    status: OCRStatus
    filename: str
    ocr_model: str = Field(..., description="Model that produced the OCR text")
    ocr_provider: str | None = Field(
        default=None, description="gemini or groq"
    )
    ocr_data: OCRData | None = None
    metadata: OCRMetadata | None = None
    metadata_model: str | None = Field(
        default=None, description="Model that produced the metadata (may differ from ocr_model)"
    )
    metadata_status: MetadataStatus = MetadataStatus.COMPLETED
    metadata_error: str | None = Field(
        default=None, description="Set when metadata extraction failed or was skipped"
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None

class MetadataStatusResponse(BaseModel):
    """GET /python/ocr/{job_id}/metadata: same shape in every state."""

    job_id: str
    metadata_status: MetadataStatus
    metadata: OCRMetadata | None = None
    metadata_provider: str | None = Field(
        default=None, description="Model that produced the metadata"
    )
    metadata_error: str | None = None

class ErrorResponse(BaseModel):
    detail: str

class _ProviderPage(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str = ""

class GeminiOCRResult(BaseModel):
    """Schema Gemini is asked to follow (kept minimal for stability)."""

    pages: list[_ProviderPage]
    full_text: str | None = None
    # Filled by Gemini in the same request as the OCR text (single call, no second pass).
    metadata: OCRMetadata | None = None

def ocr_data_from_dict(raw: dict[str, Any]) -> OCRData:
    """Validate a parsed Gemini dict and build the OCR data stored/returned by the API."""
    # Metadata is validated separately (metadata_from_dict): malformed metadata must never
    # discard otherwise good OCR text.
    result = GeminiOCRResult.model_validate({k: v for k, v in raw.items() if k != "metadata"})
    pages = [
        OCRPage(page_number=p.page_number, text=p.text)
        for p in sorted(result.pages, key=lambda p: p.page_number)
    ]
    full_text = result.full_text or "\n\n".join(p.text for p in pages)
    return OCRData(pages=pages, full_text=full_text)


def metadata_from_dict(raw: dict[str, Any]) -> OCRMetadata | None:
    """Validated metadata from a combined OCR response, or None if absent/invalid/empty.

    None means "extract it separately": malformed or all-empty metadata is never accepted.
    """
    candidate = raw.get("metadata")
    if not isinstance(candidate, dict):
        return None
    try:
        metadata = OCRMetadata.model_validate(candidate)
    except ValueError:
        return None
    has_content = any(
        (v if isinstance(v, str) else bool(v)) for v in metadata.model_dump().values() if v is not None
    )
    return metadata if has_content else None
