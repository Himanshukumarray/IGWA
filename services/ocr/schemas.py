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
    status: OCRStatus
    filename: str
    ocr_model: str = Field(..., description="Model that produced the OCR text")
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
    """Stored OCR result as returned by GET /ocr/{document_id}."""

    success: bool = True
    document_id: str
    status: OCRStatus
    filename: str
    ocr_model: str = Field(..., description="Model that produced the OCR text")
    ocr_data: OCRData | None = None
    metadata: OCRMetadata | None = None
    metadata_model: str | None = Field(
        default=None, description="Model that produced the metadata (may differ from ocr_model)"
    )
    metadata_status: MetadataStatus = MetadataStatus.COMPLETED
    metadata_error: str | None = Field(
        default=None, description="Set when metadata extraction failed or was skipped"
    )

class ErrorResponse(BaseModel):
    detail: str

class GeminiOCRResult(BaseModel):
    """Schema Gemini is asked to follow (kept minimal for stability)."""

    pages: list[OCRPage]
    full_text: str | None = None

def ocr_data_from_dict(raw: dict[str, Any]) -> OCRData:
    """Validate a parsed Gemini dict and build the OCR data stored/returned by the API."""
    result = GeminiOCRResult.model_validate(raw)
    pages = sorted(result.pages, key=lambda p: p.page_number)
    full_text = result.full_text or "\n\n".join(p.text for p in pages)
    return OCRData(pages=pages, full_text=full_text)
