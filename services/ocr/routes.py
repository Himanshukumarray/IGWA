from fastapi import APIRouter, File, UploadFile
from services.ocr import service
from services.ocr.schemas import ErrorResponse, OCRResultResponse, OCRUploadResponse

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post(
    "/upload",
    response_model=OCRUploadResponse,
    summary="Upload a document and extract its text with OCR",
    description="Accepts a PDF, PNG, JPG, JPEG or WEBP file (multipart/form-data), "
    "runs Gemini OCR, keeps the result temporarily in memory and returns a document_id.",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported or empty file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        502: {"model": ErrorResponse, "description": "OCR provider failure"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or image to OCR"),
) -> OCRUploadResponse:
    return await service.process_upload(file)

@router.get(
    "/{document_id}",
    response_model=OCRResultResponse,
    summary="Retrieve an OCR result by document_id",
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document(document_id: str) -> OCRResultResponse:
    return service.get_ocr_result(document_id)
