import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

ALLOWED_TYPES: dict[str, set[str]] = {
    "pdf": {"application/pdf"},
    "png": {"image/png"},
    "jpg": {"image/jpeg", "image/jpg", "image/pjpeg"},
    "jpeg": {"image/jpeg", "image/jpg", "image/pjpeg"},
    "webp": {"image/webp"},
}

CANONICAL_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

_READ_CHUNK = 1024 * 1024


def sanitize_filename(filename: str | None) -> str:
    """Strip directory components and unsafe characters from a client filename."""
    name = Path((filename or "").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    return name[:150] or "upload"


def detect_type_from_content(data: bytes) -> str | None:
    """Identify the real file type from magic bytes."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def validate_upload_metadata(filename: str, content_type: str | None) -> str:
    """Validate extension and declared MIME type; return the extension."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    allowed_mimes = ALLOWED_TYPES[ext] | {"application/octet-stream"}
    if content_type and content_type.lower() not in allowed_mimes:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return ext


async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks, aborting as soon as it exceeds the size limit."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_READ_CHUNK):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413, detail="File exceeds maximum allowed size"
            )
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return b"".join(chunks)


def validate_content_matches_extension(data: bytes, ext: str) -> str:
    """Ensure magic bytes agree with the extension; return the canonical MIME type."""
    expected = "jpeg" if ext in {"jpg", "jpeg"} else ext
    if detect_type_from_content(data) != expected:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return CANONICAL_MIME[ext]


def preprocess_document(data: bytes, mime_type: str) -> tuple[bytes, str]:
    """Hook for future preprocessing (e.g. PDF splitting, image deskew).

    Currently the document is sent to Gemini unchanged.
    """
    return data, mime_type
