
import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEYS = [
    key.strip()
    for key in os.getenv("GEMINI_API_KEYS", "").split(",")
    if key.strip()
]

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

PDF_PATH = BASE_DIR / "test_ocr.pdf"
OUTPUT_DIR = BASE_DIR / "ocr_test_results"

OUTPUT_DIR.mkdir(exist_ok=True)

OCR_PROMPT = """
You are an OCR extraction engine.

Extract all visible textual information from this PDF.

Rules:
1. Extract text from every page.
2. Do not summarize.
3. Do not translate.
4. Do not rewrite.
5. Do not correct spelling or grammar.
6. Preserve the original language.
7. Preserve headings and paragraphs.
8. Preserve reading order.
9. Preserve tables where possible.
10. Do not invent missing or unreadable text.
11. Mark unreadable text as [UNREADABLE].
12. Return only the extracted document text.
13. Separate each page using a page number heading.
"""


def test_key(api_key, index, pdf_data):

    start = time.perf_counter()
    client = None

    result = {
        "key_number": index,
        "masked_key": f"****{api_key[-4:]}",
        "model": MODEL,
        "file_name": PDF_PATH.name,
        "status": "FAILED",
        "response_time_seconds": 0,
        "character_count": 0,
        "finish_reason": None,
        "ocr_text": "",
        "error_code": None,
        "error_message": None
    }

    print(f"\nTesting Key {index}: {result['masked_key']}")
    print("Processing PDF...", flush=True)

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=120000
            )
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type="application/pdf"
                ),
                OCR_PROMPT
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=16384
            )
        )

        extracted_text = response.text or ""

        finish_reason = None

        if response.candidates:
            finish_reason = str(
                response.candidates[0].finish_reason
            )

        result["finish_reason"] = finish_reason
        result["ocr_text"] = extracted_text
        result["character_count"] = len(extracted_text)

        if "MAX_TOKENS" in (finish_reason or ""):
            result["status"] = "TRUNCATED RESPONSE"

        elif not extracted_text.strip():
            result["status"] = "EMPTY OCR RESPONSE"

        elif finish_reason and finish_reason.split(".")[-1] != "STOP":
            result["status"] = "INCOMPLETE RESPONSE"

        else:
            result["status"] = "SUCCESS"

        print(f"OCR preview:\n{extracted_text[:500]}")

    except Exception as error:

        code = getattr(error, "code", None)
        message = str(error).replace(api_key, "[REDACTED]")

        result["error_code"] = code
        result["error_message"] = message

        if "API_KEY_INVALID" in message:
            result["status"] = "INVALID API KEY"

        elif code == 429:
            result["status"] = "QUOTA EXCEEDED"

        elif code == 403:
            result["status"] = "PERMISSION DENIED"

        elif code == 401:
            result["status"] = "AUTHENTICATION FAILED"

        elif code == 400:
            result["status"] = "BAD REQUEST"

        elif code == 404:
            result["status"] = "MODEL NOT FOUND"

        elif code in (500, 502, 503, 504):
            result["status"] = "SERVER ERROR"

        else:
            result["status"] = "FAILED"

        print(f"Error code: {code}")
        print(f"Error details: {message}")

    finally:

        result["response_time_seconds"] = round(
            time.perf_counter() - start, 2
        )

        if client is not None:
            client.close()

    output_file = OUTPUT_DIR / f"key_{index}_result.json"

    output_file.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print(f"Status: {result['status']}")
    print(f"Response time: {result['response_time_seconds']}s")
    print(f"Characters extracted: {result['character_count']}")
    print(f"Result saved: {output_file.name}")
    print("-" * 60)

    return result


def main():

    if not API_KEYS:
        print("No Gemini API keys found in .env")
        return

    if not PDF_PATH.is_file():
        print(f"PDF not found: {PDF_PATH}")
        return

    pdf_data = PDF_PATH.read_bytes()

    if not pdf_data.startswith(b"%PDF-"):
        print("Invalid PDF file.")
        return

    results = []

    print("=" * 60)
    print("GEMINI PDF OCR - API KEY TESTING")
    print(f"Model: {MODEL}")
    print(f"Total keys: {len(API_KEYS)}")
    print(f"PDF: {PDF_PATH.name}")
    print(f"PDF size: {len(pdf_data) / 1024:.2f} KB")
    print("=" * 60)

    for index, api_key in enumerate(API_KEYS, start=1):

        result = test_key(api_key, index, pdf_data)
        results.append(result)

        time.sleep(2)

    successful = sum(
        result["status"] == "SUCCESS"
        for result in results
    )

    summary = {
        "model": MODEL,
        "pdf": PDF_PATH.name,
        "total_keys": len(API_KEYS),
        "successful_keys": successful,
        "unsuccessful_keys": len(API_KEYS) - successful,
        "results": [
            {
                key: value
                for key, value in result.items()
                if key != "ocr_text"
            }
            for result in results
        ]
    }

    summary_file = OUTPUT_DIR / "summary.json"

    summary_file.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\nFINAL OCR TEST SUMMARY")
    print("=" * 60)
    print(f"Total keys: {len(API_KEYS)}")
    print(f"Successful keys: {successful}")
    print(f"Unsuccessful keys: {len(API_KEYS) - successful}")

    for result in results:
        print(
            f"Key {result['key_number']}: "
            f"{result['status']} | "
            f"{result['response_time_seconds']}s | "
            f"{result['character_count']} characters"
        )

    print(f"\nAll results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
