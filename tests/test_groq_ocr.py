"""MOCKED tests for the Groq OCR fallback. No network, no real credentials."""
import asyncio
from types import SimpleNamespace

import fitz
import pytest

from config.settings import get_settings
from services.ocr import groq_client, service
from services.ocr.groq_client import GroqOCRError


class ApiError(Exception):
    def __init__(self, code, message="boom", headers=None):
        super().__init__(message)
        self.status_code = code
        self.response = SimpleNamespace(status_code=code, headers=headers or {})


def make_pdf(pages):
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 100), f"PAGE-{i + 1}", fontsize=11)
    return doc.tobytes()


def fake_client(monkeypatch, create, seen=None):
    def with_options(**kw):
        if seen is not None:
            seen.update(kw)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(groq_client, "_get_client", lambda: SimpleNamespace(with_options=with_options))


def test_finished_pages_are_kept_and_a_retry_only_sends_the_missing_ones(monkeypatch):
    calls = []

    async def create(**kw):
        calls.append(1)
        if len(calls) == 2:
            raise ApiError(429, "rate limited", {"retry-after": "7"})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"page-text-{len(calls)}"))])

    fake_client(monkeypatch, create)
    done: dict[int, str] = {}
    with pytest.raises(GroqOCRError) as e:
        asyncio.run(groq_client.extract_text(make_pdf(3), "application/pdf", done))
    assert e.value.partial is done and 1 in done and 2 not in done and e.value.category == "rate_limit"
    kept = len(done)
    calls.clear()
    out = asyncio.run(groq_client.extract_text(make_pdf(3), "application/pdf", done))
    assert len(calls) == 3 - kept  # only the pages that were still missing were requested again
    assert [p["page_number"] for p in out["pages"]] == [1, 2, 3] and out["pages"][0]["text"] == "page-text-1"


def test_service_retry_resends_only_missing_pages(monkeypatch):
    seen = []

    async def fake_groq(payload, mime, done=None):
        seen.append(dict(done))
        if len(seen) == 1:
            done[1] = "a"
            err = GroqOCRError("429")
            err.__cause__ = ApiError(429, "limit", {"retry-after": "0"})
            raise err
        done[2] = "b"
        return {"pages": [{"page_number": 1, "text": done[1]}, {"page_number": 2, "text": done[2]}]}

    monkeypatch.setattr(groq_client, "extract_text", fake_groq)
    cfg = get_settings().model_copy(update=dict(ocr_groq_max_retries=1))
    out = asyncio.run(service._call_groq(b"x", "application/pdf", 30.0, cfg))
    assert seen == [{}, {1: "a"}] and [p["text"] for p in out["pages"]] == ["a", "b"]


@pytest.mark.parametrize("code", [400, 401, 402])
def test_service_does_not_retry_permanent_errors(monkeypatch, code):
    calls = []

    async def fake_groq(payload, mime, done=None):
        calls.append(1)
        err = GroqOCRError("rejected")
        err.__cause__ = ApiError(code)
        raise err

    monkeypatch.setattr(groq_client, "extract_text", fake_groq)
    with pytest.raises(GroqOCRError):
        asyncio.run(service._call_groq(b"x", "application/pdf", 30.0, get_settings()))
    assert len(calls) == 1


def test_no_sdk_retries_and_a_failure_cancels_the_queued_pages(monkeypatch):
    seen = {"calls": 0}

    async def create(**kw):
        seen["calls"] += 1
        await asyncio.sleep(0.05)
        raise RuntimeError("502")

    fake_client(monkeypatch, create, seen)

    async def scenario():
        with pytest.raises(GroqOCRError):
            await groq_client.extract_text(make_pdf(5), "application/pdf")
        calls = seen["calls"]
        await asyncio.sleep(0.2)
        assert seen["calls"] == calls  # nothing keeps calling Groq after the failure

    asyncio.run(scenario())
    assert seen["max_retries"] == 0 and seen["calls"] <= 2
