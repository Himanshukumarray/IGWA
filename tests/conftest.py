"""Shared pytest configuration: no test may reach a real provider with the keys in .env.

Every test starts with an empty Gemini key pool and a blocked Groq client; tests that need a provider
install their own fake over these.
"""
import pytest

from services.ocr import gemini_key_manager, groq_client


@pytest.fixture(autouse=True)
def _no_real_providers(monkeypatch):
    def blocked():
        raise groq_client.GroqOCRError("real Groq calls are blocked in tests")

    monkeypatch.setattr(groq_client, "_get_client", blocked)
    monkeypatch.setattr(gemini_key_manager, "_manager", gemini_key_manager.GeminiKeyManager([]))
