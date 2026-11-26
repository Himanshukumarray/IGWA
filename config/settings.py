from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "IGWA"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    gemini_api_key: str = ""
    # Ordered failover pool. Precedence: GEMINI_API_KEYS (comma list) > GEMINI_API_KEY_1..10 >
    # legacy GEMINI_API_KEY. See services/ocr/gemini_key_manager.py.
    gemini_api_keys: str = ""
    gemini_api_key_1: str = ""
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    gemini_api_key_4: str = ""
    gemini_api_key_5: str = ""
    gemini_api_key_6: str = ""
    gemini_api_key_7: str = ""
    gemini_api_key_8: str = ""
    gemini_api_key_9: str = ""
    gemini_api_key_10: str = ""
    # Per attempt. The whole document is one request, so long/scanned PDFs need a generous value.
    gemini_request_timeout_seconds: float = Field(default=150.0, gt=0)
    # Whole failover chain of one call. Keep it >= 2 x request timeout + margin so a first slow key cannot
    # leave the second key a doomed sliver of time.
    gemini_overall_budget_seconds: float = Field(default=310.0, gt=0)
    # An attempt is not started with less budget than this left (it could not finish): no doomed call, no blame.
    gemini_min_attempt_seconds: float = Field(default=15.0, ge=0)
    gemini_shared_failure_cooldown_seconds: float = Field(default=10.0, ge=0)  # fast-fail after network/5xx
    gemini_rate_limit_cooldown_seconds: float = Field(default=60.0, ge=0)  # 429 without Retry-After
    gemini_quota_cooldown_seconds: float = Field(default=3600.0, ge=0)  # daily quota exhausted
    gemini_billing_cooldown_seconds: float = Field(default=3600.0, ge=0)  # 402
    gemini_forbidden_cooldown_seconds: float = Field(default=3600.0, ge=0)  # 403
    gemini_model: str = "gemini-3.1-flash-lite"

    # --- Gemini key selection / failover (see services/ocr/gemini_key_manager.py) ---
    # random: every OCR request shuffles the healthy keys (no replacement); ordered: configured order.
    gemini_key_selection: Literal["random", "ordered"] = "random"
    gemini_failover_enabled: bool = True  # False: one key only, no switching
    gemini_failover_delay_seconds: float = Field(default=0.0, ge=0)  # pause between keys (0 = none)
    # A 5xx/timeout/network failure moves on to another key; after this many DISTINCT keys fail that
    # way in one request it is treated as a provider-wide outage (circuit opens). 1 = never rotate.
    gemini_shared_failover_max_keys: int = Field(default=2, ge=1)
    # Keys tried for one request whose response is empty/truncated/invalid before giving up on Gemini.
    gemini_max_invalid_response_attempts: int = Field(default=2, ge=1)
    ocr_validate_response: bool = True  # reject empty/truncated/invalid Gemini output (retry other key)
    ocr_retry_failed_pages: bool = True  # re-OCR only missing pages / page-by-page after truncation
    ocr_overall_deadline_seconds: float = Field(default=400.0, gt=0)  # OCR: Gemini + Groq together

    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"

    max_file_size_mb: int = 20

    # --- OCR: Gemini (ordered key failover) -> Groq, whole document ---
    ocr_enable_gemini_fallback: bool = True  # False skips Gemini entirely
    ocr_enable_groq_fallback: bool = True
    ocr_groq_max_retries: int = Field(default=1, ge=0, le=3)

    # Per-provider cap for a text-only metadata attempt (used when the OCR response carried no metadata).
    metadata_timeout_seconds: float = 90.0

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
