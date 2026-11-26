"""Ordered Gemini API key failover.

Each request tries one key at a time. The order is shuffled per request (selection="random",
sampling without replacement) or the configured order (selection="ordered"). A failure is
classified: key-specific problems (invalid key, billing, quota/rate limit, permission) and
unusable responses (empty/truncated/invalid output) move straight on to the next key with no sleep;
shared problems (network/DNS, timeouts, 5xx) rotate too, but only up to
Policy.shared_failover_max_keys distinct keys, after which the provider is treated as down
(circuit breaker) instead of burning every key; malformed requests stop immediately because another
key would fail the same way.

State (cooldowns, health counters) is in-process memory guarded by a lock. With several uvicorn
workers each worker keeps its own view: that only costs an extra 429/5xx probe per worker.

Credentials never leave this module: diagnostics expose slot numbers only, and everything that
is logged or raised passes through redact().
"""
import asyncio
import contextvars
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

_KEY_PATTERN = re.compile(r"(?:AIza|AQ\.)[0-9A-Za-z_\-]{20,}")
_PLACEHOLDER_MARKERS = ("your_", "replace_with", "changeme", "xxxx")
_MAX_COOLDOWN_SECONDS = 3600.0
_MAX_NUMBERED_KEYS = 10
_MIN_ATTEMPT_SECONDS = 0.5  # never start an attempt with less budget than this

# Correlates log lines of one OCR request (set by the service; defaults to "-").
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("ocr_request_id", default="-")


# ---- errors raised to callers (never contain credentials) ----------------------------------------


class GeminiFailure(Exception):
    """Base class. `category` is a safe failure label; `no_retry` tells callers not to retry."""

    no_retry = True

    def __init__(self, message: str, category: str = "other", attempts: int = 0):
        super().__init__(message)
        self.category = category
        self.attempts = attempts


class InvalidResponse(Exception):
    """Raised by a response validator: the call succeeded but the output is unusable
    (empty, truncated, malformed). The manager tries another key; the key is not penalized."""

    def __init__(self, message: str, category: str = "invalid_response"):
        super().__init__(message)
        self.category = category


class GeminiInvalidResponse(GeminiFailure):
    """Every key tried returned an unusable response."""


class GeminiKeysExhausted(GeminiFailure):
    """No Gemini credential could serve the request (all failed, cooling down or disabled)."""


class GeminiSharedFailure(GeminiFailure):
    """Network/DNS/timeout/overload: rotating keys would not help."""


class GeminiRequestFailure(GeminiFailure):
    """Malformed request, unsupported model, payload too large...: same for every key."""


# ---- classification ---------------------------------------------------------------------------------


class Kind(str, Enum):
    KEY_INVALID = "key_invalid"  # permanent for this process
    FORBIDDEN = "forbidden"  # 403, cooldown
    BILLING = "billing"  # 402, cooldown
    RATE_LIMIT = "rate_limit"  # 429, short cooldown / Retry-After
    QUOTA = "quota_exhausted"  # 429 daily quota or 403 quota, long cooldown
    SHARED = "shared"  # network / timeout / 5xx: do not rotate
    REQUEST = "request"  # 4xx about the request itself: do not rotate


_KEY_KINDS = {Kind.KEY_INVALID, Kind.FORBIDDEN, Kind.BILLING, Kind.RATE_LIMIT, Kind.QUOTA}
_INVALID_KEY_MARKERS = (
    "api key not valid", "api_key_invalid", "api key expired", "invalid api key",
    "reported as leaked", "api key was reported",
)
_NETWORK_NAME_MARKERS = ("timeout", "connect", "dns", "gaierror", "network", "ssl", "disconnected",
                         "protocol", "remoteprotocol", "readerror", "closed")


@dataclass(frozen=True)
class Failure:
    kind: Kind
    category: str  # safe label for logs/metrics
    status: int | None = None
    retry_after: float | None = None


def redact(text: str, extra_keys: list[str] | None = None) -> str:
    """Remove anything that looks like (or is) a configured API key from text."""
    for key in extra_keys or []:
        if key:
            text = text.replace(key, "[redacted-key]")
    return _KEY_PATTERN.sub("[redacted-key]", text)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return getattr(getattr(exc, "response", None), "status_code", None)


def _retry_after(exc: BaseException, text: str) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        if headers is not None and headers.get("retry-after") is not None:
            return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        pass
    match = re.search(r"retry(?:delay|[ _-]?in)?\W{0,12}([0-9]+(?:\.[0-9]+)?)\s*s", text)
    return float(match.group(1)) if match else None


def classify_error(exc: BaseException) -> Failure:
    """Decide whether an exception is about the key, shared infrastructure, or the request."""
    text = " ".join(str(getattr(exc, "message", None) or exc).split()).lower()
    status = _status_of(exc)
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return Failure(Kind.SHARED, "timeout", status)
    if status is None:
        name = type(exc).__name__.lower()
        if isinstance(exc, OSError) or any(m in name for m in _NETWORK_NAME_MARKERS):
            return Failure(Kind.SHARED, "network", None)
        return Failure(Kind.REQUEST, "unknown_error", None)  # unknown: never rotate blindly
    if any(m in text for m in _INVALID_KEY_MARKERS) or status == 401:
        return Failure(Kind.KEY_INVALID, "authentication_failed", status)
    if status == 402 or (status in (400, 403) and "billing" in text):
        return Failure(Kind.BILLING, "billing_required", status)
    if status == 403:
        if "quota" in text:
            return Failure(Kind.QUOTA, "quota_exhausted", status, _retry_after(exc, text))
        return Failure(Kind.FORBIDDEN, "permission_denied", status)
    if status == 429:
        retry_after = _retry_after(exc, text)
        daily = "perday" in text.replace(" ", "") or "daily" in text
        if daily:
            return Failure(Kind.QUOTA, "quota_exhausted", status, retry_after)
        return Failure(Kind.RATE_LIMIT, "rate_limited", status, retry_after)
    if status == 408 or status >= 500:
        return Failure(Kind.SHARED, "timeout" if status in (408, 504) else "server_error", status)
    labels = {400: "bad_request", 404: "model_not_found", 413: "payload_too_large", 422: "unprocessable"}
    return Failure(Kind.REQUEST, labels.get(status, "request_error"), status)


# ---- key pool --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    request_timeout_seconds: float = 90.0
    overall_budget_seconds: float = 110.0
    shared_failure_cooldown_seconds: float = 10.0
    selection: str = "ordered"  # settings default is "random"; the class default keeps legacy order
    failover_enabled: bool = True
    failover_delay_seconds: float = 0.0
    shared_failover_max_keys: int = 1  # 1 = legacy: a 5xx/timeout stops instead of rotating
    max_invalid_response_attempts: int = 2
    rate_limit_cooldown_seconds: float = 60.0
    quota_cooldown_seconds: float = 3600.0
    billing_cooldown_seconds: float = 3600.0
    forbidden_cooldown_seconds: float = 3600.0
    # An execute() attempt is only started when at least this much budget is left: a call that cannot finish is
    # doomed, burns quota, and would otherwise be blamed on the provider.
    min_attempt_seconds: float = 0.5

    @classmethod
    def from_settings(cls, s: Settings) -> "Policy":
        return cls(
            request_timeout_seconds=s.gemini_request_timeout_seconds,
            overall_budget_seconds=s.gemini_overall_budget_seconds,
            shared_failure_cooldown_seconds=s.gemini_shared_failure_cooldown_seconds,
            selection=s.gemini_key_selection,
            failover_enabled=s.gemini_failover_enabled,
            failover_delay_seconds=s.gemini_failover_delay_seconds,
            shared_failover_max_keys=s.gemini_shared_failover_max_keys,
            max_invalid_response_attempts=s.gemini_max_invalid_response_attempts,
            rate_limit_cooldown_seconds=s.gemini_rate_limit_cooldown_seconds,
            quota_cooldown_seconds=s.gemini_quota_cooldown_seconds,
            billing_cooldown_seconds=s.gemini_billing_cooldown_seconds,
            forbidden_cooldown_seconds=s.gemini_forbidden_cooldown_seconds,
            min_attempt_seconds=s.gemini_min_attempt_seconds,
        )


class Health(str, Enum):
    HEALTHY = "HEALTHY"
    COOLDOWN = "COOLDOWN"
    RATE_LIMITED = "RATE_LIMITED"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    DISABLED = "DISABLED"


@dataclass(repr=False)
class _Slot:
    number: int  # 1-based, the only identifier ever logged ("Key-N")
    key: str = field(repr=False)
    disabled_reason: str | None = None
    cooldown_until: float = 0.0
    reason: str | None = None
    client: Any = None
    # health counters (mutated only under the manager lock)
    total: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_success_at: float | None = None  # epoch seconds
    last_failure_at: float | None = None
    last_error: str | None = None
    response_seconds: float = 0.0  # summed over successes

    @property
    def label(self) -> str:
        return f"Key-{self.number}"

    def health(self, now: float) -> Health:
        if self.disabled_reason:
            return Health.AUTHENTICATION_FAILED if self.disabled_reason == "authentication_failed" else Health.DISABLED
        if now < self.cooldown_until:
            if self.reason in ("rate_limited", "quota_exhausted"):
                return Health.RATE_LIMITED
            if self.reason in ("timeout", "network", "server_error"):
                return Health.TEMPORARILY_UNAVAILABLE
            return Health.COOLDOWN
        return Health.HEALTHY

    def __repr__(self) -> str:  # never include the credential
        return f"<GeminiSlot {self.number}>"

    def available(self, now: float) -> bool:
        return self.disabled_reason is None and now >= self.cooldown_until


def _is_placeholder(value: str) -> bool:
    low = value.lower()
    return not value or any(m in low for m in _PLACEHOLDER_MARKERS)


def load_keys(s: Settings) -> list[str]:
    """Configured credentials in order. GEMINI_API_KEYS (comma list) wins, then GEMINI_API_KEY_1..10,
    then the legacy single GEMINI_API_KEY. Placeholders are dropped and duplicates ignored."""
    def clean(values: list[str]) -> list[str]:
        out: list[str] = []
        for v in values:
            v = v.strip().strip("\"'")
            if not _is_placeholder(v) and v not in out:
                out.append(v)
        return out

    listed = clean(s.gemini_api_keys.split(","))
    numbered = clean([getattr(s, f"gemini_api_key_{i}", "") for i in range(1, _MAX_NUMBERED_KEYS + 1)])
    legacy = clean([s.gemini_api_key])
    pool = listed or numbered or legacy
    if legacy and pool is not legacy and legacy[0] not in pool:
        logger.warning("GEMINI_API_KEY is ignored because GEMINI_API_KEYS / GEMINI_API_KEY_n are set")
    return pool


class GeminiKeyManager:
    def __init__(
        self,
        keys: list[str],
        policy: Policy | None = None,
        client_factory: Callable[[str, Policy], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ):
        self._slots = [_Slot(i, k) for i, k in enumerate(keys, start=1)]
        self.policy = policy or Policy()
        self._client_factory = client_factory or _default_client_factory
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.Lock()  # never held across an await
        self._shared_block_until = 0.0
        self._shared_block_category = "network"
        self.metrics: dict[str, Any] = {
            "attempts": 0, "successes": 0, "failures": {}, "success_by_slot": {},
        }

    # -- state ---------------------------------------------------------------------------------
    def redact(self, text: str) -> str:
        return redact(text, [s.key for s in self._slots])

    def counts(self) -> dict[str, int]:
        now = self._clock()
        with self._lock:
            return {"configured": len(self._slots), "available": sum(s.available(now) for s in self._slots)}

    def snapshot(self) -> list[dict[str, Any]]:
        """Safe diagnostics: slot labels, states, counters. Never credentials."""
        now = self._clock()
        wall = time.time()
        with self._lock:
            out = []
            for s in self._slots:
                remaining = max(0.0, s.cooldown_until - now)
                out.append({
                    "slot": s.number,
                    "key": s.label,
                    "state": "disabled" if s.disabled_reason else ("cooldown" if now < s.cooldown_until else "available"),
                    "health": s.health(now).value,
                    "reason": s.disabled_reason or (s.reason if now < s.cooldown_until else None),
                    "cooldown_remaining_seconds": round(remaining, 1),
                    "cooldown_expires_at": wall + remaining if remaining else None,
                    "total_requests": s.total,
                    "successful_requests": s.successes,
                    "failed_requests": s.failures,
                    "consecutive_failures": s.consecutive_failures,
                    "last_success_at": s.last_success_at,
                    "last_failure_at": s.last_failure_at,
                    "last_error": s.last_error,
                    "average_response_seconds": round(s.response_seconds / s.successes, 3) if s.successes else None,
                })
            return out

    def reset(self) -> None:
        with self._lock:
            for s in self._slots:
                s.disabled_reason, s.cooldown_until, s.reason = None, 0.0, None
            self._shared_block_until = 0.0

    def _client(self, slot: _Slot) -> Any:
        with self._lock:
            if slot.client is None:
                slot.client = self._client_factory(slot.key, self.policy)
            return slot.client

    def _next_slot(self, tried: set[int]) -> "_Slot | None":
        """An eligible key this request has not tried yet (random or configured order), counted as an
        attempt. None when every key was tried, is cooling down or is disabled."""
        now = self._clock()
        with self._lock:
            candidates = [s for s in self._slots if s.available(now) and s.number not in tried]
            if not candidates:
                return None
            slot = self._rng.choice(candidates) if self.policy.selection == "random" else candidates[0]
            slot.total += 1
            self.metrics["attempts"] += 1
            return slot

    def _record_success(self, slot: _Slot, seconds: float) -> None:
        with self._lock:
            slot.successes += 1
            slot.consecutive_failures = 0
            slot.last_success_at = time.time()
            slot.response_seconds += seconds
            self.metrics["successes"] += 1
            by_slot = self.metrics["success_by_slot"]
            by_slot[slot.number] = by_slot.get(slot.number, 0) + 1

    def _record_failure(self, slot: _Slot, category: str) -> None:
        with self._lock:
            slot.failures += 1
            slot.consecutive_failures += 1
            slot.last_failure_at = time.time()
            slot.last_error = category
            self.metrics["failures"][category] = self.metrics["failures"].get(category, 0) + 1

    def _cooldown_for(self, failure: Failure) -> float:
        p = self.policy
        base = {Kind.RATE_LIMIT: p.rate_limit_cooldown_seconds, Kind.QUOTA: p.quota_cooldown_seconds,
                Kind.BILLING: p.billing_cooldown_seconds, Kind.FORBIDDEN: p.forbidden_cooldown_seconds}[failure.kind]
        if failure.kind == Kind.RATE_LIMIT and failure.retry_after is not None:
            seconds = failure.retry_after  # honour the provider's Retry-After
        elif failure.kind == Kind.QUOTA:
            seconds = max(failure.retry_after or 0.0, base)  # a daily quota does not clear in seconds
        else:
            seconds = base
        return min(max(seconds, 1.0), _MAX_COOLDOWN_SECONDS)

    def _penalize(self, slot: _Slot, failure: Failure) -> str:
        """Record a key-specific failure; returns a safe description for the log."""
        with self._lock:
            if failure.kind == Kind.KEY_INVALID:
                slot.disabled_reason = failure.category
                return "authentication failed (key disabled)"
            seconds = self._cooldown_for(failure)
            until = self._clock() + seconds
            if until > slot.cooldown_until:  # concurrent failures never shorten a cooldown
                slot.cooldown_until, slot.reason = until, failure.category
            return f"{failure.category.replace('_', ' ')} (cooldown {seconds:.0f}s)"

    def _bench(self, slot: _Slot, failure: Failure) -> None:
        """Briefly sideline a key after a 5xx/timeout/network failure (failover mode only)."""
        with self._lock:
            until = self._clock() + self.policy.shared_failure_cooldown_seconds
            if until > slot.cooldown_until:
                slot.cooldown_until, slot.reason = until, failure.category

    def _open_breaker(self, category: str) -> None:
        with self._lock:
            self._shared_block_until = self._clock() + self.policy.shared_failure_cooldown_seconds
            self._shared_block_category = category

    # -- execution -------------------------------------------------------------------------------
    async def execute(
        self,
        call: Callable[[Any], Awaitable[T]],
        *,
        purpose: str,
        budget: float | None = None,
        validate: Callable[[T], None] | None = None,
    ) -> T:
        """Run `call(client)` with failover. Returns the first successful (and valid) result.

        `validate(result)` may raise InvalidResponse; the next key is then tried (bounded by
        Policy.max_invalid_response_attempts). There is no sleep between keys unless
        Policy.failover_delay_seconds is set.
        """
        started = self._clock()
        deadline = started + (budget if budget is not None else self.policy.overall_budget_seconds)
        rid = request_id_var.get()
        p = self.policy
        if not self._slots:
            raise GeminiKeysExhausted("No Gemini API key is configured", "not_configured", 0)
        if started < self._shared_block_until:
            raise GeminiSharedFailure(
                f"Gemini recently failed with a shared error ({self._shared_block_category}); skipping",
                self._shared_block_category, 0)

        healthy = self.counts()["available"]
        logger.info("[OCR] request=%s purpose=%s available_keys=%d/%d selection=%s", rid,
                    purpose, healthy, len(self._slots), p.selection)
        attempts = shared_fails = invalid_fails = model_404s = timeouts = 0
        last_shared: tuple[str, BaseException] | None = None
        last_404: BaseException | None = None
        last_invalid: InvalidResponse | None = None
        last_timeout: BaseException | None = None
        failed_at: float | None = None
        tried: set[int] = set()
        # processing timeouts tolerated per call: same bound as other shared failures, never a breaker trip
        max_timeouts = max(1, p.shared_failover_max_keys)

        while True:
            if attempts and p.failover_delay_seconds > 0:
                await asyncio.sleep(min(p.failover_delay_seconds, max(deadline - self._clock() - 0.001, 0)))
            remaining = deadline - self._clock()
            timeout = min(p.request_timeout_seconds, remaining)
            if timeout < min(max(p.min_attempt_seconds, _MIN_ATTEMPT_SECONDS), p.request_timeout_seconds):
                # too little budget left for a call to finish: do not waste quota on a doomed request and
                # never blame the provider for the application's own deadline
                logger.warning("[OCR] request=%s purpose=%s remaining budget %.1fs is too small for another "
                               "attempt: not starting one (key health unchanged)", rid, purpose, remaining)
                raise GeminiSharedFailure("Gemini time budget exhausted", "budget_exceeded", attempts)
            slot = self._next_slot(tried)
            if slot is None:
                break  # no eligible key left
            now = self._clock()
            budget_limited = timeout < p.request_timeout_seconds
            tried.add(slot.number)
            attempts += 1
            attempt_started = now
            switch_ms = (now - failed_at) * 1000 if failed_at is not None else 0.0
            if attempts > 1:
                logger.info("[OCR] request=%s switching to %s (failover selection %.2fms)", rid, slot.label, switch_ms)
            try:
                result = await asyncio.wait_for(call(self._client(slot)), timeout)
                if validate is not None:
                    validate(result)
            except asyncio.CancelledError:
                raise  # caller gave up: do not blame the key
            except InvalidResponse as exc:
                failed_at = self._clock()
                self._record_failure(slot, exc.category)
                invalid_fails += 1
                last_invalid = exc
                logger.warning("[OCR] request=%s attempt=%d %s unusable response: %s (%.2fs)", rid, attempts,
                               slot.label, exc.category, failed_at - attempt_started)
                if not p.failover_enabled or invalid_fails >= p.max_invalid_response_attempts:
                    raise GeminiInvalidResponse(f"Gemini returned an unusable response: {exc.category}",
                                                exc.category, attempts) from exc
                continue
            except (asyncio.TimeoutError, TimeoutError) as exc:
                failed_at = self._clock()
                if budget_limited:
                    # the application gave this attempt less than a full timeout: nobody's fault, no health or
                    # circuit-breaker consequence
                    logger.warning("[OCR] request=%s attempt=%d %s hit the application deadline after %.1fs "
                                   "(budget-limited; key health unchanged, no circuit breaker)", rid, attempts,
                                   slot.label, failed_at - attempt_started)
                    raise GeminiSharedFailure("Gemini time budget exhausted", "budget_exceeded", attempts) from exc
                timeouts += 1
                last_timeout = exc
                self._record_failure(slot, "timeout")  # counter only: a slow response never disables or benches a key
                logger.warning("[OCR] request=%s attempt=%d %s processing timeout after %.1fs (key health "
                               "unchanged, circuit breaker not used)", rid, attempts, slot.label,
                               failed_at - attempt_started)
                if not p.failover_enabled or timeouts >= max_timeouts:
                    raise GeminiSharedFailure("Gemini processing timeout", "timeout", attempts) from exc
                continue
            except Exception as exc:
                failed_at = self._clock()
                failure = classify_error(exc)
                self._record_failure(slot, failure.category)
                if failure.kind in _KEY_KINDS:
                    logger.warning("Gemini credential slot %d: %s", slot.number, self._penalize(slot, failure))
                    logger.info("[OCR] request=%s attempt=%d %s failed: %s (%.2fs)", rid, attempts, slot.label,
                                failure.category, failed_at - attempt_started)
                    if not p.failover_enabled:
                        raise GeminiKeysExhausted("Gemini failover is disabled", failure.category, attempts) from exc
                    continue  # next key immediately, no sleep
                detail = self.redact(" ".join(str(exc).split())[:160])
                if failure.kind == Kind.SHARED and failure.category == "timeout":
                    # provider-side 408/504: slow work, not a provider outage
                    timeouts += 1
                    last_timeout = exc
                    if not p.failover_enabled or timeouts >= max_timeouts:
                        raise GeminiSharedFailure("Gemini processing timeout", "timeout", attempts) from exc
                    continue
                if failure.kind == Kind.SHARED:
                    shared_fails += 1
                    last_shared = (failure.category, exc)
                    if p.shared_failover_max_keys > 1:
                        self._bench(slot, failure)  # keeps concurrent requests off a flaky key briefly
                    if not p.failover_enabled or shared_fails >= p.shared_failover_max_keys:
                        self._open_breaker(failure.category)
                        logger.error("Gemini credential slot %d: %s (shared failure, not rotating keys) %s",
                                     slot.number, failure.category, detail)
                        raise GeminiSharedFailure(f"Gemini shared failure: {failure.category}",
                                                  failure.category, attempts) from exc
                    logger.warning("[OCR] request=%s attempt=%d %s failed: %s (%.2fs); trying another key",
                                   rid, attempts, slot.label, failure.category, failed_at - attempt_started)
                    continue
                # request-level error: identical for every key ...
                if (failure.status == 404 and p.failover_enabled and p.shared_failover_max_keys > 1
                        and model_404s + 1 < p.shared_failover_max_keys):
                    model_404s += 1  # ... except a 404, confirmed once on another key (credential-specific?)
                    last_404 = exc
                    continue
                logger.error("Gemini credential slot %d: %s (request error, not rotating keys) %s",
                             slot.number, failure.category, detail)
                raise GeminiRequestFailure(f"Gemini rejected the request: {failure.category}",
                                           failure.category, attempts) from exc
            duration = self._clock() - attempt_started
            self._record_success(slot, duration)
            logger.info("Gemini credential slot %d: request succeeded (purpose=%s attempts=%d duration=%.2fs)",
                        slot.number, purpose, attempts, self._clock() - started)
            logger.info("[OCR] request=%s attempt=%d %s succeeded provider_time=%.2fs", rid, attempts, slot.label,
                        duration)
            return result

        logger.error("Gemini: no credential could serve the request (purpose=%s attempts=%d)", purpose, attempts)
        if last_shared is not None:
            self._open_breaker(last_shared[0])
            raise GeminiSharedFailure(f"Gemini shared failure: {last_shared[0]}", last_shared[0],
                                      attempts) from last_shared[1]
        if last_timeout is not None:
            raise GeminiSharedFailure("Gemini processing timeout", "timeout", attempts) from last_timeout
        if last_404 is not None:
            raise GeminiRequestFailure("Gemini rejected the request: model_not_found", "model_not_found",
                                       attempts) from last_404
        if last_invalid is not None:
            raise GeminiInvalidResponse(f"Gemini returned an unusable response: {last_invalid.category}",
                                        last_invalid.category, attempts) from last_invalid
        raise GeminiKeysExhausted("All Gemini credentials failed or are unavailable",
                                  "all_keys_unavailable", attempts)

def _default_client_factory(api_key: str, policy: Policy) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(policy.request_timeout_seconds * 1000),
            # attempts=1: the SDK would otherwise retry connection errors itself (sleeping 1-10s),
            # multiplying with our failover. Failover is the only retry layer.
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


# ---- process-wide instance --------------------------------------------------------------------------------

_manager: GeminiKeyManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> GeminiKeyManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            settings = get_settings()
            _manager = GeminiKeyManager(load_keys(settings), Policy.from_settings(settings))
            logger.info("Gemini key manager ready: %d credential slot(s)", len(_manager._slots))
        return _manager


def set_manager(manager: GeminiKeyManager | None) -> None:
    """Install (or clear with None) the process-wide manager. Used by tests and reconfiguration."""
    global _manager
    with _manager_lock:
        _manager = manager
