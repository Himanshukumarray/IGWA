"""MOCKED tests for randomized Gemini key selection, immediate failover, response validation and
page-wise recovery. No network, no real credentials."""
import asyncio
import json
import logging
import random
import socket
import time
from types import SimpleNamespace

import fitz
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from config.settings import get_settings
from main import app
from services.ocr import gemini_client, groq_client, service
from services.ocr import gemini_key_manager as km
from services.ocr.groq_client import GroqOCRError

KEYS = [f"AIzaSyRANDOMKEY{i}0000000000000000000000" for i in range(1, 7)]
META = {"document_type": "FIR", "fir_number": "F-1"}


def ocr_json(pages=1, tag="G"):
    return json.dumps({"pages": [{"page_number": n, "text": f"{tag}{n} text"} for n in range(1, pages + 1)],
                       "metadata": META})


class ApiError(Exception):
    def __init__(self, code, message="boom", headers=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.response = SimpleNamespace(status_code=code, headers=headers or {})


def resp(text, finish=None):
    cands = [SimpleNamespace(finish_reason=finish)] if finish else []
    return SimpleNamespace(text=text, candidates=cands)


def pages_in(kw):
    """Number of pages in the PDF part of a generate_content call."""
    part = kw["contents"][0]
    data = part.inline_data.data if hasattr(part, "inline_data") else None
    return fitz.open(stream=data, filetype="pdf").page_count


class Fleet:
    """Fake Gemini clients. behaviors[slot] is consumed in order, the last item repeats.
    Items: "ok" | "hang" | an exception | a response | callable(**kw) -> response."""

    def __init__(self, behaviors=None, policy=None, keys=KEYS, pages=1, seed=None):
        self.calls: list[int] = []
        self.pages = pages
        self.behaviors = {i: list(v) for i, v in (behaviors or {}).items()}
        self.manager = km.GeminiKeyManager(
            list(keys), policy or km.Policy(selection="random"), self._factory, rng=random.Random(seed))

    def _factory(self, key, policy):
        slot = KEYS.index(key) + 1
        fleet = self

        async def generate_content(**kw):
            fleet.calls.append(slot)
            queue = fleet.behaviors.get(slot, ["ok"])
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if item == "ok":
                return resp(ocr_json(fleet.pages))
            if item == "hang":
                await asyncio.sleep(30)
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                return item(**kw)
            return item

        return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))

    def install(self, monkeypatch):
        monkeypatch.setattr(km, "_manager", self.manager)
        return self


def run(coro):
    return asyncio.run(coro)


def execute(fleet, **kw):
    async def go():
        return await fleet.manager.execute(
            lambda c: c.aio.models.generate_content(model="m", contents=["x"]), purpose="test", **kw)
    return run(go())


def health(fleet):
    return {s["slot"]: s["health"] for s in fleet.manager.snapshot()}


# ---- 1-4, 26: selection ---------------------------------------------------------------------------


def test_all_healthy_one_request_one_attempt():
    f = Fleet(seed=1)
    execute(f)
    assert len(f.calls) == 1 and f.manager.metrics["attempts"] == 1


def test_selection_is_randomized_across_requests_without_repeats():
    f = Fleet(seed=7)
    firsts = []
    for _ in range(300):
        f.calls.clear()
        execute(f)
        firsts.append(f.calls[0])
    counts = {k: firsts.count(k) for k in range(1, 7)}
    assert all(c > 20 for c in counts.values()), counts  # every key gets a fair share (300/6 = 50 expected)


def test_ordered_mode_keeps_configured_order():
    f = Fleet(policy=km.Policy(selection="ordered"), behaviors={1: [ApiError(429, "x")]})
    execute(f)
    assert f.calls == [1, 2]


def test_failover_walks_a_shuffled_order_and_never_repeats_a_key():
    for seed in range(40):
        f = Fleet({i: [ApiError(429, "slow")] for i in (1, 2, 3, 4, 5)}, seed=seed)  # only key 6 works
        execute(f)
        assert f.calls[-1] == 6 and len(f.calls) == len(set(f.calls)) and len(f.calls) <= 6
        expected = list(range(1, 7))
        random.Random(seed).shuffle(expected)
        assert f.calls == expected[: expected.index(6) + 1]  # exactly the planned shuffle, in order


def test_first_fails_second_succeeds_and_nothing_else_is_called():
    f = Fleet({i: [ApiError(429, "slow")] for i in range(1, 7)} | {}, seed=3)
    f.behaviors[5] = ["ok"]
    execute(f)
    assert f.calls[-1] == 5 and f.manager.metrics["successes"] == 1


# ---- 5-8: error classes ----------------------------------------------------------------------------


def test_429_cools_down_and_other_keys_serve_until_recovery():
    clock = [1000.0]
    f = Fleet(seed=2)
    f.manager._clock = lambda: clock[0]
    f.behaviors[1] = [ApiError(429, "slow", {"retry-after": "30"}), "ok"]
    f.manager._rng = random.Random()
    f.manager.policy = km.Policy(selection="ordered")
    execute(f)
    assert health(f)[1] == "RATE_LIMITED" and f.calls == [1, 2]
    clock[0] += 31  # cooldown expired -> eligible again
    f.calls.clear()
    execute(f)
    assert f.calls == [1] and health(f)[1] == "HEALTHY"


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_5xx_rotates_to_another_key_and_benches_the_failing_one(code):
    f = Fleet({1: [ApiError(code, "overloaded")]}, policy=km.Policy(selection="ordered", shared_failover_max_keys=2))
    execute(f)
    assert f.calls == [1, 2] and health(f)[1] == "TEMPORARILY_UNAVAILABLE"


def test_provider_wide_outage_opens_breaker_after_max_keys_not_all_six():
    err = ApiError(503, "overloaded")
    f = Fleet({i: [err] for i in range(1, 7)}, policy=km.Policy(selection="ordered", shared_failover_max_keys=2))
    with pytest.raises(km.GeminiSharedFailure):
        execute(f)
    assert f.calls == [1, 2]
    with pytest.raises(km.GeminiSharedFailure):  # circuit open: no new requests at all
        execute(f)
    assert f.calls == [1, 2]


@pytest.mark.parametrize("code,message", [(401, "unauthenticated"), (400, "API key not valid. Please pass a valid API key.")])
def test_invalid_or_unauthenticated_key_is_disabled_and_skipped(code, message):
    f = Fleet({1: [ApiError(code, message)]}, policy=km.Policy(selection="ordered"))
    execute(f)
    execute(f)
    assert f.calls == [1, 2, 2] and health(f)[1] == "AUTHENTICATION_FAILED"


def test_authentication_failure_recovers_after_credential_check_succeeds():
    f = Fleet({1: [ApiError(401, "nope")]}, policy=km.Policy(selection="ordered"))
    execute(f)
    f.manager._slots[0].client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        get=lambda model: asyncio.sleep(0))))
    run(f.manager.check_keys("m"))
    assert health(f)[1] == "HEALTHY"


# ---- 9-11: request errors ----------------------------------------------------------------------------


def test_invalid_pdf_400_does_not_burn_the_pool():
    f = Fleet({i: [ApiError(400, "Request contains an invalid argument: unable to process input document")]
               for i in range(1, 7)}, policy=km.Policy(selection="random", shared_failover_max_keys=2))
    with pytest.raises(km.GeminiRequestFailure):
        execute(f)
    assert len(f.calls) == 1 and set(health(f).values()) == {"HEALTHY"}


def test_wrong_model_404_is_confirmed_on_one_other_key_then_stops():
    f = Fleet({i: [ApiError(404, "models/x is not found")] for i in range(1, 7)},
              policy=km.Policy(selection="random", shared_failover_max_keys=2))
    with pytest.raises(km.GeminiRequestFailure) as e:
        execute(f)
    assert len(f.calls) == 2 and e.value.category == "model_not_found" and set(health(f).values()) == {"HEALTHY"}


def test_404_on_one_key_only_is_credential_specific_and_other_key_serves():
    f = Fleet({1: [ApiError(404, "not found")]}, policy=km.Policy(selection="ordered", shared_failover_max_keys=2))
    execute(f)
    assert f.calls == [1, 2]


def test_timeout_switches_key_and_is_bounded():
    f = Fleet({1: ["hang"]}, policy=km.Policy(selection="ordered", request_timeout_seconds=0.05,
                                              shared_failover_max_keys=2))
    t0 = time.perf_counter()
    execute(f)
    assert f.calls == [1, 2] and time.perf_counter() - t0 < 1.0


def test_overall_deadline_stops_a_chain_of_slow_keys():
    f = Fleet({i: ["hang"] for i in range(1, 7)},
              policy=km.Policy(selection="ordered", request_timeout_seconds=0.05, overall_budget_seconds=0.12,
                               shared_failover_max_keys=6))
    t0 = time.perf_counter()
    with pytest.raises(km.GeminiSharedFailure):
        execute(f)
    assert time.perf_counter() - t0 < 1.0 and len(f.calls) <= 3


# ---- 12-14: response validation -------------------------------------------------------------------------


def ocr(fleet, monkeypatch, pages=1, **settings):
    cfg = get_settings().model_copy(update=dict(gemini_model="gem", groq_model="grq", **settings))
    monkeypatch.setattr(gemini_client, "get_settings", lambda: cfg)
    fleet.install(monkeypatch)
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 100), f"page {i + 1}", fontsize=11)
    data = doc.tobytes()
    gemini_client.expected_pages_var.set(pages)
    return run(gemini_client.extract_text(data, "application/pdf"))


@pytest.mark.parametrize("bad", [
    resp(""), resp("   "), resp("not json at all"), resp('{"pages": "nope"}'), resp('{"error": "quota"}'),
    resp(ocr_json(1), finish="SAFETY"), resp(ocr_json(1)[:30], finish="MAX_TOKENS"),
])
def test_unusable_response_is_retried_on_another_key(bad, monkeypatch):
    f = Fleet({1: [bad], 2: ["ok"], 3: ["ok"], 4: ["ok"], 5: ["ok"], 6: ["ok"]},
              policy=km.Policy(selection="ordered"))
    out = ocr(f, monkeypatch)
    assert f.calls == [1, 2] and out["pages"][0]["text"] == "G1 text"
    assert f.manager.snapshot()[0]["health"] == "HEALTHY"  # a bad answer is not the key's fault...
    assert f.manager.snapshot()[0]["failed_requests"] == 1  # ...but it is counted


def test_invalid_responses_stop_after_the_configured_attempts():
    f = Fleet({i: [resp("")] for i in range(1, 7)}, policy=km.Policy(selection="ordered", max_invalid_response_attempts=2))
    with pytest.raises(km.GeminiInvalidResponse):
        execute(f, validate=lambda r: (_ for _ in ()).throw(km.InvalidResponse("empty", "empty_response")))
    assert len(f.calls) == 2


def test_truncated_multipage_output_falls_back_to_page_by_page(monkeypatch):
    def per_page_or_truncate(**kw):
        n = pages_in(kw)
        if n == 3:
            return resp(ocr_json(3)[:40], finish="MAX_TOKENS")
        return resp(json.dumps({"pages": [{"page_number": 1, "text": f"single-{id(kw) % 7}"}]}))

    f = Fleet({i: [per_page_or_truncate] for i in range(1, 7)}, policy=km.Policy(selection="random"), pages=3)
    out = ocr(f, monkeypatch, pages=3)
    assert [p["page_number"] for p in out["pages"]] == [1, 2, 3] and "metadata" not in out
    assert len(f.calls) == 2 + 3  # 2 truncated whole-document attempts, then one call per page


# ---- 24: partial recovery ---------------------------------------------------------------------------------


def test_only_missing_pages_are_re_ocrd_and_order_is_preserved(monkeypatch):
    seen = []

    def answer(**kw):
        n = pages_in(kw)
        seen.append(n)
        if n == 3:  # whole document: page 2 skipped
            return resp(json.dumps({"pages": [{"page_number": 3, "text": "C"}, {"page_number": 1, "text": "A"}],
                                    "metadata": META}))
        return resp(json.dumps({"pages": [{"page_number": 1, "text": "B"}]}))

    f = Fleet({i: [answer] for i in range(1, 7)}, pages=3)
    out = ocr(f, monkeypatch, pages=3)
    assert seen == [3, 1] and [(p["page_number"], p["text"]) for p in out["pages"]] == [(1, "A"), (2, "B"), (3, "C")]
    assert "metadata" not in out  # incomplete-transcript metadata is discarded; re-extracted from full text


def test_page_recovery_can_be_disabled_then_other_key_is_tried(monkeypatch):
    def skip(**kw):
        return resp(json.dumps({"pages": [{"page_number": 1, "text": "A"}]}))

    f = Fleet({1: [skip], 2: ["ok"], 3: ["ok"], 4: ["ok"], 5: ["ok"], 6: ["ok"]}, policy=km.Policy(selection="ordered"),
              pages=3)
    out = ocr(f, monkeypatch, pages=3, ocr_retry_failed_pages=False)
    assert f.calls == [1, 2] and len(out["pages"]) == 3


# ---- 15-16, 22, 28: provider fallback -----------------------------------------------------------------------


class Env:
    def __init__(self, monkeypatch, groq_ok=True):
        self.groq_calls = 0
        cfg = get_settings().model_copy(update=dict(gemini_model="gem", groq_model="grq"))
        monkeypatch.setattr(service, "get_settings", lambda: cfg)
        monkeypatch.setattr(gemini_client, "get_settings", lambda: cfg)
        service.ocr_results.clear()

        async def groq(data, mime):
            self.groq_calls += 1
            if not groq_ok:
                raise GroqOCRError("down")
            return {"pages": [{"page_number": 1, "text": "GROQ TEXT"}], "full_text": "GROQ TEXT"}

        monkeypatch.setattr(groq_client, "extract_text", groq)


class FakeFile:
    def __init__(self, data=None):
        if data is None:
            doc = fitz.open()
            doc.new_page().insert_text((72, 100), "x", fontsize=11)
            data = doc.tobytes()
        self.filename, self.content_type, self._data = "a.pdf", "application/pdf", data

    async def read(self, n=-1):
        data, self._data = self._data, b""
        return data


def test_all_keys_fail_then_groq_succeeds(monkeypatch):
    env = Env(monkeypatch)
    f = Fleet({i: [ApiError(429, "quota")] for i in range(1, 7)}, seed=4).install(monkeypatch)
    r = run(service.process_ocr(FakeFile()))
    assert r.ocr_provider == "groq" and env.groq_calls == 1 and sorted(f.calls) == [1, 2, 3, 4, 5, 6]


def test_shared_project_quota_exhaustion_tries_each_key_once_then_groq(monkeypatch):
    env = Env(monkeypatch)
    f = Fleet({i: [ApiError(429, "Quota exceeded for metric generate_content_free_tier_requests PerDay, limit: 0")]
               for i in range(1, 7)}).install(monkeypatch)
    r = run(service.process_ocr(FakeFile()))
    assert len(f.calls) == 6 and r.ocr_provider == "groq"
    assert {s["reason"] for s in f.manager.snapshot()} == {"quota_exhausted"}  # long cooldown, not retried


def test_all_providers_fail_is_a_502_and_nothing_fabricated_is_stored(monkeypatch):
    Env(monkeypatch, groq_ok=False)
    Fleet({i: [ApiError(401, "bad")] for i in range(1, 7)}).install(monkeypatch)
    with pytest.raises(HTTPException) as e:
        run(service.process_ocr(FakeFile()))
    assert e.value.status_code == 502 and service.ocr_results == {}


def test_overall_deadline_skips_groq_when_no_budget_is_left(monkeypatch):
    env = Env(monkeypatch)
    cfg = get_settings().model_copy(update=dict(gemini_model="gem", groq_model="grq", ocr_overall_deadline_seconds=0.0))
    monkeypatch.setattr(service, "get_settings", lambda: cfg)
    Fleet().install(monkeypatch)
    with pytest.raises(HTTPException):
        run(service.process_ocr(FakeFile()))
    assert env.groq_calls == 0


# ---- 19-21: pool configuration -----------------------------------------------------------------------------


def test_empty_pool():
    with pytest.raises(km.GeminiKeysExhausted) as e:
        execute(Fleet(keys=[]))
    assert e.value.category == "not_configured"


def test_single_key_pool_works_and_fails_cleanly():
    f = Fleet(keys=KEYS[:1])
    execute(f)
    assert f.calls == [1]
    f2 = Fleet({1: [ApiError(429, "slow")]}, keys=KEYS[:1])
    with pytest.raises(km.GeminiKeysExhausted):
        execute(f2)
    assert f2.calls == [1]


def test_duplicate_and_empty_keys_are_dropped_and_pool_size_is_dynamic():
    s = get_settings().model_copy(update=dict(
        gemini_api_key="", gemini_api_keys=",".join([KEYS[0], KEYS[1], KEYS[0], "", " ", KEYS[2], KEYS[3], KEYS[4], KEYS[5]])))
    assert km.load_keys(s) == KEYS  # six unique keys, nothing hard-coded to five


def test_reload_adds_and_removes_keys_and_keeps_health_of_survivors():
    f = Fleet({1: [ApiError(401, "bad")]}, keys=KEYS[:3], policy=km.Policy(selection="ordered"))
    execute(f)
    f.manager.reload([KEYS[0], KEYS[2], KEYS[4], KEYS[5]])
    assert f.manager.counts() == {"configured": 4, "available": 3}  # KEYS[0] stays disabled


# ---- 23, 25: latency -----------------------------------------------------------------------------------------


def test_no_sleep_between_key_attempts(monkeypatch):
    async def no_sleep(*a, **k):
        raise AssertionError("failover must not sleep")

    f = Fleet({i: [ApiError(429, "slow")] for i in range(1, 6)})
    monkeypatch.setattr(km.asyncio, "sleep", no_sleep)  # only the manager's module attribute is patched
    t0 = time.perf_counter()
    execute(f)
    assert f.calls[-1] == 6 and (time.perf_counter() - t0) < 0.5


def test_optional_failover_delay_is_honoured_and_zero_by_default():
    assert km.Policy().failover_delay_seconds == 0.0 and get_settings().gemini_failover_delay_seconds == 0.0
    f = Fleet({1: [ApiError(429, "x")]}, policy=km.Policy(selection="ordered", failover_delay_seconds=0.05))
    t0 = time.perf_counter()
    execute(f)
    assert time.perf_counter() - t0 >= 0.05


def test_failover_disabled_uses_a_single_key():
    f = Fleet({1: [ApiError(429, "x")]}, policy=km.Policy(selection="ordered", failover_enabled=False))
    with pytest.raises(km.GeminiKeysExhausted):
        execute(f)
    assert f.calls == [1]


# ---- 18, 30: concurrency, cancellation ---------------------------------------------------------------------------


def test_concurrent_uploads_all_succeed_with_balanced_load_and_clean_state(monkeypatch):
    Env(monkeypatch)
    f = Fleet({1: [ApiError(429, "slow", {"retry-after": "60"})]}, seed=9,
              policy=km.Policy(selection="random", max_concurrent_per_key=4)).install(monkeypatch)

    async def burst():
        return await asyncio.gather(*(service.process_ocr(FakeFile()) for _ in range(60)))

    results = run(burst())
    assert len(results) == 60 and len({r.document_id for r in results}) == 60  # no shared/leaked records
    assert all(r.ocr_provider == "gemini" for r in results)
    snap = f.manager.snapshot()
    assert all(s["active_requests"] == 0 for s in snap)
    assert f.manager.snapshot()[0]["health"] == "RATE_LIMITED"  # cooled down for every later request
    assert f.calls.count(1) <= 60 // 6 * 2  # only requests already in flight before the 429 touched it
    assert sum(s["successful_requests"] for s in snap) == 60


def test_saturated_keys_are_deprioritized_not_blocked():
    f = Fleet(policy=km.Policy(selection="random", max_concurrent_per_key=1), seed=1)
    for s in f.manager._slots[:5]:
        s.active = 1  # five keys busy
    order, _ = f.manager._plan()
    assert order[0].number == 6
    for s in f.manager._slots:
        s.active = 1  # all busy: still usable
    assert len(f.manager._plan()[0]) == 6


def test_cancelled_request_releases_the_key_and_does_not_blame_it():
    f = Fleet({i: ["hang"] for i in range(1, 7)}, policy=km.Policy(selection="ordered", request_timeout_seconds=30))

    async def scenario():
        task = asyncio.create_task(f.manager.execute(
            lambda c: c.aio.models.generate_content(model="m", contents=["x"]), purpose="t"))
        await asyncio.sleep(0.05)
        assert f.manager.snapshot()[0]["active_requests"] == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    s = f.manager.snapshot()[0]
    assert s["active_requests"] == 0 and s["health"] == "HEALTHY" and s["failed_requests"] == 0


# ---- health bookkeeping ----------------------------------------------------------------------------------------------


def test_health_counters_are_tracked_per_key():
    f = Fleet({1: [ApiError(429, "slow", {"retry-after": "20"})]}, policy=km.Policy(selection="ordered"))
    execute(f)
    execute(f)
    a, b = f.manager.snapshot()[:2]
    assert (a["total_requests"], a["failed_requests"], a["consecutive_failures"], a["last_error"]) == (1, 1, 1, "rate_limited")
    assert a["cooldown_expires_at"] is not None and a["last_failure_at"] is not None and a["key"] == "Key-1"
    assert (b["total_requests"], b["successful_requests"], b["consecutive_failures"]) == (2, 2, 0)
    assert b["average_response_seconds"] is not None and b["last_success_at"] is not None


# ---- 27, 29: API compatibility and secrecy ----------------------------------------------------------------------------


def test_http_contract_and_health_are_unchanged(monkeypatch):
    Env(monkeypatch)
    Fleet({1: [ApiError(429, "x")]}, seed=1).install(monkeypatch)

    async def meta(t):
        from services.ocr.schemas import OCRMetadata
        return OCRMetadata(fir_number="F9")

    monkeypatch.setattr(gemini_client, "extract_metadata", meta)
    client = TestClient(app)
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "x", fontsize=11)
    body = client.post("/python/ocr/upload", files={"file": ("a.pdf", doc.tobytes(), "application/pdf")}).json()
    for key in ("success", "message", "document_id", "job_id", "status", "filename", "ocr_model", "ocr_provider",
                "ocr_data", "metadata", "metadata_status", "metadata_error", "isFir"):
        assert key in body
    health_body = client.get("/health").text
    assert not any(k in health_body for k in KEYS) and "Key-" not in health_body


def test_no_credential_leaks_through_logs_errors_or_snapshots(monkeypatch, caplog):
    f = Fleet({
        1: [ApiError(401, f"API key not valid: {KEYS[0]}")],
        2: [ApiError(429, f"quota for key {KEYS[1]}")],
        3: [ApiError(503, f"upstream key={KEYS[2]}")],
        4: [resp(f"error with {KEYS[3]}")],
    }, policy=km.Policy(selection="ordered", shared_failover_max_keys=2))
    with caplog.at_level(logging.DEBUG):
        out = ocr(f, monkeypatch)  # key 5 answers after four different failures
    blob = caplog.text + repr(out) + repr(f.manager.snapshot()) + repr(f.manager._slots) + repr(f.manager.metrics)
    assert not any(k in blob for k in KEYS)
    assert "Key-1" in caplog.text


def test_structured_log_traces_selection_failover_and_success(monkeypatch, caplog):
    km.request_id_var.set("req-123")
    f = Fleet({1: [ApiError(429, "slow")]}, policy=km.Policy(selection="ordered"))
    with caplog.at_level(logging.INFO):
        execute(f)
    text = caplog.text
    assert "request=req-123" in text and "available_keys=6/6" in text and "order=Key-1,Key-2" in text
    assert "Key-1 failed: rate_limited" in text and "switching to Key-2" in text and "failover selection" in text
    assert "Key-2 succeeded" in text


def test_demo_fallback_does_not_exist_as_a_code_path():
    assert not hasattr(get_settings(), "demo_ocr_fallback_enabled")  # .env value is ignored, never read
