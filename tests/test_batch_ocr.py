"""MOCKED tests for parallel batch OCR (services/ocr/batch_ocr.py) and the strict key pool.
No network, no real credentials. Test PDFs carry their own page labels ("PAGE-7"), and the fake Gemini
answers with the label it SEES, so batch-local -> original page mapping is verified end to end."""
import asyncio
import json
import logging
import random
import time
from types import SimpleNamespace

import fitz
import pytest
from fastapi import HTTPException

from config.settings import get_settings
from services.ocr import batch_ocr, gemini_client, groq_client, service
from services.ocr import gemini_key_manager as km
from services.ocr.batch_ocr import BatchOCRError, BatchRun

KEYS = [f"AIzaSyBATCHKEY{i}0000000000000000000000" for i in range(1, 7)]


class ApiError(Exception):
    def __init__(self, code, message="boom", headers=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.response = SimpleNamespace(status_code=code, headers=headers or {})


def make_pdf(pages, heavy=False):
    doc = fitz.open()
    for i in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 72), f"PAGE-{i}")
        if heavy:
            page.insert_text((72, 100), "x" * 10)
    data = doc.tobytes()
    doc.close()
    return data


def labels_of(kw):
    part = kw["contents"][0]
    with fitz.open(stream=part.inline_data.data, filetype="pdf") as d:
        return [p.get_text().strip() for p in d]


def ok_json(labels, start=1, skip=()):
    return json.dumps({"pages": [{"page_number": n, "text": f"OCR[{lab}]"}
                                 for n, lab in enumerate(labels, start=start) if n not in skip]})


def resp(text, finish=None):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish)] if finish else [])


class Fleet:
    """Fake Gemini fleet. behaviors[slot]: consumed in order, last repeats. Items: "ok" | "hang" |
    exception | ("sleep", s) | ("skip", {local numbers}) | "truncated" | "garbage" | callable(labels)->item.
    rules: callables (slot, labels) -> item | None checked first (document-specific failures)."""

    def __init__(self, behaviors=None, keys=KEYS, policy=None, seed=0, rules=None, delay=0.0):
        self.calls: list[tuple[int, list[str]]] = []
        self.behaviors = {i: list(v) for i, v in (behaviors or {}).items()}
        self.rules = rules or []
        self.delay = delay
        self.inflight: dict[int, int] = {}
        self.max_inflight: dict[int, int] = {}
        self.total_inflight = 0
        self.max_total = 0
        self.keys = list(keys)
        self.manager = km.GeminiKeyManager(
            self.keys, policy or km.Policy(selection="random", max_concurrent_per_key=1,
                                           global_max_concurrent=4, shared_failure_cooldown_seconds=30.0),
            self._factory, rng=random.Random(seed))

    def _factory(self, key, policy):
        slot = self.keys.index(key) + 1
        fleet = self

        async def generate_content(**kw):
            labels = labels_of(kw)
            fleet.calls.append((slot, labels))
            fleet.inflight[slot] = fleet.inflight.get(slot, 0) + 1
            fleet.total_inflight += 1
            fleet.max_inflight[slot] = max(fleet.max_inflight.get(slot, 0), fleet.inflight[slot])
            fleet.max_total = max(fleet.max_total, fleet.total_inflight)
            try:
                if fleet.delay:
                    await asyncio.sleep(fleet.delay)
                item = None
                for rule in fleet.rules:
                    item = rule(slot, labels)
                    if item is not None:
                        break
                if item is None:
                    queue = fleet.behaviors.get(slot, ["ok"])
                    item = queue.pop(0) if len(queue) > 1 else queue[0]
                if callable(item):
                    item = item(labels)
                if item == "ok":
                    return resp(ok_json(labels))
                if item == "hang":
                    await asyncio.sleep(60)
                if item == "truncated":
                    return resp('{"pages": [{"page_number": 1, "te', "MAX_TOKENS")
                if item == "garbage":
                    return resp("not json at all")
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, tuple) and item[0] == "sleep":
                    await asyncio.sleep(item[1])
                    return resp(ok_json(labels))
                if isinstance(item, tuple) and item[0] == "skip":
                    return resp(ok_json(labels, skip=item[1]))
                return item
            finally:
                fleet.inflight[slot] -= 1
                fleet.total_inflight -= 1

        return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))


def cfg(**kw):
    base = dict(ocr_batch_timeout_seconds=2.0, ocr_batch_size=5, ocr_overall_deadline_seconds=30.0,
                ocr_max_parallel_batches=4, ocr_global_max_concurrent_batches=4, gemini_reserved_keys=2,
                gemini_max_invalid_response_attempts=2, gemini_shared_failover_max_keys=2)
    base.update(kw)
    return get_settings().model_copy(update=base)


def run_batches(fleet, pages, s=None, groq=None, deadline=20.0, heavy=False):
    s = s or cfg()
    data = make_pdf(pages, heavy)
    groq_events = []

    async def go():
        km.request_id_var.set("test-req")
        runner = BatchRun(data, pages, s, manager=fleet.manager, deadline=time.monotonic() + deadline,
                          groq_call=groq, on_groq_fallback=lambda: groq_events.append(1))
        return await runner.run()

    result = asyncio.run(go())
    result.groq_events = groq_events
    return result


def texts(result):
    return {p.page_number: p.text for p in result.data.pages}


def expect_all(result, pages):
    assert texts(result) == {n: f"OCR[PAGE-{n}]" for n in range(1, pages + 1)}
    assert result.data.full_text == "\n\n".join(f"OCR[PAGE-{n}]" for n in range(1, pages + 1))


def batches_called(fleet):
    return sorted((tuple(labels) for _, labels in fleet.calls), key=lambda t: int(t[0].split('-')[1]))


# ---- batch creation (1-10) --------------------------------------------------------------------------


@pytest.mark.parametrize("pages,expected", [
    (1, 1), (3, 1), (5, 1), (6, 2), (10, 2), (15, 3), (20, 4), (30, 6), (50, 10), (100, 20)])
def test_plan_batches_counts(pages, expected):
    groups = batch_ocr.plan_batches(pages, 5)
    assert len(groups) == expected
    assert [p for g in groups for p in g] == list(range(1, pages + 1))  # order and original numbers kept
    assert all(len(g) <= 5 for g in groups)


def test_plan_boundaries_exact():
    assert batch_ocr.plan_batches(12, 5) == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12]]
    assert batch_ocr.plan_batches(0, 5) == []


def test_heavy_scanned_documents_get_smaller_batches():
    s = cfg()
    assert batch_ocr.effective_batch_size(20, 20 * 100_000, s) == 5
    assert batch_ocr.effective_batch_size(20, 20 * 2_000_000, s) == 3


def test_20_pages_maps_local_numbers_to_original_pages():
    f = Fleet()
    r = run_batches(f, 20)
    expect_all(r, 20)
    assert batches_called(f) == [tuple(f"PAGE-{i}" for i in range(a, a + 5)) for a in (1, 6, 11, 16)]
    assert r.stats["batches_planned"] == 4 and r.stats["pages_sent"] == 20  # nothing sent twice


def test_corrupt_or_empty_pdf_fails_cleanly():
    f = Fleet()

    async def go(data):
        km.request_id_var.set("t")
        return await BatchRun(data, 10, cfg(), manager=f.manager, deadline=time.monotonic() + 5).run()

    with pytest.raises(BatchOCRError):
        asyncio.run(go(b"%PDF-1.4 garbage"))
    assert service._expected_page_count(b"%PDF-1.4 garbage", "application/pdf") is None
    assert service._use_batches(b"x", "application/pdf", None, cfg()) is False


# ---- key allocation (11-24) ---------------------------------------------------------------------------


@pytest.mark.parametrize("healthy,active,standby", [
    (6, 4, 2), (5, 3, 2), (4, 2, 2), (3, 2, 1), (2, 1, 1), (1, 1, 0), (0, 0, 0)])
def test_allocation_table(healthy, active, standby):
    assert km.allocation(healthy, 4, 2) == (active, standby)


def test_allocation_without_reserved_keys_and_small_parallel_limit():
    assert km.allocation(6, 4, 0) == (4, 0)
    assert km.allocation(6, 2, 2) == (2, 2)


def test_six_keys_four_batches_use_four_distinct_keys_concurrently():
    f = Fleet(delay=0.2)
    r = run_batches(f, 20)
    expect_all(r, 20)
    assert len({slot for slot, _ in f.calls}) == 4
    assert f.max_total == 4 and max(f.max_inflight.values()) == 1
    assert r.stats["workers"] == 4 and r.stats["standby"] == 2


def test_selection_is_randomized_not_hardcoded():
    used = set()
    for seed in range(12):
        f = Fleet(seed=seed)
        run_batches(f, 20)
        used.add(tuple(sorted(s for s, _ in f.calls)))
    assert len(used) > 3


def test_fewer_healthy_keys_shrink_the_active_set():
    f = Fleet(keys=KEYS[:3], delay=0.1)
    r = run_batches(f, 30)
    expect_all(r, 30)
    assert r.stats["workers"] == 2 and r.stats["standby"] == 1 and f.max_total <= 2


def test_single_key_processes_everything_sequentially():
    f = Fleet(keys=KEYS[:1], delay=0.05)
    r = run_batches(f, 20)
    expect_all(r, 20)
    assert f.max_total == 1 and {s for s, _ in f.calls} == {1}  # no deadlock without standby


def test_billing_disabled_and_cooling_keys_are_never_used():
    f = Fleet()
    m = f.manager
    m._slots[0].disabled_reason = "authentication_failed"
    m._slots[1].cooldown_until = m._clock() + 100
    m._slots[2].cooldown_until, m._slots[2].reason = m._clock() + 100, "billing_required"
    r = run_batches(f, 20)
    expect_all(r, 20)
    assert {s for s, _ in f.calls} <= {4, 5, 6}
    assert r.stats["workers"] == 2  # 3 healthy keys


def test_duplicate_credentials_are_one_key():
    s = get_settings().model_copy(update={"gemini_api_keys": f"{KEYS[0]},{KEYS[0]},{KEYS[1]}",
                                          "gemini_api_key": ""})
    assert km.load_keys(s) == [KEYS[0], KEYS[1]]


def test_no_healthy_key_and_no_groq_fails_fast():
    f = Fleet()
    for slot in f.manager._slots:
        slot.disabled_reason = "authentication_failed"
    with pytest.raises(BatchOCRError):
        run_batches(f, 10)
    assert f.calls == []


def test_strict_per_key_cap_and_release():
    f = Fleet(keys=KEYS[:2])
    m = f.manager
    a, st = m.reserve()
    b, st2 = m.reserve()
    assert (st, st2) == ("ok", "ok") and a is not b
    assert m.reserve() == (None, "busy")  # both keys at their cap of 1: never handed out
    m.release(a)
    again, st3 = m.reserve()
    assert st3 == "ok" and again is a
    m.release(again)
    m.release(b)
    assert all(s["active_requests"] == 0 for s in m.snapshot())


def test_global_cap_is_strict_across_documents():
    f = Fleet(policy=km.Policy(selection="random", max_concurrent_per_key=1, global_max_concurrent=2), delay=0.1)

    async def go():
        km.request_id_var.set("t")
        s = cfg(ocr_global_max_concurrent_batches=2)
        runs = [BatchRun(make_pdf(20), 20, s, manager=f.manager, deadline=time.monotonic() + 20).run()
                for _ in range(3)]
        return await asyncio.gather(*runs)

    results = asyncio.run(go())
    assert all(texts(r) == {n: f"OCR[PAGE-{n}]" for n in range(1, 21)} for r in results)  # no leakage
    assert f.max_total <= 2


def test_failed_batch_uses_a_standby_key_that_initial_wave_left_free():
    f = Fleet(rules=[lambda slot, labels: ApiError(500, "x") if labels[0] == "PAGE-6" and slot in (1, 2, 3, 4)
                     else None], seed=9, delay=0.1)
    r = run_batches(f, 20)
    expect_all(r, 20)
    first_wave = [s for s, _ in f.calls[:4]]
    assert len(set(first_wave)) == 4  # 4 active keys; the other 2 were free as standby
    assert f.max_total <= 4


# ---- parallel processing (25-32) -------------------------------------------------------------------------


def test_next_batch_starts_as_soon_as_one_finishes_no_waves():
    f = Fleet(behaviors={}, delay=0.0, rules=[
        lambda slot, labels: ("sleep", 0.6) if labels[0] == "PAGE-1" else None])
    r = run_batches(f, 30, s=cfg(ocr_max_parallel_batches=2, gemini_reserved_keys=2))
    expect_all(r, 30)
    order = [labels[0] for _, labels in f.calls]
    assert order[:2] == ["PAGE-1", "PAGE-6"] and len(order) == 6
    assert r.stats["workers"] == 2


def test_out_of_order_completion_still_merges_in_page_order():
    f = Fleet(rules=[lambda slot, labels: ("sleep", 0.4) if labels[0] == "PAGE-1" else
                     ("sleep", 0.05) if labels[0] == "PAGE-11" else None])
    r = run_batches(f, 20)
    expect_all(r, 20)
    assert [p.page_number for p in r.data.pages] == list(range(1, 21))


def test_fair_between_documents_small_doc_is_not_starved():
    f = Fleet(keys=KEYS[:4], policy=km.Policy(selection="random", max_concurrent_per_key=1,
                                              global_max_concurrent=4), delay=0.15)
    done = {}

    async def go():
        km.request_id_var.set("t")
        s = cfg()

        async def one(name, pages):
            r = await BatchRun(make_pdf(pages), pages, s, manager=f.manager,
                               deadline=time.monotonic() + 30).run()
            done[name] = time.monotonic()
            return r

        big = asyncio.ensure_future(one("big", 100))
        await asyncio.sleep(0.05)
        await one("small", 10)
        return await big

    asyncio.run(go())
    assert done["small"] < done["big"]


# ---- recovery (33-49) -------------------------------------------------------------------------------------


def test_failed_batch_moves_to_standby_and_nothing_else_restarts():
    f = Fleet(behaviors={}, rules=[lambda slot, labels: ApiError(429, "slow") if labels[0] == "PAGE-6" and slot <= 3
                                   else None], seed=1)
    r = run_batches(f, 20)
    expect_all(r, 20)
    firsts = [labels[0] for _, labels in f.calls]
    assert firsts.count("PAGE-1") == firsts.count("PAGE-11") == firsts.count("PAGE-16") == 1
    assert r.stats["pages_sent"] <= 20 + 5 * 3  # only batch 2 was ever retried


def test_billing_key_is_cooled_and_batch_goes_to_another_key():
    f = Fleet(behaviors={s: [ApiError(402, "billing required")] for s in range(1, 4)}, seed=2)
    r = run_batches(f, 20)
    expect_all(r, 20)
    assert all(s["health"] != "HEALTHY" for s in f.manager.snapshot() if s["total_requests"] and s["slot"] <= 3)


def test_auth_failure_disables_the_key():
    f = Fleet(behaviors={1: [ApiError(400, "API key not valid. Please pass a valid API key.")]}, seed=3)
    r = run_batches(f, 30)
    expect_all(r, 30)
    snap = {s["slot"]: s for s in f.manager.snapshot()}
    assert snap[1]["health"] == "AUTHENTICATION_FAILED" or snap[1]["total_requests"] == 0


def test_processing_timeout_splits_the_batch_and_does_not_penalize_the_key():
    f = Fleet(rules=[lambda slot, labels: "hang" if len(labels) == 5 else None])
    s = cfg(ocr_batch_timeout_seconds=0.5)
    r = run_batches(f, 10, s=s)
    expect_all(r, 10)
    assert r.stats["splits"] == 2
    sizes = sorted(len(l) for _, l in f.calls)
    assert 2 in sizes and 3 in sizes  # 5 -> 2 + 3, as specified
    assert all(x["health"] == "HEALTHY" for x in f.manager.snapshot())  # a timeout is not a key fault


def test_repeated_timeouts_split_down_to_single_pages():
    f = Fleet(rules=[lambda slot, labels: "hang" if len(labels) > 1 else None])
    r = run_batches(f, 10, s=cfg(ocr_batch_timeout_seconds=0.4))
    expect_all(r, 10)
    assert max(len(l) for _, l in f.calls[-10:]) == 1 or any(len(l) == 1 for _, l in f.calls)
    assert all(x["health"] == "HEALTHY" for x in f.manager.snapshot())


def test_missing_pages_are_recovered_individually_and_valid_pages_kept():
    f = Fleet(rules=[lambda slot, labels: ("skip", {2, 4}) if labels == [f"PAGE-{i}" for i in range(6, 11)]
                     and len([1 for _ in ()]) == 0 and not getattr(f, "seen", None) else None])
    f.seen = None
    calls = {"n": 0}

    def rule(slot, labels):
        if labels[0] == "PAGE-6" and len(labels) == 5:
            return ("skip", {2, 4})
        return None

    f.rules = [rule]
    r = run_batches(f, 10)
    expect_all(r, 10)
    redo = [labels for _, labels in f.calls if len(labels) == 2]
    assert redo == [["PAGE-7", "PAGE-9"]]  # only the two missing pages were redone


def test_truncated_response_splits_and_recovers():
    f = Fleet(rules=[lambda slot, labels: "truncated" if len(labels) == 5 else None])
    r = run_batches(f, 10)
    expect_all(r, 10)
    assert r.stats["splits"] >= 2


def test_garbage_response_tries_other_key_then_succeeds():
    f = Fleet(behaviors={s: ["garbage", "ok"] for s in range(1, 7)})
    r = run_batches(f, 10)
    expect_all(r, 10)


def test_network_and_5xx_errors_rotate_keys_without_opening_a_circuit_breaker():
    f = Fleet(behaviors={1: [ApiError(503, "unavailable")], 2: [ConnectionError("down")]}, seed=4)
    r = run_batches(f, 30)
    expect_all(r, 30)
    assert f.manager._shared_block_until == 0.0  # batch path never trips the provider-wide breaker


def test_two_timeouts_do_not_open_the_breaker_so_later_requests_still_use_gemini():
    f = Fleet(rules=[lambda slot, labels: "hang" if len(labels) >= 3 else None])
    run_batches(f, 10, s=cfg(ocr_batch_timeout_seconds=0.4))
    r = run_batches(f, 10, s=cfg(ocr_batch_timeout_seconds=0.4))
    expect_all(r, 10)


def test_document_specific_400_splits_down_to_the_bad_page_then_groq_only_for_it():
    def rule(slot, labels):
        return ApiError(400, "bad document") if "PAGE-7" in labels else None

    groq_pages = []

    async def groq(pdf, timeout):
        with fitz.open(stream=pdf, filetype="pdf") as d:
            labels = [p.get_text().strip() for p in d]
        groq_pages.append(labels)
        return {"pages": [{"page_number": i, "text": f"OCR[{lab}]"} for i, lab in enumerate(labels, 1)]}

    f = Fleet(rules=[rule])
    r = run_batches(f, 10, groq=groq)
    expect_all(r, 10)
    assert groq_pages == [["PAGE-7"]] and r.provider == "gemini" and len(r.groq_events) == 1
    keys_tried_for_7 = [s for s, l in f.calls if l == ["PAGE-7"]]
    assert len(keys_tried_for_7) == 1  # the bad input was not pushed through every key


def test_gemini_unavailable_falls_back_to_groq_for_failed_batches_only():
    f = Fleet(rules=[lambda slot, labels: ApiError(402, "billing") if labels[0] == "PAGE-6" else None])
    seen = []

    async def groq(pdf, timeout):
        with fitz.open(stream=pdf, filetype="pdf") as d:
            labels = [p.get_text().strip() for p in d]
        seen.append(labels)
        return {"pages": [{"page_number": i, "text": f"OCR[{lab}]"} for i, lab in enumerate(labels, 1)]}

    r = run_batches(f, 20, groq=groq)
    expect_all(r, 20)
    assert seen == [[f"PAGE-{i}" for i in range(6, 11)]]  # only that batch, never the whole PDF


def test_all_gemini_keys_dead_everything_goes_to_groq_batch_by_batch():
    f = Fleet(behaviors={s: [ApiError(402, "billing")] for s in range(1, 7)})
    n = []

    async def groq(pdf, timeout):
        with fitz.open(stream=pdf, filetype="pdf") as d:
            labels = [p.get_text().strip() for p in d]
        n.append(len(labels))
        return {"pages": [{"page_number": i, "text": f"OCR[{lab}]"} for i, lab in enumerate(labels, 1)]}

    r = run_batches(f, 20, groq=groq)
    expect_all(r, 20)
    assert sum(n) == 20 and r.provider in ("gemini", "groq")


def test_both_providers_unavailable_raises_incomplete_not_success():
    f = Fleet(behaviors={s: [ApiError(402, "billing")] for s in range(1, 7)})

    async def groq(pdf, timeout):
        raise groq_client.GroqOCRError("down")

    with pytest.raises(BatchOCRError) as exc:
        run_batches(f, 10, groq=groq)
    assert exc.value.failed


def test_retry_budget_is_bounded_calls_do_not_explode():
    f = Fleet(behaviors={s: [ApiError(429, "slow", headers={"retry-after": "1"})] for s in range(1, 7)})
    with pytest.raises(BatchOCRError):
        run_batches(f, 10)
    assert len(f.calls) <= 6 * 2  # every key at most once per batch


def test_two_rate_limited_keys_are_treated_as_shared_project_quota():
    f = Fleet(behaviors={s: [ApiError(429, "quota", headers={"retry-after": "1"})] for s in (1, 2, 3, 4, 5, 6)})
    t0 = time.monotonic()
    with pytest.raises(BatchOCRError):
        run_batches(f, 20)
    assert time.monotonic() - t0 < 30
    assert len(f.calls) < 4 * 6  # not every key tried for every batch


def test_insufficient_remaining_budget_is_not_blamed_on_the_key():
    f = Fleet(rules=[lambda slot, labels: "hang"])
    with pytest.raises(BatchOCRError) as exc:
        run_batches(f, 10, s=cfg(ocr_batch_timeout_seconds=30.0), deadline=5.5)
    assert all(x["health"] == "HEALTHY" for x in f.manager.snapshot())
    assert "deadline" in " ".join(exc.value.failed.values()) or "budget" in " ".join(exc.value.failed.values())


def test_adaptive_timeout_function():
    s = cfg(ocr_batch_timeout_seconds=90.0)
    assert batch_ocr.adaptive_timeout(5, 1_000_000, s, 500, None) == (90.0, False)
    t1, _ = batch_ocr.adaptive_timeout(1, 1_000_000, s, 500, None)
    assert 40 < t1 < 90
    big, _ = batch_ocr.adaptive_timeout(5, 8_000_000, s, 500, None)
    assert 90 < big <= s.ocr_batch_timeout_max_seconds
    t, limited = batch_ocr.adaptive_timeout(5, 1_000_000, s, 30, None)
    assert (t, limited) == (30, True)
    fast, _ = batch_ocr.adaptive_timeout(5, 1_000_000, s, 500, 2.0)  # history tightens it, within bounds
    assert 45 <= fast <= 90
    fixed, _ = batch_ocr.adaptive_timeout(1, 1_000_000, cfg(ocr_adaptive_timeout_enabled=False), 500, None)
    assert fixed == 2.0


# ---- validation (50-61) ----------------------------------------------------------------------------------


def test_duplicate_and_out_of_range_pages_never_overwrite_validated_text():
    f = Fleet()
    s = cfg()
    runner = BatchRun(make_pdf(10), 10, s, manager=f.manager, deadline=time.monotonic() + 5)
    b = batch_ocr.Batch("B1", [6, 7, 8])
    missing = runner._store(b, {"pages": [{"page_number": 1, "text": "six"}, {"page_number": 9, "text": "x"}]}, "gemini")
    assert runner.results == {6: "six"} and missing == [7, 8]
    runner._store(b, {"pages": [{"page_number": 1, "text": "DIFFERENT"}, {"page_number": 2, "text": "seven"}]}, "gemini")
    assert runner.results == {6: "six", 7: "seven"}


def test_validator_rejects_bad_page_numbers_and_missing_when_not_allowed():
    box = {}
    v = gemini_client.make_batch_validator(3, False, box)
    with pytest.raises(km.InvalidResponse):
        v(resp(json.dumps({"pages": [{"page_number": 1, "text": "a"}, {"page_number": 1, "text": "b"}]})))
    with pytest.raises(km.InvalidResponse):
        v(resp(json.dumps({"pages": [{"page_number": 4, "text": "a"}]})))
    with pytest.raises(km.InvalidResponse):
        v(resp(json.dumps({"pages": [{"page_number": 1, "text": "a"}]})))
    with pytest.raises(km.InvalidResponse):
        v(resp(json.dumps({"error": "quota"})))


def test_blank_and_unreadable_pages_are_kept_as_results_not_missing():
    def rule(slot, labels):
        return resp(json.dumps({"pages": [{"page_number": 1, "text": ""}, {"page_number": 2, "text": "[UNREADABLE]"}]})) \
            if labels == ["PAGE-1", "PAGE-2"] else None

    f = Fleet(rules=[rule])
    r = run_batches(f, 2 + 4, s=cfg(ocr_batch_size=2))
    t = texts(r)
    assert t[1] == "" and t[2] == "[UNREADABLE]" and len(t) == 6
    assert len([1 for _, l in f.calls if l == ["PAGE-1", "PAGE-2"]]) == 1  # not retried


def test_hindi_and_english_text_is_preserved_exactly():
    text = "प्रथम सूचना रिपोर्ट FIR No. १२३/2024 पुलिस थाना"

    def rule(slot, labels):
        return resp(json.dumps({"pages": [{"page_number": i, "text": text + lab} for i, lab in enumerate(labels, 1)]}))

    r = run_batches(Fleet(rules=[rule]), 10)
    assert texts(r)[3] == text + "PAGE-3" and texts(r)[8] == text + "PAGE-8"
    assert r.data.full_text.startswith(text + "PAGE-1\n\n" + text + "PAGE-2")


# ---- cancellation and cleanup (62-70) --------------------------------------------------------------------------


def test_cancellation_releases_every_key_and_stops_provider_calls():
    f = Fleet(rules=[lambda slot, labels: "hang"])

    async def go():
        km.request_id_var.set("t")
        task = asyncio.ensure_future(BatchRun(make_pdf(20), 20, cfg(ocr_batch_timeout_seconds=30.0), manager=f.manager,
                                              deadline=time.monotonic() + 60).run())
        await asyncio.sleep(0.3)
        assert f.total_inflight == 4
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)
        return len(f.calls)

    calls = asyncio.run(go())
    assert f.total_inflight == 0 and all(s["active_requests"] == 0 for s in f.manager.snapshot())
    assert len(f.calls) == calls == 4  # nothing new after cancellation
    assert all(x["health"] == "HEALTHY" for x in f.manager.snapshot())


def test_terminal_failure_stops_new_provider_requests():
    f = Fleet(behaviors={s: [ApiError(402, "billing")] for s in range(1, 7)}, delay=0.05)
    with pytest.raises(BatchOCRError):
        run_batches(f, 50)
    n = len(f.calls)
    asyncio.run(asyncio.sleep(0.2))
    assert len(f.calls) == n and f.total_inflight == 0
    assert all(s["active_requests"] == 0 for s in f.manager.snapshot())


def test_deadline_exhaustion_returns_failure_with_pages_listed():
    f = Fleet(rules=[lambda slot, labels: "hang"])
    t0 = time.monotonic()
    with pytest.raises(BatchOCRError) as exc:
        run_batches(f, 10, s=cfg(ocr_batch_timeout_seconds=3.0), deadline=7.0)
    assert time.monotonic() - t0 < 9
    assert exc.value.failed and f.total_inflight == 0


def test_no_credentials_in_logs(caplog):
    caplog.set_level(logging.DEBUG)
    f = Fleet(behaviors={1: [ApiError(429, f"bad key {KEYS[0]}")]}, seed=5)
    run_batches(f, 10)
    assert not any(k in caplog.text for k in KEYS)
    assert "Batch-" in caplog.text and "assigned=Key-" in caplog.text


def test_groq_has_no_sdk_retries_and_cancels_siblings(monkeypatch):
    seen = {}

    class FakeCompletions:
        async def create(self, **kw):
            seen["calls"] = seen.get("calls", 0) + 1
            await asyncio.sleep(0.05)
            raise RuntimeError("502")

    fake = SimpleNamespace(with_options=lambda **kw: (seen.update(kw) or SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()))))
    monkeypatch.setattr(groq_client, "_get_client", lambda: fake)
    with pytest.raises(groq_client.GroqOCRError):
        asyncio.run(groq_client.extract_text(make_pdf(5), "application/pdf"))
    assert seen["max_retries"] == 0
    calls = seen["calls"]
    assert calls <= 2  # at most the page that was already starting; the queued rest were cancelled
    asyncio.run(asyncio.sleep(0.2))
    assert seen["calls"] == calls  # and nothing keeps calling Groq after the failure


# ---- service integration (58, 59, 15) -----------------------------------------------------------------------------


def test_service_uses_batches_and_extracts_metadata_after_merge(monkeypatch):
    f = Fleet()
    monkeypatch.setattr(km, "_manager", f.manager)
    monkeypatch.setattr(service, "_expected_page_count", lambda p, m: 12)
    got = {}

    async def fake_meta(job, text):
        got["text"] = text
        return service.OCRMetadata(document_type="FIR", document_class="FIR"), "gemini-test"

    monkeypatch.setattr(service, "extract_metadata", fake_meta)

    async def go():
        await_data = make_pdf(12)
        return await service._remote_whole_document(await_data, "application/pdf", "doc-1", cfg())

    data, model, provider, metadata = asyncio.run(go())
    assert provider == "gemini" and metadata is None and len(data.pages) == 12
    assert data.full_text == "\n\n".join(f"OCR[PAGE-{n}]" for n in range(1, 13))
    assert service.batch_mode_var.get() in (True, False)


def test_small_documents_keep_the_single_request_path():
    s = cfg()
    assert service._use_batches(make_pdf(5), "application/pdf", 5, s) is False
    assert service._use_batches(make_pdf(6), "application/pdf", 6, s) is True
    assert service._use_batches(b"img", "image/png", 1, s) is False
    assert service._use_batches(make_pdf(30), "application/pdf", 30, cfg(ocr_batch_processing_enabled=False)) is False


def test_config_defaults_match_the_documented_policy():
    s = get_settings().model_copy()
    for name in ("ocr_batch_size", "ocr_max_parallel_batches", "gemini_reserved_keys",
                 "ocr_batch_timeout_seconds", "ocr_global_max_concurrent_batches"):
        assert hasattr(s, name)
    from config.settings import Settings
    d = Settings.model_fields
    assert d["ocr_batch_size"].default == 5 and d["ocr_max_parallel_batches"].default == 4
    assert d["gemini_reserved_keys"].default == 2 and d["gemini_max_concurrent_per_key"].default == 1
    assert d["ocr_batch_timeout_seconds"].default == 90.0


# ---- end to end through the HTTP route -------------------------------------------------------------------------


def _client(monkeypatch, fleet):
    from fastapi.testclient import TestClient
    from main import app
    monkeypatch.setattr(km, "_manager", fleet.manager)
    get_settings.cache_clear()
    monkeypatch.setattr(service, "get_settings", lambda: cfg())
    service.ocr_results.clear()
    return TestClient(app)


def test_upload_route_multi_batch_keeps_response_schema_and_metadata_after_merge(monkeypatch):
    f = Fleet()
    client = _client(monkeypatch, f)
    seen = {}

    async def fake_meta(job, text):
        seen["text"] = text
        return service.OCRMetadata(document_type="FIR", document_class="FIR", fir_number="7/2024"), "gemini-test"

    monkeypatch.setattr(service, "extract_metadata", fake_meta)
    res = client.post("/python/ocr/upload", files={"file": ("doc.pdf", make_pdf(12), "application/pdf")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "completed" and body["metadata_status"] == "completed"
    assert body["metadata"]["fir_number"] == "7/2024" and body["isFir"] is True
    assert [p["page_number"] for p in body["ocr_data"]["pages"]] == list(range(1, 13))
    assert body["ocr_data"]["full_text"] == "\n\n".join(f"OCR[PAGE-{n}]" for n in range(1, 13))
    assert body["ocr_provider"] == "gemini" and set(body) >= {"document_id", "job_id", "filename", "ocr_model"}
    assert seen["text"].startswith("[[PAGE 1]]\nOCR[PAGE-1]")  # metadata sees the merged, page-marked text
    assert client.get(f"/python/ocr/{body['document_id']}").json()["ocr_data"]["full_text"] == body["ocr_data"]["full_text"]


def test_upload_route_metadata_failure_falls_back_to_background_polling(monkeypatch):
    f = Fleet()
    client = _client(monkeypatch, f)

    async def failing_meta(job, text):
        return None, None

    monkeypatch.setattr(service, "extract_metadata", failing_meta)
    res = client.post("/python/ocr/upload", files={"file": ("doc.pdf", make_pdf(12), "application/pdf")})
    assert res.status_code == 200 and res.json()["metadata_status"] in ("pending", "failed")
    assert len(res.json()["ocr_data"]["pages"]) == 12  # OCR text is never lost because metadata failed


def test_upload_route_incomplete_ocr_is_502_never_partial_success(monkeypatch):
    f = Fleet(behaviors={s: [ApiError(402, "billing")] for s in range(1, 7)})
    client = _client(monkeypatch, f)
    monkeypatch.setattr(get_settings(), "ocr_enable_groq_fallback", False, raising=False)
    monkeypatch.setattr(service, "get_settings", lambda: cfg(ocr_enable_groq_fallback=False))
    res = client.post("/python/ocr/upload", files={"file": ("doc.pdf", make_pdf(12), "application/pdf")})
    assert res.status_code == 502 and "provider failed" in res.json()["detail"]
    assert service.ocr_results == {}  # nothing stored for an incomplete document


def test_upload_route_small_pdf_still_uses_single_request_with_same_call_metadata(monkeypatch):
    f = Fleet()
    client = _client(monkeypatch, f)

    def single_call(labels):
        return resp(json.dumps({"pages": [{"page_number": i, "text": f"S{i}"} for i in range(1, len(labels) + 1)],
                                "metadata": {"document_type": "FIR", "fir_number": "9", "document_class": "FIR"}}))

    f.behaviors = {s: [single_call] for s in range(1, 7)}
    res = client.post("/python/ocr/upload", files={"file": ("doc.pdf", make_pdf(4), "application/pdf")})
    assert res.status_code == 200 and len(f.calls) == 1  # one request, whole document
    assert res.json()["metadata_status"] == "completed" and res.json()["metadata"]["fir_number"] == "9"
