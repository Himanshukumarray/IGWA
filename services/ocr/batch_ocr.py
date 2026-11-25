"""Parallel, batch-wise OCR with dynamic Gemini key allocation and batch-level recovery.

A multi-page PDF is cut into page batches (PyMuPDF `insert_pdf`: pages are copied unchanged, no
re-rendering, original order and orientation). A bounded set of workers takes batches from a queue;
every call reserves ONE Gemini key strictly (a key at its concurrency cap is never handed out) via
GeminiKeyManager.acquire(). A failed batch is handled on its own and nothing else is restarted:

    other key (billing/auth/429/5xx/invalid output) -> split in halves (timeout, truncation, rejected
    input) -> Groq (only that batch) -> terminal failure of the request.

Validated pages go into a registry keyed by ORIGINAL page number and are never overwritten or redone.
`full_text` is built in Python from the merged pages (schemas.ocr_data_from_dict -> build_full_text).

Concurrency limits (all process-local; several uvicorn workers each have their own copy, so the real
ceiling is `workers x limit`): per key (Policy.max_concurrent_per_key), process total
(Policy.global_max_concurrent) and per document (workers = allocation()). Requests are served roughly
FIFO: every document has at most `workers` waiters in the key queue, so a large document cannot starve
a small one. Progress is in memory only: a restart or a terminal failure loses it.
"""
import asyncio
import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import fitz

from config.settings import Settings
from services.ocr import gemini_client
from services.ocr import gemini_key_manager as km
from services.ocr.schemas import OCRData, ocr_data_from_dict

logger = logging.getLogger(__name__)

_MIN_USEFUL_SECONDS = 5.0  # a provider call with less time than this left is doomed: do not start it
_INVALID_CATEGORIES = {"empty_response", "invalid_json", "schema_validation", "error_response", "no_pages",
                       "bad_page_numbers", "missing_pages", "unexpected_finish_reason"}

GroqCall = Callable[[bytes, float], Awaitable[dict]]  # (sub-PDF, timeout) -> {"pages": [...]} (local numbers)


class BatchOCRError(Exception):
    """The document could not be fully OCRed. Never carries OCR text or credentials."""

    def __init__(self, message: str, failed: dict[str, str] | None = None):
        super().__init__(message)
        self.failed = failed or {}  # page range -> reason


class _Terminal(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class BatchResult:
    data: OCRData
    provider: str  # "gemini" unless every page came from Groq
    stats: dict[str, Any]


@dataclass
class Batch:
    id: str
    pages: list[int]  # ORIGINAL 1-based page numbers, in order
    depth: int = 0
    recovery: bool = False  # a retry/child: may use standby keys and is served first
    attempted: set[int] = field(default_factory=set)  # key slots already tried for THIS batch
    invalid_fails: int = 0
    shared_fails: int = 0
    timeouts: int = 0


def page_range(pages: list[int]) -> str:
    """[1,2,3,5] -> "1-3,5" (log label)."""
    out, start, prev = [], None, None
    for p in pages:
        if start is None:
            start = prev = p
        elif p == prev + 1:
            prev = p
        else:
            out.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = p
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else str(start))
    return ",".join(out)


# ---- planning -----------------------------------------------------------------------------------------


def effective_batch_size(page_count: int, nbytes: int, s: Settings) -> int:
    """Pages per batch. Scanned/high-resolution PDFs (large average page) get smaller batches because
    their OCR output and processing time per page are higher."""
    size = s.ocr_batch_size
    if page_count and nbytes / page_count > s.ocr_heavy_page_bytes:
        size = max(s.ocr_batch_min_size, -(-size * 3 // 5))  # ceil(0.6 x size): 5 -> 3
    return max(1, size)


def plan_batches(page_count: int, size: int) -> list[list[int]]:
    """Contiguous page groups of at most `size` pages with original page numbers."""
    return [list(range(i, min(i + size, page_count + 1))) for i in range(1, page_count + 1, size)]


def adaptive_timeout(n_pages: int, nbytes: int, s: Settings, remaining: float,
                     seconds_per_page: float | None) -> tuple[float, bool]:
    """(timeout, limited_by_budget). Scales the configured full-batch timeout by batch size and PDF size,
    tightens it with the latency measured earlier in this request, bounds it, and never exceeds the
    remaining overall budget (limited=True then: a timeout is the application's fault, not the key's)."""
    base = s.ocr_batch_timeout_seconds
    if s.ocr_adaptive_timeout_enabled:
        t = base * (0.4 + 0.6 * min(n_pages / s.ocr_batch_size, 1.0))
        t += 10.0 * max(0.0, nbytes / 1_000_000 - 2.0)
        if seconds_per_page is not None:
            t = min(t, max(4.0 * seconds_per_page * n_pages + 15.0, 0.5 * t))
        t = min(max(t, min(20.0, base)), s.ocr_batch_timeout_max_seconds)
    else:
        t = base
    return min(t, remaining), remaining < t


class PreparedPdf:
    """The uploaded PDF opened once; sub-PDFs are copied out of it (never modifies the upload)."""

    def __init__(self, data: bytes):
        self._doc = fitz.open(stream=data, filetype="pdf")
        self._lock = threading.Lock()
        self.page_count = self._doc.page_count

    def sub_pdf(self, pages: list[int]) -> bytes:
        with self._lock, fitz.open() as out:
            for number in pages:
                out.insert_pdf(self._doc, from_page=number - 1, to_page=number - 1)
            return out.tobytes()

    def close(self) -> None:
        with self._lock:
            self._doc.close()


_groq_sems: "weakref.WeakKeyDictionary[Any, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _groq_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _groq_sems.get(loop)
    if sem is None:
        sem = _groq_sems[loop] = asyncio.Semaphore(limit)
    return sem


# ---- the run -------------------------------------------------------------------------------------------


class BatchRun:
    def __init__(self, data: bytes, page_count: int, s: Settings, *, manager: "km.GeminiKeyManager",
                 deadline: float, groq_call: GroqCall | None = None,
                 on_groq_fallback: Callable[[], None] | None = None):
        self.data, self.page_count, self.s, self.mgr = data, page_count, s, manager
        self.deadline = deadline  # time.monotonic() value: no provider call starts after it
        self.groq_call, self.on_groq_fallback = groq_call, on_groq_fallback
        self.min_useful = min(_MIN_USEFUL_SECONDS, s.ocr_batch_timeout_seconds)
        self.rid = km.request_id_var.get()
        self.results: dict[int, str] = {}  # original page number -> text (never overwritten)
        self.page_provider: dict[int, str] = {}
        self.failed: dict[str, str] = {}
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.abort = asyncio.Event()
        self.alive = 0
        self.standby = 0
        self.shrink = 0
        self.throttle_until = 0.0
        self.rl_slots: set[int] = set()
        self.slow_slots: set[int] = set()
        self._timeouts_by_slot: dict[int, int] = {}
        self.project_throttled = False
        self.seconds_per_page: float | None = None
        self.prepared: PreparedPdf | None = None
        self._pdfs: dict[tuple[int, ...], bytes] = {}
        self._seq = 0
        self.stats: dict[str, Any] = {
            "gemini_calls": 0, "groq_calls": 0, "splits": 0, "pages_sent": 0, "queue_wait_seconds": 0.0,
            "failovers": 0, "batches_planned": 0, "workers": 0, "standby": 0,
        }

    # -- logging --
    def _log(self, level: int, msg: str, *args: Any) -> None:
        logger.log(level, "[OCR] request=%s " + msg, self.rid, *args)

    # -- queue --
    def _put(self, b: Batch) -> None:
        self._seq += 1
        self.queue.put_nowait((0 if b.recovery else 1, self._seq, b))  # recovery work is served first

    async def _pdf(self, pages: list[int]) -> bytes:
        key = tuple(pages)
        if key not in self._pdfs:
            assert self.prepared is not None
            self._pdfs[key] = await asyncio.to_thread(self.prepared.sub_pdf, pages)
        return self._pdfs[key]

    # -- entry point --
    async def run(self) -> BatchResult:
        started = time.perf_counter()
        s = self.s
        t_prep = time.perf_counter()
        try:
            self.prepared = await asyncio.to_thread(PreparedPdf, self.data)
        except Exception as exc:
            raise BatchOCRError("PDF could not be split into batches", {"all": type(exc).__name__}) from exc
        workers_task: list[asyncio.Task] = []
        try:
            groups = plan_batches(self.prepared.page_count,
                                  effective_batch_size(self.prepared.page_count, len(self.data), s))
            batches = [Batch(f"B{i}", g) for i, g in enumerate(groups, start=1)]
            self.stats["batches_planned"] = len(batches)
            healthy = self.mgr.healthy_count()
            active, self.standby = km.allocation(healthy, s.ocr_max_parallel_batches, s.gemini_reserved_keys)
            workers = max(1, min(active, len(batches)))
            if healthy == 0 and self.groq_call is None:
                raise BatchOCRError("No healthy Gemini key and no fallback provider", {"all": "no_healthy_keys"})
            self.stats.update(workers=workers, standby=self.standby)
            self._log(logging.INFO, "Total pages: %d Total batches: %d Healthy Gemini keys: %d "
                      "Active key capacity: %d Standby keys: %d preprocessing=%.2fs", self.prepared.page_count,
                      len(batches), healthy, workers, self.standby, time.perf_counter() - t_prep)
            for b in batches:
                self._put(b)
            self.alive = workers
            workers_task = [asyncio.ensure_future(self._worker()) for _ in range(workers)]
            await self._wait_done()
        finally:
            for t in workers_task:
                t.cancel()
            if workers_task:
                await asyncio.gather(*workers_task, return_exceptions=True)
            if self.prepared is not None:
                await asyncio.to_thread(self.prepared.close)
            self._pdfs.clear()  # temporary sub-PDFs: released on success, failure and cancellation
        return self._finish(started)

    async def _wait_done(self) -> None:
        join = asyncio.ensure_future(self.queue.join())
        stop = asyncio.ensure_future(self.abort.wait())
        try:
            done, _ = await asyncio.wait({join, stop}, timeout=max(self.deadline - time.monotonic(), 0.0),
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done and not self.abort.is_set():
                missing = [p for p in range(1, self.page_count + 1) if p not in self.results]
                self.failed[page_range(missing)] = "deadline_exhausted"
        finally:
            for t in (join, stop):
                t.cancel()
            await asyncio.gather(join, stop, return_exceptions=True)

    def _finish(self, started: float) -> BatchResult:
        missing = [p for p in range(1, self.page_count + 1) if p not in self.results]
        if missing or self.failed:
            reasons = dict(self.failed)
            if missing and not reasons:
                reasons[page_range(missing)] = "missing"
            self._log(logging.ERROR, "Completed pages: %d/%d Missing pages: %s Status: FAILED reasons=%s",
                      len(self.results), self.page_count, page_range(missing) or "-", reasons)
            raise BatchOCRError("OCR incomplete", reasons)
        data = ocr_data_from_dict({"pages": [{"page_number": n, "text": self.results[n]}
                                             for n in range(1, self.page_count + 1)]})
        providers = set(self.page_provider.values())
        provider = "groq" if providers == {"groq"} else "gemini"
        self.stats["seconds"] = round(time.perf_counter() - started, 3)
        self.stats["providers"] = sorted(providers)
        self._log(logging.INFO, "Completed pages: %d/%d Missing pages: 0 Duplicate pages: 0 Page merge "
                  "successful duration=%.2fs gemini_calls=%d groq_calls=%d splits=%d", len(self.results),
                  self.page_count, self.stats["seconds"], self.stats["gemini_calls"], self.stats["groq_calls"],
                  self.stats["splits"])
        return BatchResult(data, provider, self.stats)

    # -- workers --
    async def _worker(self) -> None:
        while True:
            if self.shrink > 0 and self.alive > 1:  # project-level rate limit: fewer concurrent batches
                self.shrink -= 1
                self.alive -= 1
                return
            _, _, b = await self.queue.get()
            try:
                if not self.abort.is_set():
                    await self._run_item(b)
            except _Terminal as exc:
                self._terminal(b, exc.reason)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a bug must fail this request cleanly, never hang it
                self._log(logging.ERROR, "Batch-%s internal error: %s", b.id, type(exc).__name__)
                self._terminal(b, "internal_error")
            finally:
                self.queue.task_done()

    def _terminal(self, b: Batch, reason: str) -> None:
        self.failed[page_range(b.pages)] = reason
        self._log(logging.ERROR, "Batch-%s pages=%s FAILED terminally reason=%s (no further provider calls)",
                  b.id, page_range(b.pages), reason)
        self.abort.set()  # no new requests; running ones are cancelled by run()

    # -- one batch --
    async def _run_item(self, b: Batch) -> None:
        s = self.s
        mgr = self.mgr
        label = f"Batch-{b.id} pages={page_range(b.pages)}"
        while True:
            if self.abort.is_set():
                return
            if self.deadline - time.monotonic() < self.min_useful:
                raise _Terminal("deadline_exhausted")
            await self._respect_throttle()
            waited = time.monotonic()
            # Keys that timed out repeatedly in THIS request are tried last (a request-local preference, not a
            # health penalty: a timeout never disables or cools down a key).
            slot = await mgr.acquire(exclude=b.attempted | self.slow_slots, deadline=self.deadline)
            if slot is None and self.slow_slots - b.attempted:
                slot = await mgr.acquire(exclude=b.attempted, deadline=self.deadline)
            self.stats["queue_wait_seconds"] += time.monotonic() - waited
            if slot is None:
                break  # no eligible key left for this batch: Groq
            err: km.AttemptError | None = None
            parsed: dict[str, Any] | None = None
            call_started = time.monotonic()
            try:
                pdf = await self._pdf(b.pages)
                timeout, limited = adaptive_timeout(len(b.pages), len(pdf), s, self.deadline - time.monotonic(),
                                                    self.seconds_per_page)
                if timeout < self.min_useful:
                    raise _Terminal("insufficient_time_budget")
                self._log(logging.INFO, "%s assigned=%s timeout=%.0fs%s", label, slot.label, timeout,
                          " (budget-limited)" if limited else "")
                box: dict[str, Any] = {}
                validate = (gemini_client.make_batch_validator(len(b.pages), s.ocr_retry_failed_pages, box)
                            if s.ocr_validate_response else None)
                self.stats["gemini_calls"] += 1
                self.stats["pages_sent"] += len(b.pages)
                try:
                    response = await mgr.attempt(slot, gemini_client.build_batch_call(pdf, timeout),
                                                 timeout=timeout, validate=validate, budget_limited=limited,
                                                 purpose="ocr-batch")
                    parsed = box.get("parsed") or gemini_client.parse_gemini_json(response.text)
                except km.AttemptError as exc:
                    err = exc
                except gemini_client.GeminiOCRError:
                    err = km.AttemptError("invalid_json", "next_key", slot=slot.number)
            finally:
                mgr.release(slot)  # success, failure and cancellation all give the key back
            duration = time.monotonic() - call_started
            if err is None and parsed is not None:
                try:
                    missing = self._store(b, parsed, "gemini")
                except Exception:  # schema surprises after validation: treat as an unusable response
                    err = km.AttemptError("schema_validation", "next_key", slot=slot.number)
                else:
                    rate = duration / max(len(b.pages), 1)
                    self.seconds_per_page = rate if self.seconds_per_page is None else 0.5 * (self.seconds_per_page + rate)
                    self._pdfs.pop(tuple(b.pages), None)
                    self._log(logging.INFO, "%s SUCCESS duration=%.2fs (%d/%d pages)", label, duration,
                              len(b.pages) - len(missing), len(b.pages))
                    if missing:  # keep what arrived, redo only the pages that did not
                        self._log(logging.WARNING, "%s missing pages=%s: recovering only those", label,
                                  page_range(missing))
                        self._put(Batch(f"{b.id}r", missing, b.depth + 1, recovery=True))
                    return
            assert err is not None
            b.attempted.add(slot.number)
            if err.category == "timeout":
                n = self._timeouts_by_slot[slot.number] = self._timeouts_by_slot.get(slot.number, 0) + 1
                if n >= 2:
                    self.slow_slots.add(slot.number)
            self._log(logging.WARNING, "%s FAILED key=%s reason=%s action=%s duration=%.2fs", label, slot.label,
                      err.category, err.action, duration)
            if err.action == "budget":
                raise _Terminal("insufficient_time_budget")
            if err.action in ("split", "request"):
                if self._split(b, err.category):
                    return
                if err.action == "request":
                    break  # one page the provider rejects: another key would reject it too
                b.timeouts += 1
                if b.timeouts >= 3:  # one page that keeps timing out on different keys: stop wasting time
                    break
                self.stats["failovers"] += 1
                continue
            # next_key
            if err.rate_limited:
                self._note_rate_limit(err)
                if self.project_throttled:
                    break  # shared project quota: do not rotate through the remaining keys to dodge the limit
            if err.shared:
                b.shared_fails += 1
                if b.shared_fails >= max(1, s.gemini_shared_failover_max_keys):
                    break
            elif err.category in _INVALID_CATEGORIES:
                b.invalid_fails += 1
                if b.invalid_fails >= max(1, s.gemini_max_invalid_response_attempts):
                    if self._split(b, err.category):
                        return
                    break
            self.stats["failovers"] += 1
            self._log(logging.INFO, "%s switching to another eligible key", label)
        await self._groq(b, label)

    def _split(self, b: Batch, reason: str) -> bool:
        s = self.s
        if not s.ocr_auto_split_failed_batches or len(b.pages) < 2 * max(1, s.ocr_batch_min_size):
            return False
        mid = len(b.pages) // 2
        self.stats["splits"] += 1
        self._log(logging.WARNING, "Batch-%s pages=%s splitting (%s) into %s + %s", b.id, page_range(b.pages),
                  reason, page_range(b.pages[:mid]), page_range(b.pages[mid:]))
        self._put(Batch(f"{b.id}.1", b.pages[:mid], b.depth + 1, recovery=True))
        self._put(Batch(f"{b.id}.2", b.pages[mid:], b.depth + 1, recovery=True))
        return True

    # -- shared-quota protection --
    def _note_rate_limit(self, err: km.AttemptError) -> None:
        """Several Gemini keys may share ONE project quota. Two different keys rate limited in one
        request = assume project level: stop rotating blindly, pause, and halve this document's
        concurrency instead of burning the remaining keys."""
        if err.slot is not None:
            self.rl_slots.add(err.slot)
        if len(self.rl_slots) < 2:
            return
        self.throttle_until = max(self.throttle_until, time.monotonic() + min(err.retry_after or 5.0, 15.0))
        if not self.project_throttled:
            self.project_throttled = True
            self.shrink = max(self.shrink, self.alive // 2)
            self._log(logging.WARNING, "Several keys rate limited: assuming a shared project quota; "
                      "pausing and reducing concurrency")

    async def _respect_throttle(self) -> None:
        wait = self.throttle_until - time.monotonic()
        if 0 < wait < self.deadline - time.monotonic() - self.min_useful:
            await asyncio.sleep(wait)

    # -- results --
    def _store(self, b: Batch, parsed: dict[str, Any], provider: str) -> list[int]:
        """Map batch-local page numbers to original ones and register NEW pages only. Returns the
        batch pages that are still missing."""
        for page in ocr_data_from_dict(parsed).pages:
            local = page.page_number
            if not 1 <= local <= len(b.pages):
                self._log(logging.WARNING, "Batch-%s ignored out-of-range page %d", b.id, local)
                continue
            original = b.pages[local - 1]
            if original in self.results:
                self._log(logging.WARNING, "Batch-%s duplicate page %d ignored (validated result kept)",
                          b.id, original)
                continue
            self.results[original] = page.text
            self.page_provider[original] = provider
        return [p for p in b.pages if p not in self.results]

    # -- Groq (batch level) --
    async def _groq(self, b: Batch, label: str) -> None:
        if self.groq_call is None:
            raise _Terminal("gemini_unavailable")
        remaining = self.deadline - time.monotonic()
        if remaining < self.min_useful:
            raise _Terminal("deadline_exhausted")
        if self.on_groq_fallback:
            self.on_groq_fallback()
        self._log(logging.WARNING, "%s Gemini recovery exhausted: Groq fallback for this batch only", label)
        started = time.monotonic()
        try:
            async with _groq_semaphore(self.s.ocr_groq_max_concurrent):
                pdf = await self._pdf(b.pages)
                self.stats["groq_calls"] += 1
                raw = await asyncio.wait_for(self.groq_call(pdf, remaining), remaining)
            data = ocr_data_from_dict(raw)
            if [p.page_number for p in data.pages] != list(range(1, len(b.pages) + 1)):
                raise ValueError("incomplete Groq batch")
            self._store(b, raw, "groq")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            category = getattr(exc, "category", None) or type(exc).__name__
            raise _Terminal(f"gemini_exhausted_groq_{category}") from exc
        self._log(logging.INFO, "%s SUCCESS via groq duration=%.2fs", label, time.monotonic() - started)
