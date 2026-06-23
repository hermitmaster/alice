"""Narrative summarizer — feeds Alice's recent interactions to Claude
and streams back a human-readable narrative.

Uses the same OAuth path the hemispheres use (CLAUDE_CODE_OAUTH_TOKEN from
alice.env). The Agent SDK is invoked without tools — this is pure
summarization, not agentic work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass, replace
from typing import AsyncIterator

from . import aggregators, bucket_cache, sources
from .settings import Paths


DEFAULT_MODEL = os.environ.get("ALICE_VIEWER_NARRATIVE_MODEL", "")
BUCKET_MODEL = os.environ.get("ALICE_VIEWER_BUCKET_MODEL", "")
# Wall-clock cap on the streamed merge step. Covers the entire async
# iteration over the SDK's stream — if Claude (or whatever's proxying
# it) hangs without yielding, we surface an SSE ``error`` event
# instead of letting the connection silently die. Bumped from 90s →
# 10 min after #135: a healthy provider always finishes inside this,
# and the old 90s ceiling kept firing on cold/slow runs even though
# the streamed merge was still making forward progress.
DEFAULT_MAX_SECONDS = 600
# Per-bucket cap on the one-shot Haiku summarization call. One slow
# call (provider 5xx, network stall) can't stall the whole fill — on
# timeout the bucket is recorded with an empty summary so the merge
# step skips it, and the rest of the fill continues. 60s is well
# above a healthy Haiku response time for a 200-event prompt.
BUCKET_MAX_SECONDS = 60
# Final-merge cache TTL. Was 5 min when the cache was in-memory only;
# bumped to 7d now that it's disk-persisted (mirrors bucket_cache).
# Content-hash invalidation already handles "new events landed in the
# window" correctly; TTL is just a safety net for cache-dir bloat.
CACHE_TTL_SECONDS = 7 * 86400


@dataclass
class NarrativeRequest:
    window_seconds: int
    max_events: int = 500


def _ensure_auth():
    """Resolve auth from env + alice.env into os.environ for the SDK subprocess.

    Returns the resolved ``AuthEnv``; ``mode == "none"`` means no
    credentials were found and the caller should surface an error.
    """
    from core.config.auth import ensure_auth_env

    return ensure_auth_env()


def _default_mind() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("ALICE_MIND") or pathlib.Path.home() / "alice-mind"
    )


def _load_viewer_backend(*, model_override: str = ""):
    from core.config.model import load as load_model_config

    backend = load_model_config(_default_mind()).viewer
    if model_override:
        backend = replace(backend, model=model_override)
    return backend


def _backend_with_model(backend: object, model_override: str):
    if model_override:
        return replace(backend, model=model_override)
    return backend


async def _run_kernel_stream(
    prompt: str,
    *,
    backend: object,
    max_seconds: int,
) -> AsyncIterator[dict]:
    """Run one tool-less viewer call through the configured kernel."""
    from core.events import CapturingEmitter
    from core.kernel import KernelSpec, NullHandler, make_kernel

    if not getattr(backend, "model", ""):
        yield {
            "type": "error",
            "message": "viewer narrative model is not configured; set viewer.model in model.yml or ALICE_VIEWER_NARRATIVE_MODEL",
        }
        return

    queue: asyncio.Queue[dict] = asyncio.Queue()

    class _TextQueueHandler(NullHandler):
        async def on_text(self, text: str) -> None:
            await queue.put({"type": "chunk", "text": text})

    emitter = CapturingEmitter()
    kernel = make_kernel(
        backend,
        emitter,
        correlation_id=f"viewer-narrative-{int(time.time())}",
        silent=True,
    )
    spec = KernelSpec(
        model=getattr(backend, "model", ""),
        allowed_tools=[],
        cwd=pathlib.Path("/tmp"),
        max_seconds=max_seconds,
    )

    async def _run():
        return await kernel.run(prompt, spec, handlers=[_TextQueueHandler()])

    task = asyncio.create_task(_run())
    while True:
        if task.done() and queue.empty():
            break
        try:
            yield await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue

    try:
        result = await task
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return

    yield {
        "type": "result",
        "duration_ms": result.duration_ms,
        "cost_usd": result.cost_usd,
        "session_id": result.session_id,
    }
    if result.error == "timeout":
        yield {"type": "error", "message": f"timed out after {max_seconds}s"}
        return
    if result.is_error:
        yield {"type": "error", "message": result.text or "kernel returned is_error"}
        return
    yield {"type": "done"}


def build_digest(paths: Paths, window_seconds: int, max_events: int) -> dict:
    """Pull raw events + artifacts, filter to window, return a compact digest."""
    now_ts = time.time()
    cutoff = now_ts - window_seconds
    all_events = sources.load_all(paths)

    in_window = [e for e in all_events if e.ts >= cutoff]
    wakes = [w for w in aggregators.group_wakes(all_events) if w.start_ts >= cutoff]
    turns = [t for t in aggregators.group_turns(all_events) if t.start_ts >= cutoff]

    surfaces = [
        e for e in in_window if e.kind in ("surface_pending", "surface_resolved")
    ]
    emergencies = [
        e for e in in_window if e.kind in ("emergency_pending", "emergency_resolved")
    ]
    notes = [e for e in in_window if e.kind in ("note_pending", "note_consumed")]
    thoughts = [e for e in in_window if e.kind == "thought_written"]

    # Tool histogram (tight activity fingerprint).
    tools: dict[str, int] = {}
    for e in in_window:
        if e.kind == "tool_use":
            name = e.detail.get("name") or "?"
            tools[name] = tools.get(name, 0) + 1

    return {
        "window_seconds": window_seconds,
        "generated_at": now_ts,
        "event_count": len(in_window),
        "wakes": [_wake_summary(w) for w in wakes[-30:]],
        "turns": [_turn_summary(t) for t in turns[-30:]],
        "surfaces": [_artifact_summary(e) for e in surfaces[-20:]],
        "emergencies": [_artifact_summary(e) for e in emergencies[-20:]],
        "notes": [_artifact_summary(e) for e in notes[-20:]],
        "thoughts": [_artifact_summary(e) for e in thoughts[-20:]],
        "tools": tools,
        "directive": sources.read_directive(paths.inner)[:1200],
    }


def _wake_summary(w) -> dict:
    return {
        "wake_id": w.wake_id,
        "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(w.start_ts)),
        "status": w.status,
        "duration_s": (w.duration_ms or 0) / 1000.0,
        "tools": w.tools,
        "events": len(w.events),
        "cost_usd": w.total_cost_usd,
    }


def _turn_summary(t) -> dict:
    return {
        "turn_id": t.turn_id,
        "kind": t.kind,
        "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.start_ts)),
        "sender": t.sender_name,
        "surface_id": t.surface_id,
        "emergency_id": t.emergency_id,
        "inbound": _trim(t.inbound or "", 280),
        "outbound": _trim(t.outbound or "", 280),
        "error": t.error,
        "duration_s": (t.duration_ms or 0) / 1000.0,
        "tools": t.tools,
    }


def _artifact_summary(e) -> dict:
    d = e.detail or {}
    return {
        "id": e.correlation_id,
        "kind": e.kind,
        "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.ts)),
        "body": _trim(d.get("body") or "", 400),
        "trailer": d.get("trailer"),
    }


def _trim(s: str, cap: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= cap else s[: cap - 1] + "…"


def render_prompt(digest: dict, window_label: str) -> str:
    """Build the prompt Claude will narrate over.

    Body lives in ``prompts/templates/viewer/narrative.window.md.j2``
    (Plan 04 Phase 4 of the runtime refactor).
    """
    from prompts import load as load_prompt

    return load_prompt(
        "viewer.narrative.window",
        digest_json=json.dumps(digest, ensure_ascii=False, indent=2, default=str),
        window_label=window_label,
    )


# ---------------------------------------------------------------------------
# Cache — avoids re-calling Claude on page refresh.


_CACHE: dict[str, tuple[float, str]] = {}


def cache_key(digest: dict) -> str:
    # Hash the digest content so any change invalidates the cache.
    blob = json.dumps(digest, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _merge_cache_dir() -> pathlib.Path:
    """Disk location for final-merge narrative cache.

    Sibling of the bucket cache under ``$ALICE_VIEWER_CACHE_DIR``
    (default ``~/.local/state/alice/viewer-cache/merges/``). The
    bind-mounted state dir survives viewer container recreates so a
    cached narrative outlives every deploy.
    """
    override = os.environ.get("ALICE_VIEWER_CACHE_DIR")
    if override:
        return pathlib.Path(override) / "merges"
    return pathlib.Path.home() / ".local/state/alice/viewer-cache/merges"


def _merge_cache_path(key: str) -> pathlib.Path:
    safe = key.replace("/", "_").replace("..", "__")
    return _merge_cache_dir() / f"{safe}.json"


def cache_get(key: str) -> str | None:
    """Return the cached merged narrative for ``key``, or None.

    Reads from the disk cache (survives viewer recreate). Falls back
    to the legacy in-memory cache for backwards compat with anything
    that wrote there pre-deploy.
    """
    path = _merge_cache_path(key)
    if path.is_file():
        try:
            raw = json.loads(path.read_text())
            if time.time() - (raw.get("saved_at") or 0) <= CACHE_TTL_SECONDS:
                return raw.get("text")
        except (OSError, json.JSONDecodeError):
            pass
    # Legacy in-memory fallback (still served until process restarts).
    entry = _CACHE.get(key)
    if entry is None:
        return None
    saved_at, text = entry
    if time.time() - saved_at > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return text


def cache_put(key: str, text: str) -> None:
    """Persist the merged narrative to disk + in-memory cache.

    Disk write is atomic (tmp + rename) to avoid serving a half-
    written file on concurrent reads. In-memory cache is kept
    populated as a fast path for hot keys within the same process.
    """
    saved_at = time.time()
    _CACHE[key] = (saved_at, text)
    path = _merge_cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"saved_at": saved_at, "text": text}, ensure_ascii=False)
        )
        tmp.replace(path)
    except OSError:
        # Disk failure shouldn't break the streaming response — the
        # caller already has the text and the in-memory cache is set.
        pass


# ---------------------------------------------------------------------------
# Streaming via Agent SDK


async def stream_narrative(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    backend: object | None = None,
    max_seconds: int = DEFAULT_MAX_SECONDS,
) -> AsyncIterator[dict]:
    """Yield events: {"type": "chunk", "text": "..."} and finally {"type": "done"}
    or {"type": "error", "message": "..."}."""
    auth = _ensure_auth()
    if auth.mode == "none":
        yield {
            "type": "error",
            "message": "no Claude credentials in env or alice.env (set CLAUDE_CODE_OAUTH_TOKEN, or ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY)",
        }
        return
    resolved_backend = (
        _backend_with_model(backend, model) if backend is not None
        else _load_viewer_backend(model_override=model)
    )
    async for ev in _run_kernel_stream(
        prompt, backend=resolved_backend, max_seconds=max_seconds
    ):
        yield ev


WINDOW_PRESETS = {
    "1h": (3600, "hour"),
    "6h": (6 * 3600, "6 hours"),
    "24h": (24 * 3600, "24 hours"),
    "7d": (7 * 86400, "week"),
    "30d": (30 * 86400, "month"),
}


def window_from_label(label: str) -> tuple[int, str]:
    return WINDOW_PRESETS.get(label, WINDOW_PRESETS["24h"])


# ---------------------------------------------------------------------------
# Bucketed cache — each bucket is summarized once, cached to disk for 7 days,
# merged on demand. Overlapping windows (e.g. "last 24h" vs. "last 25h") share
# 95%+ of their buckets so subsequent queries are near-instant.

# bucket_seconds chosen so any window has ~6-30 buckets (a tractable merge input).
WINDOW_BUCKET_SECONDS = {
    "1h": 600,  # 10-min buckets → 6
    "6h": 1800,  # 30-min buckets → 12
    "24h": 3600,  # 1-hour buckets → 24
    "7d": 6 * 3600,  # 6-hour buckets → 28
    "30d": 86400,  # 1-day buckets → 30
}

MAX_CONCURRENT_BUCKET_GENERATIONS = 4


def bucket_seconds_for(window_label: str) -> int:
    return WINDOW_BUCKET_SECONDS.get(window_label, WINDOW_BUCKET_SECONDS["24h"])


def align_down(ts: float, step: int) -> int:
    return (int(ts) // step) * step


@dataclass
class BucketSlot:
    start: int
    end: int
    events: list  # list of UnifiedEvent
    content_hash: str

    def is_open(self, now_ts: float) -> bool:
        # A bucket is "open" if now is inside its range — its contents may still
        # grow, so cache lifetime should be treated as volatile for this one.
        return self.start <= now_ts < self.end


def build_buckets(
    paths: Paths, window_seconds: int, window_label: str, now_ts: float | None = None
) -> list[BucketSlot]:
    now_ts = now_ts or time.time()
    bucket_seconds = bucket_seconds_for(window_label)
    end = (
        align_down(now_ts, bucket_seconds) + bucket_seconds
    )  # include the current open bucket
    start = align_down(now_ts - window_seconds, bucket_seconds)

    all_events = sources.load_all(paths)
    events_in_window = [e for e in all_events if start <= e.ts < end]

    # Partition by bucket index.
    slots: dict[int, list] = {}
    bstart = start
    while bstart < end:
        slots[bstart] = []
        bstart += bucket_seconds
    for ev in events_in_window:
        idx = align_down(ev.ts, bucket_seconds)
        if idx in slots:
            slots[idx].append(ev)

    out: list[BucketSlot] = []
    for bs, evs in sorted(slots.items()):
        out.append(
            BucketSlot(
                start=bs,
                end=bs + bucket_seconds,
                events=evs,
                content_hash=_hash_events(evs),
            )
        )
    return out


def _hash_events(evs: list) -> str:
    """Stable hash over (ts, kind, summary) of events in the bucket."""
    h = hashlib.sha256()
    for e in evs:
        h.update(f"{e.ts:.3f}|{e.kind}|{e.summary}".encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def _bucket_digest(slot: BucketSlot) -> str:
    """Compact human-readable events list for the per-bucket LLM prompt."""
    if not slot.events:
        return "(no events)"
    lines = []
    for ev in slot.events[:200]:  # defensive cap
        ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))
        lines.append(f"[{ts}] {ev.hemisphere}/{ev.kind}: {ev.summary}")
    return "\n".join(lines)


def _bucket_prompt(slot: BucketSlot) -> str:
    """Body lives in ``prompts/templates/viewer/narrative.bucket.md.j2``
    (Plan 04 Phase 4)."""
    from prompts import load as load_prompt

    return load_prompt(
        "viewer.narrative.bucket",
        start=time.strftime("%Y-%m-%d %H:%M", time.localtime(slot.start)),
        end=time.strftime("%H:%M", time.localtime(slot.end)),
        events=_bucket_digest(slot),
    )


async def _summarize_bucket(
    slot: BucketSlot,
    *,
    backend: object | None = None,
) -> bucket_cache.BucketSummary:
    """Call Claude for one bucket. Returns a BucketSummary ready to cache."""
    if not slot.events:
        return bucket_cache.BucketSummary(
            bucket_start=slot.start,
            bucket_seconds=slot.end - slot.start,
            content_hash=slot.content_hash,
            event_count=0,
            summary="",  # empty → merge step skips
            cost_usd=0.0,
            duration_ms=0,
            generated_at=time.time(),
        )
    prompt = _bucket_prompt(slot)
    started = time.time()
    # Per-bucket wall-clock cap — see BUCKET_MAX_SECONDS docstring.
    async with asyncio.timeout(BUCKET_MAX_SECONDS):
        text, cost = await _run_once(
            prompt,
            backend=backend,
            model=BUCKET_MODEL,
            max_output_tokens_hint=300,
        )
    return bucket_cache.BucketSummary(
        bucket_start=slot.start,
        bucket_seconds=slot.end - slot.start,
        content_hash=slot.content_hash,
        event_count=len(slot.events),
        summary=text.strip(),
        cost_usd=cost,
        duration_ms=int((time.time() - started) * 1000),
        generated_at=time.time(),
    )


async def _run_once(
    prompt: str,
    *,
    backend: object | None = None,
    model: str = "",
    max_output_tokens_hint: int = 500,
) -> tuple[str, float]:
    """Non-streaming LLM call — used for per-bucket summaries."""
    auth = _ensure_auth()
    if auth.mode == "none":
        raise RuntimeError(
            "no Claude credentials (set CLAUDE_CODE_OAUTH_TOKEN, or ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY)"
        )
    resolved_backend = (
        _backend_with_model(backend, model) if backend is not None
        else _load_viewer_backend(model_override=model)
    )
    chunks: list[str] = []
    cost = 0.0
    async for ev in _run_kernel_stream(
        prompt,
        backend=resolved_backend,
        max_seconds=BUCKET_MAX_SECONDS,
    ):
        if ev["type"] == "chunk":
            chunks.append(ev["text"])
        elif ev["type"] == "result":
            cost = float(ev.get("cost_usd") or 0)
        elif ev["type"] == "error":
            raise RuntimeError(ev.get("message") or "viewer bucket call failed")
    return "".join(chunks), cost


async def ensure_bucket_cache(
    slots: list[BucketSlot],
    *,
    backend: object | None = None,
    progress_cb=None,
) -> list[bucket_cache.BucketSummary]:
    """For each slot, return a cached or freshly-generated BucketSummary.

    Open (in-progress) buckets always regenerate so the "now" edge stays live.
    All others prefer the on-disk cache.
    """
    now_ts = time.time()
    results: list[bucket_cache.BucketSummary | None] = [None] * len(slots)
    to_generate: list[tuple[int, BucketSlot]] = []

    for idx, slot in enumerate(slots):
        force = slot.is_open(now_ts)
        cached = (
            None
            if force
            else bucket_cache.read(
                bucket_seconds=slot.end - slot.start,
                bucket_start=slot.start,
                content_hash=slot.content_hash,
            )
        )
        if cached is not None:
            results[idx] = cached
        else:
            to_generate.append((idx, slot))

    failures: list[str] = []

    if progress_cb:
        await progress_cb(
            {
                "cached": sum(1 for r in results if r is not None),
                "total": len(slots),
                "pending": len(to_generate),
                "failed": 0,
            }
        )

    sem = asyncio.Semaphore(MAX_CONCURRENT_BUCKET_GENERATIONS)

    async def _one(idx: int, slot: BucketSlot):
        async with sem:
            try:
                summary = await _summarize_bucket(slot, backend=backend)
                persist = not slot.is_open(now_ts)
            except Exception as exc:  # noqa: BLE001 — includes asyncio.TimeoutError
                # One bucket failing is recoverable: record an empty
                # placeholder so the merge step skips it (empty
                # summaries are dropped by render_merge_prompt) and
                # the rest of the fill keeps going. Don't persist
                # failure placeholders to disk — a future request
                # should retry.
                failures.append(
                    f"bucket {slot.start}: {type(exc).__name__}: {exc}"
                )
                summary = bucket_cache.BucketSummary(
                    bucket_start=slot.start,
                    bucket_seconds=slot.end - slot.start,
                    content_hash=slot.content_hash,
                    event_count=len(slot.events),
                    summary="",
                    cost_usd=0.0,
                    duration_ms=0,
                    generated_at=time.time(),
                )
                persist = False
            if persist:
                try:
                    bucket_cache.write(summary)
                except OSError:
                    pass
            results[idx] = summary
            if progress_cb:
                done = sum(1 for r in results if r is not None)
                await progress_cb(
                    {
                        "cached": done,
                        "total": len(slots),
                        "pending": len(slots) - done,
                        "failed": len(failures),
                    }
                )

    await asyncio.gather(*(_one(i, s) for i, s in to_generate))
    return [r for r in results if r is not None]


def render_merge_prompt(
    summaries: list[bucket_cache.BucketSummary], window_label: str
) -> str:
    """Body lives in ``prompts/templates/viewer/narrative.weave.md.j2``
    (Plan 04 Phase 4)."""
    from prompts import load as load_prompt

    lines = []
    for s in summaries:
        if not s.summary:
            continue
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.bucket_start))
        lines.append(f"[{ts}] ({s.event_count} events) {s.summary}")
    body = "\n".join(lines) if lines else "(no activity in window)"
    return load_prompt(
        "viewer.narrative.weave",
        body=body,
        window_label=window_label,
    )
