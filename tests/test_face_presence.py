"""Tests for :mod:`alice_speaking.infra.face_presence`.

The pusher must:

- No-op when ``ALICE_FACE_URL`` is empty.
- POST /state with the right body on a state change.
- Coalesce duplicate consecutive states (don't spam the face).
- Swallow HTTP errors silently.
- Choose ``sleep`` vs ``idle`` based on the quiet-hours hook.
- Push ``speaking`` on enter / ``thinking`` on exit of the context
  manager.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from alice_speaking.infra.face_presence import FacePresence


class _Recorder:
    """A tiny ASGI-ish callable that records POSTs and returns a fixed status."""

    def __init__(self, status: int = 200) -> None:
        self.posts: list[dict[str, Any]] = []
        self._status = status

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8") or "{}")
        self.posts.append({"url": str(request.url), "body": body})
        return httpx.Response(self._status)


@pytest.fixture
def patch_httpx(monkeypatch: pytest.MonkeyPatch):
    """Replace httpx.AsyncClient with one backed by MockTransport."""
    rec = _Recorder()

    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(rec.handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("alice_speaking.infra.face_presence.httpx.AsyncClient", factory)
    return rec


@pytest.mark.asyncio
async def test_disabled_when_url_empty():
    fp = FacePresence(url="")
    assert not fp.enabled
    fp.set_state("thinking")
    # No raise, no schedule.


@pytest.mark.asyncio
async def test_set_state_posts(patch_httpx: _Recorder):
    fp = FacePresence(url="http://face.test")
    fp.set_state("thinking")
    await asyncio.sleep(0.05)
    assert patch_httpx.posts == [
        {"url": "http://face.test/state", "body": {"state": "thinking"}}
    ]


@pytest.mark.asyncio
async def test_coalesces_duplicate_state(patch_httpx: _Recorder):
    fp = FacePresence(url="http://face.test")
    fp.set_state("thinking")
    await asyncio.sleep(0.05)
    fp.set_state("thinking")
    await asyncio.sleep(0.05)
    fp.set_state("idle")
    await asyncio.sleep(0.05)
    states = [p["body"]["state"] for p in patch_httpx.posts]
    assert states == ["thinking", "idle"]


@pytest.mark.asyncio
async def test_invalid_state_ignored(patch_httpx: _Recorder):
    fp = FacePresence(url="http://face.test")
    fp.set_state("definitely-not-a-state")
    await asyncio.sleep(0.05)
    assert patch_httpx.posts == []


@pytest.mark.asyncio
async def test_set_idle_picks_sleep_in_quiet_hours(patch_httpx: _Recorder):
    fp = FacePresence(url="http://face.test", quiet_hours_fn=lambda: True)
    fp.set_idle()
    await asyncio.sleep(0.05)
    assert [p["body"]["state"] for p in patch_httpx.posts] == ["sleep"]


@pytest.mark.asyncio
async def test_set_idle_picks_idle_outside_quiet_hours(patch_httpx: _Recorder):
    fp = FacePresence(url="http://face.test", quiet_hours_fn=lambda: False)
    fp.set_idle()
    await asyncio.sleep(0.05)
    assert [p["body"]["state"] for p in patch_httpx.posts] == ["idle"]


@pytest.mark.asyncio
async def test_speaking_context_manager(patch_httpx: _Recorder):
    fp = FacePresence(url="http://face.test")
    async with fp.speaking():
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.05)
    assert [p["body"]["state"] for p in patch_httpx.posts] == [
        "speaking",
        "thinking",
    ]


@pytest.mark.asyncio
async def test_http_errors_are_swallowed(monkeypatch: pytest.MonkeyPatch):
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("alice_speaking.infra.face_presence.httpx.AsyncClient", factory)
    fp = FacePresence(url="http://face.test")
    fp.set_state("thinking")
    await asyncio.sleep(0.05)
    # No raise; reaching here means errors were swallowed.
