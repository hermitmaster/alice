"""Tests for the /context viewer route + the context_probe_client helpers.

Two layers:

- ``decompose`` is a pure function over a snapshot dict; tested
  directly with synthetic input.
- The route + ``fetch_snapshot`` is exercised end-to-end with the real
  FastAPI app (via httpx.AsyncClient) and a stubbed alice binary that
  emits the JSON snapshot on stdout — no docker, no daemon.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import stat
import textwrap

import pytest

from alice_viewer import context_probe_client


# ----------------------------------------------------------------------------
# decompose() — pure function tests


def _example_snapshot() -> dict:
    return {
        "ts": 1000.0,
        "model": "claude-sonnet-4-5",
        "backend": "subscription",
        "session_id": "s-1",
        "system_prompt": {"chars": 14, "text": "you are alice."},
        "tools": {
            "builtin": ["Bash", "Read"],
            "custom": ["mcp__alice__send_message"],
            "count": 3,
        },
        "mcp_servers": {
            "alice": {
                "type": "stdio",
                "tool_count": 1,
                "tool_names": ["send_message"],
            },
        },
        "pending_preamble": None,
        "in_flight": None,
    }


def test_decompose_emits_one_component_per_category():
    out = context_probe_client.decompose(_example_snapshot())
    names = [c["name"] for c in out["components"]]
    assert "system prompt" in names
    assert "builtin tools" in names
    assert "custom tools" in names
    assert "mcp:alice" in names
    # No preamble in this snapshot
    assert "pending preamble" not in names


def test_decompose_total_matches_component_sum():
    out = context_probe_client.decompose(_example_snapshot())
    assert out["total_tokens"] == sum(c["tokens"] for c in out["components"])


def test_decompose_skips_zero_token_components():
    snap = _example_snapshot()
    snap["tools"]["custom"] = []
    out = context_probe_client.decompose(snap)
    names = [c["name"] for c in out["components"]]
    assert "custom tools" not in names


def test_decompose_includes_pending_preamble_when_present():
    snap = _example_snapshot()
    snap["pending_preamble"] = {"chars": 200, "text": "previous turns: ..."}
    out = context_probe_client.decompose(snap)
    names = [c["name"] for c in out["components"]]
    assert "pending preamble" in names


def test_decompose_falls_back_to_chars_when_text_omitted():
    """When the snapshot was fetched with --no-text the text fields are
    None; decompose() should still produce a (rougher) estimate."""
    snap = _example_snapshot()
    snap["system_prompt"] = {"chars": 400, "text": None}
    out = context_probe_client.decompose(snap)
    sp = next(c for c in out["components"] if c["name"] == "system prompt")
    # 400 chars / 4 ≈ 100 tokens. Allow a small range.
    assert 50 <= sp["tokens"] <= 200


# ----------------------------------------------------------------------------
# fetch_snapshot() — subprocess-shape tests using a stubbed binary


def _write_stub(
    bin_path: pathlib.Path,
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    sleep_seconds: float = 0.0,
) -> None:
    """Write an executable shell script that prints ``stdout`` and exits
    with ``exit_code``. Used to stand in for the real bin/alice. The
    body is left-aligned (no leading whitespace) because Linux kernels
    refuse to exec a script whose shebang isn't at column 0."""
    lines = [
        "#!/bin/sh",
        f"sleep {sleep_seconds:g}" if sleep_seconds > 0 else "",
        "cat <<'__OUT__'",
        stdout,
        "__OUT__",
    ]
    if stderr:
        lines += ["cat >&2 <<'__ERR__'", stderr, "__ERR__"]
    lines.append(f"exit {exit_code}")
    body = "\n".join(line for line in lines if line is not None) + "\n"
    bin_path.write_text(body)
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_fetch_snapshot_parses_first_snapshot_event(tmp_path):
    snapshot = _example_snapshot()
    payload = "\n".join(
        [
            json.dumps({"type": "ack"}),
            json.dumps({"type": "context_snapshot", "data": snapshot}),
            json.dumps({"type": "done"}),
        ]
    )
    stub = tmp_path / "alice"
    _write_stub(stub, stdout=payload)
    result = asyncio.run(
        context_probe_client.fetch_snapshot(alice_bin=str(stub))
    )
    assert result == snapshot


def test_fetch_snapshot_raises_on_nonzero_exit(tmp_path):
    stub = tmp_path / "alice"
    _write_stub(stub, stdout="", stderr="boom", exit_code=3)
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(context_probe_client.fetch_snapshot(alice_bin=str(stub)))
    assert "exited 3" in str(exc_info.value)


def test_fetch_snapshot_raises_on_missing_binary(tmp_path):
    """When the executable doesn't exist on PATH or as an absolute file,
    callers see FileNotFoundError so the route can return a friendly
    'no worker' status."""
    missing = tmp_path / "definitely-not-installed-anywhere"
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            context_probe_client.fetch_snapshot(alice_bin=str(missing))
        )


def test_fetch_snapshot_raises_when_no_snapshot_event(tmp_path):
    """The wrapper exited 0 but produced no context_snapshot — caller
    sees a clear error rather than getting None back."""
    payload = json.dumps({"type": "ack"}) + "\n" + json.dumps({"type": "done"})
    stub = tmp_path / "alice"
    _write_stub(stub, stdout=payload)
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(context_probe_client.fetch_snapshot(alice_bin=str(stub)))
    assert "no context_snapshot" in str(exc_info.value)


def test_fetch_snapshot_times_out(tmp_path):
    stub = tmp_path / "alice"
    _write_stub(stub, stdout="", sleep_seconds=2.0)
    with pytest.raises(TimeoutError):
        asyncio.run(
            context_probe_client.fetch_snapshot(
                alice_bin=str(stub), timeout=0.3
            )
        )


# ----------------------------------------------------------------------------
# /context route — page render only (the JSON API exercised separately)


def test_context_page_renders(tmp_path, monkeypatch):
    """The page itself shouldn't depend on the worker being up — it
    just renders chrome and lets the JS fetch /api/context."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from alice_viewer.main import create_app
    from alice_viewer.settings import Paths

    # Minimal Paths fixture pointing at a tempdir; the route doesn't
    # actually touch any of these files unless _state_context() does.
    mind = tmp_path / "mind"
    (mind / "inner" / "state").mkdir(parents=True)
    (mind / "memory").mkdir()
    (mind / "memory" / "events.jsonl").write_text("")
    (mind / "inner" / "directive.md").write_text("")

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (mind / "inner" / "state").mkdir(parents=True, exist_ok=True)
    paths = Paths(
        mind_dir=mind,
        state_dir=state_dir,
        thinking_log=mind / "memory" / "events.jsonl",
        speaking_log=mind / "memory" / "events.jsonl",
        turn_log=mind / "inner" / "state" / "speaking-turns.jsonl",
    )
    app = create_app(paths=paths)
    client = TestClient(app)

    response = client.get("/context")
    assert response.status_code == 200
    assert "context · live snapshot" in response.text
    assert "/api/context" in response.text
