"""Viewer-side glue for the speaking daemon's ``alice context`` RPC.

The viewer runs on the host while the speaking daemon runs in the
worker container. This module shells out to ``bin/alice context --json``
(which docker-execs into the worker, hits the CLI socket, and prints
the snapshot) and turns the result into a structure suitable for the
``/context`` template — including a tiktoken-based estimate of the
token weight of each component so the donut renders in proportional
slices.

Why tiktoken instead of the official anthropic ``count_tokens``: the
viewer doesn't (and shouldn't) ship the anthropic SDK, and the donut
only needs proportional accuracy. The cl100k_base encoder is close
enough for "look, this MCP server's tool definitions are eating ~30%
of the context" — and it's offline.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import shutil
from typing import Any, Optional


log = logging.getLogger(__name__)


SNAPSHOT_TIMEOUT_SECONDS = 15.0


@functools.lru_cache(maxsize=1)
def _encoder():
    """Lazy + cached tiktoken encoder. cl100k_base is the GPT-4-class
    encoder — not Claude's, but close enough for the proportional
    breakdown the donut uses. Pre-cached so per-request cost is just
    encode() calls."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _tokens(text: Optional[str]) -> int:
    """Approximate token count for ``text``. Falls back to a chars/4
    heuristic if tiktoken errors out (unlikely — the encoder is
    self-contained)."""
    if not text:
        return 0
    try:
        return len(_encoder().encode(text))
    except Exception:  # noqa: BLE001
        log.exception("tiktoken encode failed; falling back to chars/4")
        return max(1, len(text) // 4)


async def fetch_snapshot(
    *,
    alice_bin: str = "alice",
    timeout: float = SNAPSHOT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run ``alice context --json`` and return the parsed snapshot.

    Raises:
        FileNotFoundError: ``alice_bin`` isn't on PATH (or absolute
            path doesn't exist).
        TimeoutError: the subprocess didn't return within ``timeout``.
        RuntimeError: the wrapper exited nonzero or didn't emit a
            ``context_snapshot`` event.
    """
    if shutil.which(alice_bin) is None and not _is_existing_path(alice_bin):
        raise FileNotFoundError(
            f"{alice_bin!r} not found on PATH — cannot reach the worker."
        )
    proc = await asyncio.create_subprocess_exec(
        alice_bin,
        "context",
        "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise TimeoutError(
            f"alice context did not return in {timeout}s"
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"alice context exited {proc.returncode}: "
            f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
        )

    snapshot: Optional[dict] = None
    for line in stdout_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "context_snapshot":
            snapshot = event.get("data")
            break
    if snapshot is None:
        raise RuntimeError(
            "alice context produced no context_snapshot event in its output"
        )
    return snapshot


def _is_existing_path(p: str) -> bool:
    import os.path

    return os.path.isabs(p) and os.path.isfile(p) and os.access(p, os.X_OK)


def decompose(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Bucket a snapshot into the categories the /context donut renders.

    Each component carries ``tokens`` (approximate) and ``label`` so the
    template doesn't need to know about the snapshot's internal shape.
    Returns:
        {
            "components": [{name, tokens, detail}, ...],
            "total_tokens": int,
            "snapshot": {raw snapshot for the data table},
        }
    """
    sp = snapshot.get("system_prompt") or {}
    sp_tokens = _tokens(sp.get("text")) if sp.get("text") is not None else (
        _approx_tokens_from_chars(sp.get("chars", 0))
    )

    preamble = snapshot.get("pending_preamble") or {}
    preamble_tokens = (
        _tokens(preamble.get("text"))
        if preamble.get("text") is not None
        else _approx_tokens_from_chars(preamble.get("chars", 0))
    )

    tools = snapshot.get("tools") or {}
    builtin = tools.get("builtin") or []
    custom = tools.get("custom") or []
    # Tool *names* are tiny — most of the cost is the JSON Schema for
    # each tool's input. We don't have that here (the daemon doesn't
    # ship schemas through the probe), so this number underweights
    # tools relative to reality. Honest about it in the template
    # ("name-only estimate"). Each name takes ~3-5 tokens.
    builtin_tokens = sum(_tokens(name) for name in builtin)
    custom_tokens = sum(_tokens(name) for name in custom)

    mcp = snapshot.get("mcp_servers") or {}
    mcp_components: list[dict[str, Any]] = []
    mcp_total = 0
    for name in sorted(mcp.keys()):
        spec = mcp[name] or {}
        names = spec.get("tool_names") or []
        # Pure name-token estimate; same caveat as above.
        toks = sum(_tokens(n) for n in names) + _tokens(name)
        mcp_total += toks
        mcp_components.append(
            {
                "name": f"mcp:{name}",
                "tokens": toks,
                "detail": (
                    f"{spec.get('tool_count', 0)} tools "
                    f"({spec.get('type', '?')})"
                ),
            }
        )

    components: list[dict[str, Any]] = [
        {
            "name": "system prompt",
            "tokens": sp_tokens,
            "detail": f"{sp.get('chars', 0)} chars",
        },
        {
            "name": "builtin tools",
            "tokens": builtin_tokens,
            "detail": f"{len(builtin)} tools (name-only estimate)",
        },
        {
            "name": "custom tools",
            "tokens": custom_tokens,
            "detail": f"{len(custom)} tools (name-only estimate)",
        },
        *mcp_components,
    ]
    if preamble_tokens:
        components.append(
            {
                "name": "pending preamble",
                "tokens": preamble_tokens,
                "detail": f"{preamble.get('chars', 0)} chars",
            }
        )
    components = [c for c in components if c["tokens"] > 0]
    total = sum(c["tokens"] for c in components)
    return {
        "components": components,
        "total_tokens": total,
        "snapshot": snapshot,
    }


def _approx_tokens_from_chars(chars: int) -> int:
    """Fallback when the snapshot was fetched with --no-text. Rough
    chars-per-token of ~4 for English-ish text."""
    return max(0, chars // 4)


__all__ = ["fetch_snapshot", "decompose", "SNAPSHOT_TIMEOUT_SECONDS"]
