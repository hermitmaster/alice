"""ContextProbe — snapshot the live daemon's context composition.

Used by the CLI socket's ``{"type": "context"}`` request to give
operators a real-time view of what's loaded in the speaking agent's
context window: the rendered system_persona prompt, the allowed-tools
list, MCP server inventory, pending bootstrap preamble, current
session_id, and whether a turn is in flight.

Per-turn token totals (input/output/cache_read/cache_creation) are NOT
included here — those land in events.jsonl as ``result`` events and
the viewer reads them via ``aggregators.latest_speaking_usage``. The
probe complements that historical view with the live-state pieces
events.jsonl can't show.

Design constraints:

- Never block. The probe runs synchronously in the CLI dispatch
  task; if a turn is mid-run, return what's known immediately.
- Never mutate. Pure read-only snapshot.
- Cheap. No tokenizer dep, no MCP introspection round-trips —
  pre-computed strings + name lists only.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ContextSnapshot:
    """Read-only view of the daemon's current context composition."""

    ts: float
    model: Optional[str]
    backend: Optional[str]
    mind_dir: Optional[str]
    skills_cwd: Optional[str]
    session_id: Optional[str]
    system_prompt: dict[str, Any]
    tools: dict[str, Any]
    mcp_servers: dict[str, dict[str, Any]]
    pending_preamble: Optional[dict[str, Any]]
    in_flight: Optional[dict[str, Any]]
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextProbe:
    """Snapshot the live speaking daemon's context composition.

    Constructed once by :class:`SpeakingDaemon` with callable accessors
    for every piece of live state — the daemon mutates state through
    its normal paths and the probe reads it on demand. No reverse
    coupling: the probe knows nothing about handlers, dispatch, or
    transports.
    """

    def __init__(
        self,
        *,
        get_system_prompt: Callable[[], Optional[str]],
        get_builtin_tools: Callable[[], list[str]],
        get_custom_tool_names: Callable[[], list[str]],
        get_mcp_servers: Callable[[], dict[str, Any]],
        get_session_id: Callable[[], Optional[str]],
        get_pending_preamble: Callable[[], Optional[str]],
        get_current_turn_kind: Callable[[], Optional[str]],
        get_model: Callable[[], Optional[str]],
        get_backend: Callable[[], Optional[str]],
        get_mind_dir: Callable[[], Optional[str]],
        get_skills_cwd: Callable[[], Optional[str]],
    ) -> None:
        self._get_system_prompt = get_system_prompt
        self._get_builtin_tools = get_builtin_tools
        self._get_custom_tool_names = get_custom_tool_names
        self._get_mcp_servers = get_mcp_servers
        self._get_session_id = get_session_id
        self._get_pending_preamble = get_pending_preamble
        self._get_current_turn_kind = get_current_turn_kind
        self._get_model = get_model
        self._get_backend = get_backend
        self._get_mind_dir = get_mind_dir
        self._get_skills_cwd = get_skills_cwd

    def snapshot(self, *, include_text: bool = True) -> ContextSnapshot:
        """Capture a point-in-time view of context composition.

        Args:
            include_text: include the rendered system_prompt and the
                pending preamble as full strings. The CLI request
                defaults to True (the operator wants to see the actual
                content); callers can flip it off if they only need
                sizes.
        """
        system_prompt_text = self._get_system_prompt() or ""
        pending_preamble_text = self._get_pending_preamble()

        custom = self._get_custom_tool_names() or []
        builtin = self._get_builtin_tools() or []
        mcp_servers_raw = self._get_mcp_servers() or {}
        mcp_servers = _summarize_mcp_servers(mcp_servers_raw, custom)

        in_flight_kind = self._get_current_turn_kind()

        return ContextSnapshot(
            ts=time.time(),
            model=self._get_model(),
            backend=self._get_backend(),
            mind_dir=self._get_mind_dir(),
            skills_cwd=self._get_skills_cwd(),
            session_id=self._get_session_id(),
            system_prompt={
                "chars": len(system_prompt_text),
                "text": system_prompt_text if include_text else None,
            },
            tools={
                "builtin": builtin,
                "custom": custom,
                "count": len(builtin) + len(custom),
            },
            mcp_servers=mcp_servers,
            pending_preamble=(
                {
                    "chars": len(pending_preamble_text),
                    "text": pending_preamble_text if include_text else None,
                }
                if pending_preamble_text
                else None
            ),
            in_flight=(
                {"turn_kind": in_flight_kind} if in_flight_kind else None
            ),
        )


def _summarize_mcp_servers(
    mcp_servers: dict[str, Any],
    custom_tool_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Group ``custom_tool_names`` (which look like ``mcp__<server>__<tool>``)
    by server, and merge with the configured ``mcp_servers`` dict so the
    operator sees both "what's wired" and "what tools each server
    exposes." Servers with no tools listed (e.g. http-only servers the
    SDK hasn't introspected yet) still appear with an empty tool list.
    """
    by_server: dict[str, list[str]] = {}
    for full in custom_tool_names:
        if not full.startswith("mcp__"):
            continue
        rest = full[len("mcp__"):]
        if "__" not in rest:
            continue
        server, tool = rest.split("__", 1)
        by_server.setdefault(server, []).append(tool)

    summary: dict[str, dict[str, Any]] = {}
    for server, spec in (mcp_servers or {}).items():
        tools = sorted(by_server.get(server, []))
        summary[server] = {
            "type": _server_type(spec),
            "tool_count": len(tools),
            "tool_names": tools,
        }
    # Include any servers we discovered from tool prefixes but that
    # aren't in the configured dict — defensive against config-vs-runtime
    # drift; the operator should be able to see both.
    for server, tools in by_server.items():
        if server in summary:
            continue
        summary[server] = {
            "type": "unknown",
            "tool_count": len(tools),
            "tool_names": sorted(tools),
        }
    return summary


def _server_type(spec: Any) -> str:
    """Classify an MCP server spec without dragging in SDK types."""
    if isinstance(spec, dict):
        return str(spec.get("type", "unknown"))
    type_attr = getattr(spec, "type", None)
    if type_attr:
        return str(type_attr)
    return type(spec).__name__


__all__ = ["ContextProbe", "ContextSnapshot"]
