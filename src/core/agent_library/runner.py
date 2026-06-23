"""Async runner for :class:`AgentSpec`.

The runner is the thin layer between an :class:`AgentSpec` and the
backend kernel. It applies the spec's constraint layers (tool
policy + behavioral rules → effective :class:`KernelSpec`), picks a
:class:`Kernel` impl via :func:`core.kernel.make_kernel`, and
dispatches one turn.

Backend selection: the caller passes a ``backend`` object
(:class:`core.config.model.BackendSpec` in practice) that the kernel
factory uses to pick AnthropicKernel vs PiKernel. The runner stays
duck-typed on ``backend`` to match :func:`make_kernel`'s shape —
this avoids a hard import cycle and keeps the runner trivially
testable with a stub backend.

OutputSchema validation is deliberately a no-op in Phase 1. The
:class:`OutputSchema` field exists on :class:`AgentSpec` so registry
entries can record expected output shapes, but the validator wires
in during Phase 2+.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from ..kernel import KernelResult, make_kernel
from .types import AgentSpec


if TYPE_CHECKING:
    from ..events import EventEmitter


__all__ = ["run_agent"]


def _default_backend_for_agent(agent: AgentSpec) -> object:
    """Resolve a configured backend for legacy callers that omit one."""
    from ..config.model import load as load_model_config

    mind = pathlib.Path(
        os.environ.get("ALICE_MIND") or pathlib.Path.home() / "alice-mind"
    )
    cfg = load_model_config(mind)
    if agent.persona == "thinking" or agent.name in {
        "designer",
        "research-writer",
    }:
        return cfg.thinking
    if agent.persona == "cozylobe":
        return cfg.viewer
    return cfg.speaking


async def run_agent(
    agent: AgentSpec,
    *,
    prompt: str,
    emitter: "EventEmitter",
    backend: object = None,
    correlation_id: Optional[str] = None,
) -> KernelResult:
    """Dispatch one turn through ``agent``'s constraint-applied kernel
    spec.

    Steps:

    1. :meth:`AgentSpec.build_spec` applies the tool policy and
       merges behavioral rules into ``append_system_prompt``,
       producing the effective :class:`KernelSpec`.
    2. :func:`core.kernel.make_kernel` picks the backend impl from
       ``backend``. If omitted, the runner resolves a hemisphere
       backend from ``mind/config/model.yml`` based on the agent
       persona.
    3. ``await kernel.run(prompt, spec)`` runs the turn and returns
       the :class:`KernelResult`.

    Raises :class:`core.agent_library.types.PolicyViolation` (from
    :meth:`AgentSpec.effective_tools` via :meth:`build_spec`) when
    the tool policy leaves no tools available. The runner does not
    catch it — the caller decides whether that's a fatal error or a
    fallback signal.

    **Per-call overrides.** Registered specs are immutable
    (``@dataclass(frozen=True)``) by design; per-call tweaks go
    through :func:`dataclasses.replace`. Wrap the registry entry
    rather than mutating it:

    .. code-block:: python

        from dataclasses import replace
        from core.agent_library import default_registry, run_agent

        spec = default_registry.get("reviewer")
        # Bump the model + per-turn time cap for this dispatch only.
        kernel = replace(spec.kernel_spec, model=os.environ["ALICE_AGENT_MODEL"], max_seconds=120)
        agent = replace(spec, kernel_spec=kernel)
        result = await run_agent(agent, prompt="...", emitter=emitter)

    Common override targets: ``kernel_spec.model`` (downgrade /
    upgrade), ``kernel_spec.max_seconds`` (per-turn budget),
    ``kernel_spec.allowed_tools`` (narrow the surface — pair with
    ``tool_policy=None`` so the registered policy's allowlist does
    not reintroduce dropped tools at :meth:`AgentSpec.effective_tools`
    time), ``kernel_spec.append_system_prompt`` (inject a caller-
    specific prompt body; the registered behavioral_constraints still
    merge in after the base via :meth:`AgentSpec.assembled_system_prompt`).
    """
    spec = agent.build_spec()

    if backend is None:
        backend = _default_backend_for_agent(agent)
    if not spec.model and getattr(backend, "model", ""):
        spec = replace(spec, model=getattr(backend, "model"))

    kernel = make_kernel(backend, emitter, correlation_id=correlation_id)
    return await kernel.run(prompt, spec)
