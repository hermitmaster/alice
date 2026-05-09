"""Diagnostics — read-only probes into the live speaking daemon.

Modules here expose snapshot-style views over daemon state for
operator tooling (the viewer's /context tab, ad-hoc shell debugging
via ``bin/alice context``). They never mutate state, never block on
in-flight turns, and tolerate partial information.
"""

from .context_probe import ContextProbe, ContextSnapshot


__all__ = ["ContextProbe", "ContextSnapshot"]
