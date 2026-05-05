"""Tests for the compaction helper module."""

from __future__ import annotations

import pathlib

from alice_speaking.pipeline import compaction
from alice_speaking.domain.turn_log import Turn


def _turn(inbound: str = "hi", outbound: str = "hello") -> Turn:
    return Turn(
        ts=0.0,
        sender_number="+1",
        sender_name="Owner",
        inbound=inbound,
        outbound=outbound,
    )


# ---------------------------------------------------------------- should_compact


def test_should_compact_fires_above_threshold() -> None:
    assert compaction.should_compact({"input_tokens": 200_000}, 150_000) is True


def test_should_compact_false_below_threshold() -> None:
    assert compaction.should_compact({"input_tokens": 50_000}, 150_000) is False


def test_should_compact_false_at_threshold() -> None:
    # Strict >, not >=, per the design prose.
    assert compaction.should_compact({"input_tokens": 150_000}, 150_000) is False


def test_should_compact_missing_usage() -> None:
    assert compaction.should_compact(None, 150_000) is False
    assert compaction.should_compact({}, 150_000) is False


def test_should_compact_malformed_usage() -> None:
    assert compaction.should_compact({"input_tokens": "lots"}, 150_000) is False
    assert compaction.should_compact({"input_tokens": None}, 150_000) is False
    assert compaction.should_compact("not a dict", 150_000) is False


def test_should_compact_reads_last_iteration() -> None:
    """When per-iteration usage is present, ``should_compact`` reads the
    *last* iteration (≈ post-turn prompt size), not the cumulative
    top-level fields. The top-level cache_read is summed across every
    internal API call inside the agent loop and inflates linearly with
    tool-call count — using it as the trigger metric was the source of
    the "compacts after every message" bug."""
    # Top-level is huge (cumulative across 10 calls), but the LAST call's
    # prompt is small → no real context pressure → must not fire.
    assert (
        compaction.should_compact(
            {
                "input_tokens": 100,
                "cache_read_input_tokens": 800_000,  # cumulative inflated
                "cache_creation_input_tokens": 50_000,
                "iterations": [
                    {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 80_000,
                        "cache_creation_input_tokens": 5_000,
                    }
                ],
            },
            150_000,
        )
        is False
    )
    # Last iteration alone crosses threshold → fire.
    assert (
        compaction.should_compact(
            {
                "input_tokens": 50,
                "cache_read_input_tokens": 1_000_000,
                "iterations": [
                    {"input_tokens": 5, "cache_read_input_tokens": 60_000},
                    {
                        "input_tokens": 5,
                        "cache_read_input_tokens": 160_000,
                    },  # last call
                ],
            },
            150_000,
        )
        is True
    )


def test_should_compact_falls_back_to_cumulative() -> None:
    """When iterations is missing (older event-log lines, pi backend),
    fall back to the top-level fields. The fallback over-counts on
    tool-heavy turns, but that's strictly the old behavior — preserved
    so callers without iteration data still trigger eventually rather
    than silently never firing."""
    assert (
        compaction.should_compact(
            {"input_tokens": 10, "cache_read_input_tokens": 800_000}, 150_000
        )
        is True
    )
    assert (
        compaction.should_compact(
            {"input_tokens": 10, "cache_creation_input_tokens": 200_000}, 150_000
        )
        is True
    )
    assert (
        compaction.should_compact(
            {"input_tokens": 10, "cache_read_input_tokens": 50_000}, 150_000
        )
        is False
    )
    # Empty iterations list also falls through to cumulative.
    assert (
        compaction.should_compact(
            {"input_tokens": 10, "cache_read_input_tokens": 800_000, "iterations": []},
            150_000,
        )
        is True
    )


# -------------------------------------------------------- preamble builders


def test_bootstrap_preamble_with_turns() -> None:
    result = compaction.build_bootstrap_preamble([_turn("hi", "hello")])
    assert "Recent conversation" in result
    assert "[Owner] hi" in result
    assert "[alice] hello" in result


def test_bootstrap_preamble_empty() -> None:
    assert compaction.build_bootstrap_preamble([]) == ""


def test_summary_preamble_includes_summary_and_tail() -> None:
    result = compaction.build_summary_preamble(
        "four-part summary body",
        [_turn("i ate breakfast", "logged")],
    )
    assert "Context summary" in result
    assert "four-part summary body" in result
    assert "Recent turns:" in result
    assert "[alice] logged" in result


def test_summary_preamble_without_recent_turns() -> None:
    result = compaction.build_summary_preamble("summary only", [])
    assert "summary only" in result
    # With no turn tail we skip the "Recent turns:" divider entirely.
    assert "Recent turns:" not in result


# -------------------------------------------------------------- read/write summary


def test_read_summary_missing(tmp_path: pathlib.Path) -> None:
    assert compaction.read_summary_if_any(tmp_path / "nope.md") is None


def test_write_then_read_summary(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "summary.md"
    compaction.write_summary(path, "body text")
    assert compaction.read_summary_if_any(path) == "body text"


def test_write_summary_creates_parent(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "a" / "b" / "summary.md"
    compaction.write_summary(path, "hi")
    assert path.is_file()


def test_read_empty_summary_returns_none(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "summary.md"
    path.write_text("   \n\n")
    assert compaction.read_summary_if_any(path) is None
