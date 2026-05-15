"""Smoke test for the tournament runner."""

from __future__ import annotations

from src.tournament import run_tournament


def test_tournament_baseline_smoke():
    stats = run_tournament(runs=3, rounds=15, mode="baseline")
    # Five baseline agents — all must appear in the stats.
    assert len(stats) == 5
    total_wins = sum(int(s["wins"]) for s in stats.values())
    assert total_wins == 3  # exactly one winner per run
    for s in stats.values():
        assert 1.0 <= s["avg_rank"] <= 5.0
        assert s["avg_equity"] > 0
