"""Tournament runner: replay the game N times and aggregate outcomes.

Useful for comparing baseline strategies, or for evaluating LLM providers
under a single configuration with many random seeds.

Usage:
    python -m src.tournament --runs 20 --rounds 50 --mode baseline
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from collections import defaultdict
from typing import Callable

from .agents.base import Agent
from .agents.baseline import (
    BuyAndHoldAgent,
    MeanReversionAgent,
    MomentumAgent,
    RandomAgent,
)
from .arena.world import World, WorldConfig
from .config import load_env
from .main import build_agents


def _baseline_factory(seed_offset: int) -> Callable[[], list[Agent]]:
    """Build a fresh agent roster seeded with a per-run offset."""

    def factory() -> list[Agent]:
        return [
            RandomAgent("random-A", seed=1 + seed_offset, display_name="Random-A"),
            MomentumAgent("momentum-B", display_name="Momentum-B"),
            MeanReversionAgent("meanrev-C", display_name="MeanRev-C"),
            BuyAndHoldAgent("buyhold-D", display_name="BuyHold-D"),
            RandomAgent("random-E", seed=42 + seed_offset, display_name="Random-E"),
        ]

    return factory


def run_tournament(
    *,
    runs: int,
    rounds: int,
    mode: str,
) -> dict[str, dict[str, float]]:
    """Run `runs` games and return per-agent stats keyed by agent_id."""

    wins: dict[str, int] = defaultdict(int)
    ranks: dict[str, list[int]] = defaultdict(list)
    equities: dict[str, list[float]] = defaultdict(list)
    display_names: dict[str, str] = {}

    for run_idx in range(runs):
        if mode == "baseline":
            agents = _baseline_factory(seed_offset=run_idx)()
        else:
            agents = build_agents(mode)

        for agent in agents:
            display_names[agent.agent_id] = agent.display_name

        config = WorldConfig(total_rounds=rounds)
        world = World(config, agents)
        result = world.run()

        leaderboard = result.leaderboard()
        winner = leaderboard[0][0]
        wins[winner] += 1
        for rank, (aid, equity) in enumerate(leaderboard, start=1):
            ranks[aid].append(rank)
            equities[aid].append(equity)

    stats: dict[str, dict[str, float]] = {}
    for aid in display_names:
        rank_list = ranks.get(aid, [])
        eq_list = equities.get(aid, [])
        stats[aid] = {
            "display_name": display_names[aid],
            "wins": wins.get(aid, 0),
            "win_rate": wins.get(aid, 0) / runs if runs else 0.0,
            "avg_rank": statistics.mean(rank_list) if rank_list else 0.0,
            "avg_equity": statistics.mean(eq_list) if eq_list else 0.0,
            "stddev_equity": statistics.stdev(eq_list) if len(eq_list) > 1 else 0.0,
        }
    return stats


def _render(stats: dict[str, dict[str, float]]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Tournament Results")
        table.add_column("Agent")
        table.add_column("Wins", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg Rank", justify="right")
        table.add_column("Avg Equity", justify="right")
        table.add_column("σ Equity", justify="right")

        ordered = sorted(stats.items(), key=lambda kv: kv[1]["wins"], reverse=True)
        for _, s in ordered:
            table.add_row(
                str(s["display_name"]),
                f"{int(s['wins'])}",
                f"{s['win_rate']:.2%}",
                f"{s['avg_rank']:.2f}",
                f"{s['avg_equity']:.4f}",
                f"{s['stddev_equity']:.4f}",
            )
        console.print(table)
    except ImportError:
        print("Tournament Results:")
        for aid, s in stats.items():
            print(
                f"  {s['display_name']:14s} wins={int(s['wins'])} "
                f"rate={s['win_rate']:.2%} avg_rank={s['avg_rank']:.2f} "
                f"avg_equity={s['avg_equity']:.4f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a multi-game tournament.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--mode", choices=["baseline", "llm", "mixed"], default="baseline")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    load_env()
    print(f"Tournament: {args.runs} runs × {args.rounds} rounds × mode={args.mode}")

    start = time.monotonic()
    stats = run_tournament(runs=args.runs, rounds=args.rounds, mode=args.mode)
    elapsed = time.monotonic() - start
    print(f"Completed in {elapsed:.2f}s\n")
    _render(stats)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
