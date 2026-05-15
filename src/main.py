"""CLI entrypoint: `python -m src.main`.

Modes
-----
- baseline: 5 non-LLM agents (random, momentum, mean-reversion, buy-and-hold,
  random) — fast, no API keys required. Good for verifying the arena.
- llm: 5 LLM agents, one per provider, configured via .env. Skips providers
  whose API key isn't set and falls back to a baseline so the game always has 5
  players.
- mixed: pairs LLM agents (where keys are set) with baselines for the rest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .agent_factory import PROVIDER_CLASSES as YAML_PROVIDER_CLASSES, build_agent
from .agent_spec import AgentSpec
from .agents.base import Agent
from .agents.baseline import (
    BuyAndHoldAgent,
    MeanReversionAgent,
    MomentumAgent,
    RandomAgent,
)
from .agents.llm import (
    AnthropicAgent,
    DeepSeekAgent,
    GoogleAgent,
    OpenAIAgent,
    QwenAgent,
)
from .arena.visualize import price_sparkline
from .arena.world import World, WorldConfig, WorldResult
from .config import PROVIDERS, llm_config_for, load_env
from .config_loader import load_config


PROVIDER_CLASSES = {
    "openai": OpenAIAgent,
    "anthropic": AnthropicAgent,
    "google": GoogleAgent,
    "deepseek": DeepSeekAgent,
    "qwen": QwenAgent,
}


def _baseline_roster() -> list[Agent]:
    return [
        RandomAgent("random-A", seed=1, display_name="Random-A"),
        MomentumAgent("momentum-B", display_name="Momentum-B"),
        MeanReversionAgent("meanrev-C", display_name="MeanRev-C"),
        BuyAndHoldAgent("buyhold-D", display_name="BuyHold-D"),
        RandomAgent("random-E", seed=42, display_name="Random-E"),
    ]


def _baseline_substitute(slot: int) -> Agent:
    return [
        RandomAgent("random-A", seed=1, display_name="Random-A"),
        MomentumAgent("momentum-B", display_name="Momentum-B"),
        MeanReversionAgent("meanrev-C", display_name="MeanRev-C"),
        BuyAndHoldAgent("buyhold-D", display_name="BuyHold-D"),
        RandomAgent("random-E", seed=42, display_name="Random-E"),
    ][slot]


def _llm_roster(
    allow_baseline_fallback: bool,
    *,
    world_facts: dict,
) -> list[Agent]:
    agents: list[Agent] = []
    for slot, spec in enumerate(PROVIDERS):
        cfg = llm_config_for(spec, **world_facts)
        cls = PROVIDER_CLASSES[spec.name]
        if cfg is None:
            if allow_baseline_fallback:
                fallback = _baseline_substitute(slot)
                logging.warning("Missing %s key — substituting %s", spec.env_key, fallback.agent_id)
                agents.append(fallback)
                continue
            raise SystemExit(
                f"Missing API key for {spec.name} ({spec.env_key}). "
                f"Set it in .env or use --mode mixed/baseline."
            )
        agent = cls(spec.name, cfg, display_name=spec.name.title())
        agents.append(agent)
    return agents


def build_agents(mode: str, *, world_facts: dict | None = None) -> list[Agent]:
    facts = world_facts or {}
    if mode == "baseline":
        return _baseline_roster()
    if mode == "llm":
        return _llm_roster(allow_baseline_fallback=False, world_facts=facts)
    if mode == "mixed":
        return _llm_roster(allow_baseline_fallback=True, world_facts=facts)
    raise SystemExit(f"unknown mode: {mode}")


def _dump_result(result: WorldResult, path: Path) -> None:
    """Legacy single-file JSON dump (kept for replay.py compatibility)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": asdict(result.config) if is_dataclass(result.config) else result.config,
        "final_price": result.final_price,
        "final_portfolios": [p.model_dump() for p in result.final_portfolios],
        "leaderboard": result.leaderboard(),
        "rounds": [r.model_dump() for r in result.round_reports],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _persist(
    result: WorldResult, agents: list[Agent], config: WorldConfig, args: argparse.Namespace
) -> None:
    """Route the run to the configured sink(s)."""
    from .storage import JsonFileSink, NullSink, RunAgent, RunStart
    from .storage.base import RunSink

    run_id = uuid.uuid4().hex
    meta = RunStart(
        run_id=run_id,
        total_rounds=config.total_rounds,
        initial_coin=config.initial_coin,
        initial_shares=config.initial_shares,
        matching_mode=config.matching_mode,
        amm_coin_reserve=(config.amm_coin_reserve if config.matching_mode == "amm" else None),
        amm_share_reserve=(config.amm_share_reserve if config.matching_mode == "amm" else None),
    )
    agent_rows = [
        RunAgent(
            run_id=run_id,
            agent_id=a.agent_id,
            display_name=a.display_name,
            provider=getattr(a, "provider", "baseline"),
            model=getattr(getattr(a, "config", None), "model", None),
        )
        for a in agents
    ]

    sinks: list[RunSink] = []
    if args.storage in ("json", "both"):
        sinks.append(JsonFileSink())
    if args.storage in ("mysql", "both"):
        try:
            from .storage import MySQLSink

            sinks.append(
                MySQLSink(
                    host=os.environ.get("MYSQL_HOST", "mysql"),
                    port=int(os.environ.get("MYSQL_PORT", "3306")),
                    user=os.environ.get("MYSQL_USER", "arena"),
                    password=os.environ.get("MYSQL_PASSWORD", "arena"),
                    database=os.environ.get("MYSQL_DATABASE", "ai_game_theory"),
                )
            )
        except ImportError:
            print("MySQL storage requested but PyMySQL not installed — skipping.")

    if not sinks:
        sinks.append(NullSink())

    try:
        for sink in sinks:
            sink.start_run(meta, agent_rows)
        for report in result.round_reports:
            for sink in sinks:
                sink.record_round(run_id, report)
        for sink in sinks:
            sink.finish_run(run_id, result.final_price)

        # Legacy single-file dump for backward-compat with src.replay.
        if args.storage in ("json", "both") and args.out is not None:
            _dump_result(result, args.out)

        print(f"Run {run_id} persisted via {[type(s).__name__ for s in sinks]}.")
    finally:
        for sink in sinks:
            sink.close()


def _render_summary(result: WorldResult, agents: list[Agent]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Final Leaderboard")
        table.add_column("Rank", style="bold")
        table.add_column("Agent")
        table.add_column("Coin", justify="right")
        table.add_column("Shares", justify="right")
        table.add_column("Equity", justify="right", style="bold green")

        name_by_id = {a.agent_id: a.display_name for a in agents}
        port_by_id = {p.agent_id: p for p in result.final_portfolios}
        for rank, (aid, equity) in enumerate(result.leaderboard(), start=1):
            p = port_by_id[aid]
            table.add_row(
                str(rank),
                name_by_id.get(aid, aid),
                f"{p.coin:.4f}",
                f"{p.shares}",
                f"{equity:.4f}",
            )
        console.print(table)
        console.print(f"Final price: [bold]{result.final_price:.4f}[/bold] GC")
        console.print("\n[bold]Price trajectory[/bold]")
        console.print(price_sparkline(result.round_reports))
    except ImportError:
        print("Final Leaderboard:")
        for rank, (aid, equity) in enumerate(result.leaderboard(), start=1):
            print(f"  {rank}. {aid}: {equity:.4f}")
        print(f"Final price: {result.final_price:.4f}")
        print("\nPrice trajectory:")
        print(price_sparkline(result.round_reports))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AI Game Theory trading arena.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml. If set, takes precedence over individual CLI flags.",
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--initial-coin", type=float, default=100.0)
    parser.add_argument("--initial-shares", type=int, default=10)
    parser.add_argument("--initial-price", type=float, default=10.0)
    parser.add_argument("--mode", choices=["baseline", "llm", "mixed"], default="baseline")
    parser.add_argument(
        "--matching",
        choices=["call_auction", "amm"],
        default="amm",
        help="Matching engine: amm (Uniswap V2-style, no fees, default) or call_auction",
    )
    parser.add_argument(
        "--amm-coin-reserve",
        type=float,
        default=1000.0,
        help="Initial coin reserve in the AMM pool (matching=amm only)",
    )
    parser.add_argument(
        "--amm-share-reserve",
        type=int,
        default=100,
        help="Initial share reserve in the AMM pool (matching=amm only)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Path to write run log JSON")
    parser.add_argument(
        "--storage",
        choices=["json", "mysql", "both"],
        default="json",
        help="Where to persist the run. mysql/both require MYSQL_* env vars.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    load_env()

    if args.config is not None:
        config, specs = load_config(args.config)
        world_facts = {
            "matching_mode": config.matching_mode,
            "num_traders": len(specs),
            "initial_coin": config.initial_coin,
            "initial_shares": config.initial_shares,
            "total_rounds": config.total_rounds,
            "amm_coin_reserve": config.amm_coin_reserve,
            "amm_share_reserve": config.amm_share_reserve,
        }
        agents = [build_agent(spec, world_facts=world_facts) for spec in specs]
        source = f"config={args.config}"
    else:
        config = WorldConfig(
            initial_coin=args.initial_coin,
            initial_shares=args.initial_shares,
            initial_price=args.initial_price,
            total_rounds=args.rounds,
            matching_mode=args.matching,
            amm_coin_reserve=args.amm_coin_reserve,
            amm_share_reserve=args.amm_share_reserve,
        )
        world_facts = {
            "matching_mode": args.matching,
            "num_traders": 5,
            "initial_coin": args.initial_coin,
            "initial_shares": args.initial_shares,
            "total_rounds": args.rounds,
            "amm_coin_reserve": args.amm_coin_reserve,
            "amm_share_reserve": args.amm_share_reserve,
        }
        agents = build_agents(args.mode, world_facts=world_facts)
        source = f"mode={args.mode}"

    world = World(config=config, agents=agents)

    print(
        f"Starting simulation: {source}, {config.matching_mode} matching, "
        f"{config.total_rounds} rounds, {len(agents)} traders."
    )
    start = time.monotonic()
    result = world.run()
    elapsed = time.monotonic() - start
    print(f"Simulation finished in {elapsed:.2f}s.")

    _persist(result, agents, config, args)

    _render_summary(result, agents)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
