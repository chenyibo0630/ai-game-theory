"""Inspect a saved run log: `python -m src.replay runs/run-xxx.json`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _format_action(o: dict) -> str:
    return f"{o['side']} {o['quantity']} @ {o['limit_price']:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a saved arena run.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--rounds", type=int, default=10, help="Show first/last N rounds")
    parser.add_argument("--full", action="store_true", help="Show every round")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"file not found: {args.path}", file=sys.stderr)
        return 1

    log = json.loads(args.path.read_text(encoding="utf-8"))
    rounds = log["rounds"]
    print(f"Run: {args.path}  ({len(rounds)} rounds, final price {log['final_price']:.4f})")
    print(f"Config: {log['config']}\n")

    print("Leaderboard:")
    for rank, (aid, equity) in enumerate(log["leaderboard"], start=1):
        print(f"  {rank}. {aid:20s} equity={equity:.4f}")
    print()

    selected = rounds if args.full else rounds[: args.rounds] + rounds[-args.rounds :]
    seen_indices = set()
    for r in selected:
        if r["round_index"] in seen_indices:
            continue
        seen_indices.add(r["round_index"])
        print(
            f"R{r['round_index']:3d}: open={r['opening_price']:.4f} "
            f"clear={r['clearing_price']:.4f} vol={r['cleared_volume']}"
        )
        if r["orders"]:
            for o in r["orders"]:
                rationale = r["rationales"].get(o["agent_id"], "")
                print(f"    {o['agent_id']:14s} {_format_action(o)}  — {rationale}")
        if r["trades"]:
            for t in r["trades"]:
                print(f"    TRADE {t['buyer_id']} ← {t['seller_id']}  qty={t['quantity']} @ {t['price']:.4f}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
