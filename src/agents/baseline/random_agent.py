"""Coin-flip baseline: useful for sanity-checking the matching engine."""

from __future__ import annotations

import random

from ...arena.types import AgentDecision
from ..base import Agent, MarketView


class RandomAgent(Agent):
    def __init__(self, agent_id: str, *, seed: int = 0, display_name: str | None = None) -> None:
        super().__init__(agent_id, display_name)
        self._rng = random.Random(seed)

    def decide(self, view: MarketView) -> AgentDecision:
        roll = self._rng.random()
        price = view.current_price

        if roll < 0.33 and view.portfolio.coin >= price:
            qty = max(1, int(view.portfolio.coin // price // 4))
            limit = price * self._rng.uniform(1.0, 1.05)
            return AgentDecision(
                action="BUY",
                quantity=qty,
                limit_price=round(limit, 4),
                rationale="random buy",
            )

        if roll < 0.66 and view.portfolio.shares > 0:
            qty = max(1, view.portfolio.shares // 4)
            limit = price * self._rng.uniform(0.95, 1.0)
            return AgentDecision(
                action="SELL",
                quantity=qty,
                limit_price=round(limit, 4),
                rationale="random sell",
            )

        return AgentDecision(action="HOLD", rationale="random hold")
