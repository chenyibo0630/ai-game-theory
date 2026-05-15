"""Always-active random mock trader.

Unlike :class:`RandomAgent` (which can also HOLD), this agent picks BUY or
SELL with equal probability every round and trades a random fraction of
its budget. It exists to inject deterministic-but-non-trivial flow into
otherwise-frozen LLM markets, breaking the all-HOLD equilibrium so other
agents can see a non-flat price history to react to.

Limit prices are deliberately loose (within ±20% of spot) so the order
almost always survives the AMM's all-or-nothing filter.
"""

from __future__ import annotations

import random

from ...arena.types import AgentDecision
from ..base import Agent, MarketView


class MockAgent(Agent):
    """Random buy/sell every round, no HOLD."""

    def __init__(
        self,
        agent_id: str,
        *,
        seed: int | None = None,
        max_fraction: float = 0.4,
        price_slack: float = 0.20,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, display_name)
        # ``seed=None`` → non-deterministic across runs (fresh entropy).
        self._rng = random.Random(seed)
        if not (0.0 < max_fraction <= 1.0):
            raise ValueError("max_fraction must be in (0, 1]")
        if price_slack < 0.0:
            raise ValueError("price_slack must be non-negative")
        self._max_fraction = max_fraction
        self._price_slack = price_slack

    def decide(self, view: MarketView) -> AgentDecision:
        spot = view.current_price
        if spot <= 0:
            return AgentDecision(action="HOLD", rationale="non-positive spot")

        wants_buy = self._rng.random() < 0.5
        fraction = self._rng.uniform(0.05, self._max_fraction)

        if wants_buy:
            return self._make_buy(view, spot, fraction)
        return self._make_sell(view, spot, fraction)

    def _make_buy(self, view: MarketView, spot: float, fraction: float) -> AgentDecision:
        # Affordable share count is bounded by cash / (spot · 1+slack) so the
        # order won't be clamped down to zero by the budget guard upstream.
        ceiling_price = spot * (1.0 + self._price_slack)
        max_affordable = int(view.portfolio.coin // ceiling_price)
        if max_affordable <= 0:
            # Out of cash for a BUY at the loose price; flip to SELL if possible.
            if view.portfolio.shares > 0:
                return self._make_sell(view, spot, fraction)
            return AgentDecision(action="HOLD", rationale="no cash, no shares")
        qty = max(1, int(max_affordable * fraction))
        limit = round(spot * (1.0 + self._price_slack), 4)
        return AgentDecision(
            action="BUY",
            quantity=qty,
            limit_price=limit,
            rationale=f"mock buy q={qty} @≤{limit}",
        )

    def _make_sell(self, view: MarketView, spot: float, fraction: float) -> AgentDecision:
        if view.portfolio.shares <= 0:
            if view.portfolio.coin > spot:
                return self._make_buy(view, spot, fraction)
            return AgentDecision(action="HOLD", rationale="no shares, no cash")
        qty = max(1, int(view.portfolio.shares * fraction))
        floor_price = max(0.01, spot * (1.0 - self._price_slack))
        limit = round(floor_price, 4)
        return AgentDecision(
            action="SELL",
            quantity=qty,
            limit_price=limit,
            rationale=f"mock sell q={qty} @≥{limit}",
        )
