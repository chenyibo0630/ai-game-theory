"""Momentum: buys after price rises, sells after price falls."""

from __future__ import annotations

from ...arena.types import AgentDecision
from ..base import Agent, MarketView


class MomentumAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        *,
        lookback: int = 5,
        aggression: float = 0.02,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, display_name)
        self.lookback = lookback
        self.aggression = aggression

    def decide(self, view: MarketView) -> AgentDecision:
        history = view.price_history[-self.lookback :]
        if len(history) < 2:
            return AgentDecision(action="HOLD", rationale="not enough history")

        start, end = history[0], history[-1]
        if start == 0:
            return AgentDecision(action="HOLD", rationale="degenerate history")
        change = (end - start) / start
        price = view.current_price

        if change > self.aggression and view.portfolio.coin >= price:
            qty = max(1, int(view.portfolio.coin // price // 3))
            return AgentDecision(
                action="BUY",
                quantity=qty,
                limit_price=round(price * 1.02, 4),
                rationale=f"momentum up {change:.2%}",
            )

        if change < -self.aggression and view.portfolio.shares > 0:
            qty = max(1, view.portfolio.shares // 3)
            return AgentDecision(
                action="SELL",
                quantity=qty,
                limit_price=round(price * 0.98, 4),
                rationale=f"momentum down {change:.2%}",
            )

        return AgentDecision(action="HOLD", rationale=f"momentum flat {change:.2%}")
