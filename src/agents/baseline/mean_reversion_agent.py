"""Mean-reversion: fades extreme moves toward the lookback average."""

from __future__ import annotations

from ...arena.types import AgentDecision
from ..base import Agent, MarketView


class MeanReversionAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        *,
        lookback: int = 10,
        threshold: float = 0.03,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, display_name)
        self.lookback = lookback
        self.threshold = threshold

    def decide(self, view: MarketView) -> AgentDecision:
        history = view.price_history[-self.lookback :]
        if len(history) < 3:
            return AgentDecision(action="HOLD", rationale="not enough history")

        avg = sum(history) / len(history)
        price = view.current_price
        if avg == 0:
            return AgentDecision(action="HOLD", rationale="degenerate avg")

        deviation = (price - avg) / avg

        if deviation < -self.threshold and view.portfolio.coin >= price:
            qty = max(1, int(view.portfolio.coin // price // 3))
            return AgentDecision(
                action="BUY",
                quantity=qty,
                limit_price=round(price * 1.01, 4),
                rationale=f"buy dip {deviation:.2%}",
            )

        if deviation > self.threshold and view.portfolio.shares > 0:
            qty = max(1, view.portfolio.shares // 3)
            return AgentDecision(
                action="SELL",
                quantity=qty,
                limit_price=round(price * 0.99, 4),
                rationale=f"sell rip {deviation:.2%}",
            )

        return AgentDecision(action="HOLD", rationale=f"in range {deviation:.2%}")
