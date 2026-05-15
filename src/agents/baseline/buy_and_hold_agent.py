"""Spend half the cash in the first round, then hold forever."""

from __future__ import annotations

from ...arena.types import AgentDecision
from ..base import Agent, MarketView


class BuyAndHoldAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        *,
        deploy_round: int = 0,
        deploy_fraction: float = 0.5,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, display_name)
        self.deploy_round = deploy_round
        self.deploy_fraction = deploy_fraction
        self._deployed = False

    def decide(self, view: MarketView) -> AgentDecision:
        if self._deployed or view.round_index < self.deploy_round:
            return AgentDecision(action="HOLD", rationale="holding")

        price = view.current_price
        if view.portfolio.coin < price:
            return AgentDecision(action="HOLD", rationale="too poor")

        budget = view.portfolio.coin * self.deploy_fraction
        qty = max(1, int(budget // price))
        self._deployed = True
        return AgentDecision(
            action="BUY",
            quantity=qty,
            limit_price=round(price * 1.05, 4),
            rationale="initial deployment",
        )
