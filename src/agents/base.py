"""Agent base interface and the public information passed in each round.

Information an agent is allowed to see:

- Per-round system message: round number, own cash balance, own share count,
  current price.
- Own trade history (every fill from past rounds — quantity, price, side).
- Recent public price history (not other-agent specific; derivable from
  observing the past).

Information an agent must NOT see:

- Any other agent's orders, holdings, decisions, identity, or rationale.

This is the fairness contract from the world spec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..arena.types import AgentDecision, PortfolioState, Trade


@dataclass(frozen=True)
class MarketView:
    """Everything an agent is allowed to know before making its decision.

    `pool_reserves` is the live ``(coin_reserve, share_reserve)`` of the AMM
    pool. It's public information — every agent sees the same numbers — but
    it's only present when the world is running with an AMM-style engine; for
    order-book engines (call auction) it stays ``None``.
    """

    round_index: int
    total_rounds: int
    current_price: float
    price_history: tuple[float, ...]
    portfolio: PortfolioState
    pool_reserves: tuple[float, int] | None = None

    @property
    def rounds_remaining(self) -> int:
        return self.total_rounds - self.round_index


class Agent(ABC):
    """Abstract trader. One instance per participant per simulation."""

    def __init__(self, agent_id: str, display_name: str | None = None) -> None:
        self.agent_id = agent_id
        self.display_name = display_name or agent_id

    @abstractmethod
    def decide(self, view: MarketView) -> AgentDecision:
        """Return the agent's chosen action for the upcoming round.

        Implementations MUST be deterministic w.r.t. their own seed/state — the
        World is responsible for the outer event loop; agents only respond to
        their MarketView.
        """

    def on_round_end(
        self,
        view: MarketView,
        decision: AgentDecision,
        fills: tuple[Trade, ...] = (),
    ) -> None:  # noqa: D401
        """Optional hook: called after a round closes.

        Arguments:
            view: portfolio + price *after* the round cleared.
            decision: what the agent submitted earlier this round.
            fills: trades the agent actually participated in this round
                   (possibly empty if order was rejected, partial, or not
                   crossed). For call_auction this is at most one side; for
                   AMM the counterparty is the POOL sentinel.
        """
