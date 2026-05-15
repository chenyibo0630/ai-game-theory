"""Common contract every matching engine has to satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..types import Order, Trade

POOL_ID: str = "POOL"
"""Sentinel agent_id used by the AMM as the counterparty for every trade.

The World skips portfolio updates for this id so the pool's bookkeeping is
self-contained inside the engine.
"""


@dataclass(frozen=True)
class ClearingResult:
    """Output of a single round of matching."""

    clearing_price: float
    """Price used to mark portfolios after the round. For call auction this is
    the single fill price; for AMM this is the post-round spot price."""

    cleared_volume: int
    """Total shares that changed hands this round."""

    trades: tuple[Trade, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form engine-specific info (e.g. AMM reserves)."""


class MatchingEngine(ABC):
    """One round of matching: orders in, clearing result out.

    Engines may hold internal state between rounds (e.g. AMM pool reserves).
    They MUST be deterministic for replay: given the same prior state and the
    same orders, repeated calls must produce the same trades.
    """

    name: str = "matching"

    def initial_price(self) -> float | None:
        """Return the engine's intrinsic initial price, or None if the World
        should fall back to :attr:`WorldConfig.initial_price`.

        Default: None (most engines don't dictate an opening price).
        """
        return None

    def pool_state(self) -> tuple[float, int] | None:
        """Return public (coin_reserve, share_reserve) for AMM-style engines.

        Order-book engines have no shared pool — they return ``None``. The
        World surfaces this to each agent so it can compute precise slippage.
        """
        return None

    @abstractmethod
    def clear(self, orders: list[Order], opening_price: float) -> ClearingResult:
        """Process one round's orders and return the resulting trades."""
