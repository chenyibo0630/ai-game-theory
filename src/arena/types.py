"""Shared immutable types for the trading arena.

All models are pydantic v2 BaseModels with `model_config = {"frozen": True}` so
they behave as value objects — copy on change rather than mutate. This follows
the immutability principle in the user's coding-style rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------- enums & primitives ----------


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


Action = Literal["BUY", "SELL", "HOLD"]


# ---------- decision / order / trade ----------


class AgentDecision(BaseModel):
    """The raw decision an Agent returns each round.

    `action == "HOLD"` means no order. Otherwise `quantity` and `limit_price`
    define a limit order. The World converts this into an :class:`Order`.
    """

    model_config = ConfigDict(frozen=True)

    action: Action
    quantity: int = 0
    limit_price: float = 0.0
    rationale: str = ""

    def is_trade(self) -> bool:
        return self.action != "HOLD" and self.quantity > 0


class Order(BaseModel):
    """A submitted order in a single round's call auction."""

    model_config = ConfigDict(frozen=True)

    round_index: int
    agent_id: str
    side: OrderSide
    quantity: int = Field(gt=0)
    limit_price: float = Field(gt=0)


class Trade(BaseModel):
    """One executed fill produced by the call auction."""

    model_config = ConfigDict(frozen=True)

    round_index: int
    buyer_id: str
    seller_id: str
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)

    @property
    def notional(self) -> float:
        return self.quantity * self.price


# ---------- state ----------


class PortfolioState(BaseModel):
    """A snapshot of one agent's holdings."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    coin: float
    shares: int

    def equity(self, mark_price: float) -> float:
        return self.coin + self.shares * mark_price

    def credit(self, *, coin: float = 0.0, shares: int = 0) -> "PortfolioState":
        return self.model_copy(update={"coin": self.coin + coin, "shares": self.shares + shares})


class PriceTick(BaseModel):
    """Per-round price record."""

    model_config = ConfigDict(frozen=True)

    round_index: int
    price: float
    volume: int = 0


class RoundReport(BaseModel):
    """Full record of a single round, suitable for replay / analysis."""

    model_config = ConfigDict(frozen=True)

    round_index: int
    opening_price: float
    clearing_price: float
    cleared_volume: int
    orders: list[Order] = Field(default_factory=list)
    trades: list[Trade] = Field(default_factory=list)
    portfolios: list[PortfolioState] = Field(default_factory=list)
    rationales: dict[str, str] = Field(default_factory=dict)
    # Post-trade AMM pool state, when the engine is AMM. None under
    # call_auction (which has no shared liquidity pool).
    pool_coin: float | None = None
    pool_shares: int | None = None
    # Per-agent market price BEFORE and AFTER applying that agent's order.
    # Defaulted at the world layer to (opening_price, opening_price) for
    # HOLD or no-order agents so every decision row has both values.
    decision_prices: dict[str, tuple[float, float]] = Field(default_factory=dict)
