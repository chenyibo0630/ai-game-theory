"""Uniswap V2-style constant-product AMM, fee-free.

The pool holds reserves `(R_c, R_s)` of coin and shares. Spot price is
`R_c / R_s`. Every order trades against the pool — `POOL` is the counterparty
for every fill.

Within a round, orders execute in an order derived from `hash(round_index,
agent_id)`. This breaks the structural advantage that alphabetically-early
agent ids would otherwise enjoy: each round shuffles execution order in a
way agents cannot predict before submission, while still being deterministic
for replay given the round index.

Trade math (no fee)
-------------------
BUY q shares (pool gives up `q` shares, receives `Δ_c` coin):
    constant product:  R_c * R_s = (R_c + Δ_c) * (R_s - q)
    →  Δ_c = q * R_c / (R_s - q)
    →  effective avg price = Δ_c / q = R_c / (R_s - q)

The largest q satisfying the buyer's limit (effective ≤ limit):
    R_c / (R_s - q) ≤ limit   ⟺   q ≤ R_s - R_c / limit

SELL q shares (pool receives `q` shares, gives up `Δ_c` coin):
    R_c * R_s = (R_c - Δ_c) * (R_s + q)
    →  Δ_c = q * R_c / (R_s + q)
    →  effective avg price = R_c / (R_s + q)

The largest q satisfying the seller's limit (effective ≥ limit):
    R_c / (R_s + q) ≥ limit   ⟺   q ≤ R_c / limit - R_s

Fairness
--------
All orders within a round are sorted by `agent_id` and executed sequentially
against the pool. Ordering is deterministic and not influenced by submission
time. No agent can front-run another — the round's orders are revealed
together at match time.
"""

from __future__ import annotations

import hashlib
import math

from ..types import Order, OrderSide, Trade
from .base import POOL_ID, ClearingResult, MatchingEngine


def _order_priority(order: Order) -> tuple:
    """Stable, deterministic per-order key that mixes round_index with
    agent_id so the execution order rotates from one round to the next."""
    digest = hashlib.sha256(
        f"{order.round_index}:{order.agent_id}".encode("utf-8")
    ).digest()
    # Tie-break by side then agent_id when two hash digests happen to collide.
    return (digest, order.side.value, order.agent_id)


class AMMEngine(MatchingEngine):
    name = "amm"

    def __init__(
        self,
        *,
        initial_coin_reserve: float,
        initial_share_reserve: int,
    ) -> None:
        if initial_coin_reserve <= 0:
            raise ValueError("coin reserve must be positive")
        if initial_share_reserve <= 1:
            raise ValueError("share reserve must be at least 2 (need headroom)")
        self.coin_reserve: float = float(initial_coin_reserve)
        self.share_reserve: int = int(initial_share_reserve)
        self._initial_k: float = self.coin_reserve * self.share_reserve

    @property
    def spot_price(self) -> float:
        return self.coin_reserve / self.share_reserve

    def initial_price(self) -> float:
        return self.spot_price

    def pool_state(self) -> tuple[float, int]:
        return (self.coin_reserve, self.share_reserve)

    def clear(self, orders: list[Order], opening_price: float) -> ClearingResult:
        trades: list[Trade] = []
        # Per-round pseudo-random order keyed on (round_index, agent_id). This
        # is deterministic for replay yet rotates execution order across rounds
        # so no agent gets a permanent first-mover advantage.
        ordered = sorted(orders, key=_order_priority)
        for order in ordered:
            trade = self._execute(order)
            if trade is not None:
                trades.append(trade)

        return ClearingResult(
            clearing_price=self.spot_price,
            cleared_volume=sum(t.quantity for t in trades),
            trades=tuple(trades),
            metadata={
                "coin_reserve": self.coin_reserve,
                "share_reserve": self.share_reserve,
                "k": self.coin_reserve * self.share_reserve,
                "k_drift": (self.coin_reserve * self.share_reserve - self._initial_k)
                / self._initial_k,
            },
        )

    # ---- internals ----

    def _execute(self, order: Order) -> Trade | None:
        if order.side == OrderSide.BUY:
            return self._execute_buy(order)
        return self._execute_sell(order)

    def _execute_buy(self, order: Order) -> Trade | None:
        if order.limit_price <= 0:
            return None
        # Largest qty s.t. effective avg price ≤ limit.
        q_by_limit = math.floor(self.share_reserve - self.coin_reserve / order.limit_price)
        # Pool can never drop below 1 share or pricing diverges.
        q_by_pool = self.share_reserve - 1
        qty = min(order.quantity, q_by_limit, q_by_pool)
        if qty <= 0:
            return None

        delta_coin = qty * self.coin_reserve / (self.share_reserve - qty)
        effective = delta_coin / qty
        # Defensive: floating-point may bump us slightly over the limit.
        if effective > order.limit_price + 1e-9:
            return None

        self.coin_reserve += delta_coin
        self.share_reserve -= qty
        return Trade(
            round_index=order.round_index,
            buyer_id=order.agent_id,
            seller_id=POOL_ID,
            quantity=qty,
            price=round(effective, 8),
        )

    def _execute_sell(self, order: Order) -> Trade | None:
        if order.limit_price <= 0:
            return None
        q_by_limit = math.floor(self.coin_reserve / order.limit_price - self.share_reserve)
        qty = min(order.quantity, q_by_limit)
        if qty <= 0:
            return None

        delta_coin = qty * self.coin_reserve / (self.share_reserve + qty)
        effective = delta_coin / qty
        if effective + 1e-9 < order.limit_price:
            return None

        self.coin_reserve -= delta_coin
        self.share_reserve += qty
        return Trade(
            round_index=order.round_index,
            buyer_id=POOL_ID,
            seller_id=order.agent_id,
            quantity=qty,
            price=round(effective, 8),
        )
