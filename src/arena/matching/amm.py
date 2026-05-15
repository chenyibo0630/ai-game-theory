"""Uniswap V2-style AMM, batch-cleared with uniform pricing and all-or-nothing fills.

Within a round there is NO order-of-execution. All orders are batched and
cleared at one uniform price ``P*``, derived ONCE from the pool's snapshot at
the start of the round and the net flow of all submitted orders.

Single-shot algorithm (no iteration)
------------------------------------
1. Δ_all = Σ buy_qty − Σ sell_qty                (over ALL submitted orders)
2. P*    = R_c / (R_s − Δ_all)                   (the post-trade pool spot
                                                  if every order filled)
3. Filter ONCE — keep only orders whose limit accepts P*:
       BUY  with limit_price ≥ P*
       SELL with limit_price ≤ P*
4. Surviving orders fill their FULL quantity at the uniform price P* (no
   partial fills). The pool then settles the SURVIVING net flow:
       R_c ← R_c + Δ_kept · P*
       R_s ← R_s − Δ_kept

Why no iteration
----------------
Sealed-bid call auctions compute ONE clearing price from the initial book.
Re-pricing after dropping orders would let some agents be passively squeezed
out: they accepted P*₁ at submission time, but a recomputed P*₂ silently
disqualifies them. Sealed bids must be honoured at the price they were
priced against. The cost is a small drift in the AMM's ``k`` invariant when
the surviving net flow ≠ submitted net flow — bounded by the dropped
orders, usually well under 1%.
"""

from __future__ import annotations

from ..types import Order, OrderSide, Trade
from .base import POOL_ID, ClearingResult, MatchingEngine

# Pool keeps at least 1 share so the price formula stays finite.
_MIN_POOL_SHARES = 1


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
        if not orders:
            return ClearingResult(
                clearing_price=self.spot_price,
                cleared_volume=0,
                trades=(),
                metadata=self._metadata(),
            )

        # 1. Net flow against pool from EVERY submitted order.
        net_q_all = self._net_flow(orders)

        # 2. Capacity guard: pool can release at most (R_s - 1) shares.
        if net_q_all >= self.share_reserve - _MIN_POOL_SHARES:
            # Net buy too large to clear without draining the pool. Round
            # voided — every submitted order rejected.
            return ClearingResult(
                clearing_price=self.spot_price,
                cleared_volume=0,
                trades=(),
                metadata=self._metadata(reason="pool_capacity"),
            )

        # 3. Single clearing price computed from the initial pool snapshot
        #    + the full submitted net flow.
        clearing_price = self.coin_reserve / (self.share_reserve - net_q_all)

        # 4. One-pass filter: orders whose limit accepts P*.
        survivors = [o for o in orders if _satisfies(o, clearing_price)]
        if not survivors:
            return ClearingResult(
                clearing_price=clearing_price,
                cleared_volume=0,
                trades=(),
                metadata=self._metadata(uniform_price=clearing_price),
            )

        net_q_kept = self._net_flow(survivors)
        # Settle the surviving net flow against the pool at the uniform P*.
        # When net_q_kept != net_q_all, k drifts slightly — by design.
        self.coin_reserve += net_q_kept * clearing_price
        self.share_reserve -= net_q_kept

        # 5. Emit one Trade per surviving order, all at P*.
        trades: list[Trade] = []
        for o in sorted(survivors, key=lambda x: x.agent_id):
            if o.side == OrderSide.BUY:
                trades.append(
                    Trade(
                        round_index=o.round_index,
                        buyer_id=o.agent_id,
                        seller_id=POOL_ID,
                        quantity=o.quantity,
                        price=round(clearing_price, 8),
                    )
                )
            else:
                trades.append(
                    Trade(
                        round_index=o.round_index,
                        buyer_id=POOL_ID,
                        seller_id=o.agent_id,
                        quantity=o.quantity,
                        price=round(clearing_price, 8),
                    )
                )

        return ClearingResult(
            clearing_price=self.spot_price,
            cleared_volume=sum(o.quantity for o in survivors),
            trades=tuple(trades),
            metadata=self._metadata(
                uniform_price=clearing_price,
                net_flow_submitted=net_q_all,
                net_flow_settled=net_q_kept,
            ),
        )

    # ---- helpers ----

    @staticmethod
    def _net_flow(orders: list[Order]) -> int:
        buys = sum(o.quantity for o in orders if o.side == OrderSide.BUY)
        sells = sum(o.quantity for o in orders if o.side == OrderSide.SELL)
        return buys - sells

    def _metadata(
        self,
        *,
        uniform_price: float | None = None,
        net_flow_submitted: int = 0,
        net_flow_settled: int = 0,
        reason: str | None = None,
    ) -> dict:
        md = {
            "coin_reserve": self.coin_reserve,
            "share_reserve": self.share_reserve,
            "k": self.coin_reserve * self.share_reserve,
            "k_drift": (self.coin_reserve * self.share_reserve - self._initial_k)
            / self._initial_k,
            "net_flow_submitted": net_flow_submitted,
            "net_flow_settled": net_flow_settled,
        }
        if uniform_price is not None:
            md["uniform_price"] = uniform_price
        if reason is not None:
            md["reason"] = reason
        return md


def _satisfies(order: Order, price: float) -> bool:
    """A buy accepts price ≤ its limit; a sell accepts price ≥ its limit."""
    if order.side == OrderSide.BUY:
        return order.limit_price + 1e-9 >= price
    return order.limit_price - 1e-9 <= price
