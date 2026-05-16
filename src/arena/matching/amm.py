"""Uniswap V2-style AMM, sequential per-order clearing against a shared pool.

Within a round, orders are applied one at a time against the current pool
state. Each fill exactly satisfies the constant product invariant (modulo
discrete-share rounding), so ``k = R_c · R_s`` is preserved trade-by-trade —
there is no "batch drift".

Execution order is a two-level sort:

1. **Priority agents go first** (``priority_agent_ids`` — typically the mock
   baselines). When present, they always trade before any other agent, so
   they act as a deterministic noise source: their pool movement is visible
   in the ``price_before`` field of every other agent's decision record.
2. **Within each tier**, orders sort by
   ``sha256("{round_index}:{agent_id}").digest()`` — a permutation seeded
   by the round index. Properties:

   - **Reproducible**: same round_index and same agents produce the same
     order, so replays of a saved run are bit-identical.
   - **Fair across rounds**: each agent's position within its tier is
     uniformly distributed, so the small "earlier-mover gets a better fill
     on same-side cohort" advantage averages out instead of persistently
     favouring lex-early ids.

For each order, against the pool's CURRENT ``(R_c, R_s)``:

    BUY  q shares: effective price = R_c / (R_s − q)
        - max q satisfying limit: q_max = floor(R_s − R_c / limit_price)
        - capped by pool capacity (must leave at least 1 share in the pool)
        - capped by the order's requested quantity
        - if q_max > 0 the order fills q_max (full fill when q_max ≥ qty,
          partial fill otherwise); if q_max ≤ 0 the order is dropped this round

    SELL q shares: effective price = R_c / (R_s + q)
        - max q satisfying limit: q_max = floor(R_c / limit_price − R_s)
        - capped by the order's requested quantity
        - same drop rule when q_max ≤ 0

After a fill the pool is updated with the exact amounts:

    BUY:  R_c ← R_c + Δ_c,  R_s ← R_s − q   (Δ_c = q · R_c / (R_s − q))
    SELL: R_c ← R_c − Δ_c,  R_s ← R_s + q   (Δ_c = q · R_c / (R_s + q))

The clearing_price reported back to the world is the post-round spot price
(``R_c / R_s``), used as the next round's opening price and pushed into the
public price history.

Ordering and fairness
---------------------
Same-side cohorts pay strictly worse prices the later they appear in the
``(agent_id, side)`` lex order: each preceding buy pushes the pool's R_c/R_s
up, raising the next buyer's effective price. Because agent roles (buyer vs
seller, large vs small qty) flip round-to-round, the positional advantage of a
lex-early id is small and averages out over many rounds. A future enhancement
could seed a per-round permutation if stricter fairness is needed.
"""

from __future__ import annotations

import hashlib
import math

from ..types import Order, OrderSide, Trade
from .base import POOL_ID, ClearingResult, MatchingEngine


def _hash_key(order: Order) -> bytes:
    """Per-(round, agent) hash that rotates fairly across rounds."""
    return hashlib.sha256(f"{order.round_index}:{order.agent_id}".encode()).digest()

# Pool keeps at least 1 share so the buy-price formula stays finite.
_MIN_POOL_SHARES = 1


class AMMEngine(MatchingEngine):
    name = "amm"

    def __init__(
        self,
        *,
        initial_coin_reserve: float,
        initial_share_reserve: int,
        priority_agent_ids: frozenset[str] | set[str] | None = None,
    ) -> None:
        if initial_coin_reserve <= 0:
            raise ValueError("coin reserve must be positive")
        if initial_share_reserve <= 1:
            raise ValueError("share reserve must be at least 2 (need headroom)")
        self.coin_reserve: float = float(initial_coin_reserve)
        self.share_reserve: int = int(initial_share_reserve)
        self._initial_k: float = self.coin_reserve * self.share_reserve
        # Agents in this set always sort before non-priority agents within a
        # round; among themselves they still rotate by the per-round hash.
        self.priority_agent_ids: frozenset[str] = frozenset(priority_agent_ids or ())

    def _sort_key(self, order: Order) -> tuple[int, bytes]:
        tier = 0 if order.agent_id in self.priority_agent_ids else 1
        return (tier, _hash_key(order))

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

        # Two-tier sort: priority agents first (mock baselines as noise
        # source), everyone else after; within each tier sort by per-round
        # SHA-256 hash so positions rotate fairly across rounds.
        ordered = sorted(orders, key=self._sort_key)

        trades: list[Trade] = []
        cleared_volume = 0
        decision_prices: dict[str, tuple[float, float]] = {}
        for order in ordered:
            price_before = self.spot_price
            q_filled, fill_price = self._fill_one(order)
            price_after = self.spot_price
            decision_prices[order.agent_id] = (price_before, price_after)
            if q_filled <= 0:
                continue
            cleared_volume += q_filled
            if order.side == OrderSide.BUY:
                trades.append(
                    Trade(
                        round_index=order.round_index,
                        buyer_id=order.agent_id,
                        seller_id=POOL_ID,
                        quantity=q_filled,
                        price=round(fill_price, 8),
                    )
                )
            else:
                trades.append(
                    Trade(
                        round_index=order.round_index,
                        buyer_id=POOL_ID,
                        seller_id=order.agent_id,
                        quantity=q_filled,
                        price=round(fill_price, 8),
                    )
                )

        return ClearingResult(
            clearing_price=self.spot_price,
            cleared_volume=cleared_volume,
            trades=tuple(trades),
            metadata=self._metadata(),
            decision_prices=decision_prices,
        )

    # ---- helpers ----

    def _fill_one(self, order: Order) -> tuple[int, float]:
        """Compute the (quantity, effective_price) for `order` against the
        current pool state, mutate the pool, and return the fill. Returns
        (0, 0.0) when nothing executes."""
        r_c, r_s = self.coin_reserve, self.share_reserve

        if order.side == OrderSide.BUY:
            # q must satisfy R_c / (R_s − q) ≤ limit_price
            # ⟺ q ≤ R_s − R_c / limit_price
            q_max_by_limit = math.floor(r_s - r_c / order.limit_price)
            # Pool capacity: keep at least _MIN_POOL_SHARES inside.
            q_max_by_pool = r_s - _MIN_POOL_SHARES
            q = min(order.quantity, q_max_by_limit, q_max_by_pool)
            if q <= 0:
                return (0, 0.0)
            delta_c = q * r_c / (r_s - q)
            self.coin_reserve = r_c + delta_c
            self.share_reserve = r_s - q
            return (q, delta_c / q)

        # SELL: q satisfies R_c / (R_s + q) ≥ limit_price
        # ⟺ q ≤ R_c / limit_price − R_s
        q_max_by_limit = math.floor(r_c / order.limit_price - r_s)
        q = min(order.quantity, q_max_by_limit)
        if q <= 0:
            return (0, 0.0)
        delta_c = q * r_c / (r_s + q)
        self.coin_reserve = r_c - delta_c
        self.share_reserve = r_s + q
        return (q, delta_c / q)

    def _metadata(self) -> dict:
        return {
            "coin_reserve": self.coin_reserve,
            "share_reserve": self.share_reserve,
            "k": self.coin_reserve * self.share_reserve,
            "k_drift": (self.coin_reserve * self.share_reserve - self._initial_k)
            / self._initial_k,
        }
