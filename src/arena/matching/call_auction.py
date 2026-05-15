"""Call-auction matching engine, all-or-nothing fills.

A single clearing price P* is computed each round that maximises traded
volume. Orders that cross at P* are eligible to fill. Fills are
**all-or-nothing**: an order either trades its full quantity or not at all.

If the eligible buy-side total ≠ eligible sell-side total at P*, the larger
side is iteratively trimmed (lowest priority first) until the two sides
match exactly. Lower-priority for buys = lower limit_price; for sells =
higher limit_price; ties broken by agent_id.
"""

from __future__ import annotations

from ..types import Order, OrderSide, Trade
from .base import ClearingResult, MatchingEngine


def _candidate_prices(orders: list[Order], fallback_price: float) -> list[float]:
    """All distinct limit prices, plus the fallback (last clearing/opening price)."""
    prices = {o.limit_price for o in orders}
    prices.add(fallback_price)
    return sorted(prices)


def _demand_at(price: float, buys: list[Order]) -> int:
    """Total shares buyers want at `price` (limit >= price)."""
    return sum(o.quantity for o in buys if o.limit_price >= price)


def _supply_at(price: float, sells: list[Order]) -> int:
    """Total shares sellers offer at `price` (limit <= price)."""
    return sum(o.quantity for o in sells if o.limit_price <= price)


class CallAuctionExchange(MatchingEngine):
    """Stateless matching engine — clears one round of orders at a time."""

    name = "call_auction"

    def clear(self, orders: list[Order], opening_price: float) -> ClearingResult:
        buys = [o for o in orders if o.side == OrderSide.BUY]
        sells = [o for o in orders if o.side == OrderSide.SELL]

        if not buys or not sells:
            return ClearingResult(clearing_price=opening_price, cleared_volume=0, trades=())

        candidates = _candidate_prices(orders, opening_price)
        best_volume = 0
        best_imbalance = float("inf")
        best_anchor_distance = float("inf")
        best_price = opening_price

        for price in candidates:
            d = _demand_at(price, buys)
            s = _supply_at(price, sells)
            volume = min(d, s)
            if volume <= 0:
                continue
            imbalance = abs(d - s)
            anchor_dist = abs(price - opening_price)
            better = (
                volume > best_volume
                or (volume == best_volume and imbalance < best_imbalance)
                or (
                    volume == best_volume
                    and imbalance == best_imbalance
                    and anchor_dist < best_anchor_distance
                )
            )
            if better:
                best_volume = volume
                best_imbalance = imbalance
                best_anchor_distance = anchor_dist
                best_price = price

        if best_volume == 0:
            return ClearingResult(clearing_price=opening_price, cleared_volume=0, trades=())

        eligible_buys = [o for o in buys if o.limit_price >= best_price]
        eligible_sells = [o for o in sells if o.limit_price <= best_price]
        cleared_buys, cleared_sells = _balance_aon(
            eligible_buys, eligible_sells, price=best_price
        )
        if not cleared_buys or not cleared_sells:
            return ClearingResult(clearing_price=best_price, cleared_volume=0, trades=())

        trades = _pair(cleared_buys, cleared_sells, price=best_price)
        cleared_volume = sum(t.quantity for t in trades)
        return ClearingResult(
            clearing_price=best_price,
            cleared_volume=cleared_volume,
            trades=tuple(trades),
        )


def _balance_aon(
    buys: list[Order],
    sells: list[Order],
    *,
    price: float,
) -> tuple[list[Order], list[Order]]:
    """Trim the larger side until Σ buy_qty == Σ sell_qty.

    Priority for KEEPING orders: buys with the highest limit fill first
    (most aggressive), sells with the lowest limit fill first; ties broken
    by agent_id. The lowest-priority orders on the over-supplied side are
    dropped one at a time until totals match.
    """
    buys_sorted = sorted(buys, key=lambda o: (-o.limit_price, o.agent_id))
    sells_sorted = sorted(sells, key=lambda o: (o.limit_price, o.agent_id))

    while True:
        bq = sum(o.quantity for o in buys_sorted)
        sq = sum(o.quantity for o in sells_sorted)
        if bq == sq:
            return buys_sorted, sells_sorted
        if bq == 0 or sq == 0:
            return [], []
        if bq > sq:
            # Drop the lowest-priority buy (last after sort).
            buys_sorted = buys_sorted[:-1]
        else:
            sells_sorted = sells_sorted[:-1]


def _pair(
    buys: list[Order],
    sells: list[Order],
    *,
    price: float,
) -> list[Trade]:
    """Walk both lists in lock-step. Since AON guarantees Σbuys == Σsells
    after _balance_aon, every fill quantity is fully accounted for."""
    trades: list[Trade] = []
    buy_iter = iter(sorted(buys, key=lambda o: o.agent_id))
    sell_iter = iter(sorted(sells, key=lambda o: o.agent_id))
    cur_buy = next(buy_iter, None)
    cur_sell = next(sell_iter, None)
    buy_remaining = cur_buy.quantity if cur_buy else 0
    sell_remaining = cur_sell.quantity if cur_sell else 0

    while cur_buy and cur_sell:
        if buy_remaining == 0:
            cur_buy = next(buy_iter, None)
            buy_remaining = cur_buy.quantity if cur_buy else 0
            continue
        if sell_remaining == 0:
            cur_sell = next(sell_iter, None)
            sell_remaining = cur_sell.quantity if cur_sell else 0
            continue

        qty = min(buy_remaining, sell_remaining)
        trades.append(
            Trade(
                round_index=cur_buy.round_index,
                buyer_id=cur_buy.agent_id,
                seller_id=cur_sell.agent_id,
                quantity=qty,
                price=price,
            )
        )
        buy_remaining -= qty
        sell_remaining -= qty

    return trades
