"""Call-auction matching engine.

A single clearing price is computed each round that maximizes traded volume.
Orders strictly better than the clearing price all execute. Orders exactly at
the clearing price split the remaining capacity pro-rata.

This is the auction mechanism real exchanges use at open/close — it's a clean
fit for simultaneous-move agent games because there's no time priority within
a round and every winner pays the same price.
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

        cleared_buys = _allocate(buys, side=OrderSide.BUY, price=best_price, capacity=best_volume)
        cleared_sells = _allocate(sells, side=OrderSide.SELL, price=best_price, capacity=best_volume)
        trades = _pair(cleared_buys, cleared_sells, price=best_price)

        return ClearingResult(
            clearing_price=best_price,
            cleared_volume=best_volume,
            trades=tuple(trades),
        )


def _allocate(
    orders: list[Order],
    *,
    side: OrderSide,
    price: float,
    capacity: int,
) -> list[tuple[Order, int]]:
    """Return [(order, filled_qty), ...] summing to `capacity`.

    Strictly-better orders (BUY limit > P, SELL limit < P) get filled in full
    first. Marginal orders at limit == P split the remaining capacity pro-rata,
    with any rounding remainder going to lowest agent_id for determinism.
    """
    if side == OrderSide.BUY:
        in_the_money = [o for o in orders if o.limit_price > price]
        marginal = [o for o in orders if o.limit_price == price]
    else:
        in_the_money = [o for o in orders if o.limit_price < price]
        marginal = [o for o in orders if o.limit_price == price]

    in_the_money.sort(key=lambda o: o.agent_id)
    marginal.sort(key=lambda o: o.agent_id)

    fills: list[tuple[Order, int]] = []
    remaining = capacity
    for o in in_the_money:
        take = min(o.quantity, remaining)
        if take > 0:
            fills.append((o, take))
            remaining -= take
        if remaining == 0:
            return fills

    if remaining == 0 or not marginal:
        return fills

    total_marginal = sum(o.quantity for o in marginal)
    if total_marginal == 0:
        return fills

    assigned: list[tuple[Order, int]] = []
    floor_total = 0
    for o in marginal:
        share = (o.quantity * remaining) // total_marginal
        assigned.append((o, share))
        floor_total += share

    leftover = remaining - floor_total
    i = 0
    while leftover > 0 and i < len(assigned):
        order, qty = assigned[i]
        if qty < order.quantity:
            assigned[i] = (order, qty + 1)
            leftover -= 1
        i += 1
        if i == len(assigned) and leftover > 0:
            i = 0

    fills.extend([(o, q) for o, q in assigned if q > 0])
    return fills


def _pair(
    buys: list[tuple[Order, int]],
    sells: list[tuple[Order, int]],
    *,
    price: float,
) -> list[Trade]:
    """Walk buy/sell lists in lock-step, producing Trade records."""
    trades: list[Trade] = []
    buy_iter = iter(buys)
    sell_iter = iter(sells)
    cur_buy = next(buy_iter, None)
    cur_sell = next(sell_iter, None)
    buy_remaining = cur_buy[1] if cur_buy else 0
    sell_remaining = cur_sell[1] if cur_sell else 0

    while cur_buy and cur_sell:
        if buy_remaining == 0:
            cur_buy = next(buy_iter, None)
            buy_remaining = cur_buy[1] if cur_buy else 0
            continue
        if sell_remaining == 0:
            cur_sell = next(sell_iter, None)
            sell_remaining = cur_sell[1] if cur_sell else 0
            continue

        qty = min(buy_remaining, sell_remaining)
        trades.append(
            Trade(
                round_index=cur_buy[0].round_index,
                buyer_id=cur_buy[0].agent_id,
                seller_id=cur_sell[0].agent_id,
                quantity=qty,
                price=price,
            )
        )
        buy_remaining -= qty
        sell_remaining -= qty

    return trades
