"""Unit tests for the call-auction matching engine."""

from __future__ import annotations

import pytest

from src.arena.matching import CallAuctionExchange
from src.arena.types import Order, OrderSide


def _buy(agent_id: str, qty: int, price: float, round_index: int = 0) -> Order:
    return Order(
        round_index=round_index,
        agent_id=agent_id,
        side=OrderSide.BUY,
        quantity=qty,
        limit_price=price,
    )


def _sell(agent_id: str, qty: int, price: float, round_index: int = 0) -> Order:
    return Order(
        round_index=round_index,
        agent_id=agent_id,
        side=OrderSide.SELL,
        quantity=qty,
        limit_price=price,
    )


@pytest.fixture
def exchange() -> CallAuctionExchange:
    return CallAuctionExchange()


def test_no_orders_returns_opening_price(exchange):
    result = exchange.clear([], opening_price=10.0)
    assert result.cleared_volume == 0
    assert result.clearing_price == 10.0
    assert result.trades == ()


def test_only_buys_no_trade(exchange):
    result = exchange.clear([_buy("a", 5, 11.0)], opening_price=10.0)
    assert result.cleared_volume == 0
    assert result.trades == ()


def test_only_sells_no_trade(exchange):
    result = exchange.clear([_sell("a", 5, 9.0)], opening_price=10.0)
    assert result.cleared_volume == 0


def test_non_crossing_no_trade(exchange):
    # Highest buy = 9, lowest sell = 11 → no overlap.
    result = exchange.clear(
        [_buy("a", 3, 9.0), _sell("b", 3, 11.0)],
        opening_price=10.0,
    )
    assert result.cleared_volume == 0


def test_simple_crossing_trade(exchange):
    # Buy 3 @ 11, Sell 3 @ 9 → cross, clear at price that maximizes volume.
    result = exchange.clear(
        [_buy("a", 3, 11.0), _sell("b", 3, 9.0)],
        opening_price=10.0,
    )
    assert result.cleared_volume == 3
    # Clearing price should fall in [9, 11] inclusive of the tie-break anchor.
    assert 9.0 <= result.clearing_price <= 11.0
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.buyer_id == "a"
    assert trade.seller_id == "b"
    assert trade.quantity == 3


def test_clearing_price_maximizes_volume(exchange):
    # Buyer wants 5 @ 12, Seller offers 5 @ 8 → clears 5 shares
    # Buyer wants 10 @ 9, Seller offers 10 @ 11 → no overlap at this band
    result = exchange.clear(
        [_buy("a", 5, 12.0), _buy("c", 10, 9.0),
         _sell("b", 5, 8.0), _sell("d", 10, 11.0)],
        opening_price=10.0,
    )
    # Demand@10 = 5 (only buyer-a since c's limit is 9 < 10), Supply@10 = 5 (b only) → 5
    # Demand@9 = 15, Supply@9 = 5 → 5
    # Demand@11 = 5, Supply@11 = 15 → 5
    # Many prices yield 5; the engine should pick one of them with smallest |D-S|.
    assert result.cleared_volume == 5


def test_partial_fill_rationed(exchange):
    # 10 buyers @ 10, 3 sellers @ 10 — only 3 can fill.
    orders = [_buy(f"b{i}", 1, 10.0) for i in range(10)]
    orders += [_sell("s1", 3, 10.0)]
    result = exchange.clear(orders, opening_price=10.0)
    assert result.cleared_volume == 3
    buyer_ids = [t.buyer_id for t in result.trades]
    assert len(buyer_ids) == 3
    # Earliest agent_ids should fill first under deterministic tie-break.
    assert set(buyer_ids) == {"b0", "b1", "b2"}


def test_strictly_in_the_money_filled_first(exchange):
    # Buyer with limit 12 must fill before buyer with limit 10 when clearing@10.
    result = exchange.clear(
        [_buy("rich", 2, 12.0), _buy("poor", 5, 10.0), _sell("seller", 2, 9.0)],
        opening_price=10.0,
    )
    assert result.cleared_volume == 2
    fills = {t.buyer_id: t.quantity for t in result.trades}
    assert fills.get("rich") == 2
    assert "poor" not in fills or fills.get("poor") == 0


def test_clearing_is_deterministic_across_calls(exchange):
    orders = [_buy("a", 3, 11.0), _buy("b", 2, 11.0), _sell("c", 4, 9.0)]
    r1 = exchange.clear(list(orders), opening_price=10.0)
    r2 = exchange.clear(list(orders), opening_price=10.0)
    assert r1.cleared_volume == r2.cleared_volume
    assert r1.clearing_price == r2.clearing_price
    assert [t.model_dump() for t in r1.trades] == [t.model_dump() for t in r2.trades]
