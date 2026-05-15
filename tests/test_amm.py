"""Unit tests for the constant-product AMM engine."""

from __future__ import annotations

import math

import pytest

from src.arena.matching import POOL_ID, AMMEngine
from src.arena.types import Order, OrderSide


def _buy(agent: str, qty: int, price: float, r: int = 0) -> Order:
    return Order(round_index=r, agent_id=agent, side=OrderSide.BUY, quantity=qty, limit_price=price)


def _sell(agent: str, qty: int, price: float, r: int = 0) -> Order:
    return Order(round_index=r, agent_id=agent, side=OrderSide.SELL, quantity=qty, limit_price=price)


@pytest.fixture
def amm() -> AMMEngine:
    # 1000 coin, 100 shares → spot price 10.0
    return AMMEngine(initial_coin_reserve=1000.0, initial_share_reserve=100)


def test_initial_price_matches_reserves(amm):
    assert amm.spot_price == pytest.approx(10.0)
    assert amm.initial_price() == pytest.approx(10.0)


def test_invalid_reserves_rejected():
    with pytest.raises(ValueError):
        AMMEngine(initial_coin_reserve=0, initial_share_reserve=100)
    with pytest.raises(ValueError):
        AMMEngine(initial_coin_reserve=100, initial_share_reserve=1)


def test_buy_executes_against_pool(amm):
    # Limit well above effective price for a 5-share buy.
    result = amm.clear([_buy("a", 5, 12.0)], opening_price=10.0)
    assert result.cleared_volume == 5
    trade = result.trades[0]
    assert trade.buyer_id == "a"
    assert trade.seller_id == POOL_ID
    # effective price = R_c / (R_s - q) = 1000 / 95 ≈ 10.526
    assert trade.price == pytest.approx(1000 / 95, rel=1e-6)
    # pool moves toward higher price.
    assert amm.spot_price > 10.0


def test_sell_executes_against_pool(amm):
    result = amm.clear([_sell("a", 5, 9.0)], opening_price=10.0)
    assert result.cleared_volume == 5
    trade = result.trades[0]
    assert trade.buyer_id == POOL_ID
    assert trade.seller_id == "a"
    # effective price = R_c / (R_s + q) = 1000 / 105 ≈ 9.524
    assert trade.price == pytest.approx(1000 / 105, rel=1e-6)
    assert amm.spot_price < 10.0


def test_buy_capped_by_limit(amm):
    # limit 10.0 means we want effective price ≤ 10.0
    # R_c / (R_s - q) ≤ 10  →  1000 / (100 - q) ≤ 10  →  q ≤ 0
    # So no fill at all.
    result = amm.clear([_buy("a", 10, 10.0)], opening_price=10.0)
    assert result.cleared_volume == 0
    assert result.trades == ()


def test_buy_partial_fill_at_tight_limit(amm):
    # limit 10.5  →  1000 / (100 - q) ≤ 10.5  →  q ≤ 100 - 95.238 = 4.76  →  4
    result = amm.clear([_buy("a", 10, 10.5)], opening_price=10.0)
    assert result.cleared_volume == 4
    assert result.trades[0].price <= 10.5


def test_sell_capped_by_limit(amm):
    # limit 10.0  →  R_c / (R_s + q) ≥ 10  →  q ≤ 1000/10 - 100 = 0
    result = amm.clear([_sell("a", 10, 10.0)], opening_price=10.0)
    assert result.cleared_volume == 0


def test_constant_product_invariant_holds_approximately(amm):
    """k should stay close to its initial value (drift only from integer rounding)."""
    k0 = amm.coin_reserve * amm.share_reserve
    amm.clear(
        [_buy("a", 3, 12.0), _sell("b", 2, 9.0), _buy("c", 1, 12.0)],
        opening_price=10.0,
    )
    k1 = amm.coin_reserve * amm.share_reserve
    # AMM math is exact for continuous quantities; integer shares introduce a
    # small drift. Bound it loosely.
    assert math.isclose(k0, k1, rel_tol=0.05)


def test_pool_never_drains_completely(amm):
    # Massive buy that would otherwise consume the pool.
    result = amm.clear([_buy("greedy", 999_999, 1e9)], opening_price=10.0)
    assert amm.share_reserve >= 1
    assert result.cleared_volume <= 99  # leaves at least 1 share in the pool


def test_multiple_orders_execute_sequentially(amm):
    # Buy then sell — with NO fees, a buy followed by an equal-size sell is a
    # perfect round-trip and the pool returns to its original state.
    result = amm.clear(
        [_buy("a", 5, 12.0), _sell("b", 5, 9.0)],
        opening_price=10.0,
    )
    assert result.cleared_volume == 10  # 5 + 5
    assert amm.spot_price == pytest.approx(10.0)
    # With no fees + identical-size mirror trades, the effective price is the
    # same on both sides (1000/95 going up == 1052.63/100 coming back).
    buy_trade = next(t for t in result.trades if t.buyer_id == "a")
    sell_trade = next(t for t in result.trades if t.seller_id == "b")
    assert buy_trade.price == pytest.approx(sell_trade.price, rel=1e-9)
    # And both are above the resting spot price of 10 because of slippage.
    assert buy_trade.price > 10.0


def test_no_orders_returns_spot_price(amm):
    result = amm.clear([], opening_price=10.0)
    assert result.cleared_volume == 0
    assert result.clearing_price == pytest.approx(10.0)


def test_determinism_under_repeated_calls():
    # Two fresh engines with the same orders should yield identical pool state.
    a = AMMEngine(initial_coin_reserve=1000.0, initial_share_reserve=100)
    b = AMMEngine(initial_coin_reserve=1000.0, initial_share_reserve=100)
    orders = [_buy("x", 3, 12.0), _sell("y", 2, 9.0), _buy("z", 1, 11.0)]
    ra = a.clear(orders, opening_price=10.0)
    rb = b.clear(orders, opening_price=10.0)
    assert a.coin_reserve == b.coin_reserve
    assert a.share_reserve == b.share_reserve
    assert [t.model_dump() for t in ra.trades] == [t.model_dump() for t in rb.trades]


def test_metadata_includes_reserves(amm):
    result = amm.clear([_buy("a", 2, 12.0)], opening_price=10.0)
    md = result.metadata
    assert md["coin_reserve"] == pytest.approx(amm.coin_reserve)
    assert md["share_reserve"] == amm.share_reserve
    assert "k" in md
