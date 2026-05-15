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
    # Solo BUY 5: net Δ = 5 → P* = R_c/(R_s − Δ) = 1000/95 ≈ 10.526.
    # Post-trade pool: R_c' = R_c·R_s/(R_s−Δ) = 1052.63, R_s' = 95.
    # k preserved (1000·100 = 1052.63·95 ≈ 100000); post-trade spot > P*.
    result = amm.clear([_buy("a", 5, 12.0)], opening_price=10.0)
    assert result.cleared_volume == 5
    trade = result.trades[0]
    assert trade.buyer_id == "a"
    assert trade.seller_id == POOL_ID
    assert trade.price == pytest.approx(1000 / 95, rel=1e-6)
    assert amm.spot_price > 10.0
    # k is preserved by uniform-price batch settlement.
    assert math.isclose(amm.coin_reserve * amm.share_reserve, 100_000.0, rel_tol=1e-9)


def test_sell_executes_against_pool(amm):
    # Solo SELL 5: net Δ = -5 → P* = 1000/(100−(−5)) = 1000/105 ≈ 9.524.
    result = amm.clear([_sell("a", 5, 9.0)], opening_price=10.0)
    assert result.cleared_volume == 5
    trade = result.trades[0]
    assert trade.buyer_id == POOL_ID
    assert trade.seller_id == "a"
    assert trade.price == pytest.approx(1000 / 105, rel=1e-6)
    assert amm.spot_price < 10.0
    assert math.isclose(amm.coin_reserve * amm.share_reserve, 100_000.0, rel_tol=1e-9)


def test_buy_dropped_when_limit_below_clearing(amm):
    # net Δ = 10 → P* = 1000/90 ≈ 11.11. Buyer's limit 10 < P* → dropped.
    result = amm.clear([_buy("a", 10, 10.0)], opening_price=10.0)
    assert result.cleared_volume == 0
    assert result.trades == ()


def test_buy_dropped_when_limit_still_below_clearing(amm):
    # net Δ = 10 → P* = 1000/90 ≈ 11.11. Buyer's limit 10.5 < P* → dropped.
    result = amm.clear([_buy("a", 10, 10.5)], opening_price=10.0)
    assert result.cleared_volume == 0
    assert result.trades == ()


def test_sell_dropped_when_limit_above_clearing(amm):
    # net Δ = -10 → P* = 1000/110 ≈ 9.09. Seller wants ≥ 10 → dropped.
    result = amm.clear([_sell("a", 10, 10.0)], opening_price=10.0)
    assert result.cleared_volume == 0


def test_constant_product_invariant_when_all_orders_fill(amm):
    """When every submitted order is kept, k is preserved exactly because
    P* = R_c / (R_s − Δ) by construction."""
    k0 = amm.coin_reserve * amm.share_reserve
    # Sized so net Δ = 2, P* = 1000/98 ≈ 10.204, all limits satisfy it.
    amm.clear(
        [_buy("a", 3, 12.0), _sell("b", 2, 9.0), _buy("c", 1, 12.0)],
        opening_price=10.0,
    )
    k1 = amm.coin_reserve * amm.share_reserve
    assert math.isclose(k0, k1, rel_tol=1e-9)


def test_pool_voids_round_when_capacity_exceeded(amm):
    # Massive buy that would drain the pool → P* infinite, round voided.
    result = amm.clear([_buy("greedy", 999_999, 1e9)], opening_price=10.0)
    assert amm.share_reserve == 100  # unchanged
    assert amm.coin_reserve == pytest.approx(1000.0)  # unchanged
    assert result.cleared_volume == 0
    assert result.trades == ()


def test_buy_and_sell_clear_at_uniform_price(amm):
    # net Δ = 5 - 5 = 0 → P* = R_c / R_s = 10.0 exactly. Both fill at 10.0.
    result = amm.clear(
        [_buy("a", 5, 12.0), _sell("b", 5, 9.0)],
        opening_price=10.0,
    )
    assert result.cleared_volume == 10  # 5 buy + 5 sell
    assert amm.spot_price == pytest.approx(10.0)
    # Uniform price → both legs trade at the same P* = 10.0.
    buy_trade = next(t for t in result.trades if t.buyer_id == "a")
    sell_trade = next(t for t in result.trades if t.seller_id == "b")
    assert buy_trade.price == pytest.approx(10.0, rel=1e-9)
    assert sell_trade.price == pytest.approx(10.0, rel=1e-9)


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
