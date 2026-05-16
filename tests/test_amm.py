"""Unit tests for the constant-product AMM engine (sequential per-order)."""

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
    # Solo BUY 5 against (1000, 100): effective price = 1000/(100-5) = 10.526.
    # Post-trade pool: R_c' = 1000 + 5·10.526 = 1052.63, R_s' = 95.
    # k preserved exactly (1052.63·95 = 100000).
    result = amm.clear([_buy("a", 5, 12.0)], opening_price=10.0)
    assert result.cleared_volume == 5
    trade = result.trades[0]
    assert trade.buyer_id == "a"
    assert trade.seller_id == POOL_ID
    assert trade.price == pytest.approx(1000 / 95, rel=1e-6)
    assert amm.spot_price > 10.0
    assert math.isclose(amm.coin_reserve * amm.share_reserve, 100_000.0, rel_tol=1e-9)


def test_sell_executes_against_pool(amm):
    # Solo SELL 5: effective price = 1000/(100+5) = 9.524.
    result = amm.clear([_sell("a", 5, 9.0)], opening_price=10.0)
    assert result.cleared_volume == 5
    trade = result.trades[0]
    assert trade.buyer_id == POOL_ID
    assert trade.seller_id == "a"
    assert trade.price == pytest.approx(1000 / 105, rel=1e-6)
    assert amm.spot_price < 10.0
    assert math.isclose(amm.coin_reserve * amm.share_reserve, 100_000.0, rel_tol=1e-9)


def test_buy_dropped_when_limit_at_spot(amm):
    # BUY at limit = current spot is impossible — any positive quantity needs
    # a price above spot, so q_max = floor(100 − 1000/10) = 0 → dropped.
    result = amm.clear([_buy("a", 10, 10.0)], opening_price=10.0)
    assert result.cleared_volume == 0
    assert result.trades == ()


def test_buy_partial_fill_when_limit_binds(amm):
    # BUY 10 limit 10.5: q_max = floor(100 − 1000/10.5) = floor(4.76) = 4.
    # Effective price for 4 shares = 1000/96 ≈ 10.417 ≤ 10.5 ✓
    # Pool moves to (1041.67, 96), k preserved.
    result = amm.clear([_buy("a", 10, 10.5)], opening_price=10.0)
    assert result.cleared_volume == 4
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.quantity == 4
    assert trade.price == pytest.approx(1000 / 96, rel=1e-6)
    assert trade.price <= 10.5 + 1e-9
    assert math.isclose(amm.coin_reserve * amm.share_reserve, 100_000.0, rel_tol=1e-9)


def test_sell_dropped_when_limit_above_post_trade_spot(amm):
    # SELL 10 limit 10.0: q_max = floor(1000/10 − 100) = 0 → dropped (can't
    # sell anything without pushing the price below the limit).
    result = amm.clear([_sell("a", 10, 10.0)], opening_price=10.0)
    assert result.cleared_volume == 0


def test_constant_product_preserved_across_sequential_trades(amm):
    # Multiple orders mixed across sides; each fill exactly preserves k
    # (modulo integer-share rounding, which is negligible at these scales).
    k0 = amm.coin_reserve * amm.share_reserve
    amm.clear(
        [_buy("a", 3, 12.0), _sell("b", 2, 9.0), _buy("c", 1, 12.0)],
        opening_price=10.0,
    )
    k1 = amm.coin_reserve * amm.share_reserve
    # Integer-share rounding leaves a sub-permille drift at most.
    assert math.isclose(k0, k1, rel_tol=1e-3)


def test_same_side_cohort_pays_two_distinct_prices(amm):
    # Two buyers in one round: whoever the shuffle puts first fills against
    # the initial pool, the other against the post-first-fill pool. We don't
    # assert which agent wins the better price — that depends on the SHA-256
    # of (round_index, agent_id) and rotates per round.
    result = amm.clear(
        [_buy("a", 5, 20.0), _buy("b", 5, 20.0)],
        opening_price=10.0,
    )
    prices = sorted(t.price for t in result.trades)
    # The cheaper buyer fills at 1000/(100-5) = 10.526; the second buys
    # against pool (1052.63, 95) at 1052.63/90 = 11.696.
    assert prices[0] == pytest.approx(1000 / 95, rel=1e-6)
    assert prices[1] == pytest.approx(1052.6315789 / 90, rel=1e-3)
    assert prices[0] < prices[1]


def test_priority_agent_always_runs_first():
    # priority_agent_ids pins those agents ahead of everyone else, regardless
    # of the hash. Within the priority tier and within the non-priority tier
    # the hash still rotates per round.
    for r in range(8):
        engine = AMMEngine(
            initial_coin_reserve=1000.0,
            initial_share_reserve=100,
            priority_agent_ids={"noise"},
        )
        orders = [
            _buy("noise", 1, 20.0, r=r),
            _buy("alpha", 1, 20.0, r=r),
            _buy("beta", 1, 20.0, r=r),
        ]
        result = engine.clear(orders, opening_price=10.0)
        # Noise always fills against the pristine pool ⇒ cheapest buy price.
        cheapest = min(result.trades, key=lambda t: t.price)
        assert cheapest.buyer_id == "noise", f"round {r}: {cheapest.buyer_id} got cheapest"


def test_per_round_order_rotates_with_round_index():
    # The shuffle key is SHA-256(round_index, agent_id), so an agent's
    # position is not pinned by id. Across many rounds with the same agent
    # roster, the "first slot" must land on each agent at least once.
    first_buyers: set[str] = set()
    for r in range(20):
        engine = AMMEngine(initial_coin_reserve=1000.0, initial_share_reserve=100)
        result = engine.clear(
            [_buy("a", 5, 20.0, r=r), _buy("b", 5, 20.0, r=r)],
            opening_price=10.0,
        )
        cheaper = min(result.trades, key=lambda t: t.price).buyer_id
        first_buyers.add(cheaper)
    assert first_buyers == {"a", "b"}


def test_pool_capacity_caps_oversized_buy(amm):
    # Massive buy: pool keeps at least 1 share, so qty caps at R_s - 1 = 99.
    # Old behaviour was to void the round; new behaviour fills 99 shares with
    # the buyer's generous limit absorbing whatever slippage is needed.
    result = amm.clear([_buy("greedy", 999_999, 1e9)], opening_price=10.0)
    assert result.cleared_volume == 99
    assert amm.share_reserve == 1  # _MIN_POOL_SHARES
    assert amm.coin_reserve > 1000.0  # took on a lot of coin
    assert math.isclose(amm.coin_reserve * amm.share_reserve, 100_000.0, rel_tol=1e-9)


def test_buy_then_sell_returns_pool_to_start(amm):
    # Symmetric BUY 5 + SELL 5 round-trip: regardless of which order the
    # shuffle picks, the pool is restored to (1000, 100) exactly. Fill prices
    # differ between the two orderings (10.526 if buy first, 9.524 if sell
    # first), so we only assert the pool invariant here.
    result = amm.clear(
        [_buy("a", 5, 12.0), _sell("b", 5, 9.0)],
        opening_price=10.0,
    )
    assert result.cleared_volume == 10
    assert amm.coin_reserve == pytest.approx(1000.0, rel=1e-9)
    assert amm.share_reserve == 100
    # Both fills happen at the same price within a round (round-trip property
    # of two equal-size opposite-side orders against constant-product AMM).
    prices = {t.price for t in result.trades}
    assert len(prices) == 1


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


def test_input_order_does_not_affect_outcome():
    # Sorting is internal, so submitting the same orders in a different list
    # order must still produce the same trades and pool state.
    a = AMMEngine(initial_coin_reserve=1000.0, initial_share_reserve=100)
    b = AMMEngine(initial_coin_reserve=1000.0, initial_share_reserve=100)
    orders = [_buy("a", 3, 12.0), _sell("b", 2, 9.0), _buy("c", 1, 12.0)]
    ra = a.clear(orders, opening_price=10.0)
    rb = b.clear(list(reversed(orders)), opening_price=10.0)
    assert a.coin_reserve == b.coin_reserve
    assert a.share_reserve == b.share_reserve
    assert [t.model_dump() for t in ra.trades] == [t.model_dump() for t in rb.trades]


def test_metadata_includes_reserves(amm):
    result = amm.clear([_buy("a", 2, 12.0)], opening_price=10.0)
    md = result.metadata
    assert md["coin_reserve"] == pytest.approx(amm.coin_reserve)
    assert md["share_reserve"] == amm.share_reserve
    assert "k" in md
    assert "k_drift" in md
    assert abs(md["k_drift"]) < 1e-9
