"""The World: holds the round loop, the matching engine, and per-agent state.

Sequence per round:
    1. Build a MarketView for each agent containing public state + their
       private portfolio.
    2. Ask each agent for an AgentDecision (in parallel via thread pool).
    3. Validate the decision (sufficient coin / shares for the limit order).
       Reject invalid orders and record a warning. The agent's intent for that
       round becomes HOLD.
    4. Pass valid orders to the configured :class:`MatchingEngine`.
    5. Apply trades to portfolios (skipping the AMM ``POOL`` counterparty),
       emit a RoundReport.

The World instance is reusable across rounds, but each call to `step` advances
one round.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from ..agents.base import Agent, MarketView
from .matching import (
    AMMEngine,
    CallAuctionExchange,
    ClearingResult,
    MatchingEngine,
    POOL_ID,
)
from .types import (
    AgentDecision,
    Order,
    OrderSide,
    PortfolioState,
    PriceTick,
    RoundReport,
    Trade,
)

log = logging.getLogger("arena.world")

MatchingMode = Literal["call_auction", "amm"]


@dataclass(frozen=True)
class WorldConfig:
    """Per-simulation parameters.

    The default endowment is 100 coin + 10 shares at price 10 GC, which gives
    every agent equal starting equity of 200 GC and ensures shares exist in
    circulation from round 0 — otherwise no SELL orders can be submitted and
    nothing would ever trade.

    For ``matching_mode="amm"`` agents trade against a constant-product pool;
    pre-allocating shares to each agent is optional but the pool reserves must
    be positive.
    """

    initial_coin: float = 100.0
    initial_shares: int = 10
    initial_price: float = 10.0
    total_rounds: int = 100
    parallel_decisions: bool = True
    decision_workers: int = 5
    matching_mode: MatchingMode = "amm"
    amm_coin_reserve: float = 1000.0
    amm_share_reserve: int = 100


@dataclass(frozen=True)
class WorldResult:
    config: WorldConfig
    round_reports: list[RoundReport]
    final_portfolios: list[PortfolioState]
    final_price: float

    def leaderboard(self) -> list[tuple[str, float]]:
        scored = [(p.agent_id, p.equity(self.final_price)) for p in self.final_portfolios]
        return sorted(scored, key=lambda kv: kv[1], reverse=True)


def _build_engine(config: WorldConfig) -> MatchingEngine:
    if config.matching_mode == "call_auction":
        return CallAuctionExchange()
    if config.matching_mode == "amm":
        return AMMEngine(
            initial_coin_reserve=config.amm_coin_reserve,
            initial_share_reserve=config.amm_share_reserve,
        )
    raise ValueError(f"unknown matching_mode: {config.matching_mode}")


@dataclass
class World:
    """Mutable holder for one simulation run."""

    config: WorldConfig
    agents: list[Agent]
    engine: MatchingEngine | None = None
    _portfolios: dict[str, PortfolioState] = field(default_factory=dict, init=False)
    _price_history: list[PriceTick] = field(default_factory=list, init=False)
    _reports: list[RoundReport] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("World needs at least one agent")
        if len({a.agent_id for a in self.agents}) != len(self.agents):
            raise ValueError("Agent ids must be unique")
        if any(a.agent_id == POOL_ID for a in self.agents):
            raise ValueError(f"agent_id {POOL_ID!r} is reserved for the AMM pool")
        if self.engine is None:
            self.engine = _build_engine(self.config)
        for agent in self.agents:
            self._portfolios[agent.agent_id] = PortfolioState(
                agent_id=agent.agent_id,
                coin=self.config.initial_coin,
                shares=self.config.initial_shares,
            )
        opening = self.engine.initial_price() or self.config.initial_price
        self._price_history.append(
            PriceTick(round_index=-1, price=opening, volume=0)
        )

    # ---- accessors ----

    @property
    def current_price(self) -> float:
        return self._price_history[-1].price

    @property
    def round_index(self) -> int:
        return len(self._reports)

    def portfolio(self, agent_id: str) -> PortfolioState:
        return self._portfolios[agent_id]

    def price_history(self) -> tuple[float, ...]:
        return tuple(t.price for t in self._price_history)

    # ---- per-round step ----

    def step(self) -> RoundReport:
        if self.round_index >= self.config.total_rounds:
            raise RuntimeError("simulation already finished")

        round_index = self.round_index
        opening_price = self.current_price
        history = self.price_history()

        assert self.engine is not None
        pool_reserves = self.engine.pool_state()
        views: dict[str, MarketView] = {
            agent.agent_id: MarketView(
                round_index=round_index,
                total_rounds=self.config.total_rounds,
                current_price=opening_price,
                price_history=history,
                portfolio=self._portfolios[agent.agent_id],
                pool_reserves=pool_reserves,
            )
            for agent in self.agents
        }

        decisions: dict[str, AgentDecision] = self._collect_decisions(views)
        orders: list[Order] = []
        for agent_id, decision in decisions.items():
            order = self._to_order(agent_id, round_index, decision)
            if order is not None:
                orders.append(order)

        assert self.engine is not None  # for type checkers — set in __post_init__
        result = self.engine.clear(orders, opening_price=opening_price)
        self._apply_trades(result.trades)

        clearing_price = result.clearing_price if result.cleared_volume > 0 else opening_price
        self._price_history.append(
            PriceTick(
                round_index=round_index,
                price=clearing_price,
                volume=result.cleared_volume,
            )
        )

        report = RoundReport(
            round_index=round_index,
            opening_price=opening_price,
            clearing_price=clearing_price,
            cleared_volume=result.cleared_volume,
            orders=orders,
            trades=list(result.trades),
            portfolios=[self._portfolios[a.agent_id] for a in self.agents],
            rationales={a: d.rationale for a, d in decisions.items()},
        )
        self._reports.append(report)

        # Compute per-agent fills so each agent gets only its own trade history.
        fills_by_agent: dict[str, list[Trade]] = {a.agent_id: [] for a in self.agents}
        for trade in result.trades:
            if trade.buyer_id in fills_by_agent:
                fills_by_agent[trade.buyer_id].append(trade)
            if trade.seller_id in fills_by_agent:
                fills_by_agent[trade.seller_id].append(trade)

        # Per-agent post-round hook.
        post_reserves = self.engine.pool_state()
        for agent in self.agents:
            view = MarketView(
                round_index=round_index,
                total_rounds=self.config.total_rounds,
                current_price=clearing_price,
                price_history=self.price_history(),
                portfolio=self._portfolios[agent.agent_id],
                pool_reserves=post_reserves,
            )
            agent.on_round_end(
                view,
                decisions[agent.agent_id],
                tuple(fills_by_agent[agent.agent_id]),
            )

        return report

    def run(self) -> WorldResult:
        while self.round_index < self.config.total_rounds:
            self.step()
        return WorldResult(
            config=self.config,
            round_reports=list(self._reports),
            final_portfolios=[self._portfolios[a.agent_id] for a in self.agents],
            final_price=self.current_price,
        )

    # ---- internals ----

    def _collect_decisions(
        self, views: dict[str, MarketView]
    ) -> dict[str, AgentDecision]:
        """Ask every agent for a decision, optionally in parallel.

        Parallelism is keyed on `config.parallel_decisions`. Sequential mode
        keeps things deterministic for tests; threaded mode is essential when
        agents make blocking network calls (LLM mode).
        """

        def _ask(agent: Agent) -> tuple[str, AgentDecision]:
            try:
                return agent.agent_id, agent.decide(views[agent.agent_id])
            except Exception:
                log.exception("Agent %s crashed; treating as HOLD", agent.agent_id)
                return agent.agent_id, AgentDecision(action="HOLD", rationale="agent crashed")

        if self.config.parallel_decisions and len(self.agents) > 1:
            workers = min(self.config.decision_workers, len(self.agents))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_ask, self.agents))
        else:
            results = [_ask(a) for a in self.agents]

        if len(results) != len(self.agents):
            raise RuntimeError(
                f"decision phase returned {len(results)} results for "
                f"{len(self.agents)} agents — possible thread failure"
            )

        decisions: dict[str, AgentDecision] = {}
        for agent_id, decision in results:
            if agent_id in decisions:
                # Should be impossible because __post_init__ rejects duplicates,
                # but guarding here makes the invariant explicit.
                raise RuntimeError(f"duplicate decision for agent_id {agent_id!r}")
            decisions[agent_id] = decision
        return {a.agent_id: decisions[a.agent_id] for a in self.agents}

    def _to_order(
        self, agent_id: str, round_index: int, decision: AgentDecision
    ) -> Order | None:
        if not decision.is_trade():
            return None

        if decision.quantity <= 0 or decision.limit_price <= 0:
            log.warning(
                "Agent %s submitted non-positive order: %s — dropped",
                agent_id,
                decision,
            )
            return None

        portfolio = self._portfolios[agent_id]
        if decision.action == "BUY":
            cost = decision.quantity * decision.limit_price
            if portfolio.coin + 1e-9 < cost:
                log.info(
                    "Agent %s BUY %d @ %.4f rejected — insufficient coin (have %.4f, need %.4f)",
                    agent_id,
                    decision.quantity,
                    decision.limit_price,
                    portfolio.coin,
                    cost,
                )
                return None
            return Order(
                round_index=round_index,
                agent_id=agent_id,
                side=OrderSide.BUY,
                quantity=decision.quantity,
                limit_price=decision.limit_price,
            )

        if decision.action == "SELL":
            if portfolio.shares < decision.quantity:
                log.info(
                    "Agent %s SELL %d rejected — insufficient shares (have %d)",
                    agent_id,
                    decision.quantity,
                    portfolio.shares,
                )
                return None
            return Order(
                round_index=round_index,
                agent_id=agent_id,
                side=OrderSide.SELL,
                quantity=decision.quantity,
                limit_price=decision.limit_price,
            )

        return None

    def _apply_trades(self, trades: tuple[Trade, ...]) -> None:
        for trade in trades:
            if trade.buyer_id != POOL_ID:
                buyer = self._portfolios[trade.buyer_id]
                self._portfolios[trade.buyer_id] = buyer.credit(
                    coin=-trade.notional, shares=trade.quantity
                )
            if trade.seller_id != POOL_ID:
                seller = self._portfolios[trade.seller_id]
                self._portfolios[trade.seller_id] = seller.credit(
                    coin=trade.notional, shares=-trade.quantity
                )

    @property
    def reports(self) -> list[RoundReport]:
        return list(self._reports)

    @property
    def last_clearing(self) -> ClearingResult | None:
        if not self._reports:
            return None
        last = self._reports[-1]
        return ClearingResult(
            clearing_price=last.clearing_price,
            cleared_volume=last.cleared_volume,
            trades=tuple(last.trades),
        )
