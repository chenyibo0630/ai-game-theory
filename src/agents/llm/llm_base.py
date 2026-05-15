"""Shared scaffolding for LLM-backed trading agents.

Every LLM agent receives the SAME engine-specific system prompt (world rules
+ win goal + allowed operations + matching-engine details). Each round the
user message tells the agent:

- the current round number (and total rounds)
- its own cash holdings
- its own share holdings
- the current market price
- the public price history of the game so far

The agent's current portfolio already encodes the result of its past trades,
so we deliberately do NOT pass per-trade memory: the current state plus the
public price trajectory contains all the information needed to act. No other
agent's state is ever revealed.

Subclasses implement `_call_provider(system, user) -> str` returning raw model
text; the base class handles prompt construction, JSON parsing with a regex
fallback, decision validation, and graceful HOLD fallback on any failure.
"""

from __future__ import annotations

import json
import logging
import re
from abc import abstractmethod
from dataclasses import dataclass

from ...arena.types import AgentDecision, Trade
from ..base import Agent, MarketView

log = logging.getLogger("agents.llm")


def _format_rules(
    *,
    num_traders: int,
    initial_coin: float,
    initial_shares: int,
    total_rounds: int,  # accepted but intentionally NOT shown to the agent
) -> str:
    del total_rounds  # hidden from the prompt — see "long horizon" note below
    return f"""You are a trader in a closed multi-agent economic game.

WORLD RULES
- {num_traders} traders compete. There is exactly one tradable stock named
  WORLD, and one currency named GameCoin (GC).
- Every trader starts with {initial_coin:g} GC and {initial_shares} shares
  (so the initial per-trader equity equals the initial price × shares + cash).
- The game runs for a number of rounds you are NOT told in advance. Treat
  every round as if many more remain — there is no "last round" you can
  plan around.
- In each round every trader independently picks ONE of three actions:
    BUY  — submit a buy limit order (quantity + max acceptable price)
    SELL — submit a sell limit order (quantity + min acceptable price)
    HOLD — do nothing
- Orders are revealed and matched at the end of each round. You will NEVER
  see anyone else's orders, holdings, decisions, or identity.
- A submitted order may fill fully, partially, or not at all depending on
  what the matching engine produces.

YOUR GOAL — long-run wealth accumulation
- Your objective is to GROW your total wealth (cash + share value) across
  the long horizon of the game through skilful trading. Think of every
  round as one move in a long positional game.
- You will be ranked at the end by total wealth. Because you don't know
  when the game ends, you cannot game any deadline — sustained, well-timed
  decisions across many rounds compound far better than a single large
  move.
- Avoid value-destroying trades. Sells against an AMM pool cost slippage
  AND lower the pool spot for the rest of the game (including for your own
  remaining holdings). Trade because the position is genuinely
  advantageous, not because you feel a clock running out.
- Treat capital preservation as a precondition for compounding: avoid
  trades whose expected post-slippage value is worse than HOLD.

OUTPUT FORMAT (STRICT JSON, single object, no markdown, no commentary)
{{
  "action": "BUY" | "SELL" | "HOLD",
  "quantity": <integer, 0 if HOLD>,
  "limit_price": <number, 0 if HOLD>,
  "rationale": "<one short sentence>"
}}

CONSTRAINTS
- For BUY: quantity * limit_price must be ≤ your current cash.
- For SELL: quantity must be ≤ your current shares.
- quantity must be a non-negative integer; limit_price must be positive.
"""


def _format_amm(
    *,
    num_traders: int,
    initial_coin: float,
    initial_shares: int,
    amm_coin_reserve: float,
    amm_share_reserve: int,
) -> str:
    spot = amm_coin_reserve / amm_share_reserve
    total_cash = num_traders * initial_coin + amm_coin_reserve
    total_shares = num_traders * initial_shares + amm_share_reserve
    return f"""
MATCHING ENGINE: Uniswap V2-style automated market maker (AMM). Zero fees.

INITIAL STATE
- Pool starts with R_c = {amm_coin_reserve:g} GC and R_s = {amm_share_reserve} shares.
- Initial spot price = R_c / R_s = {spot:.4f} GC per share.
- System-wide totals are constant across the game:
    cash:   {total_cash:g} GC   = {num_traders} traders × {initial_coin:g} + pool {amm_coin_reserve:g}
    shares: {total_shares}      = {num_traders} traders × {initial_shares} + pool {amm_share_reserve}

CORE MECHANICS
- All trades happen against the shared liquidity POOL — there is no
  agent-to-agent matching. POOL is the implicit counterparty for every fill.
- Constant-product invariant: R_c * R_s ≈ k.
- Current market price = R_c / R_s.
- For a BUY of q shares the pool receives cash, releases shares:
      Δ_c = q * R_c / (R_s − q)
      effective avg price = R_c / (R_s − q)
- For a SELL of q shares the pool releases cash, receives shares:
      Δ_c = q * R_c / (R_s + q)
      effective avg price = R_c / (R_s + q)
- Zero transaction fee.

PARTIAL FILLS (driven by your limit_price)
- BUY at limit L: max fillable q = floor(R_s − R_c / L). If ≤ 0 → no fill.
- SELL at limit L: max fillable q = floor(R_c / L − R_s). If ≤ 0 → no fill.
- All fills are floored to integer shares.
- The pool always keeps at least 1 share to prevent price divergence, so
  you can never drain it completely.

EXECUTION ORDER WITHIN A ROUND
- All orders in a round execute SEQUENTIALLY against the pool, in a
  deterministic order keyed on agent_id. You cannot front-run by
  submitting earlier — every order is revealed at the same instant.
- If others in the same round buy before your buy, spot has already moved
  up by the time yours executes (and you pay the higher post-impact price).
- Each round the live (R_c, R_s) will be provided to you under "Pool".

STRATEGIC IMPLICATIONS
- Slippage is real: bigger orders pay worse marginal prices. Splitting a
  large position across rounds usually beats one big order.
- limit_price is your slippage protection. Close to spot ⇒ tight discipline
  with risk of partial / no fill; loose ⇒ guaranteed fill at a worse price.
- Buys raise spot, sells lower spot. Anticipate that other traders react to
  the same public price history you do.
- Cash and shares are both productive: cash earns optionality (you can buy
  when prices dip), shares earn appreciation (you participate when price
  drifts up). A balanced book usually compounds better than an all-in tilt.
- Long-game discipline: do not engineer a panic-sell in the last few rounds
  to "lock in" a number. The AMM punishes large terminal sells with steep
  slippage, the spot crash hurts your remaining shares too, and the
  resulting loss of absolute wealth is rarely worth the marginal change in
  ranking. Trade on conviction about value, not on countdown.
"""


def _format_call_auction(*, num_traders: int) -> str:
    return f"""
MATCHING ENGINE: Call auction (batched).

CORE MECHANICS
- All {num_traders} traders' orders are collected each round and cleared at a
  single uniform price P* that maximises traded volume.
- BUY orders with limit_price ≥ P* fill in full at P*.
- SELL orders with limit_price ≤ P* fill in full at P*.
- Orders at limit_price = P* share the residual capacity pro-rata
  (deterministic agent_id tie-break).
- Time priority does NOT matter within a round.
- Unmatched orders are cancelled at round end (no carry-over book).

TIE-BREAK
- If multiple prices produce the same max volume, the engine prefers the
  one with smallest |demand − supply|, then the one closest to the
  previous-round clearing price.

STRATEGIC IMPLICATIONS
- Your limit_price is a participation cutoff, not the price you pay: every
  filled trader pays the SAME P*.
- Aggressive limits (loose) increase your chance of filling; tight limits
  risk missing the clear.
- The clearing price anchors near the previous price when supply and
  demand are roughly balanced — large directional flow is needed to
  break the anchor.
"""


def build_system_prompt(
    matching_mode: str,
    *,
    num_traders: int = 5,
    initial_coin: float = 100.0,
    initial_shares: int = 10,
    total_rounds: int = 100,
    amm_coin_reserve: float = 1000.0,
    amm_share_reserve: int = 100,
) -> str:
    """Compose the engine-specific system prompt with concrete world values."""
    rules = _format_rules(
        num_traders=num_traders,
        initial_coin=initial_coin,
        initial_shares=initial_shares,
        total_rounds=total_rounds,
    )
    mode = matching_mode.lower()
    if mode == "amm":
        return rules + _format_amm(
            num_traders=num_traders,
            initial_coin=initial_coin,
            initial_shares=initial_shares,
            amm_coin_reserve=amm_coin_reserve,
            amm_share_reserve=amm_share_reserve,
        )
    if mode == "call_auction":
        return rules + _format_call_auction(num_traders=num_traders)
    raise ValueError(f"unknown matching_mode: {matching_mode}")


# Convenience default for ad-hoc tests / smoke checks that don't go through
# LLMConfig — uses the default AMM world parameters.
SYSTEM_PROMPT = build_system_prompt("amm")


def _fmt_price_history(history: tuple[float, ...], window: int = 30) -> str:
    """Format the recent slice of public clearing prices for the prompt.

    `view.price_history` always starts with the engine's initial price (a
    PriceTick recorded before round 0). We show up to `window` most-recent
    entries, oldest first.
    """
    if not history:
        return "(no prior rounds)"
    tail = history[-window:]
    return ", ".join(f"{p:.4f}" for p in tail)


def _build_user_prompt(view: MarketView) -> str:
    history_str = _fmt_price_history(view.price_history)
    pool_line = ""
    if view.pool_reserves is not None:
        r_c, r_s = view.pool_reserves
        pool_line = f"Pool: R_c = {r_c:.4f} GC, R_s = {r_s} shares\n"

    return (
        # The per-round system message: round number (counter only — total is
        # intentionally hidden), own cash, own shares, price, plus the live
        # pool state when running on an AMM engine.
        f"=== This round ===\n"
        f"Round number: {view.round_index + 1}\n"
        f"Your cash: {view.portfolio.coin:.4f} GC\n"
        f"Your shares: {view.portfolio.shares}\n"
        f"Current market price: {view.current_price:.4f} GC\n"
        f"{pool_line}"
        "\n"
        f"=== Public price history (oldest → newest) ===\n"
        f"{history_str}\n\n"
        "Reply with one JSON object as specified."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _sanitize_rationale(raw: object) -> str:
    """Strip control characters and clamp length so untrusted model output
    can't break terminal rendering or downstream parsers."""
    return _CTRL_RE.sub(" ", str(raw))[:240]


def _parse_decision(text: str) -> AgentDecision:
    """Extract a JSON object from a model response and validate it."""
    if not text:
        raise ValueError("empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise ValueError(f"no JSON object found in response: {text!r}")
        payload = json.loads(match.group(0))

    action = str(payload.get("action", "HOLD")).upper()
    if action not in {"BUY", "SELL", "HOLD"}:
        raise ValueError(f"invalid action: {action!r}")

    quantity = int(payload.get("quantity", 0) or 0)
    limit_price = float(payload.get("limit_price", 0) or 0)
    rationale = _sanitize_rationale(payload.get("rationale", ""))

    if action == "HOLD":
        quantity = 0
        limit_price = 0.0
    if quantity <= 0 or limit_price <= 0:
        action = "HOLD"
        quantity = 0
        limit_price = 0.0

    return AgentDecision(
        action=action,  # type: ignore[arg-type]
        quantity=quantity,
        limit_price=limit_price,
        rationale=rationale,
    )


@dataclass
class LLMConfig:
    """Per-agent provider config plus the world parameters needed to write a
    well-grounded system prompt (initial endowments, pool reserves, etc.).

    `api_key` is intentionally masked in __repr__ so accidental logging or
    serialisation of this dataclass cannot leak credentials.
    """

    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.4
    max_tokens: int = 256
    timeout: float = 30.0

    # World facts that go into the system prompt.
    matching_mode: str = "amm"
    num_traders: int = 5
    initial_coin: float = 100.0
    initial_shares: int = 10
    total_rounds: int = 100
    amm_coin_reserve: float = 1000.0
    amm_share_reserve: int = 100

    def __repr__(self) -> str:  # security: never reveal the api_key
        return (
            "LLMConfig("
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"matching_mode={self.matching_mode!r}, api_key='***')"
        )


class LLMAgent(Agent):
    """Base for any LLM-backed agent.

    Stateless across rounds — every decision is built solely from the
    MarketView (current portfolio + public price history). This matches the
    world spec: an agent's prior decisions are not given back to it explicitly,
    but the resulting portfolio state and the price trajectory are.
    """

    provider: str = "llm"

    def __init__(
        self,
        agent_id: str,
        config: LLMConfig,
        *,
        display_name: str | None = None,
    ) -> None:
        super().__init__(agent_id, display_name)
        self.config = config
        self._system_prompt = build_system_prompt(
            config.matching_mode,
            num_traders=config.num_traders,
            initial_coin=config.initial_coin,
            initial_shares=config.initial_shares,
            total_rounds=config.total_rounds,
            amm_coin_reserve=config.amm_coin_reserve,
            amm_share_reserve=config.amm_share_reserve,
        )
        # Per-agent logger so file handlers attached externally
        # (via config.yaml `log_file:` field) emit only this agent's prompts,
        # responses, and decisions.
        self.log = logging.getLogger(f"agent.{agent_id}")

    def decide(self, view: MarketView) -> AgentDecision:
        user_prompt = _build_user_prompt(view)
        self.log.debug("round=%d user_prompt:\n%s", view.round_index, user_prompt)
        try:
            raw = self._call_provider(self._system_prompt, user_prompt)
            self.log.debug("round=%d raw_response: %r", view.round_index, raw[:500])
            decision = _parse_decision(raw)
        except Exception as exc:
            # Only the exception class is recorded — `exc` strings can contain
            # the raw model response, which we don't want bleeding into logs.
            self.log.warning(
                "round=%d decision failed (%s) — fallback HOLD",
                view.round_index,
                exc.__class__.__name__,
            )
            log.warning(
                "%s (%s) failed at round %d: %s — HOLD",
                self.agent_id,
                self.provider,
                view.round_index,
                exc.__class__.__name__,
            )
            return AgentDecision(action="HOLD", rationale=f"fallback ({exc.__class__.__name__})")

        clamped = self._clamp_to_portfolio(view, decision)
        self.log.info(
            "round=%d decision=%s qty=%d limit=%.4f rationale=%s",
            view.round_index,
            clamped.action,
            clamped.quantity,
            clamped.limit_price,
            clamped.rationale,
        )
        return clamped

    @abstractmethod
    def _call_provider(self, system: str, user: str) -> str:
        """Return raw model text."""

    @staticmethod
    def _clamp_to_portfolio(view: MarketView, decision: AgentDecision) -> AgentDecision:
        """Bring an over-ambitious order back inside the agent's budget."""
        if decision.action == "BUY":
            max_qty_by_coin = int(view.portfolio.coin // decision.limit_price)
            qty = min(decision.quantity, max_qty_by_coin)
            if qty <= 0:
                return AgentDecision(action="HOLD", rationale="clamped: no budget")
            return decision.model_copy(update={"quantity": qty})

        if decision.action == "SELL":
            qty = min(decision.quantity, view.portfolio.shares)
            if qty <= 0:
                return AgentDecision(action="HOLD", rationale="clamped: no shares")
            return decision.model_copy(update={"quantity": qty})

        return decision
