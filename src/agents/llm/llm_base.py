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
    del total_rounds  # 故意不告诉 agent 总轮数 —— 见下方"长周期"说明
    return f"""你是一个博弈论专家。

【世界规则】
- 共有 {num_traders} 名交易员相互竞争。市场中只有一只可交易股票，名为
  WORLD；只有一种货币，名为 GameCoin（GC）。
- 每名交易员初始持有 {initial_coin:g} GC 和 {initial_shares} 股
  （因此初始个人净值 = 初始价格 × 持股数 + 现金）。
- 每一轮每名交易员独立地从以下三种动作中选择一个：
    BUY  —— 提交买入限价单（数量 + 可接受的最高价）
    SELL —— 提交卖出限价单（数量 + 可接受的最低价）
    HOLD —— 不操作
- 所有订单在每轮结束时统一揭示并撮合。你可以看到价格的变化，但是看不到其他人的订单、持仓、
  决策或身份。
- 一个订单要么按你提交的数量全额成交，要么本轮 0 成交（不存在部分成交）。

【你的目标 —— 成为世界首富】
- 你的目标是理解交易的规则，利用低价买入高价卖出，在游戏的长周期中持续增长你的总财富，成为博弈论世界的首富（现金 + 持股市值）。把每一轮都看作一盘长棋中的一手。

【输出格式（严格 JSON，单个对象，不要 markdown、不要任何额外说明）】
{{
  "action": "BUY" | "SELL" | "HOLD",
  "quantity": <整数，HOLD 时为 0>,
  "limit_price": <数字，HOLD 时为 0>,
  "rationale": "<一句简短的中文理由>"
}}

【约束】
- BUY：quantity * limit_price 必须 ≤ 你当前的现金。
- SELL：quantity 必须 ≤ 你当前的持股数。
- quantity 必须是非负整数；limit_price 必须为正数。
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
【撮合引擎】Uniswap V2 风格的自动做市商（AMM），批量统一定价。零手续费。

【初始状态】
- 流动池初始储备：R_c = {amm_coin_reserve:g} GC，R_s = {amm_share_reserve} 股。
- 初始现货价 = R_c / R_s = {spot:.4f} GC/股。
- 整局游戏的系统总量恒定：
    现金：{total_cash:g} GC = {num_traders} 名交易员 × {initial_coin:g} + 流动池 {amm_coin_reserve:g}
    股票：{total_shares}    = {num_traders} 名交易员 × {initial_shares} + 流动池 {amm_share_reserve}

【核心机制】
- 所有交易都对手共享流动池 —— 交易员之间不存在直接撮合，流动池是每一笔
  成交的隐含对手方。
- 当前市场价 = R_c / R_s。
- 同一轮内所有订单**同时密封提交、同时清算**，**没有任何执行顺序**。
  你无法靠"先提交"抢跑，也不会被同一轮其他人的下单冲击到。

【出清算法（每轮只算一次，不迭代）】
1. 用本轮所有提交订单计算净流量：
       Δ = Σ 买入数量 − Σ 卖出数量
2. 计算唯一的统一出清价：
       P* = R_c / (R_s − Δ)
   （即：池子吸收净流量 Δ 之后的新现货价）
3. 一次性筛选订单（依据本轮初始 P*）：
       BUY  保留：limit_price ≥ P*
       SELL 保留：limit_price ≤ P*
   不满足者本轮直接作废，**不重新计算 P***。
4. 所有幸存订单按数量**全额成交**（不存在部分成交），统一以 P* 结算：
       买家支付 q · P* GC，得 q 股
       卖家收到 q · P* GC，给 q 股
5. 池子最终更新（按筛选后实际净流量 Δ_kept）：
       R_c ← R_c + Δ_kept · P*
       R_s ← R_s − Δ_kept

【你的 limit_price 的含义】
- 是参与门槛，不是你的成交价。所有成交者都按同一个 P* 结算。
- 限价过紧（买太低 / 卖太高）→ 本轮被淘汰，0 成交。
- 限价宽松 → 必定成交，但你可能获得比限价更好的实际价格 P*。

【策略含义】
- 一轮内净买入大 → P* 上推；净卖出大 → P* 下压。所有人面对的是同一个
  公开历史，要预判其他交易员可能怎么动。
- 限价是二元的："过门槛 → 全额"或"不过门槛 → 0"，不存在"成交一半"的情况。
- 每轮的实时 (R_c, R_s) 会在用户消息的"流动池"字段中提供给你。

"""


def _format_call_auction(*, num_traders: int) -> str:
    return f"""
【撮合引擎】集合竞价（批量撮合，全 or 不成）。

【核心机制】
- 每轮收集全部 {num_traders} 名交易员的订单，按"使成交量最大化"的统一
  出清价 P* 一次性出清。
- 所有 limit_price ≥ P* 的 BUY 订单**全额**按 P* 成交。
- 所有 limit_price ≤ P* 的 SELL 订单**全额**按 P* 成交。
- 不存在"部分成交"——若某一侧总量超出对手侧总量，按优先级（买价从高到
  低、卖价从低到高，再按 agent_id）逐单剔除最低优先级订单，直到两侧总
  量相等为止。被剔除的订单本轮 0 成交。
- 同一轮内时间先后没有任何优先级。
- 未成交订单在本轮结束时撤销（不留挂单簿）。

【破平规则】
- 若多个价格产生相同的最大成交量，引擎优先选 |需求 − 供给| 最小的那个，
  再优先选与上一轮出清价最接近的那个。

【策略含义】
- 你的 limit_price 是"是否参与 + 优先级排序"的依据：买价越高、卖价越低，
  越不容易被剔除。
- 一旦入选，就按 P* 全额成交；不存在"成交一半"的情况。
- 当供需大体平衡时，出清价会锚定在上一轮价格附近 —— 想打破这个锚需要
  明显的方向性单边流量。
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
        return "（暂无历史轮次）"
    tail = history[-window:]
    return ", ".join(f"{p:.4f}" for p in tail)


def _build_user_prompt(view: MarketView) -> str:
    history_str = _fmt_price_history(view.price_history)
    pool_line = ""
    if view.pool_reserves is not None:
        r_c, r_s = view.pool_reserves
        pool_line = f"流动池：R_c = {r_c:.4f} GC，R_s = {r_s} 股\n"

    return (
        # 每轮用户消息：轮次序号（只是计数器 —— 总轮数故意隐藏）、自己的现金、
        # 自己的持股、当前价格，以及 AMM 引擎下的实时流动池状态。
        f"=== 本轮信息 ===\n"
        f"轮次序号：{view.round_index + 1}\n"
        f"你的现金：{view.portfolio.coin:.4f} GC\n"
        f"你的持股：{view.portfolio.shares} 股\n"
        f"当前市场价：{view.current_price:.4f} GC\n"
        f"{pool_line}"
        "\n"
        f"=== 公开价格历史（由旧到新）===\n"
        f"{history_str}\n\n"
        "请按规定输出一个 JSON 对象作为回复。"
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
